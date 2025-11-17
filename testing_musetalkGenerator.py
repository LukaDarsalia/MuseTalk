import math
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from typing import List
from einops import rearrange
from musetalk.models.unet import PositionalEncoding


class MuseTalkGenerator(nn.Module):
    """Unified generator that wraps positional encoding, UNet, and VAE decoding."""

    def __init__(
        self,
        *,
        whisper_encoder: nn.Module,
        vae: nn.Module,
        unet: nn.Module,
        pe: nn.Module,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
        video_fps: int = 25,
    ) -> None:
        super().__init__()
        self.whisper_encoder = whisper_encoder
        self.whisper_encoder_device = next(self.whisper_encoder.parameters()).device

        self.vae = vae
        self.vae_device = next(self.vae.parameters()).device
        self.scaling_factor = self.vae.config.scaling_factor
        self.unet = unet
        self.unet_device = next(self.unet.parameters()).device
        self.timesteps = torch.tensor([0], device=self.unet_device)
        self.pe = pe

        self.audio_padding_length_left = audio_padding_length_left
        self.audio_padding_length_right = audio_padding_length_right
        self.video_fps = video_fps

    def decode_latents(self, latents):
        """
        Decode latent variables back into an image.
        :param latents: The latent variables to decode.
        :return: A NumPy array representing the decoded image.
        """
        latents = (1/  self.scaling_factor) * latents
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = image[...,::-1] # RGB to BGR
        return image

    def get_whisper_chunk(
        self,
        whisper_input_features,
        device,
        weight_dtype,
        librosa_length,
        fps=25,
    ):
        audio_feature_length_per_frame = 2 * (self.audio_padding_length_left + self.audio_padding_length_right + 1)
        whisper_feature = []
        # Process multiple 30s mel input features
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(device).to(weight_dtype)
            audio_feats = self.whisper_encoder(input_feature, output_hidden_states=True).hidden_states
            audio_feats = torch.stack(audio_feats, dim=2)
            whisper_feature.append(audio_feats)

        whisper_feature = torch.cat(whisper_feature, dim=1)
        # Trim the last segment to remove padding
        sr = 16000
        audio_fps = 50
        fps = int(fps)
        whisper_idx_multiplier = audio_fps / fps
        num_frames = math.floor((librosa_length / sr) * fps)
        actual_length = math.floor((librosa_length / sr) * audio_fps)
        whisper_feature = whisper_feature[:,:actual_length,...]

        # Calculate padding amount
        padding_nums = math.ceil(whisper_idx_multiplier)
        # Add padding at start and end
        whisper_feature = torch.cat([
            torch.zeros_like(whisper_feature[:, :padding_nums * self.audio_padding_length_left]),
            whisper_feature,
            # Add extra padding to prevent out of bounds
            torch.zeros_like(whisper_feature[:, :padding_nums * 3 * self.audio_padding_length_right])
        ], 1)

        audio_prompts = []
        for frame_index in range(num_frames):
            audio_index = math.floor(frame_index * whisper_idx_multiplier)
            audio_clip = whisper_feature[:, audio_index: audio_index + audio_feature_length_per_frame]
            audio_prompts.append(audio_clip)

        audio_prompts = torch.cat(audio_prompts, dim=0)  # T, 10, 5, 384
        audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')
        return audio_prompts

    def forward(
        self,
        whisper_input_features: List[torch.Tensor],
        latent_inputs: torch.Tensor,
        frame_idx: int,
        librosa_length: int,
    ):
        whisper_chunks = self.get_whisper_chunk(
            whisper_input_features=whisper_input_features,
            device=self.whisper_encoder_device,
            weight_dtype=self.whisper_encoder.dtype,
            librosa_length=librosa_length,
            fps=self.video_fps,
        )

        batch_size = latent_inputs.shape[0]
        context_whisper_chunk = whisper_chunks[frame_idx: frame_idx+batch_size]  # shape [batch_size, 50, 384]
        
        audio_feature_batch = self.pe(context_whisper_chunk.to(self.unet_device))
        latent_batch = latent_inputs.to(device=self.unet_device, dtype=self.unet.dtype)

        pred_latents = self.unet(
            latent_batch,
            self.timesteps,
            encoder_hidden_states=audio_feature_batch,
        ).sample

        pred_latents = pred_latents.to(device=self.vae_device, dtype=self.vae.dtype)
        recon = self.decode_latents(pred_latents)
        return recon



def simple_full_audio_inference(args):
    """Compact full-audio inference that mirrors streaming style without chunking.

    Steps:
    1. Load models & avatar cache
    2. Extract entire audio feature once
    3. Iterate frames in batches using MuseTalkGenerator
    4. Blend & save frames, then mux audio
    """
    import cv2
    import copy
    from tqdm import tqdm
    from musetalk.utils.utils import load_all_model
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.streaming.avatar import Avatar, AvatarConfig, AvatarRuntime
    from musetalk.utils.blending import get_image_blending
    from transformers import WhisperModel

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # 1. Models
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=unet.model.dtype).eval()
    whisper.requires_grad_(False)
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    fp = FaceParsing(left_cheek_width=args.left_cheek_width, right_cheek_width=args.right_cheek_width) if args.version == "v15" else FaceParsing()
    timesteps = torch.tensor([0], device=device)

    avatar_runtime = AvatarRuntime(
        device=device,
        vae=vae,
        unet=unet,
        pe=pe,
        audio_processor=audio_processor,
        whisper=whisper,
        face_parser=fp,
        timesteps=timesteps,
        weight_dtype=unet.model.dtype,
    )
    avatar_config = AvatarConfig(
        version=args.version,
        result_root=args.result_dir,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        skip_save_images=True,
        fps=args.fps,
        batch_size=args.batch_size,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )

    # 2. Avatar cache (no recreation)
    avatar_path_override = os.path.join(args.avatar_root, args.avatar_id) if args.avatar_root else None
    avatar = Avatar(
        avatar_id=args.avatar_id,
        video_path=None,  # Use cached frames/latents only
        bbox_shift=0 if args.version == "v15" else args.bbox_shift,
        preparation=False,
        config=avatar_config,
        runtime=avatar_runtime,
        interactive=False,
        force_recreate=False,
        avatar_path_override=avatar_path_override,
    )
    print(f"Avatar loaded: {args.avatar_id} | cached frames cycle length = {avatar.prepared_length()}\n")

    # 3. Audio feature extraction (entire file)
    print("Extracting full audio features ...")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(
        args.audio_path,
        weight_dtype=unet.model.dtype,
    )
    total_frames = math.floor((librosa_length / 16000) * args.fps)
    print(f"Total frames to generate: {total_frames}\n")

    # 4. Generator
    generator = MuseTalkGenerator(
        whisper_encoder=whisper.encoder,
        vae=vae.vae,
        unet=unet.model,
        pe=pe,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
        video_fps=args.fps,
    ).eval()

    frames_dir = os.path.join(args.result_dir, "simple_tmp", args.avatar_id)
    os.makedirs(frames_dir, exist_ok=True)

    # 5. Inference loop
    frame_index = 0
    with torch.no_grad():
        for start in tqdm(range(0, total_frames, args.batch_size), desc="Generating"):
            current_bs = min(args.batch_size, total_frames - start)
            # Cycle latents from avatar cache
            latent_indices = [(start + i) % len(avatar.input_latent_list_cycle) for i in range(current_bs)]
            latent_batch = torch.cat([avatar.input_latent_list_cycle[i] for i in latent_indices], dim=0)
            recon_batch = generator(
                whisper_input_features=whisper_input_features,
                latent_inputs=latent_batch,
                frame_idx=start,
                librosa_length=librosa_length,
            )
            # Blend each frame
            for recon in recon_batch:
                bbox = avatar.coord_list_cycle[frame_index % len(avatar.coord_list_cycle)]
                ori = copy.deepcopy(avatar.frame_list_cycle[frame_index % len(avatar.frame_list_cycle)])
                mask = avatar.mask_list_cycle[frame_index % len(avatar.mask_list_cycle)]
                mask_coords = avatar.mask_coords_list_cycle[frame_index % len(avatar.mask_coords_list_cycle)]
                x1, y1, x2, y2 = bbox
                try:
                    recon = cv2.resize(recon.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    frame_index += 1
                    continue
                blended = get_image_blending(ori, recon, bbox, mask, mask_coords)
                cv2.imwrite(f"{frames_dir}/{frame_index:08d}.png", blended)
                frame_index += 1

    # 6. Encode & mux
    print("Encoding video & adding audio ...")
    temp_video = os.path.join(args.result_dir, f"{args.avatar_id}_raw.mp4")
    final_video = args.output_path
    os.makedirs(os.path.dirname(final_video), exist_ok=True)
    img2vid_cmd = (
        f"ffmpeg -y -v warning -r {args.fps} -f image2 -i {frames_dir}/%08d.png "
        f"-vcodec libx264 -vf format=yuv420p -crf 18 {temp_video}"
    )
    audio_mux_cmd = (
        f"ffmpeg -y -v warning -i {args.audio_path} -i {temp_video} -c:v copy -c:a aac -shortest {final_video}"
    )
    os.system(img2vid_cmd)
    os.system(audio_mux_cmd)

    # Cleanup temp video & optionally frames
    try:
        os.remove(temp_video)
    except Exception:
        pass
    print(f"Done. Output saved to {final_video}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Simple full-audio MuseTalk inference using cached avatar")
    p.add_argument("--audio_path", type=str, default="./data/audio/piradoba.wav")
    p.add_argument("--output_path", type=str, default="./results/testing_output.mp4")
    p.add_argument("--avatar_id", type=str, required=True)
    p.add_argument("--avatar_root", type=str, default="testing_avatar_creation/v15/avatars")
    p.add_argument("--result_dir", type=str, default="./results")
    p.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    p.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json")
    p.add_argument("--unet_model_path", type=str, default="./models/musetalk/pytorch_model.bin")
    p.add_argument("--vae_type", type=str, default="sd-vae")
    p.add_argument("--whisper_dir", type=str, default="./models/whisper")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--bbox_shift", type=int, default=0)
    p.add_argument("--audio_padding_length_left", type=int, default=2)
    p.add_argument("--audio_padding_length_right", type=int, default=2)
    p.add_argument("--extra_margin", type=int, default=10)
    p.add_argument("--parsing_mode", type=str, default="jaw")
    p.add_argument("--left_cheek_width", type=int, default=90)
    p.add_argument("--right_cheek_width", type=int, default=90)
    return p
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    simple_full_audio_inference(args)