"""Utility script to precompute avatar assets.

Run this script inside the legacy `avatar_venv` environment where mmpose is
available. It prepares all intermediate files (frames, masks, latents, etc.) so
that other scripts can load the avatar without requiring mmpose.
"""

import argparse
import os
import sys

import torch

# Ensure repository root is on the path when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from musetalk.streaming import Avatar, AvatarConfig, AvatarRuntime  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402
from musetalk.utils.utils import load_all_model  # noqa: E402


def build_runtime(device: torch.device, version: str, left_cheek_width: int, right_cheek_width: int) -> AvatarRuntime:
    """Create a minimal AvatarRuntime for asset preparation."""
    vae, unet, pe = load_all_model(device=device)

    face_parser = (
        FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
        if version == "v15"
        else FaceParsing()
    )

    # Positional encoding and diffusion UNet expect tensors on the same device.
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    return AvatarRuntime(
        device=device,
        vae=vae,
        unet=unet,
        pe=pe,
        audio_processor=None,
        whisper=None,
        face_parser=face_parser,
        timesteps=torch.tensor([0], device=device),
        weight_dtype=unet.model.dtype,
    )


def prepare_avatar(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    runtime = build_runtime(
        device=device,
        version=args.version,
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )

    config = AvatarConfig(
        version=args.version,
        result_root=args.result_dir,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        skip_save_images=args.skip_save_images,
        fps=args.fps,
        batch_size=args.batch_size,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )

    avatar_id = args.avatar_id or os.path.splitext(os.path.basename(args.video_path))[0]

    print("Preparing avatar assets...")
    Avatar(
        avatar_id=avatar_id,
        video_path=args.video_path,
        bbox_shift=args.bbox_shift,
        preparation=True,
        config=config,
        runtime=runtime,
        interactive=False,
        force_recreate=args.force_recreate,
    )
    print(f"Avatar '{avatar_id}' assets stored under {config.result_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute avatar assets (requires mmpose)")
    parser.add_argument("--video_path", type=str, required=True, help="Source video for the avatar")
    parser.add_argument("--avatar_id", type=str, default=None, help="Identifier for the avatar folder")
    parser.add_argument("--result_dir", type=str, default="./results", help="Root directory for results")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Model version")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding-box adjustment for v1")
    parser.add_argument("--extra_margin", type=int, default=10, help="Additional pixels for v15 crops")
    parser.add_argument("--parsing_mode", type=str, default="jaw", help="Face parsing mode")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate for cached assets")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size stored in avatar config")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding used during inference")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding used during inference")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Face parsing parameter (v15 only)")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Face parsing parameter (v15 only)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index to use if CUDA is available")
    parser.add_argument("--skip_save_images", action="store_true", help="Store avatar config with skip-save flag")
    parser.add_argument("--force_recreate", action="store_true", help="Overwrite existing avatar assets")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_avatar(parse_args())
