"""
Whisper Stream Processor V3
Hybrid approach: Uses HuggingFace Whisper model but with original slicing logic
"""

import torch
import numpy as np
from transformers import WhisperModel, AutoFeatureExtractor
from collections import deque
from typing import Optional
import librosa


class WhisperStreamProcessorV3:
    """
    Streaming Whisper processor using HuggingFace model but original feature slicing.

    Key: Uses the EXACT slicing logic from audio2feature.py get_sliced_feature()
    This gives us the proven 200ms context window.
    """

    def __init__(
        self,
        whisper_model_path: str,
        input_sample_rate: int = 24000,
        whisper_sample_rate: int = 16000,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize processor.

        Args:
            whisper_model_path: Path to HuggingFace Whisper model
            input_sample_rate: Input audio sample rate (24kHz)
            whisper_sample_rate: Whisper's required sample rate (16kHz)
            device: Device to run on
            dtype: Data type for inference
        """
        self.input_sample_rate = input_sample_rate
        self.whisper_sample_rate = whisper_sample_rate
        self.device = device
        self.dtype = dtype
        self.audio_fps = 50  # Whisper outputs at 50 Hz

        # Load HuggingFace Whisper
        print(f"Loading Whisper from {whisper_model_path}...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_model_path)
        self.whisper = WhisperModel.from_pretrained(whisper_model_path).to(device=device, dtype=dtype)
        self.whisper.eval()
        self.whisper.requires_grad_(False)

        # Buffering - reduced for lower latency
        self.audio_buffer = deque()
        self.min_buffer_duration_s = 1.0  # 1 second minimum
        self.processing_window_duration_s = 3.0  # 3 second window

        chunk_duration_s = 0.08  # 80ms
        self.min_buffer_chunks = int(self.min_buffer_duration_s / chunk_duration_s)
        self.process_window_chunks = int(self.processing_window_duration_s / chunk_duration_s)

        # Feature storage
        self.whisper_feature_array = None  # Shape: [seq_len, num_layers, hidden_dim]
        self.processed_chunks_count = 0

    def resample_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Resample single chunk from 24kHz to 16kHz"""
        if self.input_sample_rate == self.whisper_sample_rate:
            return audio_chunk

        resampled = librosa.resample(
            audio_chunk,
            orig_sr=self.input_sample_rate,
            target_sr=self.whisper_sample_rate
        )
        return resampled

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Process incoming audio chunk through Whisper.

        Args:
            audio_chunk: Audio at input_sample_rate

        Returns:
            bool: True if new features were computed
        """
        # Resample to 16kHz
        resampled_chunk = self.resample_chunk(audio_chunk)

        # Buffer
        self.audio_buffer.append(resampled_chunk)

        # Check if ready to process
        if len(self.audio_buffer) < self.min_buffer_chunks:
            return False

        # Get recent audio window
        window_size = min(self.process_window_chunks, len(self.audio_buffer))
        recent_chunks = list(self.audio_buffer)[-window_size:]
        recent_audio = np.concatenate(recent_chunks)

        # Extract Mel features
        input_features = self.feature_extractor(
            recent_audio,
            return_tensors="pt",
            sampling_rate=self.whisper_sample_rate
        ).input_features.to(device=self.device, dtype=self.dtype)

        # Run through Whisper encoder
        with torch.no_grad():
            encoder_outputs = self.whisper.encoder(
                input_features,
                output_hidden_states=True
            )
            hidden_states = encoder_outputs.hidden_states

            # Stack: [1, seq_len, num_layers, hidden_dim]
            audio_feats = torch.stack(hidden_states, dim=2)

            # Transpose to: [1, num_layers, seq_len, hidden_dim]
            audio_feats = audio_feats.permute(0, 2, 1, 3)

            # Remove batch dim and convert to numpy for compatibility
            # Shape: [num_layers, seq_len, hidden_dim]
            audio_feats = audio_feats[0].cpu().numpy()

            # Convert to audio2feature format: [seq_len, num_layers, hidden_dim]
            audio_feats = np.transpose(audio_feats, (1, 0, 2))

        self.whisper_feature_array = audio_feats
        self.processed_chunks_count += 1

        return True

    def get_sliced_feature(
        self,
        feature_array: np.ndarray,
        vid_idx: int,
        audio_feat_length: list = [2, 2],
        fps: int = 25
    ) -> np.ndarray:
        """
        Get sliced features for a single frame.
        EXACT implementation from audio2feature.py lines 16-45.

        Args:
            feature_array: Whisper features [seq_len, num_layers, hidden_dim]
            vid_idx: Video frame index
            audio_feat_length: [left_padding, right_padding]
            fps: Video FPS

        Returns:
            Selected features reshaped to [-1, 384]
        """
        length = len(feature_array)
        selected_feature = []

        # EXACT logic from original code
        center_idx = int(vid_idx * 50 / fps)
        left_idx = center_idx - audio_feat_length[0] * 2
        right_idx = center_idx + (audio_feat_length[1] + 1) * 2

        for idx in range(left_idx, right_idx):
            idx = max(0, idx)
            idx = min(length - 1, idx)
            x = feature_array[idx]
            selected_feature.append(x)

        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)  # [50, 384]
        return selected_feature

    def get_features_for_frames(
        self,
        num_frames: int,
        video_fps: int = 25,
        audio_feat_length: list = [2, 2]
    ) -> Optional[torch.Tensor]:
        """
        Extract features for multiple frames using original slicing logic.

        Args:
            num_frames: Number of frames to generate
            video_fps: Video FPS
            audio_feat_length: [left, right] padding

        Returns:
            Tensor [num_frames, 50, 384] or None
        """
        if self.whisper_feature_array is None:
            return None

        feature_array = self.whisper_feature_array
        length = len(feature_array)

        # Check maximum frames possible
        whisper_idx_multiplier = 50.0 / video_fps
        audio_feature_length = 2 * (audio_feat_length[0] + audio_feat_length[1] + 1)
        max_frames = int((length - audio_feature_length) / whisper_idx_multiplier)

        if max_frames <= 0:
            return None

        actual_frames = min(num_frames, max_frames)

        whisper_chunks = []
        for i in range(actual_frames):
            # Use original slicing method
            selected_feature = self.get_sliced_feature(
                feature_array,
                vid_idx=i,
                audio_feat_length=audio_feat_length,
                fps=video_fps
            )
            whisper_chunks.append(selected_feature)

        if len(whisper_chunks) == 0:
            return None

        # Stack: [num_frames, 50, 384]
        whisper_chunks = np.stack(whisper_chunks, axis=0)

        # Convert to torch
        whisper_chunks = torch.from_numpy(whisper_chunks).float()

        return whisper_chunks

    def reset(self):
        """Reset processor state"""
        self.audio_buffer.clear()
        self.whisper_feature_array = None
        self.processed_chunks_count = 0

    def get_stats(self) -> dict:
        """Get processing statistics"""
        buffer_duration_s = len(self.audio_buffer) * 0.08

        return {
            'buffer_chunks': len(self.audio_buffer),
            'buffer_duration_s': buffer_duration_s,
            'processed_chunks': self.processed_chunks_count,
            'has_features': self.whisper_feature_array is not None,
            'feature_length': len(self.whisper_feature_array) if self.whisper_feature_array is not None else 0,
            'min_buffer_ready': len(self.audio_buffer) >= self.min_buffer_chunks
        }
