"""
Streaming module for MuseTalk
Provides clean interfaces for audio streaming and real-time inference
"""

from .audio_stream import AudioStream, PseudoAudioStream, LiveAudioStream
from .avatar import Avatar, AvatarConfig, AvatarRuntime
from .whisper_processor import WhisperStreamProcessor

__all__ = [
    'AudioStream',
    'PseudoAudioStream',
    'LiveAudioStream',
    'Avatar',
    'AvatarConfig',
    'AvatarRuntime',
    'WhisperStreamProcessor',
]
