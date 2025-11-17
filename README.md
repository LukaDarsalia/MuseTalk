# MuseTalk Dual-Environment Workflow

This README focuses on a practical two-environment setup for MuseTalk:

1. An "avatar creation" environment (full stack) that includes the original dependencies and optional MMLab pose/parse toolchain for generating avatar caches.
2. A lightweight "streaming inference" environment (no mmpose/mmcv/mmdet) for fast real-time lip-sync generation using pre-built avatar assets and a newer (or alternative) PyTorch build.

Both environments operate on the same `models/` weight directory and the avatar cache directories you generate once in the full environment.

---
## Why Two Environments?

The heaviest dependencies (mmcv, mmpose, mmdet, mmengine, openmim) are only needed while constructing avatars (face parsing, landmark / pose helpers, masks). Once an avatar is prepared, streaming inference only needs:

* Core model weights (VAE, UNet, MuseTalk configs)
* Whisper tiny model (for audio features)
* Diffusers / accelerate / transformers / numpy / cv2
* Optional `sounddevice` for synchronized audio playback

Separating environments keeps the live/production runtime lean, reduces container image size, and lets you pin or experiment with a newer PyTorch version without rebuilding heavy CUDA wheels for all MMLab packages.

---
## Directory Essentials

```
models/
  musetalk/            # v1 model (pytorch_model.bin + musetalk.json)
  musetalkV15/         # v15 model (unet.pth + musetalk.json)
  whisper/             # whisper tiny config + weights
  sd-vae/              # VAE weights (config.json + diffusion_pytorch_model.bin)
  dwpose/, syncnet/, face-parse-bisent/  # used only in full env for avatar prep

testing_avatar_creation/
  v15/avatars/<avatar_id>/
    latents.pt
    coords.pkl
    mask_coords.pkl
    full_imgs/
    mask/
```

The `avatars/<avatar_id>` directory produced by `scripts/create_avatar.py` is what the streaming environment consumes.

---
## Environment Files

We keep two requirement files to make intent explicit:

* `requirements.txt` (original) — base libs. MMLab stack is installed via `openmim` commands, not listed here.
* `requirements_streaming.txt` (new) — trimmed set without MMLab plus a (potentially) newer Torch.

### 1. Avatar Creation Environment (full)

```fish
# Create & activate (fish shell)
conda create -n musetalk_avatar python=3.10 -y
conda activate musetalk_avatar

# Install a stable Torch (match original examples)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Core libs
pip install -r requirements.txt

# MMLab stack (needed only here)
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

# (Optional) verify ffmpeg
ffmpeg -version
```

### 2. Streaming Inference Environment (lean)

Choose any compatible Torch build (example below uses a newer one). You do NOT install mmcv/mmpose/etc.

```fish
conda create -n musetalk_stream python=3.10 -y
conda activate musetalk_stream

# Newer Torch example (adjust CUDA index URL as needed for your GPU):
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Lean requirements
pip install -r requirements_streaming.txt

# Optional for live playback sync
pip install sounddevice
```

If you want pure CPU streaming, skip the CUDA index URL; Torch will install a CPU wheel.

---
## Avatar Creation (Full Environment)

Run inside `musetalk_avatar` environment. This produces the reusable avatar cache under `testing_avatar_creation/v15/avatars/<avatar_id>`.

```fish
python scripts/create_avatar.py \
  --video_path data/video/piradoba.mp4 \
  --result_dir testing_avatar_creation \
  --version v15 \
  --fps 25 \
  --batch_size 8
```

Notes:
* Ensure weights are placed under `models/` as documented.
* The command will parse / segment frames and generate latent + mask artifacts.
* You only need to rerun if source video or parsing params change.

Resulting path of interest for streaming:

```
testing_avatar_creation/v15/avatars/piradoba/
  latents.pt
  coords.pkl
  mask_coords.pkl
  full_imgs/
  mask/
```

---
## Streaming Inference (Lean Environment)

Run inside `musetalk_stream` after avatar creation is complete.

```fish
python scripts/streaming_inference_v2.py \
  --audio_path data/audio/piradoba.wav \
  --output_path results/video.mp4 \
  --version v15 \
  --batch_size 8 \
  --avatar_id piradoba \
  --avatar_root testing_avatar_creation/v15/avatars \
  --lookahead_chunks 2
```

Flags explained:
* `--avatar_root` points to directory containing `<avatar_id>` subfolder with cached assets.
* `--lookahead_chunks` buffers future 80 ms audio chunks for smoother prosody (increase for potentially better alignment at cost of latency).
* `--batch_size` controls frame generation batching.
* `--fps` (default 25) should match training fps for best results.

When audio finishes, the script flushes remaining buffered frames and muxes video + audio via ffmpeg.

Optional additions:
```fish
# Disable live audio monitoring
--no_audio_playback

# Change avatar source if you built a v1 avatar
--version v1
```

---
## Minimal Requirements Contrast

| Component            | Avatar Creation Env | Streaming Env |
|----------------------|--------------------|---------------|
| mmcv / mmpose / mmdet| Yes                | No            |
| openmim / mmengine   | Yes                | No            |
| face parsing extras  | Yes                | No (precomputed) |
| Torch version        | Pinned (2.0.1)     | Flexible (e.g. 2.2.x) |
| sounddevice          | Optional           | Optional (recommended) |

---
## Updating / Rebuilding Avatars

If you want to regenerate an avatar with new parsing margins or cheek widths, do so only in the full environment and re-run the `create_avatar.py` command. The streaming environment does not need changes unless model weights change.

---
## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| Missing `latents.pt` | Avatar not created | Re-run create_avatar.py in full env |
| Runtime import error for mmpose in streaming | Used wrong environment | Activate `musetalk_stream` (no MMLab) |
| CUDA mismatch Torch | Different driver vs build | Reinstall Torch matching local CUDA or use CPU wheel |
| Low FPS | GPU in P8 state / thermal throttle | Ensure `nvidia-smi` shows performance mode; reduce batch size |
| Audio desync | lookahead too high / playback disabled | Lower `--lookahead_chunks` or enable playback |

---
## Citation

Please cite the original MuseTalk work:
```bibtex
@article{musetalk,
  title={MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling},
  author={Zhang, Yue and Zhong, Zhizhou and Liu, Minhao and Chen, Zhaokang and Wu, Bin and Zeng, Yubin and Zhan, Chao and He, Yingjie and Huang, Junxin and Zhou, Wenjiang},
  journal={arXiv},
  year={2025}
}
```

---
## License & Use

* Code: MIT
* Models: Usable for academic & commercial purposes (respect third-party model licenses: Whisper, VAE, etc.)
* Avatar source videos: Ensure you have rights for any redistribution or commercial usage.

---
## Acknowledgements

MuseTalk builds upon open-source components including Whisper, diffusers, LatentSync, and the MMLab ecosystem. Thanks to all upstream contributors.

---
## Next Steps / Ideas

* Add a script to automatically export only necessary weights for streaming.
* Provide a Docker multi-stage build (full -> lean runtime image).
* Optional integration with MuseV for an end-to-end generation pipeline.

---
Happy lip-syncing! If this dual setup helps you, consider contributing improvements or opening issues.

---
## Simple Full-Audio Generator Script (`testing_musetalkGenerator.py`)

In addition to the streaming chunked pipeline (`scripts/streaming_inference_v2.py`), the repository includes a **compact full-audio inference script** that uses the `MuseTalkGenerator` class directly to process an entire audio file in one pass. This is useful when:

* You already prepared an avatar cache and just want a quick deterministic render for a single audio clip.
* You don't need real-time chunk buffering, latency tuning, or lookahead behavior.
* You prefer minimal code flow for experimentation or extension.

### What It Does
1. Loads VAE, UNet, positional encoding, and Whisper encoder.
2. Loads an existing avatar cache (latents, masks, coords) – no recreation or parsing.
3. Extracts Whisper features for the whole audio file once.
4. Iterates through all frames derived from audio length (fps-based) in batches.
5. Generates latent predictions, decodes them, blends with original avatar frame regions, and writes PNG frames.
6. Muxes the rendered frames with the original audio via `ffmpeg` into a final MP4.

### When to Use vs Streaming
| Use Case | Choose `testing_musetalkGenerator.py` | Choose `streaming_inference_v2.py` |
|----------|----------------------------------------|------------------------------------|
| Offline full render | ✅ | ✅ (more verbose) |
| Need lowest code complexity | ✅ | ❌ |
| Real-time / progressive display | ❌ | ✅ |
| Latency / lookahead tuning | ❌ | ✅ |
| Single audio file known end | ✅ | ✅ |

### Requirements
Run this in either environment; the lean streaming environment is sufficient as long as the avatar cache exists. Ensure:
* `models/` weights are present
* Avatar cache directory: `testing_avatar_creation/v15/avatars/<avatar_id>/`

### Command (fish)
```fish
python testing_musetalkGenerator.py \
  --avatar_id piradoba \
  --avatar_root testing_avatar_creation/v15/avatars \
  --audio_path data/audio/piradoba.wav \
  --output_path results/piradoba_simple.mp4 \
  --fps 25 \
  --batch_size 8
```

Key Flags:
* `--avatar_id` – name of the prebuilt avatar folder.
* `--avatar_root` – root directory containing avatar cache subfolder.
* `--fps` – should match training fps (25 recommended).
* `--batch_size` – generation batch size (balance VRAM vs speed).

### Output
Creates a temporary frame directory, encodes it to a raw MP4, then muxes audio and deletes the intermediate video (frames are retained for inspection). Final file: `results/piradoba_simple.mp4`.

### Extending
You can hook in post-processing (e.g., super-resolution) by intercepting the saved PNG frames before muxing. The script purposefully keeps logic linear and commented for easier modification.

### Internals
The `MuseTalkGenerator` class is untouched; only a thin driver function wraps:
```text
Full Audio -> Whisper Features -> Batched UNet Forward -> Decode -> Blend -> Save -> Mux
```

If you later need incremental generation or playback synchronization, switch to `streaming_inference_v2.py`.
