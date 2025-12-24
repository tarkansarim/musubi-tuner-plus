# Musubi Tuner Setup Notes & Troubleshooting

This document contains critical setup information and common pitfalls when setting up training for Musubi Tuner, particularly for Wan2.2.

## Critical Setup Requirements

### 1. Virtual Environment Activation
**ALWAYS activate the `.venv` before running any commands!**
- Windows: `call .venv\Scripts\Activate.bat` or `.venv\Scripts\Activate.ps1` (PowerShell)
- Batch files MUST start with `@call .venv\Scripts\Activate.bat`

### 2. Project Installation
The project must be installed in editable mode for imports to work:
```bash
pip install -e .
```

### 3. Wan2.2 Model Requirements

#### Models Needed:
1. **DiT Low-Noise Model** (fp16): `wan2.2_t2v_low_noise_14B_fp16.safetensors`
2. **DiT High-Noise Model** (fp16): `wan2.2_t2v_high_noise_14B_fp16.safetensors`
3. **VAE** (Wan2.1 VAE works for 2.2): `wan_2.1_vae.safetensors`
4. **T5 Text Encoder** (MUST be encoder-only): `models_t5_umt5-xxl-enc-bf16.pth`

#### Critical T5 Model Note:
- **DO NOT use generic UMT5-XXL models** (e.g., `umt5_xxl_fp16.safetensors`)
- Must use the **Wan-specific encoder-only T5**: `models_t5_umt5-xxl-enc-bf16.pth`
- Download from: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
- Wrong T5 will cause: `RuntimeError: Error(s) in loading state_dict for T5Encoder: Missing key(s)...`

### 4. Dataset Configuration (TOML)

#### Minimal Working Config for Wan2.2:
```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
video_directory = "datasets/video_instance"
cache_directory = "datasets/video_instance_latents"
target_frames = [1, 17]
frame_extraction = "head"
```

#### Critical Config Rules:
- **DO NOT add extra keys!** The `voluptuous` schema is EXTREMELY strict
- Invalid keys that will cause errors: `path`, `type`, `num_frames`, `width`, `height`, `resolution` (in datasets section)
- For Wan: `target_frames` must be `[1, N*4+1]` (e.g., `[1, 17]` for 16 frames)
- `cache_directory` is auto-generated based on `video_directory` if not specified

### 5. Caching Latents & Text Encodings

#### Latent Caching:
```bash
python src/musubi_tuner/wan_cache_latents.py \
  --dataset_config datasets/video_instance/config.toml \
  --vae "path/to/wan_2.1_vae.safetensors" \
  --batch_size 1 \
  --num_workers 4 \
  --skip_existing \
  --vae_cache_cpu
```

**Unsupported arguments for `wan_cache_latents.py`:**
- ❌ `--vae_chunk_size`
- ❌ `--vae_tiling`
- ❌ `--fp8_vae`
- ❌ `--output_dir` (auto-generated)
- ❌ `--max_workers` (use `--num_workers` instead)

#### Text Encoder Caching:
```bash
python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config datasets/video_instance/config.toml \
  --t5 "path/to/models_t5_umt5-xxl-enc-bf16.pth" \
  --batch_size 4 \
  --skip_existing
```

**Note:** Use `--t5` (not `--text_encoder2`)

### 6. Training Configuration

#### Minimal Working Training Command:
```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
  src/musubi_tuner/wan_train_network.py \
  --dit "path/to/wan2.2_t2v_low_noise_14B_fp16.safetensors" \
  --dit_high_noise "path/to/wan2.2_t2v_high_noise_14B_fp16.safetensors" \
  --timestep_boundary 875 \
  --vae "path/to/wan_2.1_vae.safetensors" \
  --dataset_config datasets/video_instance/config.toml \
  --sdpa \
  --network_module networks.lora_wan \
  --network_dim 4 \
  --network_alpha 2 \
  --optimizer_type AdamW8bit \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 10 \
  --max_train_steps 100 \
  --save_every_n_steps 50 \
  --mixed_precision fp16 \
  --output_dir outputs/test_wan_lora \
  --logging_dir logs/test_wan \
  --fp8_base \
  --gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --seed 42 \
  --timestep_sampling shift \
  --discrete_flow_shift 5.0 \
  --preserve_distribution_shape
```

#### Critical Training Arguments:

1. **Mixed Precision:** MUST match DiT dtype
   - fp16 models → `--mixed_precision fp16` (in both accelerate and script)
   - bf16 models → `--mixed_precision bf16`
   - Mismatch causes: `AssertionError: DiT weights are in fp16, mixed precision must be fp16 or no`

2. **Timestep Boundary:** (for dual-model Wan2.2)
   - Required when using `--dit_high_noise`
   - Integer format (0-1000): `--timestep_boundary 875` (default for T2V)
   - Or float format (0.0-1.0): `--timestep_boundary 0.875`
   - Missing causes: `ValueError: timestep_boundary is not specified for high noise model`

3. **Attention Mechanism:** (REQUIRED - must specify one)
   - `--sdpa` (recommended, most compatible)
   - `--xformers` (may cause CUDA errors on some GPUs)
   - `--flash_attn` / `--flash3` (requires specific setup)
   - Missing causes: `ValueError: either --sdpa, --flash-attn, --flash3, --sage-attn or --xformers must be specified`

4. **Optimizer:** (REQUIRED)
   - `--optimizer_type AdamW8bit` (recommended for LoRA)
   - Missing causes: `AttributeError: module 'torch.optim' has no attribute ''`

5. **Invalid Arguments for `wan_train_network.py`:**
   - ❌ `--latents_dir` (loaded from dataset config's `cache_directory`)
   - ❌ `--text_encoder_dir` (auto-detected)
   - ❌ `--v2_2` (detected from `--dit_high_noise`)
   - ❌ `--resolution` (from dataset config)
   - ❌ `--train_batch_size` (use `batch_size` in config.toml)
   - ❌ `--blocks_to_swap "transformer_blocks"` (expects integer, not string)

## Common Errors & Fixes

### Error: "found 0 videos"
**Cause:** Wrong `video_directory` path in config.toml
**Fix:** Use absolute path or correct relative path from project root (e.g., `datasets/video_instance`)

### Error: "extra keys not allowed @ data['datasets'][0]['path']"
**Cause:** Invalid keys in TOML config (strict voluptuous validation)
**Fix:** Use ONLY the minimal config shown above. Remove any extra keys.

### Error: "Missing key(s) in state_dict: 'token_embedding.weight'..."
**Cause:** Using wrong T5 model (full model instead of encoder-only)
**Fix:** Download correct Wan-specific T5: `models_t5_umt5-xxl-enc-bf16.pth`

### Error: "CUDA error (flash-attention): invalid argument"
**Cause:** xformers flash-attention incompatibility
**Fix:** Use `--sdpa` instead of `--xformers`

### Error: "ModuleNotFoundError: No module named 'musubi_tuner'"
**Cause:** Project not installed in editable mode
**Fix:** Run `pip install -e .` from project root

### Error: Pip packages installing outside venv
**Cause:** Global Anaconda pip interfering
**Fix:** 
1. Deactivate conda: `conda deactivate`
2. Delete venv: `rm -r .venv` (or `Remove-Item -Recurse .venv` in PowerShell)
3. Recreate venv: `python -m venv .venv`
4. Activate and reinstall: `.venv\Scripts\Activate.bat` then `pip install -e .`

## Batch File Best Practices

### Windows Batch Files (.bat):
```batch
@echo off
call .venv\Scripts\Activate.bat

echo Starting training...
python src\musubi_tuner\wan_train_network.py ^
  --arg1 value1 ^
  --arg2 value2

if %errorlevel% neq 0 (
    echo Error occurred
    pause
    exit /b %errorlevel%
)

echo Complete!
pause
```

**Key Points:**
- Start with `@echo off` for clean output
- **ALWAYS** activate venv first: `call .venv\Scripts\Activate.bat`
- Use `^` for line continuation (NOT backslash)
- Use backslashes `\` in paths (not forward slashes)
- Check error level: `if %errorlevel% neq 0`
- Use `pause` at end to see output

## Model Download Links

### Wan2.2 Models:
- **Low-Noise DiT (fp16):** https://huggingface.co/ai-toolkit/Wan2.2-T2V-A14B/resolve/main/wan2.2_t2v_low_noise_14B_fp16.safetensors
- **High-Noise DiT (fp16):** https://huggingface.co/ai-toolkit/Wan2.2-T2V-A14B/resolve/main/wan2.2_t2v_high_noise_14B_fp16.safetensors

### Shared Models (Wan2.1 works for 2.2):
- **VAE:** https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/Wan2.1_VAE.pth
- **T5 (encoder-only):** https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
- **CLIP (for I2V):** https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/pytorch_model.bin

## Recommended Workflow

1. **Setup Environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.bat
   pip install -e .
   ```

2. **Prepare Dataset:**
   - Place videos in `datasets/video_instance/`
   - Create `.txt` caption files (same name as videos)
   - Create minimal `config.toml`

3. **Cache Data:**
   ```bash
   # Latents
   python src/musubi_tuner/wan_cache_latents.py --dataset_config datasets/video_instance/config.toml --vae "path/to/vae.safetensors" --batch_size 1 --num_workers 4 --skip_existing --vae_cache_cpu
   
   # Text encodings
   python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config datasets/video_instance/config.toml --t5 "path/to/models_t5_umt5-xxl-enc-bf16.pth" --batch_size 4 --skip_existing
   ```

4. **Train:**
   ```bash
   accelerate launch --mixed_precision fp16 src/musubi_tuner/wan_train_network.py [args...]
   ```

5. **Monitor:**
   ```bash
   tensorboard --logdir logs/test_wan
   ```

## Tips for Future Agents

1. **Read project context first:** Check `READFIRST.md` or `.ai/context/overview.md`
2. **Use absolute paths for models** to avoid path resolution issues
3. **Start with minimal configs** - add complexity only when needed
4. **Check error messages carefully** - they often indicate exact argument issues
5. **Activate venv FIRST** - most import/package errors are from this
6. **Use batch files for complex commands** - easier to debug and reproduce
7. **Test caching before training** - catch config issues early
8. **Use `--skip_existing` in cache commands** - avoid re-encoding
9. **Check GPU compatibility** - use `--sdpa` if xformers fails
10. **Keep batch size at 1 for video** - high VRAM usage per sample

## Windows Multi-GPU Limitations

### Known Issue: libuv Error
Windows multi-GPU training with PyTorch may fail with:
```
torch.distributed.DistStoreError: use_libuv was requested but PyTorch was built without libuv support
```

**Root Cause:**
- PyTorch's TCPStore on Windows has a libuv dependency conflict
- Environment variable `USE_LIBUV=0` doesn't propagate to subprocess
- No NCCL support on Windows (Linux-only fast GPU backend)

**Workarounds:**

1. **Single-GPU Training (Recommended)**
   - Works perfectly on Windows
   - 96GB VRAM is more than enough for Wan2.2
   - Use your existing `run_test_train.bat`

2. **Sequential Multi-GPU**
   - Train separate runs on each GPU with different seeds
   - Merge LoRAs afterward using `merge_lora.py`
   ```batch
   set CUDA_VISIBLE_DEVICES=0
   run_test_train.bat --output_dir outputs/lora_gpu0
   
   set CUDA_VISIBLE_DEVICES=1  
   run_test_train.bat --output_dir outputs/lora_gpu1 --seed 43
   ```

3. **WSL2 (Best Solution)**
   - Install WSL2 with Ubuntu + CUDA
   - Full NCCL multi-GPU support
   - Near-native Linux performance
   - Proper DDP with 2x speed improvement

4. **Increase Batch Size on Single GPU**
   - With 96GB VRAM, try `batch_size = 2` in config.toml
   - Or use `--gradient_accumulation_steps 2` for larger effective batch

**Note:** Linux has no such issues - multi-GPU works out of the box with accelerate/torchrun.

---

*Last updated: 2025-10-08*
*Session: Wan2.2 T2V training setup with fp16 models + Windows multi-GPU investigation*

