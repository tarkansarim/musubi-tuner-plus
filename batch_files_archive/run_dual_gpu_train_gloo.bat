@echo off
call .venv\Scripts\Activate.bat

REM Fix for Windows multi-GPU: disable libuv
set USE_LIBUV=0
set MASTER_ADDR=127.0.0.1

echo Starting Wan2.2 DUAL-GPU training (100 steps) with gloo backend...
echo Using 2x RTX PRO 6000 Blackwell GPUs (192GB total VRAM)

echo.
echo Launching dual-GPU training with gloo backend...
python src\musubi_tuner\wan_train_network.py ^
  --multi_gpu ^
  --dit "E:\AI\models\diffusion_models\wan2.2_t2v_low_noise_14B_fp16.safetensors" ^
  --dit_high_noise "E:\AI\models\diffusion_models\wan2.2_t2v_high_noise_14B_fp16.safetensors" ^
  --timestep_boundary 875 ^
  --vae "E:\AI\models\vae\WAN2.1\wan_2.1_vae.safetensors" ^
  --dataset_config datasets\video_instance\config.toml ^
  --sdpa ^
  --network_module networks.lora_wan ^
  --network_dim 4 ^
  --network_alpha 2 ^
  --optimizer_type AdamW8bit ^
  --learning_rate 1e-4 ^
  --lr_scheduler cosine ^
  --lr_warmup_steps 10 ^
  --max_train_steps 100 ^
  --save_every_n_steps 50 ^
  --mixed_precision fp16 ^
  --output_dir outputs\test_wan_lora_dual_gpu ^
  --logging_dir logs\test_wan_dual_gpu ^
  --fp8_base ^
  --gradient_checkpointing ^
  --gradient_accumulation_steps 1 ^
  --seed 42 ^
  --timestep_sampling shift ^
  --discrete_flow_shift 5.0 ^
  --preserve_distribution_shape

if %errorlevel% neq 0 (
    echo Error in training. Check paths, VRAM, or multi-GPU setup.
    pause
    exit /b %errorlevel%
)

echo.
echo Dual-GPU training complete! LoRA weights in outputs\test_wan_lora_dual_gpu\
echo Run: tensorboard --logdir logs\test_wan_dual_gpu to view loss.
pause

