import gradio as gr
import os
import subprocess
import sys
from pathlib import Path

from .common_gui import (
    get_file_path,
    get_folder_path,
    output_message,
    scriptdir,
)
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
document_symbol = "\U0001F4C4"  # 📄

# Tooltip dictionary for hover tooltips
TOOLTIPS = {
    # Keys must match EXACT label text from the GUI
    "GPU Device(s)": "GPU device index(es) to use. Single GPU: '0' or '1'. Multiple GPUs: '0,1'",
    "Training Task": "T2V: Text-to-video | I2V: Image-to-video with first-frame conditioning",
    "Training Mode (DiT)": "Train low/high noise models separately or together",
    "Low-Noise DiT Model": "Path to WAN 2.2 low-noise DiT model (final denoising steps)",
    "High-Noise DiT Model": "Path to WAN 2.2 high-noise DiT model (initial denoising steps)",
    "VAE Model": "Path to VAE model for encoding/decoding video frames",
    "T5 Text Encoder": "Path to T5 text encoder for converting prompts to embeddings",
    "Dataset Directory": "Directory containing training videos and caption files",
    "Output Directory": "Directory where trained LoRA weights will be saved",
    "Logging Directory (TensorBoard)": "Directory for TensorBoard logs",
    "Optimizer": "AdamW8bit: Memory-efficient | AdamW: Standard | Adafactor: Adaptive",
    "Mixed Precision": "fp16: Faster, less VRAM | bf16: Better stability | no: Full precision",
    "Gradient Accumulation Steps": "Accumulate gradients over N steps. Effective batch = batch_size × this",
    "Gradient Checkpointing": "Trade computation for memory - saves VRAM",
    "FP8 Base Model": "Use FP8 precision for base model to save VRAM",
    "Timestep Boundary": "Timestep where training switches from high to low noise model (0-1000)",
    "Discrete Flow Shift": "Shift parameter for flow matching. Higher = harder timesteps",
    "Seed": "Random seed for reproducibility. -1 = random",
    "LR Scheduler": "Learning rate schedule: cosine decreases, constant stays fixed",
    "LR Warmup Steps": "Gradually increase LR over first N steps",
    "Max Gradient Norm": "Clip gradients to prevent exploding. 0 = no clipping",
    "Network Dim (LoRA Rank)": "LoRA rank. Higher = more capacity but larger file. Common: 4,8,16,32,64",
    "Network Alpha": "Scaling factor for LoRA weights. Typically rank/2 or equal to rank",
    "Learning Rate": "Step size for updates. Too high = unstable, too low = slow",
    "Max Train Steps": "Total training steps. More = longer training",
    "Batch Size": "Samples per batch. Higher = faster but more VRAM",
    "Target Frames": "Frames per video clip. More = better motion but more VRAM",
    "Resolution Width": "Maximum width for training videos",
    "Resolution Height": "Maximum height for training videos",
    "Output Name": "Base filename for saved models (e.g., 'my_lora' → 'my_lora-100.safetensors')",
    "Save Every N Steps": "Save checkpoint every N steps",
    "Save Last N Steps": "Keep only last N checkpoints. 0 = keep all",
    "Save Training State": "Save full training state to resume later",
    "Save Last N Steps State": "Keep only last N states. 0 = keep all",
    "Save State on Train End": "Save final state when training completes",
    "Max Data Loader Workers": "Parallel workers for loading data. More = faster but more RAM",
    "Persistent Data Loader Workers": "Keep workers alive between epochs for faster loading",
    "Enable Sampling": "Generate sample images/videos during training to monitor progress",
    "Sample Every N Steps": "Generate samples every N steps. 0 = disabled. Warning: Very slow!",
    "Sample Type": "Video: Generate video clips | Image: Generate single frames (faster)",
    "Sample Prompts (one per line)": "Enter prompts directly here, one per line. Auto-saved when training starts.",
    "Or Load from File": "Optional: Load prompts from an existing text file",
    "Resume from State": "Path to saved training state to resume from",
    "Pretrained Network Weights": "Start from existing LoRA weights",
    "Network Dropout": "Randomly drop LoRA weights to prevent overfitting. 0 = no dropout",
    "Scale Weight Norms": "Normalize LoRA weights to prevent dominating when merged",
    "Optimizer Args": "Additional optimizer arguments (key=value, space-separated)",
    "LR Scheduler Args": "Additional scheduler arguments (key=value, space-separated)",
    "LR Scheduler Num Cycles": "Number of restart cycles for cosine_with_restarts scheduler",
    "LR Scheduler Power": "Power factor for polynomial scheduler",
    "LR Scheduler Min LR Ratio": "Minimum LR as ratio of initial LR. 0 = decay to zero",
    "Timestep Sampling Method": "How to sample timesteps. 'shift' recommended for flow matching",
    "Weighting Scheme": "Weight different timesteps. 'none' = uniform weighting",
    "Min Timestep": "Minimum timestep (0-999). Lower = cleaner images",
    "Max Timestep": "Maximum timestep (1-1000). Higher = noisier images",
    "Preserve Distribution Shape": "Maintain original timestep distribution with discrete flow shift",
    "Log With": "Logging backend: TensorBoard (local), WandB (cloud), or both",
    "Log Config": "Save training config to logs for reproducibility",
    "WandB API Key": "Your Weights & Biases API key (from wandb.ai/authorize)",
    "WandB Run Name": "Name for this training run in WandB dashboard",
    "Training Comment": "Comment about this training run (saved in metadata)",
    "Author": "Your name or username (saved in model metadata)",
    "License": "License for the trained model (e.g., 'apache-2.0', 'mit', 'cc-by-4.0')",
    "Description": "Detailed description of what this model does",
    "Tags": "Comma-separated tags for categorization (e.g., 'anime,character,style')",
}

def wan22_train_tab():
    """Create WAN2.2 training tab"""
    
    # Configuration File Management (at the very top, like Kohya)
    with gr.Accordion("Configuration file", open=True, elem_classes=["preset_background"]):
        with gr.Row():
            config_file_name = gr.Textbox(
                label="",
                placeholder="Configuration file name (e.g., my_wan22_config.toml)",
                value="",
                interactive=True,
                scale=3,
            )
            config_file_picker = gr.Button(
                document_symbol,
                elem_id="open_folder_small",
                scale=0,
                min_width=40,
            )
            config_load_button = gr.Button(
                document_symbol + " Load",
                elem_id="open_folder",
                scale=1,
            )
            config_save_as_button = gr.Button(
                "💾 Save as",
                elem_id="open_folder",
                scale=1,
            )
            config_save_button = gr.Button(
                "💾 Save",
                elem_id="open_folder",
                scale=1,
            )
            config_reset_button = gr.Button(
                "🔄 Reset",
                elem_id="open_folder",
                scale=1,
            )
        
        config_status = gr.Textbox(
            label="",
            value="No configuration loaded",
            interactive=False,
            lines=1,
        )
    
    # GPU Device Selection
    with gr.Row():
        gpu_device = gr.Textbox(
            label="GPU Device(s)",
            value="0",
            interactive=True,
            scale=2,
        )
    
    # Model paths
    with gr.Accordion("Model", open=False, elem_classes=["basic_background"]) as model_accordion:
        with gr.Row(equal_height=True):
            with gr.Column():
                task_mode = gr.Radio(
                    choices=["t2v-14B (Text-to-Video)", "i2v-14B (Image-to-Video)"],
                    value="t2v-14B (Text-to-Video)",
                    label="Training Task",
                    interactive=True,
                )
            with gr.Column():
                training_mode = gr.Radio(
                    choices=["Both (alternating)", "Both (separate GPUs)", "Low-noise only", "High-noise only"],
                    value="Both (alternating)",
                    label="Training Mode (DiT)",
                    interactive=True,
                )
        
        task_info = gr.Markdown(
            value="**T2V Mode:** Training LoRA for text-to-video generation. Trains on video clips and/or images with text prompts.",
            visible=True,
        )
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    dit_low_noise = gr.Textbox(
                        label="Low-Noise DiT Model",
                        placeholder="E:/AI/models/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
                        interactive=True,
                    )
                    dit_low_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                    )
                    dit_low_button.click(
                        get_file_path,
                        inputs=[dit_low_noise, gr.Textbox(value="*.safetensors", visible=False), gr.Textbox(value="Model", visible=False)],
                        outputs=dit_low_noise,
                        show_progress=False,
                    )
            
            with gr.Column():
                with gr.Row():
                    dit_high_noise = gr.Textbox(
                        label="High-Noise DiT Model",
                        placeholder="E:/AI/models/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors",
                        interactive=True,
                    )
                    dit_high_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                    )
                    dit_high_button.click(
                        get_file_path,
                        inputs=[dit_high_noise, gr.Textbox(value="*.safetensors", visible=False), gr.Textbox(value="Model", visible=False)],
                        outputs=dit_high_noise,
                        show_progress=False,
                    )
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    vae_path = gr.Textbox(
                        label="VAE Model",
                        placeholder="E:/AI/models/vae/WAN2.1/wan_2.1_vae.safetensors",
                        interactive=True,
                    )
                    vae_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                    )
                    vae_button.click(
                        get_file_path,
                        inputs=[vae_path, gr.Textbox(value="*.safetensors *.pth", visible=False), gr.Textbox(value="VAE", visible=False)],
                        outputs=vae_path,
                        show_progress=False,
                    )
            
            with gr.Column():
                with gr.Row():
                    t5_path = gr.Textbox(
                        label="T5 Text Encoder",
                        placeholder="E:/AI/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth",
                        interactive=True,
                    )
                    t5_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                    )
                    t5_button.click(
                        get_file_path,
                        inputs=[t5_path, gr.Textbox(value="*.pth", visible=False), gr.Textbox(value="T5", visible=False)],
                        outputs=t5_path,
                        show_progress=False,
                    )
    
    # Folders
    with gr.Accordion("Folders", open=False, elem_classes=["basic_background"]) as folders_accordion:
        with gr.Row():
            video_directory = gr.Textbox(
                label="Dataset Directory",
                placeholder="G:/Projects/dataset/videos",
                interactive=True,
            )
            video_dir_button = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
            )
            video_dir_button.click(
                get_folder_path,
                inputs=video_directory,
                outputs=video_directory,
                show_progress=False,
            )
        
        with gr.Row():
            output_dir = gr.Textbox(
                label="Output Directory",
                value="outputs/wan_lora",
                interactive=True,
            )
            output_dir_button = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
            )
            output_dir_button.click(
                get_folder_path,
                inputs=output_dir,
                outputs=output_dir,
                show_progress=False,
            )
        
        with gr.Row():
            logging_dir = gr.Textbox(
                label="Logging Directory (TensorBoard)",
                value="logs/wan22_training",
                interactive=True,
            )
            logging_dir_button = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
            )
            logging_dir_button.click(
                get_folder_path,
                inputs=logging_dir,
                outputs=logging_dir,
                show_progress=False,
            )
    
    # Training Configuration (main parameters)
    with gr.Accordion("Training configuration", open=False, elem_classes=["advanced_background"]) as training_config_accordion:
        with gr.Row():
            optimizer_type = gr.Dropdown(
                choices=["AdamW", "AdamW8bit", "Adafactor"],
                value="AdamW8bit",
                label="Optimizer",
                interactive=True,
            )
            mixed_precision = gr.Dropdown(
                choices=["no", "fp16", "bf16"],
                value="fp16",
                label="Mixed Precision",
                interactive=True,
            )
        
        with gr.Row():
            gradient_accumulation_steps = gr.Number(
                label="Gradient Accumulation Steps",
                value=4,
                precision=0,
                interactive=True,
            )
            gradient_checkpointing = gr.Checkbox(
                label="Gradient Checkpointing",
                value=True,
                interactive=True,
            )
            fp8_base = gr.Checkbox(
                label="FP8 Base Model",
                value=True,
                interactive=True,
            )
        
        with gr.Row():
            timestep_boundary = gr.Number(
                label="Timestep Boundary",
                value=875,
                precision=0,
                interactive=True,
            )
            discrete_flow_shift = gr.Number(
                label="Discrete Flow Shift",
                value=5.0,
                interactive=True,
            )
        
        with gr.Row():
            seed = gr.Number(
                label="Seed",
                value=42,
                precision=0,
                interactive=True,
            )
            lr_scheduler = gr.Dropdown(
                choices=["constant", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"],
                value="cosine",
                label="LR Scheduler",
                interactive=True,
            )
        
        with gr.Row():
            lr_warmup_steps = gr.Number(
                label="LR Warmup Steps",
                value=20,
                precision=0,
                interactive=True,
            )
            max_grad_norm = gr.Number(
                label="Max Gradient Norm",
                value=1.0,
                interactive=True,
            )
    
    # Parameters (additional settings)
    with gr.Accordion("Parameters", open=False, elem_classes=["basic_background"]) as parameters_accordion:
        
        with gr.Row():
            network_dim = gr.Number(
                label="Network Dim (LoRA Rank)",
                value=16,
                precision=0,
                interactive=True,
            )
            network_alpha = gr.Number(
                label="Network Alpha",
                value=8,
                precision=0,
                interactive=True,
            )
            learning_rate = gr.Number(
                label="Learning Rate",
                value=8e-5,
                interactive=True,
            )
        
        with gr.Row():
            max_train_steps = gr.Number(
                label="Max Train Steps",
                value=3000,
                precision=0,
                interactive=True,
            )
            batch_size = gr.Number(
                label="Batch Size",
                value=1,
                precision=0,
                interactive=True,
            )
        
        with gr.Row():
            target_frames = gr.Number(
                label="Target Frames",
                value=121,
                precision=0,
                interactive=True,
            )
            resolution_width = gr.Number(
                label="Resolution Width",
                value=854,
                precision=0,
                interactive=True,
            )
            resolution_height = gr.Number(
                label="Resolution Height",
                value=480,
                precision=0,
                interactive=True,
            )
        
        with gr.Row():
            output_name = gr.Textbox(
                label="Output Name",
                placeholder="my_wan_lora",
                interactive=True,
            )
        
        # State Saving Section
        gr.Markdown("### Checkpoint & State Saving")
        
        with gr.Row():
            save_every_n_steps = gr.Number(
                label="Save Every N Steps",
                value=100,
                precision=0,
                interactive=True,
            )
            save_last_n_steps = gr.Number(
                label="Save Last N Steps",
                value=10,
                precision=0,
                interactive=True,
            )
        
        with gr.Row():
            save_state = gr.Checkbox(
                label="Save Training State",
                value=True,
                interactive=True,
            )
            save_last_n_steps_state = gr.Number(
                label="Save Last N Steps State",
                value=3,
                precision=0,
                interactive=True,
            )
            save_state_on_train_end = gr.Checkbox(
                label="Save State on Train End",
                value=True,
                interactive=True,
            )
        
        with gr.Row():
            max_data_loader_n_workers = gr.Number(
                label="Max Data Loader Workers",
                value=4,
                precision=0,
                interactive=True,
            )
            persistent_data_loader_workers = gr.Checkbox(
                label="Persistent Data Loader Workers",
                value=True,
                interactive=True,
            )
    
    # Advanced Parameters
    with gr.Accordion("Advanced Parameters", open=False, elem_classes=["basic_background"]) as advanced_params_accordion:
        with gr.Tab("Sampling"):
            enable_sampling = gr.Checkbox(
                label="Enable Sampling",
                value=True,
                interactive=True,
            )
            
            with gr.Row():
                sample_every_n_steps = gr.Number(
                    label="Sample Every N Steps",
                    value=100,
                    precision=0,
                    interactive=True,
                    scale=2,
                )
                sample_type = gr.Radio(
                    choices=["Video", "Image"],
                    value="Image",
                    label="Sample Type",
                    scale=1,
                    interactive=True,
                )
            
            gr.Markdown("### Sample Prompts")
            gr.Markdown("Add multiple prompts with individual settings. Each prompt can have custom guidance scale, steps, and seed.")
            
            # Container for dynamic prompt rows
            sample_prompts_container = gr.Column(elem_classes=["sample_prompts_container"])
            
            with sample_prompts_container:
                # Initial prompt row
                with gr.Row(elem_classes=["sample_prompt_row"]):
                    prompt_text = gr.Textbox(
                        label="Prompt",
                        placeholder="a beautiful sunset over the ocean",
                        scale=6,
                        elem_classes=["sample_prompt_text"],
                        interactive=True,
                    )
                    guidance_scale = gr.Number(
                        label="Guidance",
                        value=3.5,
                        minimum=1.0,
                        maximum=20.0,
                        step=0.1,
                        scale=1,
                        interactive=True,
                    )
                    sample_steps = gr.Number(
                        label="Steps",
                        value=20,
                        minimum=1,
                        maximum=100,
                        precision=0,
                        scale=1,
                        interactive=True,
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=43,
                        precision=0,
                        scale=1,
                        interactive=True,
                    )
                    remove_btn = gr.Button("🗑️", elem_id="remove_prompt_btn", scale=0, min_width=40)
            
            with gr.Row():
                add_prompt_btn = gr.Button("➕ Add Prompt", variant="secondary")
                clear_all_prompts_btn = gr.Button("Clear All", variant="stop")
            
            with gr.Row():
                sample_prompts_file = gr.Textbox(
                    label="Or Load from File",
                    placeholder="path/to/prompts.txt (optional)",
                    scale=4,
                    interactive=True,
                )
                sample_prompts_button = gr.Button(document_symbol, elem_id="open_folder_small", scale=1)
                load_prompts_button = gr.Button("Load", scale=1)
            
            # Hidden state to store all prompts as JSON
            sample_prompts_state = gr.State(value=[])
            
            # Wire up enable/disable sampling functionality
            def toggle_sampling(enabled):
                """Enable or disable all sampling controls"""
                return [
                    gr.update(interactive=enabled),  # sample_every_n_steps
                    gr.update(interactive=enabled),  # sample_type
                    gr.update(interactive=enabled),  # add_prompt_btn
                    gr.update(interactive=enabled),  # clear_all_prompts_btn
                    gr.update(interactive=enabled),  # sample_prompts_file
                    gr.update(interactive=enabled),  # sample_prompts_button
                    gr.update(interactive=enabled),  # load_prompts_button
                ]
            
            enable_sampling.change(
                fn=toggle_sampling,
                inputs=[enable_sampling],
                outputs=[
                    sample_every_n_steps,
                    sample_type,
                    add_prompt_btn,
                    clear_all_prompts_btn,
                    sample_prompts_file,
                    sample_prompts_button,
                    load_prompts_button,
                ],
            )
        
        with gr.Tab("Resume & Weights"):
            with gr.Row():
                resume = gr.Textbox(
                    label="Resume from State",
                    placeholder="path/to/state",
                    interactive=True,
                )
                resume_button = gr.Button(document_symbol, elem_id="open_folder_small")
            with gr.Row():
                network_weights = gr.Textbox(
                    label="Pretrained Network Weights",
                    placeholder="path/to/lora.safetensors",
                    interactive=True,
                )
                network_weights_button = gr.Button(document_symbol, elem_id="open_folder_small")
            with gr.Row():
                network_dropout = gr.Slider(
                    label="Network Dropout",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                )
                scale_weight_norms = gr.Number(
                    label="Scale Weight Norms",
                    value=1.0,
                    interactive=True,
                )
        
        with gr.Tab("Optimizer & Scheduler"):
            optimizer_args = gr.Textbox(
                label="Optimizer Args",
                placeholder='weight_decay=0.01 betas=0.9,0.999',
                interactive=True,
            )
            lr_scheduler_args = gr.Textbox(
                label="LR Scheduler Args",
                placeholder='',
                interactive=True,
            )
            with gr.Row():
                lr_scheduler_num_cycles = gr.Number(
                    label="LR Scheduler Num Cycles",
                    value=1,
                    precision=0,
                    interactive=True,
                )
                lr_scheduler_power = gr.Number(
                    label="LR Scheduler Power",
                    value=1.0,
                    interactive=True,
                )
                lr_scheduler_min_lr_ratio = gr.Number(
                    label="LR Scheduler Min LR Ratio",
                    value=0.0,
                    interactive=True,
                )
        
        with gr.Tab("Timestep Sampling"):
            with gr.Row():
                timestep_sampling = gr.Dropdown(
                    choices=["shift", "uniform", "sigmoid", "sigma", "logsnr", "qinglong_flux", "qinglong_qwen"],
                    value="shift",
                    label="Timestep Sampling Method",
                    interactive=True,
                )
                weighting_scheme = gr.Dropdown(
                    choices=["none", "logit_normal", "mode", "cosmap", "sigma_sqrt"],
                    value="none",
                    label="Weighting Scheme",
                    interactive=True,
                )
            with gr.Row():
                min_timestep = gr.Number(
                    label="Min Timestep",
                    value=0,
                    precision=0,
                    minimum=0,
                    maximum=999,
                    interactive=True,
                )
                max_timestep = gr.Number(
                    label="Max Timestep",
                    value=1000,
                    precision=0,
                    minimum=1,
                    maximum=1000,
                    interactive=True,
                )
                preserve_distribution_shape = gr.Checkbox(
                    label="Preserve Distribution Shape",
                    value=True,
                    interactive=True,
                )
        
        with gr.Tab("Logging"):
            with gr.Row():
                log_with = gr.Dropdown(
                    choices=["tensorboard", "wandb", "all"],
                    value="tensorboard",
                    label="Log With",
                    interactive=True,
                )
                log_config = gr.Checkbox(
                    label="Log Config",
                    value=False,
                    interactive=True,
                )
            with gr.Row():
                wandb_api_key = gr.Textbox(
                    label="WandB API Key",
                    type="password",
                    interactive=True,
                )
                wandb_run_name = gr.Textbox(
                    label="WandB Run Name",
                    placeholder="my_training_run",
                    interactive=True,
                )
        
        with gr.Tab("Metadata"):
            training_comment = gr.Textbox(
                label="Training Comment",
                placeholder="Optional comment",
                interactive=True,
            )
            with gr.Row():
                metadata_author = gr.Textbox(
                    label="Author",
                    interactive=True,
                )
                metadata_license = gr.Textbox(
                    label="License",
                    placeholder="e.g., openrail++",
                    interactive=True,
                )
            metadata_description = gr.Textbox(
                label="Description",
                lines=2,
                interactive=True,
            )
            metadata_tags = gr.Textbox(
                label="Tags",
                placeholder="character, anime, style",
                interactive=True,
            )
    
    # Action Buttons
    with gr.Row():
        start_training_btn = gr.Button("Start Training", elem_id="myTensorButton", variant="primary")
        stop_btn = gr.Button("Stop All", elem_id="myTensorButtonStop", variant="stop")
    
    # Output Console
    output_console = gr.Textbox(
        label="Output Console",
        lines=20,
        interactive=False,
        max_lines=50,
    )
    
    # TensorBoard
    with gr.Accordion("TensorBoard", open=False):
        with gr.Row():
            start_tb_btn = gr.Button("Start TensorBoard", elem_id="myTensorButton")
            stop_tb_btn = gr.Button("Stop TensorBoard", elem_id="myTensorButtonStop")
    
    # Button handlers
    def reset_to_defaults_handler():
        """Reset all parameters to their default values"""
        return [
            # GPU Device
            "0",  # gpu_device
            # Model section
            "t2v-14B (Text-to-Video)",  # task_mode
            "Both (alternating)",  # training_mode
            "",  # dit_low_noise
            "",  # dit_high_noise
            "",  # vae_path
            "",  # t5_path
            # Folders
            "",  # video_directory
            "outputs/wan_lora",  # output_dir
            "logs/wan22_training",  # logging_dir
            # Training Configuration
            "AdamW8bit",  # optimizer_type
            "fp16",  # mixed_precision
            4,  # gradient_accumulation_steps
            True,  # gradient_checkpointing
            True,  # fp8_base
            875,  # timestep_boundary
            5.0,  # discrete_flow_shift
            42,  # seed
            "cosine",  # lr_scheduler
            20,  # lr_warmup_steps
            1.0,  # max_grad_norm
            # Parameters
            16,  # network_dim
            8,  # network_alpha
            8e-5,  # learning_rate
            3000,  # max_train_steps
            1,  # batch_size
            121,  # target_frames
            854,  # resolution_width
            480,  # resolution_height
            "",  # output_name
            100,  # save_every_n_steps
            10,  # save_last_n_steps
            True,  # save_state
            3,  # save_last_n_steps_state
            True,  # save_state_on_train_end
            4,  # max_data_loader_n_workers
            True,  # persistent_data_loader_workers
            # Advanced Parameters
            0,  # sample_every_n_steps
            "",  # sample_prompts
            "",  # resume
            "",  # network_weights
            0.0,  # network_dropout
            1.0,  # scale_weight_norms
            "",  # optimizer_args
            "",  # lr_scheduler_args
            1,  # lr_scheduler_num_cycles
            1.0,  # lr_scheduler_power
            0.0,  # lr_scheduler_min_lr_ratio
            "shift",  # timestep_sampling
            "none",  # weighting_scheme
            0,  # min_timestep
            1000,  # max_timestep
            True,  # preserve_distribution_shape
            "tensorboard",  # log_with
            False,  # log_config
            "",  # wandb_api_key
            "",  # wandb_run_name
            "",  # training_comment
            "",  # metadata_author
            "",  # metadata_license
            "",  # metadata_description
            "",  # metadata_tags
            "✓ Reset to default values",  # config_status
        ]
    
    def check_cache_exists(video_dir, target_frames, resolution_width, resolution_height):
        """Check if cache files already exist for this dataset"""
        if not video_dir or not os.path.exists(video_dir):
            return False, "Video directory not found"
        
        # Determine cache directory (default: video_dir + "_latents")
        cache_dir = Path(video_dir).parent / f"{Path(video_dir).name}_latents"
        
        if not cache_dir.exists():
            return False, f"Cache directory not found: {cache_dir}"
        
        # Check if cache files exist
        video_files = list(Path(video_dir).glob("*.mp4")) + list(Path(video_dir).glob("*.avi")) + \
                      list(Path(video_dir).glob("*.mov")) + list(Path(video_dir).glob("*.mkv"))
        
        if not video_files:
            return False, "No video files found in directory"
        
        # Check for at least one cached latent file matching the resolution pattern
        cache_files = list(cache_dir.glob("*.npz"))
        
        if not cache_files:
            return False, "No cache files found"
        
        # Check if cache files match current resolution and frame count
        expected_pattern = f"_{resolution_width:04d}x{resolution_height:04d}_"
        matching_caches = [f for f in cache_files if expected_pattern in f.name]
        
        if not matching_caches:
            return False, f"No cache files found matching resolution {resolution_width}x{resolution_height}"
        
        log.info(f"Found {len(matching_caches)} existing cache files in {cache_dir}")
        return True, f"Cache exists: {len(matching_caches)} files found"
    
    def start_training_handler(
        dit_low, dit_high, vae, t5, video_dir, output_dir, logging_dir,
        task_mode, training_mode, network_dim, network_alpha, learning_rate, max_train_steps,
        save_every_n_steps, batch_size, target_frames, resolution_width, resolution_height,
        optimizer_type, mixed_precision, gradient_accumulation_steps, gradient_checkpointing,
        fp8_base, timestep_boundary, discrete_flow_shift, save_state, seed,
        sample_every_n_steps_val, sample_type_val, sample_prompts_text_val
    ):
        """Start training with automatic caching if needed"""
        output_lines = []
        
        # Save sample prompts to file if provided
        sample_prompts_path = None
        if sample_every_n_steps_val > 0 and sample_prompts_text_val:
            # sample_prompts_text_val should be a list of dicts from the state
            if isinstance(sample_prompts_text_val, list) and len(sample_prompts_text_val) > 0:
                # Create prompts directory in output folder
                prompts_dir = os.path.join(output_dir, "sample_prompts")
                os.makedirs(prompts_dir, exist_ok=True)
                
                # Save prompts to file in AI Toolkit format
                sample_prompts_path = os.path.join(prompts_dir, "prompts.txt")
                try:
                    with open(sample_prompts_path, 'w', encoding='utf-8') as f:
                        for prompt_data in sample_prompts_text_val:
                            # Format: prompt|guidance|steps|seed
                            line = f"{prompt_data['prompt']}|{prompt_data['guidance']}|{prompt_data['steps']}|{prompt_data['seed']}\n"
                            f.write(line)
                    
                    sample_type_str = "video" if sample_type_val == "Video" else "image"
                    output_lines.append(f"✓ Saved {len(sample_prompts_text_val)} {sample_type_str} sample prompts to: {sample_prompts_path}")
                    output_lines.append("")
                except Exception as e:
                    output_lines.append(f"⚠ Warning: Failed to save prompts: {e}")
                    output_lines.append("")
        
        # Validate required paths
        if not dit_low or not os.path.exists(dit_low):
            return "Error: Low-Noise DiT model path is required and must exist"
        if not vae or not os.path.exists(vae):
            return "Error: VAE model path is required and must exist"
        if not video_dir or not os.path.exists(video_dir):
            return "Error: Dataset directory is required and must exist"
        
        # Check training mode requirements
        if training_mode in ["High-noise only", "Both (alternating)", "Both (separate GPUs)"]:
            if not dit_high or not os.path.exists(dit_high):
                return "Error: High-Noise DiT model is required for this training mode"
        
        # Extract task name from task_mode
        task = task_mode.split(" ")[0]  # "t2v-14B" or "i2v-14B"
        
        output_lines.append("=== Musubi Tuner - WAN2.2 Training ===")
        output_lines.append(f"Task: {task}")
        output_lines.append(f"Training Mode: {training_mode}")
        output_lines.append("")
        
        # Check if cache exists
        cache_exists, cache_msg = check_cache_exists(video_dir, target_frames, resolution_width, resolution_height)
        
        if cache_exists:
            output_lines.append(f"✓ {cache_msg}")
            output_lines.append("Skipping caching step (cache already exists)")
        else:
            output_lines.append(f"⚠ Cache check: {cache_msg}")
            output_lines.append("Caching will be performed automatically before training...")
            output_lines.append("")
            output_lines.append("TODO: Implement automatic caching:")
            output_lines.append(f"  1. Run wan_cache_latents.py --task {task} for {video_dir}")
            output_lines.append(f"  2. Run wan_cache_text_encoder_outputs.py --task {task}")
            output_lines.append(f"  3. Target: {target_frames} frames @ {resolution_width}x{resolution_height}")
        
        output_lines.append("")
        output_lines.append("TODO: Implement training:")
        output_lines.append(f"  - Task: {task}")
        output_lines.append(f"  - LoRA Rank: {network_dim}, Alpha: {network_alpha}")
        output_lines.append(f"  - Learning Rate: {learning_rate}")
        output_lines.append(f"  - Steps: {max_train_steps}, Save every: {save_every_n_steps}")
        output_lines.append(f"  - Optimizer: {optimizer_type}, Precision: {mixed_precision}")
        output_lines.append("")
        if task.startswith("i2v"):
            output_lines.append("Note: I2V training uses video clips with first-frame conditioning")
            output_lines.append("      The trained LoRA will work with I2V models for image-to-video generation")
        
        return "\n".join(output_lines)
    
    def stop_handler():
        return "Stop not yet implemented"
    
    start_training_btn.click(
        fn=start_training_handler,
        inputs=[
            dit_low_noise, dit_high_noise, vae_path, t5_path, video_directory, output_dir, logging_dir,
            task_mode, training_mode, network_dim, network_alpha, learning_rate, max_train_steps,
            save_every_n_steps, batch_size, target_frames, resolution_width, resolution_height,
            optimizer_type, mixed_precision, gradient_accumulation_steps, gradient_checkpointing,
            fp8_base, timestep_boundary, discrete_flow_shift, save_state, seed,
            sample_every_n_steps, sample_type, sample_prompts_state
        ],
        outputs=output_console,
    )
    
    stop_btn.click(
        fn=stop_handler,
        inputs=[],
        outputs=output_console,
    )
    
    # Configuration file handlers
    def load_config_handler(config_path):
        """Load configuration from TOML file"""
        if not config_path:
            return {
                config_status: "Error: No configuration file specified",
            }
        
        if not os.path.exists(config_path):
            return {
                config_status: f"Error: Configuration file not found: {config_path}",
            }
        
        try:
            import toml
            config = toml.load(config_path)
            
            # TODO: Apply loaded config to all UI components
            # For now, just show success message and expand all accordions
            
            return {
                config_status: f"✓ Loaded configuration from: {config_path}",
                config_file_name: config_path,
                model_accordion: gr.Accordion(open=True),
                folders_accordion: gr.Accordion(open=True),
                training_config_accordion: gr.Accordion(open=True),
                parameters_accordion: gr.Accordion(open=True),
                advanced_params_accordion: gr.Accordion(open=True),
            }
        except Exception as e:
            return {
                config_status: f"Error loading configuration: {str(e)}",
            }
    
    def save_config_handler(config_path, *args):
        """Save current configuration to TOML file"""
        if not config_path:
            return "Error: No configuration file name specified"
        
        # Add .toml extension if not present
        if not config_path.endswith('.toml'):
            config_path = config_path + '.toml'
        
        try:
            import toml
            
            # TODO: Gather all current UI values and save to TOML
            # For now, just create a placeholder config
            config = {
                "note": "Configuration save functionality not yet fully implemented",
                "file": config_path,
            }
            
            with open(config_path, 'w') as f:
                toml.dump(config, f)
            
            return f"✓ Configuration saved to: {config_path}"
        except Exception as e:
            return f"Error saving configuration: {str(e)}"
    
    def save_as_config_handler():
        """Open file dialog for Save As"""
        from .common_gui import get_saveasfile_path
        file_path = get_saveasfile_path(
            file_path="",
            defaultextension=".toml",
            extension_name="Config files"
        )
        if file_path:
            # Save the config to the new file path
            try:
                # TODO: Actually save the config here
                return file_path
            except Exception as e:
                log.error(f"Failed to save config: {e}")
                return ""
        return ""
    
    # Connect config buttons
    config_file_picker.click(
        fn=get_file_path,
        inputs=[config_file_name, gr.Textbox(value="*.toml", visible=False), gr.Textbox(value="Config files", visible=False)],
        outputs=config_file_name,
        show_progress=False,
    )
    
    config_load_button.click(
        fn=load_config_handler,
        inputs=[config_file_name],
        outputs=[config_status, config_file_name, model_accordion, folders_accordion, training_config_accordion, parameters_accordion, advanced_params_accordion],
    )
    
    config_save_button.click(
        fn=save_config_handler,
        inputs=[config_file_name],
        outputs=config_status,
    )
    
    config_save_as_button.click(
        fn=save_as_config_handler,
        inputs=[],
        outputs=config_file_name,
    )
    
    config_reset_button.click(
        fn=reset_to_defaults_handler,
        inputs=[],
        outputs=[
            gpu_device,
            task_mode, training_mode, dit_low_noise, dit_high_noise, vae_path, t5_path,
            video_directory, output_dir, logging_dir,
            optimizer_type, mixed_precision, gradient_accumulation_steps, gradient_checkpointing,
            fp8_base, timestep_boundary, discrete_flow_shift, seed,
            lr_scheduler, lr_warmup_steps, max_grad_norm,
            network_dim, network_alpha, learning_rate, max_train_steps,
            batch_size, target_frames, resolution_width, resolution_height,
            output_name, save_every_n_steps, save_last_n_steps, save_state, save_last_n_steps_state, save_state_on_train_end,
            max_data_loader_n_workers, persistent_data_loader_workers,
            sample_every_n_steps, sample_type, sample_prompts_state, sample_prompts_file, resume, network_weights, network_dropout,
            scale_weight_norms, optimizer_args, lr_scheduler_args, lr_scheduler_num_cycles,
            lr_scheduler_power, lr_scheduler_min_lr_ratio, timestep_sampling, weighting_scheme,
            min_timestep, max_timestep, preserve_distribution_shape, log_with, log_config,
            wandb_api_key, wandb_run_name, training_comment, metadata_author, metadata_license,
            metadata_description, metadata_tags, config_status,
        ],
    )
    
    # Dynamic prompt management functions
    def add_prompt_row():
        """Add a new prompt row (handled by JavaScript)"""
        return gr.update()
    
    def remove_prompt_row():
        """Remove a prompt row (handled by JavaScript)"""
        return gr.update()
    
    def clear_all_prompts():
        """Clear all prompts (handled by JavaScript)"""
        return gr.update()
    
    def load_prompts_from_file(file_path):
        """Load prompts from file and parse into list"""
        if not file_path or not os.path.exists(file_path):
            return []
        try:
            prompts = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Simple format: just the prompt text
                        # Advanced format: prompt|guidance|steps|seed
                        parts = line.split('|')
                        prompt_data = {
                            'prompt': parts[0].strip(),
                            'guidance': float(parts[1]) if len(parts) > 1 else 3.5,
                            'steps': int(parts[2]) if len(parts) > 2 else 20,
                            'seed': int(parts[3]) if len(parts) > 3 else -1,
                        }
                        prompts.append(prompt_data)
            return prompts
        except Exception as e:
            log.error(f"Failed to load prompts from {file_path}: {e}")
            return []
    
    # Wire up file picker buttons for advanced parameters
    sample_prompts_button.click(
        fn=get_file_path,
        inputs=[sample_prompts_file, gr.Textbox(value="*.txt", visible=False), gr.Textbox(value="Prompts file", visible=False)],
        outputs=sample_prompts_file,
        show_progress=False,
    )
    
    # Wire up prompt management buttons
    add_prompt_btn.click(
        fn=None,
        inputs=[],
        outputs=[],
        js="""
        () => {
            // Find the container
            const container = document.querySelector('.sample_prompts_container');
            if (!container) return;
            
            // Clone the first prompt row
            const firstRow = container.querySelector('.sample_prompt_row');
            if (!firstRow) return;
            
            const newRow = firstRow.cloneNode(true);
            
            // Clear values in the cloned row
            newRow.querySelectorAll('input, textarea').forEach(input => {
                if (input.type === 'number') {
                    // Reset to defaults
                    if (input.closest('[label="Guidance"]')) input.value = '3.5';
                    else if (input.closest('[label="Steps"]')) input.value = '20';
                    else if (input.closest('[label="Seed"]')) input.value = '43';
                } else {
                    input.value = '';
                }
            });
            
            // Add the new row
            container.appendChild(newRow);
        }
        """
    )
    
    # Note: Remove button functionality will be handled by event delegation in JavaScript
    # Clear all will be handled similarly
    
    load_prompts_button.click(
        fn=load_prompts_from_file,
        inputs=[sample_prompts_file],
        outputs=[sample_prompts_state],
        show_progress=False,
    )
    
    resume_button.click(
        fn=get_folder_path,
        inputs=resume,
        outputs=resume,
        show_progress=False,
    )
    
    network_weights_button.click(
        fn=get_file_path,
        inputs=[network_weights, gr.Textbox(value="*.safetensors *.pt *.pth", visible=False), gr.Textbox(value="Network weights", visible=False)],
        outputs=network_weights,
        show_progress=False,
    )
    
    # Task mode change handler
    def on_task_mode_change(task):
        task_name = task.split(" ")[0]  # Extract "t2v-14B" or "i2v-14B"
        if task_name.startswith("i2v"):
            return "**I2V Mode:** Training LoRA to improve likeness consistency in image-to-video generation. Trains on video clips and/or images to help the model better maintain character/object identity from the starting frame."
        else:
            return "**T2V Mode:** Training LoRA for text-to-video generation. Can train on video clips and/or images with text prompts."
    
    task_mode.change(
        fn=on_task_mode_change,
        inputs=[task_mode],
        outputs=task_info,
    )

