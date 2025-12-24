import os
import sys
import argparse
import contextlib
import inspect
import gradio as gr

from musubi_gui.wan22_train_tab import wan22_train_tab
from musubi_gui.custom_logging import setup_logging

PYTHON = sys.executable
project_dir = os.path.dirname(os.path.abspath(__file__))

# Set up logging
log = setup_logging()

# Function to read file content, suppressing any FileNotFoundError
def read_file_content(file_path):
    with contextlib.suppress(FileNotFoundError):
        with open(file_path, "r", encoding="utf8") as file:
            return file.read()
    return ""

# Function to initialize the Gradio UI interface
def initialize_ui_interface():
    # Load custom CSS if available
    css = read_file_content("./musubi_gui/assets/style.css")

    # Create the main Gradio Blocks interface with JS parameter for tooltips
    tooltip_js = """
    function() {
        console.log('🔧 Tooltip script loaded');
        
        function applyTooltips() {
            const tooltips = {
                'GPU Device': 'Which GPU(s) to use for training. Single: "0" or "1". Both: "0,1". Check with nvidia-smi.',
                'Training Task': 'T2V: Trains on text+video pairs for text-to-video generation. I2V: Trains on videos with first-frame conditioning for better likeness consistency in image-to-video.',
                'Training Mode': 'Both alternating: Single GPU switches between models (recommended). Low-noise only: Trains final denoising steps. High-noise only: Trains initial denoising steps.',
                'Low-Noise DiT': 'The 14B parameter diffusion transformer for final denoising steps (clean details). Required for all training modes. Use fp16 version for 96GB VRAM.',
                'High-Noise DiT': 'The 14B parameter diffusion transformer for initial denoising steps (structure/composition). Required for "Both" and "High-noise only" modes.',
                'VAE Model': 'Video Auto-Encoder that compresses videos into latent space. Use wan_2.1_vae.safetensors. Same for WAN 2.1 and 2.2.',
                'T5 Text Encoder': 'Text encoder that converts captions to embeddings. Use models_t5_umt5-xxl-enc-bf16.pth. Processes your .txt caption files.',
                'Dataset Directory': 'Folder with your training videos (.mp4) and matching caption files (.txt). Example: video001.mp4 needs video001.txt with description.',
                'Output Directory': 'Where trained LoRA .safetensors files are saved. Load these in ComfyUI/Forge to use your trained style/character.',
                'Logging Directory': 'TensorBoard logs location. Run "tensorboard --logdir <this_path>" to monitor training loss, learning rate, and progress graphs.',
                'Optimizer': 'AdamW8bit: Best for VRAM efficiency, minimal quality loss. AdamW: Standard, uses more VRAM. Adafactor: Adaptive learning rate, experimental.',
                'Optimizer Type': 'AdamW8bit: Best for VRAM efficiency, minimal quality loss. AdamW: Standard, uses more VRAM. Adafactor: Adaptive learning rate, experimental.',
                'Mixed Precision': 'fp16: Fastest, lowest VRAM, required for fp16 models. bf16: Better numerical stability for fp32 models. no: Full precision, highest VRAM.',
                'Gradient Accumulation': 'Simulates larger batch sizes. Effective batch = batch_size × this value. Use 2-4 to fit training in VRAM. Higher = more stable but slower.',
                'Gradient Checkpointing': 'Saves VRAM by recomputing activations during backward pass. Trades ~20% speed for ~40% VRAM savings. Enable if OOM errors.',
                'FP8 Base': 'Runs base model in FP8 precision to save VRAM. Experimental. Can reduce quality. Try if OOM with fp16.',
                'Timestep Boundary': 'Timestep (0-1000) where training switches from high-noise to low-noise model. 875 = high-noise handles 0-875, low-noise handles 876-1000. Lower = more low-noise training.',
                'Discrete Flow Shift': 'Shifts timestep distribution for flow matching. Higher (5.0-7.0) = trains on harder timesteps. 3.0 = balanced. Affects convergence speed.',
                'Seed': 'Random seed for reproducibility. Same seed + settings = identical results. Use -1 for random. Set specific number to reproduce training.',
                'LR Scheduler': 'cosine: Smooth decay to 0. constant: Fixed LR. cosine_with_restarts: Periodic resets. polynomial: Gradual decay. Cosine recommended for most cases.',
                'LR Warmup Steps': 'Gradually increases LR from 0 to target over N steps. Prevents early training instability. 10-50 typical. Higher for large models.',
                'Max Gradient Norm': 'Clips gradients to prevent exploding. 1.0 = standard. 0 = no clipping (risky). Increase to 2.0 if training diverges.',
                'Network Dim': 'LoRA rank = model capacity. 4: Fast, small files, less detail. 8-16: Balanced (recommended). 32-64: High detail, large files, slower. Higher ≠ always better.',
                'Network Alpha': 'Scales LoRA weight influence. Typically rank/2 or equal to rank. Lower = subtler effect when merged. Higher = stronger effect but risk overfitting.',
                'Learning Rate': 'Step size for weight updates. 1e-4 to 8e-5 typical for video LoRA. Too high = unstable/divergence. Too low = slow convergence. Start conservative.',
                'Max Train Steps': 'Total training iterations. More steps = longer training. 500-3000 typical for video LoRA. Monitor loss curve to know when to stop.',
                'Batch Size': 'Videos processed per step. Higher = faster but more VRAM. 1 typical for 121-frame video @ 480p on 96GB. Use gradient accumulation for effective larger batch.',
                'Target Frames': 'Frames per video clip. 17: Fast training, less motion. 121: Better motion learning, more VRAM. For IMAGE training: Set to 1. Must match your cache. Affects training time significantly.',
                'Resolution Width': 'Max width for training videos. Videos are downscaled to fit. 854 (480p), 1280 (720p), 1920 (1080p). Higher = more detail but more VRAM.',
                'Resolution Height': 'Max height for training videos. 480 (480p), 720 (720p), 1080 (1080p). Aspect ratio preserved with bucketing. Higher = more VRAM.',
                'Output Name': 'Base filename for saved models. Example: "my_character" → "my_character-100.safetensors", "my_character-200.safetensors", etc.',
                'Save Every N Steps': 'Checkpoint frequency. 100 = save every 100 steps. Lower = more checkpoints (disk space). Higher = fewer checkpoints. Balance between safety and storage.',
                'Save Last N Steps': 'Keep only last N checkpoints to save disk space. 0 = keep all. 5-10 typical. Old checkpoints auto-deleted.',
                'Save Training State': 'Saves optimizer state, scheduler, RNG for exact resume. Large files (~20GB). Enable if training takes days and may need to pause.',
                'Save Last N Steps State': 'Keep only last N training states. States are huge. 1-3 typical. 0 = keep all (not recommended).',
                'Save State on Train End': 'Saves final training state when complete. Useful for continuing training later or fine-tuning further.',
                'Max Data Loader Workers': 'Parallel CPU threads for loading data. 4-8 typical. Higher = faster data loading but more CPU/RAM. 0 = main thread only.',
                'Persistent Data Loader Workers': 'Keeps data loader workers alive between epochs. Faster epoch transitions but uses more RAM. Recommended for multi-epoch training.',
                'Sample Every N Steps': 'Generate test videos every N steps to monitor progress. 0 = disabled. 100-500 typical. WARNING: Very slow, adds 2-5 min per sample.',
                'Sample Prompts': 'Text file with prompts for sample generation. One prompt per line. Used with "Sample Every N Steps". Helps visualize training progress.',
                'Resume from State': 'Path to saved training state to continue from. Must match model architecture. Resumes from exact step, optimizer state, and scheduler.',
                'Pretrained Network': 'Start from existing LoRA weights instead of random init. Useful for fine-tuning or transfer learning. Must match network dim/alpha.',
                'Network Dropout': 'Randomly drops LoRA weights during training to prevent overfitting. 0 = no dropout. 0.1-0.3 typical if overfitting. Usually not needed.',
                'Scale Weight Norms': 'Normalizes LoRA weights to prevent them from dominating base model when merged. 1.0 = standard normalization. 0 = disabled.',
                'Optimizer Args': 'Additional optimizer arguments as key=value pairs. Example: "weight_decay=0.01 eps=1e-8". Advanced users only.',
                'LR Scheduler Args': 'Additional scheduler arguments. Example for cosine_with_restarts: "num_cycles=3". Check scheduler docs for options.',
                'LR Scheduler Num Cycles': 'Number of restart cycles for cosine_with_restarts scheduler. 1 = no restarts. 2-4 = periodic LR resets. Can help escape local minima.',
                'LR Scheduler Power': 'Power factor for polynomial scheduler. 1.0 = linear decay. 2.0 = quadratic. Higher = slower initial decay, faster final decay.',
                'LR Scheduler Min LR Ratio': 'Minimum LR as ratio of initial LR. 0.0 = decay to zero. 0.1 = decay to 10% of initial. Prevents LR from getting too small.',
                'Timestep Sampling': 'How to sample timesteps during training. shift: Flow matching (recommended for WAN 2.2). uniform: Equal probability. logit_normal: Focuses on mid-range.',
                'Weighting Scheme': 'Weights different timesteps during loss calculation. none: Uniform (recommended). snr: Signal-to-noise ratio weighting. Can affect convergence.',
                'Min Timestep': 'Minimum timestep for training (0-999). 0 = cleanest images. Higher = skip early denoising steps. Usually keep at 0.',
                'Max Timestep': 'Maximum timestep for training (1-1000). 1000 = noisiest images. Lower = skip late denoising steps. Usually keep at 1000.',
                'Preserve Distribution': 'Maintains original timestep distribution when using discrete flow shift. Recommended to enable for WAN 2.2 training.',
                'Log With': 'Logging backend. tensorboard: Local web UI (recommended). wandb: Cloud tracking with team features. all: Both.',
                'Log Config': 'Saves training config (all parameters) to logs for reproducibility. Recommended to enable. Helps track what settings produced which results.',
                'WandB API Key': 'Your Weights & Biases API key from wandb.ai/authorize. Required only if using wandb logging. Free tier available.',
                'WandB Run Name': 'Name for this training run in WandB dashboard. Helps organize experiments. Example: "character_v1_highres".',
                'Training Comment': 'Personal note about this training run. Saved in model metadata. Example: "First attempt with 121 frames, 720p".',
                'Metadata Author': 'Your name/username. Saved in LoRA metadata. Shows up when others inspect your model. Optional but recommended.',
                'Metadata License': 'License for your LoRA. Example: "openrail++", "cc-by-4.0", "proprietary". Important for sharing models publicly.',
                'Metadata Description': 'What your LoRA does. Example: "Anime character style LoRA trained on 200 clips". Helps users understand your model.',
                'Metadata Tags': 'Comma-separated tags for categorization. Example: "character,anime,style,fantasy". Helps with discovery and organization.'
            };
            
            let count = 0;
            let tooltipEl = null;
            let tooltipTimeout = null;
            
            // Apply tooltips - search ALL elements with text matching our keys
            const allElements = document.querySelectorAll('*');
            allElements.forEach(el => {
                // Only check text nodes, skip if has children (to avoid parent containers)
                if (el.children.length > 0) return;
                
                const text = el.textContent.trim();
                // Skip if text is empty or too long
                if (!text || text.length > 100) return;
                
                for (const [key, tooltip] of Object.entries(tooltips)) {
                    // Match if text equals key OR if text contains key (for labels with extra formatting)
                    if (text === key || text.includes(key)) {
                        el.setAttribute('data-tooltip', tooltip);
                        el.style.cursor = 'help';
                        count++;
                        console.log(`Applied tooltip to "${key}" on element:`, el.tagName, el.className, `(text: "${text}")`);
                        
                        // Add hover listeners to show/hide tooltip with delay
                        el.addEventListener('mouseenter', (e) => {
                            // Clear any existing timeout
                            if (tooltipTimeout) {
                                clearTimeout(tooltipTimeout);
                            }
                            
                            // Wait 1 second before showing tooltip
                            tooltipTimeout = setTimeout(() => {
                                // Create tooltip element
                                tooltipEl = document.createElement('div');
                                tooltipEl.textContent = tooltip;
                                tooltipEl.style.cssText = `
                                    position: fixed;
                                    background: #0f0f0f;
                                    color: #b0b0b0;
                                    padding: 8px 12px;
                                    border: 1px solid #3a3a3a;
                                    border-radius: 6px;
                                    font-size: 13px;
                                    line-height: 1.5;
                                    max-width: 350px;
                                    z-index: 999999;
                                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
                                    pointer-events: none;
                                `;
                                
                                // Position it below and to the right of mouse cursor (stays fixed)
                                tooltipEl.style.left = (e.clientX + 15) + 'px';
                                tooltipEl.style.top = (e.clientY + 15) + 'px';
                                
                                document.body.appendChild(tooltipEl);
                            }, 1000);
                        });
                        
                        el.addEventListener('mouseleave', () => {
                            // Clear timeout if mouse leaves before tooltip appears
                            if (tooltipTimeout) {
                                clearTimeout(tooltipTimeout);
                                tooltipTimeout = null;
                            }
                            
                            // Remove tooltip if it exists
                            if (tooltipEl) {
                                tooltipEl.remove();
                                tooltipEl = null;
                            }
                        });
                        
                        break;
                    }
                }
            });
            
            console.log('✅ Tooltips applied to ' + count + ' labels');
        }
        
        setTimeout(applyTooltips, 500);
        setTimeout(applyTooltips, 1500);
        
        // Handle dynamic prompt row removal
        document.addEventListener('click', function(e) {
            if (e.target.textContent === '🗑️' && e.target.tagName === 'BUTTON') {
                const row = e.target.closest('.sample_prompt_row');
                if (row) {
                    const container = row.parentElement;
                    // Don't allow removing the last row
                    if (container.querySelectorAll('.sample_prompt_row').length > 1) {
                        row.remove();
                    }
                }
            }
        });
        
        // Collect all prompts before training starts
        function collectAllPrompts() {
            const prompts = [];
            document.querySelectorAll('.sample_prompt_row').forEach(row => {
                const promptInput = row.querySelector('.sample_prompt_text input, .sample_prompt_text textarea');
                const guidanceInput = row.querySelector('[label="Guidance"] input');
                const stepsInput = row.querySelector('[label="Steps"] input');
                const seedInput = row.querySelector('[label="Seed"] input');
                
                if (promptInput && promptInput.value.trim()) {
                    prompts.push({
                        prompt: promptInput.value.trim(),
                        guidance: parseFloat(guidanceInput?.value || 3.5),
                        steps: parseInt(stepsInput?.value || 20),
                        seed: parseInt(seedInput?.value || -1)
                    });
                }
            });
            
            // Store in hidden state
            const stateInput = document.querySelector('[data-testid="sample_prompts_state"] input');
            if (stateInput) {
                stateInput.value = JSON.stringify(prompts);
                stateInput.dispatchEvent(new Event('input', { bubbles: true }));
            }
            
            return prompts;
        }
        
        // Hook into start training button
        setTimeout(() => {
            const startBtn = document.querySelector('button:has-text("Start Training")');
            if (startBtn) {
                startBtn.addEventListener('click', collectAllPrompts);
            }
        }, 1000);
    }
    """

    theme = gr.themes.Default()

    # Gradio 6 moved css/theme/js from Blocks() -> launch(). We'll support both.
    gradio_major = int(getattr(gr, "__version__", "0").split(".")[0] or 0)
    if gradio_major >= 6:
        ui_interface = gr.Blocks(title="Musubi Tuner GUI")
    else:
        ui_interface = gr.Blocks(css=css, title="Musubi Tuner GUI", theme=theme, js=tooltip_js)

    with ui_interface:
        gr.Markdown("# Musubi Tuner")
        gr.Markdown("*Train video LoRAs with WAN2.2*")
        
        # Create tabs for different functionalities
        with gr.Tab("WAN2.2 Training"):
            wan22_train_tab()
        with gr.Tab("About"):
            gr.Markdown("## Musubi Tuner")
            gr.Markdown("A training framework for video generation models.")
            gr.Markdown("Based on Kohya SS GUI structure.")

    return ui_interface, css, tooltip_js, theme

# Function to configure and launch the UI
def UI(**kwargs):
    log.info(f"Starting Musubi Tuner GUI")
    log.info(f"headless: {kwargs.get('headless', False)}")

    # Initialize the Gradio UI interface
    ui_interface, css, tooltip_js, theme = initialize_ui_interface()

    # Construct launch parameters
    launch_params = {
        "server_name": kwargs.get("listen", "127.0.0.1"),
        "server_port": kwargs.get("server_port", 7860),
        "share": kwargs.get("share", False),
        "inbrowser": kwargs.get("inbrowser", True),
    }

    # Add Gradio 6.0+ compatible parameters (css/theme/js moved to launch()) if supported.
    launch_sig = inspect.signature(ui_interface.launch)
    if "css" in launch_sig.parameters:
        launch_params["css"] = css
    if "theme" in launch_sig.parameters:
        launch_params["theme"] = theme
    if "js" in launch_sig.parameters:
        launch_params["js"] = tooltip_js

    # Launch the Gradio interface
    ui_interface.launch(**launch_params)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
        help="IP to listen on for connections to Gradio",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--inbrowser", action="store_true", help="Open in browser"
    )
    parser.add_argument(
        "--share", action="store_true", help="Share the gradio UI"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode"
    )

    args = parser.parse_args()

    UI(
        listen=args.listen,
        server_port=args.server_port,
        inbrowser=args.inbrowser,
        share=args.share,
        headless=args.headless,
    )


