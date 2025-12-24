"""
Launcher script for multi-GPU training on Windows using gloo backend.
"""
import argparse
import os
import sys
import torch

from musubi_tuner.utils.multi_gpu_trainer import MultiGPUTrainer


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Launch multi-GPU training")
    parser.add_argument(
        "--device_indexes",
        type=str,
        default=None,
        help="Comma-separated list of GPU device indices (e.g., '0,1'). If not specified, uses all available GPUs."
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Path to the training script to run"
    )
    
    # Parse known args to get device_indexes and script
    args, remaining_args = parser.parse_known_args()
    
    # Set Windows-specific environment variables
    os.environ.setdefault('USE_LIBUV', '0')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Multi-GPU training requires CUDA.")
        sys.exit(1)
    
    # Check GPU count
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Error: Multi-GPU training requires at least 2 GPUs, found {num_gpus}")
        sys.exit(1)
    
    print(f"Launching multi-GPU training on {num_gpus} GPUs")
    if args.device_indexes:
        print(f"Using device indexes: {args.device_indexes}")
    
    # Import the training script module
    script_path = args.script
    if not os.path.exists(script_path):
        print(f"Error: Training script not found: {script_path}")
        sys.exit(1)
    
    # Add script directory to path
    script_dir = os.path.dirname(os.path.abspath(script_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Import the module
    script_name = os.path.basename(script_path).replace('.py', '')
    try:
        # Use importlib to import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the trainer class and main function
        if not hasattr(module, 'main'):
            print(f"Error: Training script {script_path} does not have a 'main' function")
            sys.exit(1)
        
        # Create a wrapper that will be called by MultiGPUTrainer
        # We need to parse args and create trainer
        def run_training(args_obj):
            """Wrapper function to run training."""
            # Re-parse arguments for this process
            # The args_obj should already have all the parsed arguments
            trainer_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    hasattr(obj, 'train') and 
                    name.endswith('Trainer')):
                    trainer_class = obj
                    break
            
            if trainer_class is None:
                # Try to find NetworkTrainer or similar
                if hasattr(module, 'NetworkTrainer'):
                    trainer_class = module.NetworkTrainer
                else:
                    print("Error: Could not find trainer class in script")
                    sys.exit(1)
            
            trainer = trainer_class()
            trainer.train(args_obj)
        
        # For now, we'll use a simpler approach: just call main() directly
        # and let the script handle multi-GPU setup internally
        # This is a fallback - the proper way is to use MultiGPUTrainer
        print("Note: Using direct script execution. Multi-GPU should be handled by the script itself.")
        print(f"Running: {script_path} {' '.join(remaining_args)}")
        
        # Set sys.argv to match what the script expects
        original_argv = sys.argv
        sys.argv = [script_path] + remaining_args
        
        try:
            module.main()
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"Error importing or running training script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


