"""
Multi-GPU trainer wrapper for gloo-based distributed training on Windows.
Based on OneTrainer-Plus MultiTrainer implementation.
"""
import datetime
import os
import platform
import socket
import time
import traceback
from typing import Optional, List, Callable, Any

import torch
import torch.distributed as dist

from musubi_tuner.utils import multi_gpu_util as multi


class MultiGPUTrainer:
    """
    Wrapper class for multi-GPU training using torch.multiprocessing.spawn
    with gloo backend (Windows) or nccl backend (Linux).
    """
    
    def __init__(self, device_indexes: Optional[str] = None):
        """
        Initialize MultiGPUTrainer.
        
        Args:
            device_indexes: Comma-separated list of device indices (e.g., "0,1")
                          If None, uses all available GPUs.
        """
        self.device_indexes = device_indexes
        self._setup_master_addr_port()
    
    def _setup_master_addr_port(self):
        """Set up MASTER_ADDR and MASTER_PORT environment variables."""
        # Force explicit loopback to avoid hostname resolution issues on Windows
        # (e.g., hostnames resolving to IPv6 link-local addresses can break Gloo).
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        # Disable libuv store on Windows (matches existing batch-file guidance)
        os.environ.setdefault("USE_LIBUV", "0")
        
        # Select a free TCP port if none provided to avoid bind conflicts
        if 'MASTER_PORT' not in os.environ or not os.environ['MASTER_PORT']:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('127.0.0.1', 0))
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    free_port = sock.getsockname()[1]
            except Exception:
                # Fall back to a common default if dynamic allocation fails
                free_port = 29500
            os.environ['MASTER_PORT'] = str(free_port)
    
    @staticmethod
    def _train_process(
        spawn_rank: int,
        world_size: int,
        trainer_class: type,
        args: Any,
        devices: Optional[List[torch.device]] = None
    ):
        """
        Static method that runs in each spawned process.
        Must be static for torch.multiprocessing.spawn to work.
        
        Args:
            spawn_rank: Rank from spawn (0-based, but we'll use spawn_rank+1 as actual rank)
            world_size: Total number of processes
            trainer_class: The trainer class to instantiate
            args: Arguments namespace or dict to pass to train method
            devices: List of devices to use (if None, auto-detect)
        """
        # Clean up any existing process group before initializing a new one
        # This handles the case where processes were killed immediately and cleanup didn't happen
        needs_wait = False
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
                needs_wait = True
            except Exception:
                pass
        
        # If we just cleaned up, wait a bit longer for ports/sockets to be fully released
        if needs_wait:
            time.sleep(2.0)
        
        # Calculate actual rank (spawn_rank is 0-based from spawn, but we want 1-based for distributed)
        # Main process will be rank 0, spawned processes will be ranks 1, 2, ...
        rank = spawn_rank + 1 if spawn_rank >= 0 else 0
        
        # Determine device
        if devices:
            device = devices[rank] if rank < len(devices) else devices[0]
        else:
            device = torch.device(f"cuda:{rank}")
        
        # Set timeout to 24 hours for long-running operations
        timeout = datetime.timedelta(hours=24)
        
        # Wait a bit longer if we just cleaned up a process group
        if needs_wait:
            time.sleep(2.0)
        
        # Force use of localhost/127.0.0.1 to avoid issues on Windows
        master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
        master_port = os.environ.get('MASTER_PORT', '29500')
        init_method = f'tcp://{master_addr}:{master_port}'
        
        # Initialize process group
        backend = 'gloo' if platform.system() == 'Windows' else 'nccl'
        if backend == "gloo":
            # Force Gloo to bind to a concrete IPv4 address (loopback) to avoid
            # ProcessGroupGloo::makeDeviceForHostname() failures on Windows.
            # NOTE: In torch 2.7 on Windows, `ProcessGroupGloo.Options(...)` returns a generic
            # `Backend.Options` object without a public `devices` field. The gloo-specific options
            # object is `ProcessGroupGloo._Options()` and exposes `_devices` / `_timeout`.
            pg_options = dist.ProcessGroupGloo._Options()
            pg_options._timeout = timeout
            pg_options._devices = [dist.ProcessGroupGloo.create_device(hostname=master_addr)]
            dist.init_process_group(
                rank=rank,
                world_size=world_size,
                timeout=timeout,
                backend=backend,
                init_method=init_method,
                pg_options=pg_options,
            )
        else:
            dist.init_process_group(
                rank=rank,
                world_size=world_size,
                timeout=timeout,
                backend=backend,
                init_method=init_method,
            )
        
        # Set CUDA device
        if device.type == 'cuda':
            torch.cuda.set_device(device.index)
        
        # Synchronize GPUs to discover communication issues early
        if multi.is_master():
            print("Synchronizing GPUs. If this stalls, this likely means that your distributed backend is broken:")
        for _ in multi.sequential():
            device_name = torch.cuda.get_device_name(device.index) if device.type == 'cuda' else str(device)
            current_cuda = torch.cuda.current_device() if device.type == "cuda" else None
            print(f"GPU #{multi.rank()}  device: {device} ({device_name})  "
                  f"current_cuda: {current_cuda}  backend: {dist.get_backend()}  world size: {dist.get_world_size()}")
        if multi.is_master():
            print("GPUs synchronized.")
        
        # Mark args as multi-GPU mode
        if hasattr(args, '__dict__'):
            setattr(args, '_multi_gpu', True)
            setattr(args, '_multi_gpu_device', device)
            setattr(args, '_multi_gpu_rank', rank)
            setattr(args, '_multi_gpu_world_size', world_size)
        elif isinstance(args, dict):
            args['_multi_gpu'] = True
            args['_multi_gpu_device'] = device
            args['_multi_gpu_rank'] = rank
            args['_multi_gpu_world_size'] = world_size
        
        # Create trainer instance and call train method
        immediate_termination = False
        try:
            trainer = trainer_class()
            trainer.train(args)
        except (KeyboardInterrupt, SystemExit):
            immediate_termination = True
            raise
        except Exception as e:
            traceback.print_exc()
            raise
        finally:
            # Skip cleanup for immediate termination
            if not immediate_termination:
                # Synchronize all ranks before destroying process group to prevent deadlock
                if dist.is_initialized():
                    dist.barrier()
                dist.destroy_process_group()
    
    def train(self, trainer_class: type, args: Any):
        """
        Spawn multiple processes for multi-GPU training.
        
        Args:
            trainer_class: The trainer class to instantiate in each process
            args: Arguments namespace or dict to pass to train method
        """
        # Determine devices and world size
        if self.device_indexes:
            device_indexes_list = [int(d.strip()) for d in self.device_indexes.split(',') if d.strip()]
            devices = [torch.device(f"cuda:{idx}") for idx in device_indexes_list]
            world_size = len(devices)
        else:
            devices = None
            world_size = torch.cuda.device_count()
            if world_size == 0:
                raise RuntimeError("No CUDA devices available for multi-GPU training")
        
        if world_size < 2:
            raise RuntimeError(f"Multi-GPU training requires at least 2 GPUs, found {world_size}")
        
        # Spawn worker processes (world_size - 1 workers, main process is rank 0)
        workers = torch.multiprocessing.spawn(
            MultiGPUTrainer._train_process,
            args=(world_size, trainer_class, args, devices),
            nprocs=world_size - 1,
            join=False
        )
        
        # Run main process as rank 0
        MultiGPUTrainer._train_process(-1, world_size, trainer_class, args, devices)
        
        # Wait for all workers to complete
        workers.join()

