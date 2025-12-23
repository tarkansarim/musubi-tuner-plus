"""
Accelerator-like wrapper for multi-GPU training using gloo backend.
This provides a compatible interface with accelerate.Accelerator for easier integration.
"""
import os
import platform
import torch
from typing import Optional, Any, List
from contextlib import contextmanager

from musubi_tuner.utils import multi_gpu_util as multi


class MultiGPUAccelerator:
    """
    Accelerator-like wrapper for multi-GPU training.
    Provides a compatible interface with accelerate.Accelerator.
    """
    
    def __init__(
        self,
        device: torch.device,
        mixed_precision: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        log_with: Optional[str] = None,
        project_dir: Optional[str] = None,
    ):
        """
        Initialize MultiGPUAccelerator.
        
        Args:
            device: The device to use for this process
            mixed_precision: Mixed precision mode ('fp16', 'bf16', or None)
            gradient_accumulation_steps: Number of gradient accumulation steps
        """
        self._device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._log_with = log_with
        self._project_dir = project_dir
        self._step = 0
        self._sync_gradients = True
        self._scaler = None
        self._tb_writer = None
        self._wandb = None
        # accelerate-compatible: list-like container of active trackers
        self.trackers: list = []
        
        # Initialize mixed precision scaler if needed
        if mixed_precision == "fp16":
            self._scaler = torch.amp.GradScaler('cuda')
        elif mixed_precision == "bf16":
            # bf16 doesn't need a scaler
            self._scaler = None
        
        # Track prepared models/optimizers
        self._prepared_models = []
        self._prepared_optimizers = []
        self._prepared_dataloaders = []
        self._prepared_schedulers = []
    
    @property
    def device(self) -> torch.device:
        """Get the device for this process."""
        return self._device
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return multi.is_master()

    @property
    def is_local_main_process(self) -> bool:
        """On a single machine, local main process is the same as global main process."""
        return self.is_main_process
    
    @property
    def num_processes(self) -> int:
        """Get the number of processes."""
        return multi.world_size()

    @property
    def scaler(self):
        """Expose GradScaler for compatibility with code that accesses accelerator.scaler."""
        return self._scaler
    
    @property
    def sync_gradients(self) -> bool:
        """Check if gradients should be synchronized."""
        return self._sync_gradients
    
    def print(self, *args, **kwargs):
        """Print only on main process."""
        if self.is_main_process:
            print(*args, **kwargs)

    def init_trackers(self, project_name: str, config: Optional[dict] = None, init_kwargs: Optional[dict] = None):
        """
        Initialize logging trackers (tensorboard / wandb) in a minimal compatible way.

        This is called by training code even when logging is disabled; so this method must be safe no-op
        when no tracker is requested.
        """
        if not self.is_main_process:
            return

        if self._log_with is None:
            return

        init_kwargs = init_kwargs or {}

        if self._log_with in ("tensorboard", "all"):
            if self._project_dir is None:
                raise ValueError("logging_dir is required when log_with is tensorboard")
            os.makedirs(self._project_dir, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception as e:
                raise ImportError("TensorBoard is required for tensorboard logging") from e
            self._tb_writer = SummaryWriter(log_dir=self._project_dir)
            self.trackers.append("tensorboard")

        if self._log_with in ("wandb", "all"):
            try:
                import wandb
            except Exception as e:
                raise ImportError("wandb is required for wandb logging") from e

            wandb_kwargs = init_kwargs.get("wandb", {}) if isinstance(init_kwargs, dict) else {}
            # Ensure we don't double-init
            if getattr(wandb, "run", None) is None:
                wandb.init(project=project_name, config=config, **wandb_kwargs)
            self._wandb = wandb
            self.trackers.append("wandb")

    def get_tracker(self, name: str):
        """Return initialized tracker handle (currently supports 'wandb')."""
        if name == "wandb" and self._wandb is not None:
            return self._wandb
        raise ValueError(f"Tracker '{name}' is not initialized")

    def log(self, logs: dict, step: int = 0):
        """Log scalar metrics to initialized trackers."""
        if not self.is_main_process:
            return

        if not logs:
            return

        if self._tb_writer is not None:
            for k, v in logs.items():
                # Only log simple numeric values
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)

        if self._wandb is not None:
            # wandb.log accepts dict of scalars
            self._wandb.log(logs, step=step)
    
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.mixed_precision == "fp16":
            with torch.cuda.amp.autocast():
                yield
        elif self.mixed_precision == "bf16":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                yield
        else:
            yield
    
    def prepare(self, *args, device_placement: Optional[List[bool]] = None):
        """
        Prepare models, optimizers, dataloaders, etc. for distributed training.
        In multi-GPU mode, we just move models to device and wrap optimizers.
        """

        class _ScaledOptimizer:
            """
            Minimal optimizer wrapper that performs GradScaler stepping when fp16 is enabled.
            This mirrors accelerate's behavior where `optimizer.step()` is scaler-aware.
            """

            def __init__(self, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler):
                self._optimizer = optimizer
                self._scaler = scaler

            @property
            def param_groups(self):
                return self._optimizer.param_groups

            def step(self, *args, **kwargs):
                # GradScaler doesn't support optimizer closures reliably; ignore closure semantics here.
                self._scaler.step(self._optimizer)
                self._scaler.update()

            def zero_grad(self, *args, **kwargs):
                return self._optimizer.zero_grad(*args, **kwargs)

            def state_dict(self):
                return self._optimizer.state_dict()

            def load_state_dict(self, state_dict):
                return self._optimizer.load_state_dict(state_dict)

            def __getattr__(self, name):
                return getattr(self._optimizer, name)

        prepared = []
        device_placement_list = device_placement if device_placement else [True] * len(args)

        def _move_to_device(obj, device: torch.device):
            if torch.is_tensor(obj):
                return obj.to(device=device)
            if isinstance(obj, dict):
                return {k: _move_to_device(v, device) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_move_to_device(v, device) for v in obj)
            return obj

        class _DeviceDataLoader:
            """Wrap a DataLoader to move each batch to the target device."""

            def __init__(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
                self._dataloader = dataloader
                self._device = device

            def __iter__(self):
                for batch in self._dataloader:
                    yield _move_to_device(batch, self._device)

            def __len__(self):
                return len(self._dataloader)

            @property
            def dataset(self):
                return self._dataloader.dataset

            @property
            def sampler(self):
                return getattr(self._dataloader, "sampler", None)

            def set_epoch(self, epoch: int):
                samp = getattr(self._dataloader, "sampler", None)
                if hasattr(samp, "set_epoch"):
                    samp.set_epoch(epoch)
        
        for i, obj in enumerate(args):
            if isinstance(obj, torch.nn.Module):
                # Move model to device
                if device_placement_list[i] if i < len(device_placement_list) else True:
                    obj = obj.to(self.device)
                prepared.append(obj)
                self._prepared_models.append(obj)
            elif isinstance(obj, torch.optim.Optimizer):
                base_opt = obj
                if self._scaler is not None:
                    obj = _ScaledOptimizer(base_opt, self._scaler)
                prepared.append(obj)
                # Store the base optimizer for unscale_/checkpointing
                self._prepared_optimizers.append(base_opt)
            elif isinstance(obj, torch.utils.data.DataLoader):
                dataloader = obj
                # Shard dataset across ranks (accelerate does this internally).
                if multi.is_enabled():
                    try:
                        from torch.utils.data.distributed import DistributedSampler
                        from torch.utils.data import RandomSampler
                    except Exception:
                        DistributedSampler = None
                        RandomSampler = None

                    if DistributedSampler is not None and not isinstance(getattr(dataloader, "sampler", None), DistributedSampler):
                        shuffle = RandomSampler is not None and isinstance(getattr(dataloader, "sampler", None), RandomSampler)
                        sampler = DistributedSampler(
                            dataloader.dataset,
                            num_replicas=multi.world_size(),
                            rank=multi.rank(),
                            shuffle=shuffle,
                        )
                        # Rebuild DataLoader with distributed sampler
                        dataloader = torch.utils.data.DataLoader(
                            dataloader.dataset,
                            batch_size=dataloader.batch_size,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=dataloader.num_workers,
                            collate_fn=dataloader.collate_fn,
                            pin_memory=getattr(dataloader, "pin_memory", False),
                            drop_last=getattr(dataloader, "drop_last", False),
                            timeout=getattr(dataloader, "timeout", 0),
                            worker_init_fn=getattr(dataloader, "worker_init_fn", None),
                            persistent_workers=getattr(dataloader, "persistent_workers", False),
                        )

                # Move batches to device
                dataloader = _DeviceDataLoader(dataloader, self.device)
                prepared.append(dataloader)
                self._prepared_dataloaders.append(dataloader)
            elif hasattr(obj, 'step'):  # Likely a scheduler
                prepared.append(obj)
                self._prepared_schedulers.append(obj)
            else:
                prepared.append(obj)
        
        return prepared[0] if len(prepared) == 1 else tuple(prepared)
    
    def unwrap_model(self, model):
        """Unwrap model (in multi-GPU mode, models are not wrapped, so return as-is)."""
        return model
    
    def backward(self, loss):
        """Backward pass with gradient accumulation support."""
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def reduce(self, tensor, reduction: str = "mean"):
        """Reduce tensor across all processes."""
        if not multi.is_enabled():
            return tensor
        
        reduced = tensor.clone()
        if reduction == "mean":
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
            reduced /= multi.world_size()
        elif reduction == "sum":
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        return reduced
    
    def clip_grad_norm_(self, parameters, max_norm: float):
        """Clip gradient norm."""
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()
        
        if self._scaler is not None:
            self._scaler.unscale_(self._prepared_optimizers[0] if self._prepared_optimizers else None)
        
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    
    def accumulate(self, model):
        """Context manager for gradient accumulation."""
        class AccumulateContext:
            def __init__(self, accelerator):
                self.accelerator = accelerator
            
            def __enter__(self):
                gas = max(1, int(self.accelerator.gradient_accumulation_steps))
                # True on the last micro-step of the accumulation window.
                self.accelerator._sync_gradients = ((self.accelerator._step + 1) % gas == 0)
                return self
            
            def __exit__(self, *args):
                self.accelerator._step += 1
        
        return AccumulateContext(self)
    
    def save_state(self, output_dir: str):
        """Save training state (checkpoint)."""
        if not self.is_main_process:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save optimizer states
        for i, optimizer in enumerate(self._prepared_optimizers):
            torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer_{i}.pt"))
        
        # Save scaler state if exists
        if self._scaler is not None:
            torch.save(self._scaler.state_dict(), os.path.join(output_dir, "scaler.pt"))
        
        # Save step
        torch.save({"step": self._step}, os.path.join(output_dir, "step.pt"))
    
    def load_state(self, input_dir: str):
        """Load training state (checkpoint)."""
        if not os.path.exists(input_dir):
            return
        
        # Load optimizer states
        for i, optimizer in enumerate(self._prepared_optimizers):
            optimizer_path = os.path.join(input_dir, f"optimizer_{i}.pt")
            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path))
        
        # Load scaler state if exists
        if self._scaler is not None:
            scaler_path = os.path.join(input_dir, "scaler.pt")
            if os.path.exists(scaler_path):
                self._scaler.load_state_dict(torch.load(scaler_path))
        
        # Load step
        step_path = os.path.join(input_dir, "step.pt")
        if os.path.exists(step_path):
            state = torch.load(step_path)
            self._step = state.get("step", 0)
    
    def gather(self, tensor):
        """Gather tensor from all processes."""
        if not multi.is_enabled():
            return tensor
        
        gathered = [torch.zeros_like(tensor) for _ in range(multi.world_size())]
        torch.distributed.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)
    
    def register_save_state_pre_hook(self, hook):
        """Register a hook to be called before saving state."""
        self._save_state_hook = hook
    
    def register_load_state_pre_hook(self, hook):
        """Register a hook to be called before loading state."""
        self._load_state_hook = hook

    def wait_for_everyone(self):
        """Barrier across all processes."""
        if multi.is_enabled():
            torch.distributed.barrier()

    def end_training(self):
        """Flush/close any trackers."""
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
            self._wandb = None

