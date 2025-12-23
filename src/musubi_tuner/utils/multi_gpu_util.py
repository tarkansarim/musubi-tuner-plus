"""
Multi-GPU utility functions for distributed training with gloo backend.
Based on OneTrainer-Plus implementation.
"""
import torch


def is_enabled() -> bool:
    """Check if distributed training is enabled."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    """Get the current process rank."""
    return torch.distributed.get_rank() if is_enabled() else 0


def is_master() -> bool:
    """Check if this is the master process (rank 0)."""
    return rank() == 0


def world_size() -> int:
    """Get the total number of processes."""
    return torch.distributed.get_world_size() if is_enabled() else 1


def sequential(enabled: bool = True):
    """
    Execute code sequentially in all ranks, using a for loop.
    Yields once per rank, synchronizing with barriers between ranks.
    """
    if enabled:
        for current in range(world_size()):
            if current == rank():
                yield()
            if is_enabled():
                torch.distributed.barrier()
    else:
        yield()


def master_first(enabled: bool = True):
    """
    Execute code first only on rank 0, then on all other ranks in parallel.
    Yields twice: first for master, then for all ranks.
    """
    if enabled:
        for current in [True, False]:
            if current == is_master():
                yield()
            if is_enabled():
                torch.distributed.barrier()
    else:
        yield()


def distributed_enumerate(iterable, distribute: bool = True):
    """
    Enumerate an iterable, distributing items across ranks.
    Each rank only processes items where i % world_size() == rank().
    """
    if distribute:
        for i, x in enumerate(iterable):
            if i % world_size() == rank():
                yield i, x
    elif is_master():
        for i, x in enumerate(iterable):
            yield i, x


def reduce_tensor_mean(tensor):
    """Reduce a tensor across all processes and compute the mean."""
    if is_enabled():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= world_size()


def reduce_grads_mean(params: list[torch.nn.Parameter], async_op: bool = False):
    """
    Reduce gradients across all processes and compute the mean.
    
    Args:
        params: List of parameters with gradients to reduce
        async_op: Whether to perform reduction asynchronously (not fully implemented)
    """
    if not is_enabled():
        return
    
    if async_op:
        # For now, we'll do synchronous reduction even if async_op is True
        # Full async implementation would require work queue management
        pass
    
    for param in params:
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, async_op=False)
            grad /= world_size()


@torch.no_grad()
def broadcast_parameters(params: list[torch.nn.Parameter], train_device: torch.device):
    """Broadcast parameters from rank 0 to all other ranks."""
    if not is_enabled():
        return
    
    for param in params:
        gpu_param = param.to(train_device)
        torch.distributed.broadcast(gpu_param, src=0)
        if not is_master() and gpu_param is not param:
            param.copy_(gpu_param)

