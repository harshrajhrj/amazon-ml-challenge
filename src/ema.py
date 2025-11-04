import torch

class EMA:
    """
    Exponential Moving Average of model parameters.
    This version keeps the shadow weights on the CPU and is optimized
    to prevent GPU OOM errors during validation swaps.
    """
    def __init__(self, model, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().cpu().clone()

    def _cpu_update_param(self, name: str, param: torch.Tensor):
        """Helper: update one parameter on CPU."""
        if param.grad is None:
            return
        new_data = param.detach().to("cpu", non_blocking=True)
        shadow = self.shadow[name]
        if shadow.dtype != new_data.dtype:
            new_data = new_data.to(dtype=shadow.dtype)
        self.shadow[name] = shadow.mul(self.decay).add(new_data, alpha=(1.0 - self.decay))

    @torch.inference_mode()
    def update(self, model):
        """Update the shadow weights with the current model weights."""
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._cpu_update_param(name, p)

    @torch.no_grad()
    def apply_to(self, model):
        """
        Load EMA weights onto the model for evaluation.
        The model's original weights are backed up to the CPU to prevent OOM errors.
        """
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                # ✅ FIX: Backup the original GPU weights to the CPU's RAM
                self.backup[name] = p.detach().cpu().clone()
                # Copy the EMA shadow weights (from CPU) to the GPU
                p.data.copy_(self.shadow[name].to(p.device, non_blocking=True))

    @torch.no_grad()
    def restore(self, model):
        """Restore the original model weights from the CPU backup."""
        for name, p in model.named_parameters():
            if name in self.backup:
                # Copy the backup weights (from CPU) back to the GPU
                p.data.copy_(self.backup[name].to(p.device, non_blocking=True))
        self.backup.clear()
