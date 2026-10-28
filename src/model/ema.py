"""EMA model state with explicit serialization and exception-safe evaluation."""
from contextlib import contextmanager
import torch


class EMAWrapper:
    def __init__(self, model, decay=0.999, mutable_param_keywords=None):
        if not 0 <= float(decay) < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.model = model
        self.decay = float(decay)
        self.mutable_param_keywords = [s.strip() for s in (mutable_param_keywords or []) if s.strip()]
        self.shadow, self.backup = {}, {}
        self.num_updates = 0

    def register(self):
        self.shadow = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        self.num_updates = 0

    @torch.no_grad()
    def update(self):
        for name, value in self.model.state_dict().items():
            target = self.shadow[name]
            mutable = not self.mutable_param_keywords or any(k in name for k in self.mutable_param_keywords)
            if value.is_floating_point() and mutable:
                target.lerp_(value.detach(), 1 - self.decay)
            else:
                target.copy_(value)
        self.num_updates += 1

    def state_dict(self):
        return {"shadow": {k: v.detach().to(device="cpu", copy=True) for k, v in self.shadow.items()},
                "decay": self.decay, "num_updates": self.num_updates,
                "mutable_param_keywords": self.mutable_param_keywords}

    def load_state_dict(self, state):
        current = self.model.state_dict()
        if set(state["shadow"]) != set(current):
            raise ValueError("EMA state keys differ from model")
        if any(state["shadow"][k].shape != v.shape for k, v in current.items()):
            raise ValueError("EMA state shapes differ from model")
        self.shadow = {k: v.detach().to(device=current[k].device, dtype=current[k].dtype).clone()
                       for k, v in state["shadow"].items()}
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.mutable_param_keywords = list(state["mutable_param_keywords"])

    def apply_shadow(self):
        if self.backup:
            raise RuntimeError("EMA weights are already applied")
        # Swap storage rather than allocating a third full GPU model copy.
        tensors = dict(self.model.named_parameters())
        tensors.update(dict(self.model.named_buffers()))
        self.backup = {k: value.data for k, value in tensors.items() if k in self.shadow}
        for key in self.backup:
            tensors[key].data = self.shadow[key]

    def restore(self):
        if self.backup:
            tensors = dict(self.model.named_parameters())
            tensors.update(dict(self.model.named_buffers()))
            for key, value in self.backup.items():
                tensors[key].data = value
            self.backup = {}

    @contextmanager
    def average_parameters(self):
        training = self.model.training
        self.apply_shadow()
        try:
            yield self.model
        finally:
            self.restore()
            self.model.train(training)
