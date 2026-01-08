# nnUNetTrainerHFFE.py
import math
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerHFFE(nnUNetTrainer):
    """
    HFFE schedule trainer (freeze + warmup/ramp + capped scale), scheduler-safe.
    - No env vars. Edit hyperparameters in the block below.
    - Does NOT override configure_optimizers() (keeps nnUNet scheduler intact).
    - Optional HFFE gradient multiplier via safe hooks (disabled by default).
    """

    # =========================================================================
    # Hyperparameters (EDIT HERE ONLY)
    # =========================================================================
    USE_HFFE: bool = False

    # Freeze: behave exactly like baseline (use_hffe=False, alpha_scale=0)
    HFFE_FREEZE_EPOCHS: int = 30

    # After freeze: schedule alpha_scale (NOT alpha itself)
    HFFE_WARMUP_EPOCHS: int = 10
    HFFE_RAMP_EPOCHS: int = 60
    HFFE_SCALE_MAX: float = 0.35
    HFFE_RAMP_SHAPE: str = "cosine"  # "cosine" or "linear"

    # Logging
    HFFE_LOG_EVERY: int = 5
    HFFE_PRINT_MODULE_NAMES_ONCE: bool = True

    # Optional: emulate smaller LR for HFFE params (scheduler-safe)
    # Strongly建议先 False，先保证能稳定训练；需要再开。
    ENABLE_HFFE_GRAD_MULT: bool = False
    HFFE_GRAD_MULT: float = 0.1

    # =========================================================================
    # Helpers
    # =========================================================================
    @staticmethod
    def _unwrap(net):
        return net.module if hasattr(net, "module") else net

    def _iter_hffe_modules(self):
        """
        Identify HFFE modules by presence of (alpha_raw, alpha_scale).
        """
        net = self._unwrap(self.network)
        for name, m in net.named_modules():
            if hasattr(m, "alpha_raw") and hasattr(m, "alpha_scale"):
                yield name, m

    def _set_use_hffe_flag(self, enable: bool):
        """
        Your PlainConvUNet defines net.use_hffe. Toggle it if present.
        """
        net = self._unwrap(self.network)
        if hasattr(net, "use_hffe"):
            net.use_hffe = bool(enable)

    def _apply_alpha_scale(self, scale: float) -> int:
        """
        Write alpha_scale into all HFFE modules (modules exist even if some levels disabled).
        Returns number of updated modules.
        """
        n = 0
        names = []
        for name, m in self._iter_hffe_modules():
            with torch.no_grad():
                if isinstance(m.alpha_scale, torch.Tensor):
                    m.alpha_scale.fill_(float(scale))
                else:
                    m.alpha_scale = float(scale)
            n += 1
            names.append(name)

        if self.HFFE_PRINT_MODULE_NAMES_ONCE and (not hasattr(self, "_hffe_names_printed")) and n > 0:
            self._hffe_names_printed = True
            self.print_to_log_file("[HFFE] modules: " + ", ".join(names))

        return n

    def _compute_scale(self, sched_epoch: int) -> float:
        """
        sched_epoch = epoch - HFFE_FREEZE_EPOCHS
        returns alpha_scale in [0, HFFE_SCALE_MAX]
        """
        w = int(self.HFFE_WARMUP_EPOCHS)
        r = int(self.HFFE_RAMP_EPOCHS)
        smax = float(self.HFFE_SCALE_MAX)

        if sched_epoch < w:
            return 0.0
        if r <= 0:
            return smax

        t = (sched_epoch - w) / float(r)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)

        shape = str(self.HFFE_RAMP_SHAPE).lower()
        if shape == "cosine":
            return smax * 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            return smax * t

    # =========================================================================
    # Optional grad multiplier (SAFE: handles grad=None)
    # =========================================================================
    def _register_hffe_grad_hooks_once(self):
        if (not self.ENABLE_HFFE_GRAD_MULT) or hasattr(self, "_hffe_grad_hooks_registered"):
            return

        mult = float(self.HFFE_GRAD_MULT)
        if mult >= 0.999999:
            self._hffe_grad_hooks_registered = True
            return

        # collect unique params under HFFE modules
        seen = set()
        params = []
        for _, m in self._iter_hffe_modules():
            for p in m.parameters(recurse=True):
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)

        def _mul_grad(g, m):
            return None if g is None else g * m

        for p in params:
            p.register_hook(lambda g, m=mult: _mul_grad(g, m))

        self._hffe_grad_hooks_registered = True
        self.print_to_log_file(f"[HFFE] grad mult enabled: {mult} (hooks on {len(params)} params)")

    # =========================================================================
    # nnUNet hooks
    # =========================================================================
    def on_train_epoch_start(self):
        # Base will step lr_scheduler. DO NOT break it.
        super().on_train_epoch_start()

        # register optional hooks (disabled by default)
        self._register_hffe_grad_hooks_once()

        epoch = getattr(self, "current_epoch", None)
        if epoch is None:
            epoch = getattr(self, "epoch", 0)
        epoch = int(epoch)

        if (not self.USE_HFFE) or (epoch < int(self.HFFE_FREEZE_EPOCHS)):
            use_hffe = False
            scale = 0.0
        else:
            use_hffe = True
            sched_epoch = epoch - int(self.HFFE_FREEZE_EPOCHS)
            scale = float(self._compute_scale(sched_epoch))

        self._set_use_hffe_flag(use_hffe)

        # IMPORTANT: if scale==0, you can still keep use_hffe=True safely.
        # If you want "true baseline until scale>0", uncomment next 2 lines:
        # if scale <= 0.0:
        #     self._set_use_hffe_flag(False)

        updated = self._apply_alpha_scale(scale)

        if self.HFFE_LOG_EVERY > 0 and (epoch % int(self.HFFE_LOG_EVERY) == 0):
            self.print_to_log_file(
                f"[HFFE] epoch={epoch} use_hffe={int(use_hffe)} scale={scale:.4f} "
                f"(freeze={self.HFFE_FREEZE_EPOCHS}, warmup={self.HFFE_WARMUP_EPOCHS}, ramp={self.HFFE_RAMP_EPOCHS}, "
                f"scale_max={self.HFFE_SCALE_MAX}, shape={self.HFFE_RAMP_SHAPE}) updated_modules={updated}"
            )

    def on_validation_epoch_start(self):
        try:
            super().on_validation_epoch_start()
        except Exception:
            pass

        epoch = getattr(self, "current_epoch", None)
        if epoch is None:
            epoch = getattr(self, "epoch", 0)
        epoch = int(epoch)

        if (not self.USE_HFFE) or (epoch < int(self.HFFE_FREEZE_EPOCHS)):
            use_hffe = False
            scale = 0.0
        else:
            use_hffe = True
            sched_epoch = epoch - int(self.HFFE_FREEZE_EPOCHS)
            scale = float(self._compute_scale(sched_epoch))

        self._set_use_hffe_flag(use_hffe)
        _ = self._apply_alpha_scale(scale)
