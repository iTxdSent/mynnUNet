
from __future__ import annotations
import copy
import pydoc
from typing import Any, Dict
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.network_architecture.plainconvunet_hffe import PlainConvUNetWithHFFE


def _resolve_imports(arch_kwargs: Dict[str, Any], kw_requires_import):
    arch_kwargs = copy.deepcopy(arch_kwargs)
    for k in kw_requires_import:
        v = arch_kwargs[k]
        obj = pydoc.locate(v)
        if obj is None:
            raise RuntimeError(f"Failed to import {k}={v} via pydoc.locate")
        arch_kwargs[k] = obj
    return arch_kwargs


class nnUNetTrainer_HFFE13(nnUNetTrainer):
    """
    nnUNetV2 plugin trainer:
    - builds PlainConvUNetWithHFFE instead of PlainConvUNet
    - exposes HFFE knobs here for easy tuning
    """

    # -----------------------------
    # HFFE hyperparameters (edit here)
    # -----------------------------
    hffe_enabled = True
    # 只开0 1 两层。
    hffe_apply_levels = (0, 1)
    #hffe_apply_levels = (0, 1, 2, 3)     # match paper i∈[1..4] :contentReference[oaicite:26]{index=26}
    hffe_mode = "replace"               # "replace" | "add"

    hffe_sam_kernel = 7
    hffe_coord_reduction = 32
    hffe_align_corners = False
    hffe_use_cross_residual = True

    # warmup/ramp (optional). Only effective when hffe_use_residual_gate=True.
    hffe_use_residual_gate = True
    hffe_gate_warmup_epochs = 0
    hffe_gate_ramp_epochs = 10
    hffe_gate_final = 1.0

    hffe_split_fuse_gate = True   # 先建议 True（BV 两目标/两类更稳）
    hffe_swm_temperature = 2.0  # set to 2.0 to soften gates
    hffe_swm_eps = 0.0         # set to 0.0 if using temperature only

    # debug (stores per-forward gate stats in module.debug_last)
    hffe_debug = False

    @classmethod
    def build_network_architecture(cls, *args, **kwargs):
        """
        Works for both training-time (instance call) and inference-time (class call).

        Training call (from nnUNetTrainer.initialize):
            build_network_architecture(network_arch_class_name: str,
                                    arch_kwargs: dict,
                                    arch_kwargs_requires_import: list[str],
                                    num_input_channels: int,
                                    num_segmentation_heads: int,
                                    enable_deep_supervision: bool)

        Inference call (from predictor.initialize_from_trained_model_folder):
            build_network_architecture(arch_kwargs: dict,
                                    arch_kwargs_requires_import: list[str],
                                    num_input_channels: int,
                                    num_segmentation_heads: int,
                                    enable_deep_supervision: bool=False)
        """
        import copy
        import pydoc
        from nnunetv2.network_architecture.plainconvunet_hffe import PlainConvUNetWithHFFE

        def _resolve_req_imports(arch_kwargs_local, req_import_list):
            arch_kwargs_local = copy.deepcopy(arch_kwargs_local)
            for k in req_import_list:
                if k not in arch_kwargs_local:
                    continue
                v = arch_kwargs_local[k]

                # dropout_op often None
                if v is None:
                    continue
                # already a class/object
                if not isinstance(v, str):
                    continue

                v_str = v.strip()
                if v_str == "":
                    arch_kwargs_local[k] = None
                    continue

                obj = pydoc.locate(v_str)
                if obj is None:
                    raise RuntimeError(f"Failed to import {k}={v_str} via pydoc.locate")
                arch_kwargs_local[k] = obj
            return arch_kwargs_local

        enable_deep_supervision = kwargs.get("enable_deep_supervision", None)


        # Case 1: "training-like" style where first arg is network class name (str)
        # It can be either:
        #   (str, dict, list, int, int, bool)
        # or (str, dict, list, int, int) with enable_deep_supervision passed via kwargs
# -----------------------------
        if len(args) >= 5 and isinstance(args[0], str):
            arch_kwargs = args[1]
            req_import = args[2]
            num_input_channels = args[3]
            num_segmentation_heads = args[4]

            if enable_deep_supervision is None:
                # if 6th positional exists use it, else fall back to kwargs/default False
                if len(args) >= 6:
                    enable_deep_supervision = args[5]
                else:
                    enable_deep_supervision = kwargs.get("enable_deep_supervision", False)

            arch_kwargs = _resolve_req_imports(arch_kwargs, req_import)


        # -----------------------------
        # Case 2: inference style (4 positional, first is dict)
        # -----------------------------
        elif len(args) >= 4 and isinstance(args[0], dict) and isinstance(args[1], (list, tuple)):
            arch_kwargs = args[0]
            req_import = args[1]
            num_input_channels = args[2]
            num_segmentation_heads = args[3]
            if enable_deep_supervision is None:
                enable_deep_supervision = False

            arch_kwargs = _resolve_req_imports(arch_kwargs, req_import)

        else:
            raise TypeError(
                f"Unexpected build_network_architecture call. args={args}, kwargs={kwargs}. "
                "Expected either (str, dict, list, int, int, bool) or (dict, list, int, int, enable_deep_supervision=bool)."
            )

        # -----------------------------
        # Build our plugin network
        # -----------------------------
        net = PlainConvUNetWithHFFE(
            input_channels=num_input_channels,
            num_classes=num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **arch_kwargs,
            hffe_enabled=cls.hffe_enabled,
            hffe_apply_levels=cls.hffe_apply_levels,
            hffe_mode=cls.hffe_mode,
            hffe_kwargs=dict(
                sam_kernel=cls.hffe_sam_kernel,
                coord_reduction=cls.hffe_coord_reduction,
                align_corners=cls.hffe_align_corners,
                use_cross_residual=cls.hffe_use_cross_residual,
                use_residual_gate=cls.hffe_use_residual_gate,
                gate_init=0.0 if cls.hffe_use_residual_gate else 1.0,
                split_fuse_gate=cls.hffe_split_fuse_gate,
                debug=cls.hffe_debug,
                swm_temperature=cls.hffe_swm_temperature,
                swm_eps=cls.hffe_swm_eps,
            ),
        )

        if cls.hffe_debug and hasattr(net, "set_hffe_debug"):
            net.set_hffe_debug(True)
        if cls.hffe_use_residual_gate and hasattr(net, "set_hffe_gate_scale"):
            net.set_hffe_gate_scale(0.0)

        return net




    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        # optional warmup/ramp for residual gate
        if not getattr(self, "network", None):
            return
        if not self.hffe_use_residual_gate:
            return

        e = int(self.current_epoch)
        if e < self.hffe_gate_warmup_epochs:
            s = 0.0
        else:
            if self.hffe_gate_ramp_epochs <= 0:
                s = float(self.hffe_gate_final)
            else:
                t = min(1.0, (e - self.hffe_gate_warmup_epochs) / float(self.hffe_gate_ramp_epochs))
                s = float(self.hffe_gate_final) * t

        if hasattr(self.network, "set_hffe_gate_scale"):
            self.network.set_hffe_gate_scale(s)
