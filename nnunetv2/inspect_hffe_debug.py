import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as cc_label
from PIL import Image  # pip install pillow (一般环境里都有)
from dynamic_network_architectures.architectures.unet import PlainConvUNet


def save_heatmap(arr2d, out_png, title=None):
    plt.figure()
    plt.imshow(arr2d, interpolation="nearest")
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def load_2d_input_any(path: str, in_channels: int, force_grayscale: bool = True,
                      normalize_mode: str = "none") -> np.ndarray:
    """
    Returns np.ndarray with shape [C,H,W], dtype float32.

    normalize_mode:
      - "none": no normalization
      - "0_1": min-max to [0,1]
      - "uint8_0_1": divide by 255 if input looks uint8-ish
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".npy"]:
        img = np.load(path)
    elif ext in [".npz"]:
        z = np.load(path)
        if "data" in z:
            img = z["data"]
        else:
            img = z[list(z.keys())[0]]
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        im = Image.open(path)
        if force_grayscale:
            im = im.convert("L")  # grayscale
            img = np.array(im)    # [H,W]
        else:
            img = np.array(im)    # [H,W] or [H,W,3/4]
    else:
        raise ValueError(f"Unsupported input file: {path}")

    # Make shape [C,H,W]
    if img.ndim == 2:
        img = img[None, ...]  # [1,H,W]
    elif img.ndim == 3:
        # could be [H,W,C] or [C,H,W]
        if img.shape[0] == in_channels:
            # assume already [C,H,W]
            pass
        elif img.shape[-1] == in_channels:
            # [H,W,C] -> [C,H,W]
            img = np.transpose(img, (2, 0, 1))
        else:
            # if RGB/RGBA but you requested in_channels=1, convert to gray
            if in_channels == 1:
                if img.shape[-1] in (3, 4):
                    rgb = img[..., :3].astype(np.float32)
                    # luminance
                    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
                    img = gray[None, ...]
                else:
                    raise ValueError(f"3D input has unexpected channel dimension: {img.shape}")
            else:
                raise ValueError(f"Input channels mismatch: got {img.shape}, expected in_channels={in_channels}")
    else:
        raise ValueError(f"Unexpected input ndim={img.ndim}, shape={img.shape}")

    img = img.astype(np.float32)

    # Optional normalization (keep conservative by default)
    if normalize_mode == "uint8_0_1":
        # common when PNG is uint8: map to [0,1]
        if img.max() > 1.5:
            img = img / 255.0
    elif normalize_mode == "0_1":
        mn, mx = float(img.min()), float(img.max())
        if mx > mn + 1e-8:
            img = (img - mn) / (mx - mn)
        else:
            img = img * 0.0
    elif normalize_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize_mode={normalize_mode}")

    # If user requests more channels but image has 1 channel, tile it
    if in_channels > img.shape[0] and img.shape[0] == 1:
        img = np.repeat(img, repeats=in_channels, axis=0)

    assert img.shape[0] == in_channels, f"Final shape {img.shape}, expected C={in_channels}"
    return img

def pad_to_factor_chw(img_chw: np.ndarray, factor: int = 32, pad_mode: str = "constant", pad_value: float = 0.0):
    """
    img_chw: [C,H,W]
    pad to make H and W multiples of `factor`, pad on bottom/right (nnUNet常见做法也是pad后再crop回去)
    Returns: padded_img, (pad_h, pad_w)
    """
    assert img_chw.ndim == 3
    C, H, W = img_chw.shape
    pad_h = (factor - (H % factor)) % factor
    pad_w = (factor - (W % factor)) % factor

    if pad_h == 0 and pad_w == 0:
        return img_chw, (0, 0)

    if pad_mode == "constant":
        padded = np.pad(
            img_chw,
            pad_width=((0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=pad_value
        )
    elif pad_mode in ("edge", "reflect"):
        padded = np.pad(
            img_chw,
            pad_width=((0, 0), (0, pad_h), (0, pad_w)),
            mode=pad_mode
        )
    else:
        raise ValueError(f"Unknown pad_mode={pad_mode}")

    return padded, (pad_h, pad_w)
def crop_back(t: torch.Tensor, pad_h: int, pad_w: int):
    # t: [B, ... , H, W]
    if pad_h > 0:
        t = t[..., :-pad_h, :]
    if pad_w > 0:
        t = t[..., :, :-pad_w]
    return t


def save_label_png(label_2d: np.ndarray, out_png: str):
    # label_2d: [H,W] int
    plt.figure()
    plt.imshow(label_2d, interpolation="nearest")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_overlay_edges(gray_2d: np.ndarray, label_2d: np.ndarray, out_png: str):
    """
    gray_2d: [H,W] float (0..1 recommended)
    label_2d: [H,W] int
    Overlay predicted edges (boundary of non-zero labels).
    """
    # simple edge: pixel differs from any 4-neighbor
    lab = label_2d.astype(np.int32)
    edge = np.zeros_like(lab, dtype=np.uint8)
    edge[1:, :] |= (lab[1:, :] != lab[:-1, :])
    edge[:-1, :] |= (lab[:-1, :] != lab[1:, :])
    edge[:, 1:] |= (lab[:, 1:] != lab[:, :-1])
    edge[:, :-1] |= (lab[:, :-1] != lab[:, 1:])

    # normalize background image to 0..1 for display
    g = gray_2d.astype(np.float32)
    g_min, g_max = float(g.min()), float(g.max())
    if g_max > g_min + 1e-8:
        g = (g - g_min) / (g_max - g_min)
    else:
        g = g * 0.0

    plt.figure()
    plt.imshow(g, cmap="gray", interpolation="nearest")
    # draw edge in a contrasting colormap (matplotlib default colormap OK)
    plt.imshow(edge, alpha=0.6, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model checkpoint (.pth)")
    ap.add_argument("--img", required=True, help="path to a 2D input (.png/.jpg/.npy/.npz)")
    ap.add_argument("--outdir", required=True, help="output dir for heatmaps")
    ap.add_argument("--in_channels", type=int, default=1)
    ap.add_argument("--num_classes", type=int, default=3)  # adjust to your task
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--enable_levels", default="0,1", help="which HFFE levels to enable, e.g. 0,1 or 0")
    ap.add_argument("--debug_downsample", type=int, default=4, help="downsample factor for debug maps")
    ap.add_argument("--normalize", default="none", choices=["none", "uint8_0_1", "0_1"],
                    help="input normalization; prefer 'none' unless your training used 0-1 scaling")
    ap.add_argument("--force_alpha", type=float, default=None,
                help="force all HFFE alpha to a fixed value for diagnosis (e.g., 0.2)")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- instantiate network with your known arch config (must match Plans.json) ---
    n_stages = 6
    features_per_stage = [32, 64, 128, 256, 512, 512]
    kernel_sizes = [[3, 3]] * 6
    strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    n_conv_per_stage = [2] * 6
    n_conv_per_stage_decoder = [2] * 5

    net = PlainConvUNet(
        input_channels=args.in_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=torch.nn.Conv2d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=args.num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
        nonlin_first=False
    )

    # enable HFFE levels
    enable_levels = set(int(x.strip()) for x in args.enable_levels.split(",") if x.strip() != "")
    net.use_hffe = True
    net.hffe_enabled_levels = enable_levels

    # turn on debug
    for m in getattr(net, "hffe_modules", []):
        m.debug = True
        m.debug_downsample = args.debug_downsample

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "network_weights" in ckpt:
            state = ckpt["network_weights"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    # strip possible "module." prefixes
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(new_state, strict=False)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    net = net.to(device)
    net.eval()
    force_alpha = args.force_alpha
    if force_alpha is not None:
        for m in net.hffe_modules:
            if hasattr(m, "alpha"):
                with torch.no_grad():
                    m.alpha.fill_(float(force_alpha))
        print(f"[Diag] forced all HFFE alpha to {force_alpha}")
        alpha_tag = f"alpha{force_alpha:.2f}"
    else:
        alpha_tag = "alpha_trained"


    # --- Force alpha for diagnosis (no retrain) ---
    # Try: 0.0, 0.2, 0.4, 0.6 and compare outputs visually
    force_alpha = 0.8  # change this value
    for i, m in enumerate(net.hffe_modules):
        if hasattr(m, "alpha"):
            with torch.no_grad():
                m.alpha.fill_(force_alpha)
    print(f"[Diag] forced all HFFE alpha to {force_alpha}")


    # load image (png/jpg/npy/npz)
    img = load_2d_input_any(args.img, in_channels=args.in_channels, force_grayscale=True,
                            normalize_mode=args.normalize)  # [C,H,W], float32
    # pad to avoid decoder cat size mismatch (important!)
    # factor = 2^(n_stages-1) = 32 for your 6-stage UNet
    img, (pad_h, pad_w) = pad_to_factor_chw(img, factor=32, pad_mode="constant", pad_value=0.0)
    print(f"[Pad] pad_h={pad_h}, pad_w={pad_w}, new shape={img.shape}")

    x = torch.from_numpy(img)[None, ...].to(device)  # [B=1,C,H,W]

    # keep a copy of the (cropped) display image for overlay
# img is [C,H,W] after padding; for display use channel 0
    display_img = img[0].copy()

    with torch.no_grad():
        logits = net(x)  # [B,C,H,W]

    # crop back to original (unpadded) size so visualization matches the PNG you fed
    logits = crop_back(logits, pad_h, pad_w)  # [B,C,H0,W0]

    probs = torch.softmax(logits, dim=1)      # [B,C,H0,W0]
    pred = torch.argmax(probs, dim=1)         # [B,H0,W0]

    pred_np = pred[0].detach().cpu().numpy().astype(np.int32)

    # crop display image too
    if pad_h > 0:
        display_img = display_img[:-pad_h, :]
    if pad_w > 0:
        display_img = display_img[:, :-pad_w]

    # save
    alpha_tag = "unknown_alpha"
    # 如果你脚本里有 force_alpha 变量，用下面这一行替换
    # alpha_tag = f"alpha{force_alpha:.2f}"

    save_label_png(pred_np, os.path.join(args.outdir, f"pred_{alpha_tag}.png"))
    save_overlay_edges(display_img, pred_np, os.path.join(args.outdir, f"overlay_{alpha_tag}.png"))

    def cc_metrics(pred_2d: np.ndarray, cls: int):
        m = (pred_2d == cls).astype(np.uint8)
        if m.sum() == 0:
            return {"area": 0, "n_cc": 0, "largest_ratio": 0.0}
        cc, n = cc_label(m)  # 4-connected by default in scipy? (actually structure default is full connectivity for 2D = 1s)
        areas = np.bincount(cc.ravel())[1:]  # skip background=0
        area = int(m.sum())
        largest = int(areas.max()) if areas.size > 0 else 0
        return {"area": area, "n_cc": int(n), "largest_ratio": float(largest / (area + 1e-8))}

    # usage after pred_np is available
    for cls in range(1, min(args.num_classes, 3)):
        print(f"[CC] class {cls}:", cc_metrics(pred_np, cls))


    # optional: save per-class probability heatmaps (skip background=0)
    for cls in range(1, min(args.num_classes, 3)):  # e.g. save class 1/2 only
        p = probs[0, cls].detach().cpu().numpy()
        save_heatmap(p, os.path.join(args.outdir, f"prob_cls{cls}_{alpha_tag}.png"),
                    title=f"prob class {cls} ({alpha_tag})")


    # print alpha + save maps
    print("=== HFFE alpha values ===")
    for i, m in enumerate(net.hffe_modules):
        if hasattr(m, "alpha"):
            a = float(m.alpha.detach().cpu().item())
            print(f"level {i}: alpha={a:.4f}")
        else:
            print(f"level {i}: (no alpha parameter)")

    print("\n=== Debug stats + saving heatmaps ===")
    for i, m in enumerate(net.hffe_modules):
        if not getattr(m, "debug", False):
            continue
        dbg = getattr(m, "debug_last", None)
        if not dbg:
            continue

        st = dbg.get("stats", {})
        print(f"\n[HFFE level {i}] stats:")
        for k, v in st.items():
            print(f"  {k}: {v}")

        for name in ["swm_low", "swm_high", "swm_fuse_mean", "swm_fuse_max"]:
            if name not in dbg:
                continue
            t = dbg[name][0, 0].numpy()  # [h,w]
            out_png = os.path.join(args.outdir, f"hffe{i}_{name}.png")
            save_heatmap(t, out_png, title=f"HFFE{i} {name}")

    print(f"\nSaved heatmaps to: {args.outdir}")




if __name__ == "__main__":
    main()
