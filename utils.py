import io
import numpy as np
from PIL import Image
import torch

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def image_bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def compute_metrics(sr_img: Image.Image, hr_img: Image.Image):
    """PSNR / SSIM between generated SR output and a true high-res reference.
    Resizes HR to match SR exactly (they should already match if scale is set correctly)."""
    if sr_img.size != hr_img.size:
        hr_img = hr_img.resize(sr_img.size, Image.BICUBIC)

    sr = np.asarray(sr_img.convert("RGB"), dtype=np.float32)
    hr = np.asarray(hr_img.convert("RGB"), dtype=np.float32)

    if HAVE_SKIMAGE:
        psnr = float(sk_psnr(hr, sr, data_range=255))
        ssim = float(sk_ssim(hr, sr, channel_axis=2, data_range=255))
    else:
        mse = np.mean((sr - hr) ** 2)
        psnr = float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
        ssim = None  # skimage not available

    return {"psnr_db": round(psnr, 2), "ssim": round(ssim, 4) if ssim is not None else None}


def bicubic_baseline(lr_img: Image.Image, scale: int) -> Image.Image:
    w, h = lr_img.size
    return lr_img.resize((w * scale, h * scale), Image.BICUBIC)
