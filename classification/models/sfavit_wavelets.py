import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pywt
except ImportError:  # Haar/db1 can run without PyWavelets.
    pywt = None


def _next_power_of_2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _shift_minus_one_same_size(x: torch.Tensor, dim: int, wrap: bool) -> torch.Tensor:
    if wrap:
        return torch.roll(x, shifts=-1, dims=dim)

    if dim in (-1, 3):  # width
        return F.pad(x, (0, 1, 0, 0))[..., 1:]
    if dim in (-2, 2):  # height
        return F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]

    raise ValueError(f"Unsupported dim={dim}; expected width(-1/3) or height(-2/2).")


def _dwt_haar_l2_same(
    x: torch.Tensor,
    w_l: torch.Tensor,
    w_h: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, H, W = x.shape
    n = max(_next_power_of_2(H), _next_power_of_2(W))
    wrap_w = n == W
    wrap_h = n == H

    # Keep AMP/fp16 inference fast by matching the input dtype/device.
    w_l = w_l.to(device=x.device, dtype=x.dtype)
    w_h = w_h.to(device=x.device, dtype=x.dtype)

    if w_l.shape[-1] != 2 or w_h.shape[-1] != 2:
        raise ValueError("_dwt_haar_l2_same only supports 2-tap Haar/db1 filters.")

    wl0, wl1 = w_l[0, 0], w_l[0, 1]
    wh0, wh1 = w_h[0, 0], w_h[0, 1]

    x_next_w = _shift_minus_one_same_size(x, dim=-1, wrap=wrap_w)
    x_l = x_next_w * wl0 + x * wl1
    x_h = x_next_w * wh0 + x * wh1

    xl_next_h = _shift_minus_one_same_size(x_l, dim=-2, wrap=wrap_h)
    xh_next_h = _shift_minus_one_same_size(x_h, dim=-2, wrap=wrap_h)

    x_ll = xl_next_h * wl0 + x_l * wl1
    x_lh = xl_next_h * wh0 + x_l * wh1
    x_hl = xh_next_h * wl0 + x_h * wl1
    x_hh = xh_next_h * wh0 + x_h * wh1

    return x_ll, x_lh, x_hl, x_hh


class DWT_Function_FFT_L2:

    @staticmethod
    def apply(
        x: torch.Tensor,
        w_l: torch.Tensor,
        w_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return _dwt_haar_l2_same(x, w_l, w_h)


class DWT_2D_FFT_L2(nn.Module):
    def __init__(self, wave: str = "haar"):
        super().__init__()
        wave_l = wave.lower()

        if wave_l in {"haar", "db1"}:
            s = 1.0 / math.sqrt(2.0)
            w_l = torch.tensor([[s, s]], dtype=torch.float32)
            w_h = torch.tensor([[s, -s]], dtype=torch.float32)
        else:
            if pywt is None:
                raise ImportError("PyWavelets is required for non-Haar wavelets.")
            w = pywt.Wavelet(wave)
            w_l = torch.tensor(w.dec_lo[::-1], dtype=torch.float32).unsqueeze(0)
            w_h = torch.tensor(w.dec_hi[::-1], dtype=torch.float32).unsqueeze(0)

        self.register_buffer("w_l", w_l)
        self.register_buffer("w_h", w_h)

    def forward(self, x: torch.Tensor):
        return DWT_Function_FFT_L2.apply(x, self.w_l, self.w_h)
