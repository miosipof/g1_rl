from __future__ import annotations

from typing import Any

import torch


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    """Detach and convert a tensor to contiguous float32."""
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)!r}")
    return x.detach().to(dtype=torch.float32).contiguous()


def build_rgbd_proxy_observation(
    scene: Any,
    proxy: Any,
    *,
    camera_name: str = "camera",
    depth_min: float = 0.0,
    depth_max: float = 10.0,
    normalize_rgb: bool = True,
) -> dict[str, torch.Tensor]:
    """Build RGB-D + proxy proprioception observations for the command proxy.

    This is the proxy-specific counterpart of ``observations.rgbd_obs``. It does
    not require an articulated Isaac Lab robot object; proprioception comes from
    ``G1CommandProxy.observation()`` and has shape ``[N, 6]``:

    ``[x_w, y_w, yaw, vx_current, vy_current, wz_current]``.

    Returned keys:
        - ``rgb``: ``[N, H, W, 3]``
        - ``depth``: ``[N, H, W, 1]``
        - ``proprio``: ``[N, 6]``
    """
    camera = scene[camera_name]

    rgb = _to_float_tensor(camera.data.output["rgb"])
    if normalize_rgb and float(rgb.max().item()) > 1.0:
        rgb = rgb / 255.0

    depth = _to_float_tensor(camera.data.output["distance_to_image_plane"])
    depth = torch.nan_to_num(depth, nan=depth_max, posinf=depth_max, neginf=depth_min)
    depth = depth.clamp(min=depth_min, max=depth_max)

    if not hasattr(proxy, "observation"):
        raise TypeError("proxy must expose an observation() method")
    proprio = _to_float_tensor(proxy.observation())

    return {
        "rgb": rgb,
        "depth": depth,
        "proprio": proprio,
    }


def summarize_proxy_observation(obs: dict[str, torch.Tensor]) -> str:
    """Return a compact shape/dtype summary for smoke-test logging."""
    return (
        f"rgb={tuple(obs['rgb'].shape)} {obs['rgb'].dtype}, "
        f"depth={tuple(obs['depth'].shape)} {obs['depth'].dtype}, "
        f"proprio={tuple(obs['proprio'].shape)} {obs['proprio'].dtype}"
    )
