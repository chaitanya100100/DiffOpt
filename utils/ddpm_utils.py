import torch
import math
import numpy as np
import pyrender
from utils.meshviewer import makeLookAt
from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    print("USING LINEAR SCHEDULE")
    # beta_t = []
    # bla = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    # for i in range(T + 1):
    #     t1 = i / (T + 1)
    #     t2 = (i + 1) / (T + 1)
    #     beta_t.append(min(1 - bla(t2) / bla(t1), 0.999))
    # beta_t = torch.tensor(beta_t)
    # print("USING COSINE SCHEDULE")

    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


def random_camera(center):
    is_batch = center.ndim == 2
    if not is_batch:
        center = center[None]
    B = center.shape[0]

    # target = center + torch.randn_like(center) * 1.0
    target = center

    # z = torch.randn_like(center[:, 2:]) * 1
    # xy = torch.zeros_like(center[:, :2])
    # xy = xy + torch.randn_like(xy).abs() * 1 + 0
    # position = torch.cat([xy, z], -1)
    position = center + torch.tensor([2.0, 2.0, 1.0]).float()[None]
    position = position + torch.randn_like(position) * 0.0

    up = torch.zeros_like(center)
    up[..., -1] = 1

    # T = makeLookAt(position.numpy(), target.numpy(), up.numpy())
    T = np.stack(
        [
            makeLookAt(p, t, u)
            for p, t, u in zip(position.numpy(), target.numpy(), up.numpy())
        ]
    )

    T = torch.tensor(T).float()
    T2 = T.clone()
    T2[:, :3, :3] = T[:, :3, :3].transpose(-1, -2)
    T2[:, :3, 3] = (-T[:, :3, :3].transpose(-1, -2) @ T[:, :3, 3:])[..., 0]

    width = height = 600
    yfov = np.pi / 4.0
    K = pyrender.PerspectiveCamera(
        yfov=yfov, aspectRatio=float(width) / height
    ).get_projection_matrix()
    K = torch.tensor(K).float()

    K = K[None].expand(B, -1, -1)
    width = torch.tensor([width] * B).float()
    height = torch.tensor([height] * B).float()
    yfov = torch.tensor([yfov] * B).float()

    P = K @ T2

    cam = {
        "T": T,
        "K": K,
        "P": P,
        # "target": target,
        # "position": position,
        # "up": up,
        "yfov": yfov,
        "width": width,
        "height": height,
    }
    if not is_batch:
        cam = {k: v[0] for k, v in cam.items()}
    return cam


def projection(joints, cam):
    is_batch = joints.ndim == 3
    if not is_batch:
        joints = joints[None]
        cam = {k: v[None] for k, v in cam.items()}

    J = joints
    J = torch.cat([J, torch.ones_like(J[..., :1])], -1)
    J = J @ cam["P"].transpose(-1, -2)
    J = J[..., :3] / J[..., 3:]
    J = J / J[..., 2:]

    h, w = cam["height"], cam["width"]
    assert torch.allclose(w, h)
    J = J * w[:, None, None] / 2 + w[:, None, None] / 2
    J[..., 1] = h[:, None] - J[..., 1]

    conf = (
        (J[..., 0] >= 0)
        & (J[..., 0] <= w[:, None])
        & (J[..., 1] >= 0)
        & (J[..., 1] <= h[:, None])
    )
    J[..., 2] = conf.float()
    return J


def j2d_to_y(j2d, h, w):
    assert torch.allclose(h, w)
    y = j2d[..., :2].clone() / w[:, None, None] - 0.5
    y = y.view(y.shape[0], -1)
    return y


def get_cur_y(x, data, smpl):
    device = x.device

    temp = {}
    temp["global_orient"] = rotation_6d_to_matrix(x[:, :6])
    temp["body_pose"] = rotation_6d_to_matrix(x[:, 6:].unflatten(-1, (-1, 6)))
    temp["betas"] = data["betas"].to(device)
    temp["transl"] = data["transl"].to(device)

    bmout = smpl(**temp)
    cam = {
        k.replace("cam_", "", 1): v.to(device)
        for k, v in data.items()
        if k.startswith("cam_")
    }
    j2d = projection(bmout.joints, cam)
    y = j2d_to_y(j2d, cam["height"], cam["width"])
    return y
