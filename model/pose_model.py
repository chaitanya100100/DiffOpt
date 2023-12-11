import math
import torch
import torch.nn as nn
import copy
from torch.nn.utils import weight_norm as weight_norm_fn
from utils.smpl import MySMPL
from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from utils.ddpm_utils import projection, j2d_to_y


def linear_layer(inp_dim, out_dim, weight_norm):
    layer = nn.Linear(inp_dim, out_dim)
    if weight_norm:
        layer = weight_norm_fn(layer)
    return layer


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def my_groupnorm(num_hidden):
    num_g = 16
    assert num_hidden % num_g == 0
    return nn.GroupNorm(num_g, num_hidden)


def get_activation(activation):
    return {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "prelu": nn.PReLU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
    }[activation]


def get_act_normalization(act_normalization):
    return {
        "none": nn.Identity,
        "batch": nn.BatchNorm1d,
        "layer": nn.LayerNorm,
        "group": my_groupnorm,
        "instance": nn.InstanceNorm1d,
    }[act_normalization]


class MyResBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        activation,
        act_normalization,
        time_dim,
        weight_norm=False,
    ):
        super().__init__()
        self.fc1 = linear_layer(hidden_dim, hidden_dim, weight_norm)
        self.bn1 = act_normalization(hidden_dim)
        self.act1 = activation()
        self.fc2 = linear_layer(hidden_dim, hidden_dim, weight_norm)
        self.bn2 = act_normalization(hidden_dim)
        self.act2 = activation()
        self.time_fc = None
        if time_dim is not None:
            self.time_fc = nn.Sequential(
                activation(), nn.Linear(time_dim, 2 * hidden_dim)
            )

    def forward(self, x, t=None):
        out = self.bn1(self.fc1(x))

        if t is not None:
            t = self.time_fc(t)
            scale, shift = t.chunk(2, dim=-1)
            out = out * (scale + 1) + shift

        out = self.act1(out)
        out = self.bn2(self.fc2(out))
        out = self.act2(out + x)
        return out


class PaddedNorm(nn.Module):
    def __init__(self, d, act_normalization):
        super().__init__()
        self.d = d
        self.nd = ((d + 15) // 16) * 16
        self.pad = torch.nn.ConstantPad1d([0, self.nd - self.d], 0)
        self.norm_fn = act_normalization(self.nd)

    def forward(self, x):
        x = self.pad(x)
        x = self.norm_fn(x)
        return x[..., : self.d]


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MyResMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        activation,
        num_hiddens,
        act_normalization,
        time_dim,
        weight_norm=False,
    ):
        super().__init__()
        assert num_hiddens > 0

        layers = [
            linear_layer(input_dim, hidden_dim, weight_norm),
            act_normalization(hidden_dim),
            activation(),
        ]
        self.inp_block = nn.Sequential(*layers)

        layers = [
            MyResBlock(hidden_dim, activation, act_normalization, time_dim, weight_norm)
            for _ in range(num_hiddens - 1)
        ]
        self.block = mySequential(*layers)

        self.out_block = linear_layer(hidden_dim, output_dim, weight_norm)

    def forward(self, x, t):
        x = self.inp_block(x)
        x = self.block(x, t)
        x = self.out_block(x)
        return x


class PoseModelMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.denoise = MyResMLP(
            input_dim=cfg.x_dim,
            output_dim=cfg.x_dim,
            hidden_dim=cfg.hidden_dim,
            activation=get_activation(cfg.activation),
            num_hiddens=cfg.num_layers,
            act_normalization=get_act_normalization(cfg.act_normalization),
            time_dim=cfg.hidden_dim,
        )
        self.t_emb = nn.Linear(1, cfg.hidden_dim)
        self.y_emb = nn.Linear(cfg.y_dim, cfg.hidden_dim)
        self.ty_emb = MyResMLP(
            input_dim=cfg.hidden_dim * 2,
            output_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            activation=get_activation(cfg.activation),
            num_hiddens=2,
            act_normalization=get_act_normalization(cfg.act_normalization),
            time_dim=cfg.hidden_dim,
        )

    def forward(self, x, y, t, y_block, **kwargs):
        y_block = y_block.float()[:, None]
        t = self.t_emb(t[:, None])
        y = self.y_emb(y)
        y = y * (1 - y_block)

        ty = torch.cat((t, y), dim=-1)
        ty = self.ty_emb(ty, t)

        x_ret = self.denoise(x, ty)
        return x_ret


class PoseModelMLP_Res(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.denoise = MyResMLP(
            input_dim=cfg.x_dim,
            output_dim=cfg.x_dim,
            hidden_dim=cfg.hidden_dim,
            activation=get_activation(cfg.activation),
            num_hiddens=cfg.num_layers,
            act_normalization=get_act_normalization(cfg.act_normalization),
            time_dim=cfg.hidden_dim,
        )
        self.t_emb = nn.Linear(1, cfg.hidden_dim)
        self.y_emb = nn.Linear(cfg.y_dim, cfg.hidden_dim)
        self.ty_emb = MyResMLP(
            input_dim=cfg.hidden_dim * 2,
            output_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            activation=get_activation(cfg.activation),
            num_hiddens=2,
            act_normalization=get_act_normalization(cfg.act_normalization),
            time_dim=cfg.hidden_dim,
        )

        self.smpl = MySMPL()

    def get_cur_y(self, x, data):
        device = x.device

        temp = {}
        temp["global_orient"] = rotation_6d_to_matrix(x[:, :6])
        temp["body_pose"] = rotation_6d_to_matrix(x[:, 6:].unflatten(-1, (-1, 6)))
        temp["betas"] = data["betas"].to(device)
        temp["transl"] = data["transl"].to(device)

        bmout = self.smpl(**temp)
        cam = {
            k.replace("cam_", "", 1): v.to(device)
            for k, v in data.items()
            if k.startswith("cam_")
        }
        j2d = projection(bmout.joints, cam)
        y = j2d_to_y(j2d, cam["height"], cam["width"])
        return y

    def forward(self, x, y, t, y_block, data):
        cur_y = self.get_cur_y(x, data)
        y = y - cur_y
        # cur_y = self.get_cur_y(x, y, data)
        # x = torch.cat((x, cur_y), dim=-1)

        y_block = y_block.float()[:, None]
        t = self.t_emb(t[:, None])
        y = self.y_emb(y)
        y = y * (1 - y_block)

        ty = torch.cat((t, y), dim=-1)
        ty = self.ty_emb(ty, t)

        x_ret = self.denoise(x, ty)
        return x_ret
