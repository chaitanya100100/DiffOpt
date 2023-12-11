import math
import torch
import torch.nn as nn


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        num_heads = max(num_heads, 1)
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        num_heads = max(num_heads, 1)
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, CN, _ = context.shape
        kv = (
            self.kv(context)
            .reshape(B, CN, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q = (
            self.q(x)
            .reshape(B, N, 1, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        (q,) = q.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DecBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        context_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            dim if context_dim is None else context_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm3 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x, context):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class AttentionGeneral(nn.Module):
    def __init__(
        self,
        dim,
        context_dim=None,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        num_heads = max(num_heads, 1)
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.context_dim = context_dim

        if self.context_dim is None:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, N, C = x.shape
        if self.context_dim is None:
            assert context is None
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
        else:
            assert context is not None
            _, CN, _ = context.shape
            kv = (
                self.kv(context)
                .reshape(B, CN, 2, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q = (
                self.q(x)
                .reshape(B, N, 1, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            k, v = kv.unbind(0)
            (q,) = q.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TBlockGeneral(nn.Module):
    def __init__(
        self,
        dim,
        context_dim=None,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        attn_cls=AttentionGeneral,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_cls(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.context_dim = context_dim
        if context_dim is not None:
            self.norm2 = norm_layer(dim)
            self.cross_attn = attn_cls(
                dim,
                context_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )

        self.norm3 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x, context=None):
        x = x + self.attn(self.norm1(x))
        if self.context_dim is not None:
            assert context is not None
            x = x + self.cross_attn(self.norm2(x), context)
        else:
            assert context is None
        x = x + self.mlp(self.norm3(x))
        return x


class Overfit(torch.nn.Module):
    def __init__(self, num_pred_params, *args, **kwargs):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(1, num_pred_params) * 0.02)

    def forward(self, *args, **kwargs):
        return self.param


class PoseTransformer(nn.Module):
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
