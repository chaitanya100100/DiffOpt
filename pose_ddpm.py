import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

from utils.ddpm_utils import ddpm_schedules, get_cur_y
from config.pose_config import Config
from model.pose_model import PoseModelMLP, PoseModelMLP_Res
from dataset.amass import AMASS, amass_collate_fn
from utils.smpl import MySMPL


class DDPM(nn.Module):
    def __init__(self, cfg):
        super(DDPM, self).__init__()
        self.cfg = cfg

        # self.nn_model = PoseModelMLP_Res(cfg)
        self.nn_model = PoseModelMLP(cfg)
        self.pass_data = isinstance(self.nn_model, PoseModelMLP_Res)
        print("Number of parameters", sum(p.numel() for p in self.parameters()))

        betas = cfg.betas
        for k, v in ddpm_schedules(betas[0], betas[1], cfg.n_T).items():
            self.register_buffer(k, v)

        self.n_T = cfg.n_T
        self.drop_prob = cfg.drop_prob

        w = torch.tensor([cfg.joint_importance[i] for i in range(24)]).float()
        w = w.pow(2)
        w = 24 * w / w.sum()
        self.register_buffer("joint_weight", w, persistent=False)
        print(w)

    def forward(self, x, y, data=None):
        if not self.pass_data:
            data = None
        device = x.device
        B = x.shape[0]

        _ts = torch.randint(1, self.n_T + 1, (B,)).to(device)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        x_t = self.sqrtab[_ts, None] * x + self.sqrtmab[_ts, None] * noise

        y_block = torch.bernoulli(torch.zeros(B) + self.drop_prob).to(device)

        loss = (
            noise - self.nn_model(x_t, y, _ts / self.n_T, y_block, data=data)
        ).square()
        # return loss.mean()
        loss = loss.view(B, 24, 6) * self.joint_weight[None, :, None]
        return loss.mean()

    @torch.no_grad()
    def sample(self, y, guide_w=0.0, data=None):
        if not self.pass_data:
            data = None
        cfg = self.cfg
        device = y.device
        B = y.shape[0]

        x_i = torch.randn(B, cfg.x_dim).to(device)

        y_block = torch.cat([torch.zeros(B), torch.ones(B)]).to(device)
        y = torch.cat([y, y], dim=0)
        if data is not None:
            data = {k: torch.cat([v, v], dim=0) for k, v in data.items()}

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            print(f"sampling timestep {i}", end="\r")
            z = torch.randn_like(x_i).to(device) if i > 1 else 0

            # double batch
            x_i = torch.cat([x_i, x_i], dim=0)
            t_is = torch.tensor([i / self.n_T] * (2 * B)).to(device)

            # split predictions and compute weighting
            eps = self.nn_model(x_i, y, t_is, y_block, data=data)
            eps1 = eps[:B]
            eps2 = eps[B:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:B]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i)

        x_i_store = torch.stack(x_i_store)
        return x_i, x_i_store


def validate_pose(ddpm, dataloader, cfg, device, ep):
    ddpm.eval()
    dataset = dataloader.dataset
    data = next(iter(dataloader))
    data = {k: v[:8] for k, v in data.items()}
    for w_i, w in enumerate(cfg.ws_test):
        # imgs_gt = dataset.viz(data["x"], data)
        # imgs_pred = imgs_gt.copy()

        x_gen, x_gen_store = ddpm.sample(data["y"].to(device), guide_w=w, data=data)
        imgs_pred = dataset.viz(x_gen.cpu(), data)
        imgs_gt = dataset.viz(data["x"], data)

        imgs = np.concatenate([imgs_pred, imgs_gt], 2)
        for ii, img in enumerate(imgs):
            img = Image.fromarray(img)
            img.save(cfg.save_dir + f"/ep_{ep}_w_{w_i}_img_{ii}.png")


def evaluate_pose(ddpm, dataloader, cfg, device, ep):
    dataset = dataloader.dataset
    ddpm.eval()
    smpl = MySMPL().to(device)

    validate_pose(ddpm, dataloader, cfg, device, ep)

    for w_i, w in enumerate(cfg.ws_test):
        y_gt = []
        y_pred = []
        for i in range(10):
            data = dataset.get_sequence_data(i)

            x_gen, x_gen_store = ddpm.sample(data["y"].to(device), guide_w=w, data=data)
            y_gen = get_cur_y(x_gen, data, smpl)
            y_gt.append(data["y"].cpu())
            y_pred.append(y_gen.cpu())

        y_gt = torch.cat(y_gt, 0)
        y_pred = torch.cat(y_pred, 0)
        keep = y_pred.isnan().bitwise_not()
        y_gt = y_gt[keep]
        y_pred = y_pred[keep]
        err = (y_gt - y_pred).abs()
        # err = err.sort()[0]
        # err = err[: int(err.shape[0] * 0.9)]
        err = err.mean()
        print(f"w {w}, err {err} using {keep.float().mean()*100}%")


def train_pose(cfg):
    os.makedirs(cfg.save_dir, exist_ok=False)

    dataset = AMASS(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=amass_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm = DDPM(cfg)
    if cfg.ckpt_path is not None:
        print("loading ckpt from " + cfg.ckpt_path)
        ddpm.load_state_dict(torch.load(cfg.ckpt_path, map_location="cpu"))
    ddpm = ddpm.to(device)

    if cfg.validate:
        print("only validating")
        with torch.no_grad():
            evaluate_pose(ddpm, dataloader, cfg, device, 0)
        return

    optim = torch.optim.Adam(
        ddpm.parameters(), lr=cfg.lrate, weight_decay=cfg.weight_decay
    )
    scheduler = lr_scheduler.LinearLR(
        optim, start_factor=1.0, end_factor=0.01, total_iters=cfg.n_epoch
    )

    for ep in range(cfg.n_epoch):
        lr = optim.param_groups[0]["lr"]
        print(f"epoch {ep}, lr {lr}")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for data in pbar:
            optim.zero_grad()
            x = data["x"].to(device)
            y = data["y"].to(device)
            loss = ddpm(x, y, data)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        scheduler.step()
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        with torch.no_grad():
            validate_pose(ddpm, dataloader, cfg, device, ep)

        # optionally save model
        if (ep + 1) % 4 == 0:
            torch.save(ddpm.state_dict(), cfg.save_dir + f"/model_{ep}.pth")
            print("saved model at " + cfg.save_dir + f"/model_{ep}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="./data/check/")
    parser.add_argument("--subset", default="oneseq")
    parser.add_argument("--hidden_dim", default=1024, type=int)
    parser.add_argument("--num_layers", default=6, type=int)
    parser.add_argument("--n_epoch", default=50)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--lrate", default=1e-4, type=float)
    parser.add_argument("--validate", default=False, action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.save_dir = args.save_dir
    cfg.subset = args.subset
    cfg.hidden_dim = args.hidden_dim
    cfg.num_layers = args.num_layers
    cfg.n_epoch = args.n_epoch
    cfg.ckpt_path = args.ckpt_path
    cfg.lrate = args.lrate
    cfg.validate = args.validate
    cfg.batch_size = args.batch_size
    train_pose(cfg)
