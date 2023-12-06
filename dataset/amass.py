import os
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torch as th
import numpy as np
from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)
from utils.meshviewer import makeLookAt
import pyrender


class AMASS(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data_root = "/move/u/jiamanli/datasets/amass"
        self.sequences = sorted(glob.glob(self.data_root + "/CMU/*/*.npz"))
        self.sequences = self.sequences[-100:]
        print("Number of sequences", len(self.sequences))
        self.smpl = MySMPL()

    def __len__(self):
        return len(self.sequences)

    def random_camera(self, center):
        # target = center + torch.randn_like(center) * 1.0
        target = center

        # z = torch.randn_like(center[2:]) * 1
        # xy = torch.zeros_like(center[:2])
        # xy = xy + torch.randn_like(xy).abs() * 1 + 0
        # position = torch.cat([xy, z], -1)
        position = center + torch.tensor([2.0, 2.0, 1.0]).float()
        position = position + torch.randn_like(position) * 0.6

        up = torch.zeros_like(center)
        up[..., -1] = 1

        T = makeLookAt(position.numpy(), target.numpy(), up.numpy())
        # T = look_at(position.numpy(), target.numpy(), up.numpy())
        T = torch.tensor(T).float()
        T2 = T.clone()
        T2[:3, :3] = T[:3, :3].T
        T2[:3, 3] = -T[:3, :3].T @ T[:3, 3]

        width = height = 600
        yfov = np.pi / 4.0
        K = pyrender.PerspectiveCamera(
            yfov=yfov, aspectRatio=float(width) / height
        ).get_projection_matrix()
        K = torch.tensor(K).float()

        P = K @ T2

        cam = {
            "T": T,
            "K": K,
            "P": P,
            "target": target,
            "position": position,
            "up": up,
            "yfov": yfov,
            "width": width,
            "height": height,
        }

        return cam

    def projection(self, joints, cam):
        J = joints
        J = torch.cat([J, torch.ones_like(J[:, :1])], -1)
        J = J @ cam["P"].T
        J = J[:, :3] / J[:, 3:]
        J = J / J[:, 2:]

        h, w = cam["height"], cam["width"]
        assert w == h
        J = J * w / 2 + w / 2
        J[:, 1] = h - J[:, 1]

        conf = (J[:, 0] >= 0) & (J[:, 0] <= w) & (J[:, 1] >= 0) & (J[:, 1] <= h)
        J[:, 2] = conf.float()
        return J

    def __getitem__(self, idx, frame_idx=None):
        sequence = self.sequences[idx]
        bdata = np.load(sequence)

        num_frames = bdata["trans"].shape[0]
        if frame_idx is None:
            frame_idx = np.random.randint(0, num_frames)
            frame_idx = slice(frame_idx, frame_idx + 1)
        elif frame_idx == -1:
            fps = int(bdata["mocap_framerate"])
            frame_idx = slice(0, 5 * fps, fps // 30)

        poses = torch.tensor(bdata["poses"][frame_idx]).float()
        global_orient = poses[..., :3].float()
        body_pose = poses[..., 3 : 3 + 23 * 3].float()
        transl = torch.tensor(bdata["trans"][frame_idx]).float()

        global_orient = axis_angle_to_matrix(global_orient)
        body_pose = axis_angle_to_matrix(body_pose.unflatten(-1, (-1, 3)))

        betas = torch.from_numpy(bdata["betas"][:10]).float()
        betas = betas[None].expand(body_pose.shape[0], -1)

        data = {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "transl": transl,
            "betas": betas,
        }

        bmout = smpl(**{k: v for k, v in data.items()})
        joints = bmout.joints
        B, J = joints.shape[:2]
        cam = self.random_camera(joints.mean((0, 1)))
        j2d = self.projection(joints.flatten(0, 1), cam).unflatten(0, (B, J))
        cam = default_collate([cam] * B)

        data["j2d"] = j2d
        data["cam"] = cam
        data["x"], data["y"] = self.data_to_xy(data)

        if B == 1:
            cam = {k: v[0] for k, v in data.pop("cam").items()}
            data = {k: v[0] for k, v in data.items()}
            data["cam"] = cam
        return data

    def data_to_xy(self, data):
        j2d = data["j2d"]
        cam = data["cam"]
        h, w = cam["height"], cam["width"]
        assert torch.allclose(h, w)
        y = j2d[..., :2] / w[:, None, None] - 0.5
        x = torch.cat([data["global_orient"][:, None], data["body_pose"]], 1)  # BJ33
        x = matrix_to_rotation_6d(x)
        return x, y

    def get_sequence_data(self, idx):
        return self.__getitem__(idx, frame_idx=-1)

    # def get_sequence_data(self, idx):
    #     return default_collate(
    #         [self.__getitem__(idx, [i]) for i in range(0, 5 * 120, 4)]
    #     )


if __name__ == "__main__":
    from utils.smpl import MySMPL
    from utils.viz_utils import viz_smpl, save_video, show_points, dcn
    import cv2

    smpl = MySMPL()
    amass = AMASS()
    dataloader = torch.utils.data.DataLoader(
        amass, batch_size=1, shuffle=True, num_workers=0
    )

    for i, data in enumerate(dataloader):
        bmout = smpl(**data)
        img = viz_smpl(bmout, smpl.faces, data["cam"])
        img = show_points(dcn(data["j2d"]), img)
        cv2.imwrite("check.png", img[0])
        break

    # sequence
    data = amass.get_sequence_data(1)
    bmout = smpl(**data)
    img = viz_smpl(bmout, smpl.faces, data["cam"])
    img = show_points(dcn(data["j2d"]), img)
    save_video(img, "check.webm", 30)

    # camera following the center
    # imgs = []
    # for i in range(0, 5 * 120, 4):
    #     data = default_collate([amass.__getitem__(1, [i])])
    #     bmout = smpl(**data)
    #     img = viz_smpl(bmout, smpl.faces, data["cam"])
    #     img = show_points(dcn(data["j2d"]), img)
    #     imgs.append(img[0])
    # save_video(np.stack(imgs), "check.webm", 30)

    import IPython

    IPython.embed()
