import glob
import torch
import copy
import cv2
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
import pyrender
from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from utils.smpl import MySMPL
from utils.viz_utils import viz_smpl, save_video, show_points, dcn
from utils.ddpm_utils import random_camera, projection, j2d_to_y


def amass_collate_fn(data):
    data = default_collate(data)
    data = {k: v.flatten(0, 1) for k, v in data.items()}
    return data


class AMASS(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.data_root = "/move/u/jiamanli/datasets/amass"
        subset = cfg.subset
        assert subset in ["oneseq", "cmu", "full"]
        if subset == "oneseq":
            self.sequences = sorted(glob.glob(self.data_root + "/CMU/*/*_poses.npz"))
            self.sequences = self.sequences[-99:-98]
            self.sequences = self.sequences * 11000
        elif subset == "cmu":
            self.sequences = sorted(glob.glob(self.data_root + "/CMU/*/*_poses.npz"))
            self.sequences = self.sequences * 6
        elif subset == "full":
            self.sequences = sorted(glob.glob(self.data_root + "/*/*/*_poses.npz"))

        # self.sequences = self.sequences[:100]

        print("Number of sequences", len(self.sequences))
        self.smpl = MySMPL()
        self.N = 8

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx, frame_idx=None):
        sequence = self.sequences[idx]
        bdata = np.load(sequence)

        try:
            num_frames = bdata["trans"].shape[0]
        except:
            print(sequence)
            exit(0)
        is_sequence = False
        if frame_idx is None:
            # frame_idx = np.random.randint(0, num_frames)
            # frame_idx = slice(frame_idx, frame_idx + 1)
            frame_idx = np.random.randint(0, num_frames, self.N)
        elif frame_idx == -1:
            fps = int(bdata["mocap_framerate"])
            frame_idx = slice(0, 5 * fps, fps // 30)
            is_sequence = True

        poses = torch.tensor(bdata["poses"][frame_idx]).float()
        global_orient = poses[..., :3].float()
        body_pose = poses[..., 3 : 3 + 23 * 3].float()
        transl = torch.tensor(bdata["trans"][frame_idx]).float()

        global_orient = axis_angle_to_matrix(global_orient)
        body_pose = axis_angle_to_matrix(body_pose.unflatten(-1, (-1, 3)))

        betas = torch.from_numpy(bdata["betas"][:10]).float()
        betas = betas[None].expand(body_pose.shape[0], -1)
        betas = torch.zeros_like(betas)

        data = {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "transl": transl,
            "betas": betas,
        }

        bmout = self.smpl(**{k: v for k, v in data.items()})
        joints = bmout.joints
        B, J = joints.shape[:2]
        if is_sequence:
            cam = random_camera(joints.mean((0, 1)))
            cam = default_collate([cam] * B)
        else:
            cam = random_camera(joints.mean(1))

        j2d = projection(joints, cam)
        data.update({f"cam_{k}": v for k, v in cam.items()})

        data["j2d"] = j2d
        data["x"], data["y"] = self.data_to_xy(data)

        if B == 1:
            data = {k: v[0] for k, v in data.items()}
        return data

    def data_to_xy(self, data):
        x = torch.cat([data["global_orient"][:, None], data["body_pose"]], 1)  # BJ33
        x = matrix_to_rotation_6d(x).view(x.shape[0], -1)

        y = j2d_to_y(data["j2d"], data["cam_height"], data["cam_width"])
        return x, y

    def get_sequence_data(self, idx):
        return self.__getitem__(idx, frame_idx=-1)

    # def get_sequence_data(self, idx):
    #     return default_collate(
    #         [self.__getitem__(idx, [i]) for i in range(0, 5 * 120, 4)]
    #     )

    def viz(self, x, data):
        B = x.shape[0]
        data = copy.deepcopy(data)

        data["global_orient"] = rotation_6d_to_matrix(x[:, :6])
        data["body_pose"] = rotation_6d_to_matrix(x[:, 6:].unflatten(-1, (-1, 6)))

        bmout = self.smpl(**data)
        cam = {
            k.replace("cam_", "", 1): v for k, v in data.items() if k.startswith("cam_")
        }
        img = viz_smpl(bmout, self.smpl.faces, cam)
        img = show_points(dcn(data["j2d"]), img)
        return img


if __name__ == "__main__":
    amass = AMASS(None)
    smpl = amass.smpl
    dataloader = torch.utils.data.DataLoader(
        amass, batch_size=1, shuffle=True, num_workers=0, collate_fn=amass_collate_fn
    )

    for i, data in enumerate(dataloader):
        bmout = smpl(**data)
        cam = {
            k.replace("cam_", "", 1): v for k, v in data.items() if k.startswith("cam_")
        }

        img = viz_smpl(bmout, smpl.faces, cam)
        img = show_points(dcn(data["j2d"]), img)
        for ii in range(len(img)):
            cv2.imwrite(f"./data/test/check{ii}.png", img[ii])
        # cv2.imwrite("check2.png", img[1])
        break

    # sequence
    # data = amass.get_sequence_data(1)
    data = amass.__getitem__(1, frame_idx=slice(0, 5 * 120, 4))
    bmout = smpl(**data)
    cam = {k.replace("cam_", "", 1): v for k, v in data.items() if k.startswith("cam_")}
    img = viz_smpl(bmout, smpl.faces, cam)
    img = show_points(dcn(data["j2d"]), img)
    save_video(img, "./data/test/check.webm", 30)

    # # camera following the center
    # imgs = []
    # for i in range(0, 5 * 120, 4):
    #     data = default_collate([amass.__getitem__(1, [i])])
    #     bmout = smpl(**data)
    #     cam = {k.replace("cam_", "", 1): v for k, v in data.items() if k.startswith("cam_")}
    #     img = viz_smpl(bmout, smpl.faces, cam)
    #     img = show_points(dcn(data["j2d"]), img)
    #     imgs.append(img[0])
    # save_video(np.stack(imgs), "check.webm", 30)

    import IPython

    IPython.embed()
