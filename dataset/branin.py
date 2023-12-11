import glob
import torch
import copy
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
import numpy as np


class Branin(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.a = 1
        self.b = 5.1 / (4 * np.pi**2)
        self.c = 5 / np.pi
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * np.pi)

        self.range = np.array([[-5, 10], [0, 15]])

    def __len__(self):
        return 10000

    def f(self, x):
        x1 = x[..., 0] * 15 - 5
        x2 = x[..., 1] * 15
        a, b, c, r, s, t = self.a, self.b, self.c, self.r, self.s, self.t
        y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s
        return y

    def __getitem__(self, idx):
        x = torch.rand(2)
        y = self.f(x)
        return {"x": x, "y": y}

    def viz(self, x, data):
        B = x.shape[0]
        data = copy.deepcopy(data)

        # return img


if __name__ == "__main__":
    import IPython

    IPython.embed()
