import os
from distutils.command.build_scripts import first_line_re

import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F

from rife.pytorch_msssim import ssim_matlab
from rife.RIFE import Model


class Interpolation(object):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "flownet.pkl")

    def __init__(self, model: Model, scale=1.0, fp16=False, exp=1, device='cpu'):
        # use scale=0.5 for 4K frames
        # exp=1 - 2X interpolation; exp=2 - 4X interpolation
        self.model = model
        self.scale = scale
        self.fp16 = fp16
        self.exp = exp
        self.device = device

        self.model.device(self.device)
        self.model.eval()

    @classmethod
    def load(cls, model_path=model_path, scale=1.0, fp16=False, exp=1, device='cpu'):
        model = Model()
        model.load_model(model_path)
        return cls(model=model, scale=scale, fp16=fp16, exp=exp, device=device)

    @torch.inference_mode
    def __call__(self, videogen: np.ndarray):
        if self.fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        tot_frame = len(videogen)
        videoten = torch.permute(torch.from_numpy(videogen), (0, 3, 1, 2)).float() / 255.
        first_frame = videoten[0].to(device=self.device)
        _, h, w = first_frame.shape
        tmp = max(32, int(32 / self.scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        pbar = tqdm(total=tot_frame)

        I0 = first_frame.unsqueeze(0)
        I0 = self.pad_image(I0, padding=padding)
        output = list()

        j = 1
        while j < tot_frame:
            I1 = videoten[j].to(device=self.device).unsqueeze(0)
            I1 = self.pad_image(I1, padding=padding)
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

            output.append(I0.cpu())
            if ssim > 0.996 and j < tot_frame-1:
                I1 = videoten[j+1].to(device=self.device).unsqueeze(0)
                I1 = self.pad_image(I1, padding=padding)
                I1 = self.model.inference(I0, I1, self.scale)
                I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
                I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

            if ssim < 0.2:
                for i in range((2 ** self.exp) - 1):
                    output.append(I0.cpu())
            elif self.exp:
                for middle in self.make_inference(I0, I1, 2 ** self.exp - 1):
                    output.append(middle.cpu())

            pbar.update(1)
            I0 = I1
            j += 1
        output.append(I0.cpu())
        pbar.update(1)

        pbar.close()
        return (255.0 * torch.permute(torch.cat(output, dim=0), (0, 2, 3, 1))[:, :h, :w]).numpy()

    def make_inference(self, I0, I1, n):
        middle = self.model.inference(I0, I1, self.scale)
        if n == 1:
            return [middle]
        first_half = self.make_inference(I0, middle, n=n // 2)
        second_half = self.make_inference(middle, I1, n=n // 2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def pad_image(self, img, padding):
        if self.fp16:
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)
