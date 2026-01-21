import os
import cv2
import torch
import argparse
import numpy as np
import warnings

from rife import Interpolation

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, required=True)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--fp16', dest='fp16', action='store_true',
                        help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
    parser.add_argument('--device', dest='device', type=str, default='cpu')
    parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
    parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='output video extension')
    parser.add_argument('--exp', dest='exp', type=int, default=1, help="1 - for 2X, 2 - for 4X interpolation")
    args = parser.parse_args()
    assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]

    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = list()
    while cap.isOpened():
        success, image = cap.read()
        if image is None:
            break
        frames.append(image)

    interpolation = Interpolation.load(scale=args.scale, fp16=args.fp16, exp=args.exp, device=args.device)
    output = interpolation(torch.from_numpy(np.array(frames))).cpu().numpy()

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    mult = 2 ** interpolation.exp
    target_fps = int(np.round(fps*mult))
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, mult, target_fps, args.ext)
    n, h, w, c = output.shape
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, target_fps, (w, h))
    for frame in output:
        vid_out.write(frame)
    vid_out.release()


if __name__ == '__main__':
    main()
