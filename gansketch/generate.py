import os
import argparse

import random
import numpy as np
import torch
from torchvision import utils
from .training.networks.stylegan2 import Generator


def save_image_pytorch(img, name):
    """Helper function to save torch tensor into an image file."""
    utils.save_image(
        img,
        name,
        nrow=1,
        padding=0,
        normalize=True,
        range=(-1, 1),
    )


def _generate(args, netG, device, mean_latent, upload_callback=None):
    """Generates images from a generator."""
    w_shift = torch.tensor(0.0, device=device)

    ind = 0
    with torch.no_grad():
        netG.eval()

        # Generate image by sampling input noises
        for start in range(0, args.samples, args.batch_size):
            end = min(start + args.batch_size, args.samples)
            batch_sz = end - start
            sample_z = torch.randn(batch_sz, 512, device=device) + w_shift

            sample, _ = netG(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            for s in sample:
                save_path = f"{args.save_dir}/{str(ind).zfill(6)}.png"
                save_image_pytorch(s, save_path)
                if upload_callback:
                    upload_callback(save_path)
                ind += 1


def generate(args: argparse.Namespace, upload_callback=None):
    device = args.device
    # use a fixed seed if given
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    netG = Generator(args.size, 512, 8).to(device)
    netG.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    # get mean latent if truncation is applied
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = netG.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    _generate(args, netG, device, mean_latent, upload_callback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir", type=str, default="./output", help="place to save the output"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint file for the generator"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output size of the generator"
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="batch size used to generate outputs"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="number of samples to generate, will be overridden if --fixed_z is given",
    )
    parser.add_argument(
        "--truncation", type=float, default=0.5, help="strength of truncation"
    )
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of samples to calculate the mean latent for truncation",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="if specified, use a fixed random seed"
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
