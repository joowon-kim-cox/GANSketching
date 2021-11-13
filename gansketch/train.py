import argparse

import torch
import torch.multiprocessing as mp

from training.gan_trainer import GANTrainer
from training.dataset import create_dataloader, yield_data


def training_loop(args:argparse.Namespace):
    torch.backends.cudnn.benchmark = True

    args.isTrain = True

    # needs to switch to spawn mode to be compatible with evaluation
    if not args.disable_eval:
        mp.set_start_method("spawn")

    # dataloader for user sketches
    dataloader_sketch, sampler_sketch = create_dataloader(
        args.dataroot_sketch, args.size, args.batch, args.sketch_channel
    )
    # dataloader for image regularization
    dataloader_image, sampler_image = create_dataloader(
        args.dataroot_image, args.size, args.batch
    )
    data_yield_image = yield_data(dataloader_image, sampler_image)

    trainer = GANTrainer(args)
    for _ in range(args.max_epoch):
        for i, data_sketch in enumerate(
            dataloader_sketch
        ):  # inner loop within one epoch
            if i >= args.max_iter:
                return

            # makes dictionary to store all inputs
            data = {}
            data["sketch"] = data_sketch
            data_image = next(data_yield_image)
            data["image"] = data_image

            # timer for argsimization per iteration
            trainer.train_one_step(data, i)



if __name__ == "__main__":
    training_loop()
    print("Training was successfully finished.")
