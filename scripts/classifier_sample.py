"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    args.attention_resolutions = '32,16,8'
    args.class_cond = True
    args.diffusion_steps = 1000
    args.image_size = 256
    args.learn_sigma = True
    args.noise_schedule = 'linear'
    args.num_channels = 256
    args.num_head_channels = 64
    args.resblock_updown = True
    args.use_fp16 = True
    args.use_scale_shift_norm = True
    args.clip_denoised = True
    args.num_samples = 100
    args.batch_size = 4
    args.use_ddim = False
    args.model_path = "../models/256x256_diffusion.pt"
    args.classifier_path = "../models/256x256_classifier.pt"
    args.classifier_scale = 1.0
    args.timestep_respacing = '250'

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to('cuda:0')
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to('cuda:0')
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        # classes = th.randint(
        #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device='cuda:0'
        # )
        classes = th.full((args.batch_size,), 97).to('cuda:0') # create class for drake
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device='cuda:0',
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    with open("../imagenet_classes.txt", "r") as f:
        imagenet_labels = [line.strip() for line in f.readlines()]
    for i, img in enumerate(arr):
        plt.imshow(img)
        plt.title(f'{imagenet_labels[label_arr[i]]}')
        plt.savefig(f'../plots/{i}.png')
        plt.close()

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        class_cond=True,
        diffusion_steps=1000,
        image_size=256,
        learn_sigma=True,
        noise_schedule='linear',
        num_channels=256,
        num_head_channels=64,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,
        clip_denoised=True,
        num_samples=100,
        batch_size=4,
        use_ddim=False,
        model_path="../models/256x256_diffusion.pt",
        classifier_path="../models/256x256_classifier.pt",
        classifier_scale=1.0,
        timestep_respacing=250,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
