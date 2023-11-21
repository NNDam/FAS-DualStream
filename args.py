import os
import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        # default=r"../../data/domain_hawkice_damdev",
        # default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--train_txt",
        type=str,
        default='',
        help="Path to train txt dataset.",
    )
    parser.add_argument(
        "--val_txt",
        type=str,
        default='',
        help="Path to val txt dataset.",
    )
    # parser.add_argument(
    #     "--eval-datasets",
    #     default=None,
    #     type=lambda x: x.split(","),
    #     help="Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
    #          " Note that same model used for all datasets, so much have same classnames"
    #          "for zero shot.",
    # )
    parser.add_argument(
        "--total-classes",
        # default='CarBandDataset',
        type=int,
        help="Total number of classes",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=512,
        help="The size of embedding",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="openai_imagenet_template",
        help="Which prompt template is used. Leave as None for linear probe, etc.",
    )
    # parser.add_argument(
    #     "--classnames",
    #     type=str,
    #     default="openai",
    #     help="Which class names to use.",
    # )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default='openclip',
        help="The architecture type of model (e.g. openclip, dino).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='ViT-B-16',
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="The pretrained weight of model.",
    )
    # parser.add_argument(
    #     "--resume",
    #     type=str,
    #     default=None,
    #     help="Path of model to resume.",
    # )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
        help="Enable mixed precision."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--save-interval", 
        type=int, 
        default=1,
        help="Save checkpoint every save_interval epoch",
    )
    parser.add_argument(
        "--eval-train",
        default=False,
        action="store_true",
        help="Whether or not to evaluate training results."
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        '--gpu-id', default='0', type=str,
        help='id(s) for CUDA_VISIBLE_DEVICES'
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args