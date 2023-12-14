
"""Argument definitions for model training code in `trainer.model`."""

import argparse

from trainer import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        help="Batch size for training steps",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--validation_split",
        help="evalution percent",
        default=0.2,
        type=float
    )

    parser.add_argument(
        "--lr", help="learning rate for optimizer", type=float, default=0.001
    )

    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models",

    )
    parser.add_argument(
        "--train_data_path",
        help="GCS location pattern of train files containing eval URLs",

    )
    parser.add_argument(
        "--dropout_rate",
        help = "drop_rate",
        type=float,
        default=0.15
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )
    args = parser.parse_args()
    hparams = args.__dict__

    model.train_and_evaluate(hparams)
