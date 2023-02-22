import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from argparse import ArgumentParser
from LitModel import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

seed_everything(1994)


def main(args):
    """Main training routine specific for this project."""

    if args.seed_from_checkpoint:
        print("model seeded")
        model = LitModel.load_from_checkpoint(args.seed_from_checkpoint, **vars(args))
    else:
        model = LitModel(**vars(args))

    trainer = Trainer(
        gpus=1,
        max_epochs=args.epochs,
        row_log_interval=100,
        log_save_interval=100,
        benchmark=True,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.logger.log_hyperparams(model.hparams)

    trainer.fit(model)


if __name__ == "__main__":
    parent_parser = ArgumentParser(add_help=False)

    parser = LitModel.add_model_specific_args(parent_parser)

    parser.add_argument(
        "--version",
        default=None,
        type=str,
        metavar="V",
        help="version or id of the net",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        type=str,
        metavar="RFC",
        help="path to checkpoint",
    )
    parser.add_argument(
        "--seed-from-checkpoint",
        default=None,
        type=str,
        metavar="SFC",
        help="path to checkpoint seed",
    )

    args = parser.parse_args()

    main(args)
