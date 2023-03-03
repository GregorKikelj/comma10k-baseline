from argparse import ArgumentParser

from LitModel import *
from pytorch_lightning import Trainer, seed_everything

seed_everything(1994)


def main(args):
    """Main training routine specific for this project."""
    model = LitModel.load_from_checkpoint(args.seed_from_checkpoint, **vars(args))

    model.setup("test")
    dataLoader = model.val_dataloader()
    trainer = Trainer(
        gpus=1,
        benchmark=True,
    )

    trainer.logger.log_hyperparams(model.hparams)

    trainer.validate(model, dataloaders=dataLoader)


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
        "--seed-from-checkpoint",
        default=None,
        type=str,
        metavar="SFC",
        help="path to checkpoint seed",
    )

    args = parser.parse_args()

    main(args)
