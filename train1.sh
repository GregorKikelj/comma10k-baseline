# baseline training script
# default image size is 1164*874 (width*height)
# half size is 582*437


python3 train_lit_model.py --backbone resnet18 --batch-size 60 --epochs 3 --lr 1e-3 --width 291 --height 218 --version rapidTest
# python3 train_lit_model.py --backbone efficientnet-b1 --batch-size 10 --epochs 10 --lr 2e-4 --weight-decay 1e-6 --width 582 --height 437 --version itest

