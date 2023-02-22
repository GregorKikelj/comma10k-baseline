# baseline training script
# default image size is 1164*874 (width*height)

# python3 train_lit_model.py --backbone efficientnet-b1 --batch-size 2 --epochs 10 --lr 2e-4 --weight-decay 1e-6 --version itest

python3 train_lit_model.py --backbone efficientnet-b4 --version first-stage --batch-size 2 --epochs 100 --height 437 --width 582 # (2, 3, 448, 608)