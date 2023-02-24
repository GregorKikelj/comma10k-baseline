# baseline training script
# default image size is 1164*874 (width*height)
# half size is 582*437


# python3 train_lit_model.py --backbone resnet18 --batch-size 60 --epochs 10 --lr 1e-3 --width 291 --height 218 --version rapidTest
python3 train_lit_model.py --backbone resnet50 --batch-size 10 --epochs 10 --lr 2e-4 --weight-decay 1e-6 --width 582 --height 437 --version half_size_test_1
# python3 train_lit_model.py --backbone resnet50 --batch-size 10 --epochs 10 --lr 2e-4 --weight-decay 1e-6 --width 582 --height 437 --augmentation-level l1 --version augmentation

