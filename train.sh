python3 train_lit_model.py --backbone resnet50 --batch-size 8 --epochs 40 --lr 7e-5 --weight-decay 1e-4 --width 582 --height 437 --network unet --augmentation-level hard --version train

python3 train_lit_model.py --backbone resnet50 --batch-size 4 --epochs 40 --lr 7e-5 --weight-decay 1e-4 --width 582 --height 437 --network unet --augmentation-level hard --version tune --seed-from-checkpoint a.ckpt --tune # Change the actual a.ckpt file here

