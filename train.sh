# baseline training script
# default image size is 1164*874 (width*height)
# padded to 32 is 1184*896
# half size is 582*437 (608 * 448)
# quarter size is 291*218 (320 * 224)


# python3 train_lit_model.py --backbone efficientnet-b0 --batch-size 50 --epochs 10 --lr 1e-3 --width 291 --height 218 --version rapidTest


# baseline
# python3 train_lit_model.py --backbone efficientnet-b3 --batch-size 5 --epochs 10 --lr 2e-4 --weight-decay 1e-6 --width 582 --height 437 --network unet --version hst_baseline
# python3 train_lit_model.py --backbone efficientnet-b3 --batch-size 5 --epochs 10 --lr 2e-4 --weight-decay 1e-6 --width 582 --height 437 --network deeplab --version hst_deeplab


# Custom transform? v1 wasn't good, might get better but let's test v2
# python3 train_lit_model.py --backbone efficientnet-b3 --batch-size 5 --epochs 10 --lr 2e-4 --weight-decay 1e-6 --width 582 --height 437 --network deeplab --augmentation-level custom --version hst_aalb

# Let's try the better augmentation
# python3 train_lit_model.py --backbone efficientnet-b3 --batch-size 5 --epochs 10 --lr 2e-4 --weight-decay 1e-6 --width 582 --height 437 --network deeplab --augmentation-level customv2 --version hst_aalb2
# python3 train_lit_model.py --backbone efficientnet-b3 --batch-size 5 --epochs 10 --lr 1e-4 --weight-decay 1e-5 --width 582 --height 437 --network deeplab --augmentation-level customv2 --version hst_aalb2 --seed-from-checkpoint "/home/gregor/logs/segnet/hst_aalb2 03.03-09:07/sn epoch=09-val_loss=0.051.ckpt"

# b4 with higher decay works well, don't like the augmentation though
# python3 train_lit_model.py --backbone efficientnet-b4 --batch-size 5 --epochs 10 --lr 2e-4 --weight-decay 1e-5 --width 582 --height 437 --network deeplab --augmentation-level customv2 --version hst_aalb21

# let's see if hard is a better augmentation: this worked extremely well, double testing because I don't trust it; double testing confirmed it...
# python3 train_lit_model.py --backbone efficientnet-b4 --batch-size 5 --epochs 10 --lr 2e-4 --weight-decay 1e-5 --width 582 --height 437 --network deeplab --augmentation-level hard --version hst_hard

# python3 train_lit_model.py --backbone efficientnet-b4 --batch-size 5 --epochs 10 --lr 2e-4 --weight-decay 1e-5 --width 582 --height 437 --network deeplab --augmentation-level customv3hard --version hst_chard


# This is a new baseline, 0.0436 (0.0431 on hard, v4hard might have more potential, bayesian search in progress)
# python3 train_lit_model.py --backbone efficientnet-b3 --batch-size 5 --epochs 20 --lr 2e-4 --weight-decay 1e-5 --width 582 --height 437 --network deeplab --augmentation-level customv4 --version hst_c4

# Custom tuning
# python3 train_lit_model.py --backbone efficientnet-b3 --batch-size 5 --epochs 10 --lr 2e-4 --weight-decay 1e-5 --width 582 --height 437 --network deeplab --augmentation-level customv4 --version hst_c4_SWA --seed-from-checkpoint "/home/gregor/logs/segnet/hst_c4 05.03-15:00/sn epoch=19-val_loss=0.044.ckpt"
# python3 train_lit_model.py --backbone efficientnet-b3 --batch-size 5 --epochs 10 --lr 2e-4 --weight-decay 1e-5 --width 582 --height 437 --network deeplab --augmentation-level hard --version hst_hard_SWA --seed-from-checkpoint "/home/gregor/logs/segnet/hst_hard20 05.03-19:42/sn epoch=19-val_loss=0.043.ckpt"

# python3 train_lit_model.py --backbone resnet50 --batch-size 5 --epochs 25 --lr 1.5e-4 --weight-decay 5e-4 --width 582 --height 437 --network unet --augmentation-level customv4 --version hst_c4_sweep
# python3 train_lit_model.py --backbone resnet50 --batch-size 5 --epochs 25 --lr 1.5e-4 --weight-decay 5e-4 --width 582 --height 437 --network deeplab --augmentation-level customv4 --version hst_c4_sweep2

# Some model eval testing
# python3 eval_model.py --backbone efficientnet-b3 --batch-size 5 --epochs 25 --lr 1.5e-4 --weight-decay 5e-4 --width 582 --height 437 --network deeplab --augmentation-level customv4 --version hst_c4_sweep2 --seed-from-checkpoint "val.ckpt"

# python3 train_lit_model.py --backbone resnet50 --batch-size 8 --epochs 40 --lr 7e-5 --weight-decay 1e-4 --width 582 --height 437 --network unet --augmentation-level hard --version 419_repro

python3 eval_model.py --backbone resnet50 --batch-size 4 --epochs 40 --lr 7e-5 --weight-decay 1e-4 --width 582 --height 437 --network unet --augmentation-level hard --version 419_test --seed-from-checkpoint a.ckpt
