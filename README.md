# comma10k-baseline 

Iterating on [yassine](https://github.com/YassineYousfi/comma10k-baseline)'s baseline

Replacing efficientnet with resnet seems to achieve a better valuation loss with much lower training and inference time.
We also use stochastic weight averaging to improve the final model. We achieve a validation loss of 0.0417 training on the half-resolution images

## How to use
The model can be trained in 10 hours on a gtx 1080ti
To train the model simply run `train.sh` script and change the data path and log paths. 
Requirements.txt lists the versions that make autoalbument work though it would probably be a better idea to use their docker image. Generally newer packages should work fine(and autoupgrading works as of this readme creation)

## Possible improvements
- train on bigger gpu with 16 bit support to allow for full resolution and larger batch sizes

## Failures
- deeplabv3+ doesn't seem better
- [autoalbument](https://albumentations.ai/docs/autoalbument/) is better than not using augmentations. However even after training for about 2 days the augmentation still performs (slightly) worse than yassine's baseline `hard` augmentation. My hypothesis is that autoalbument tries to only make augmentations that don't significantly "worsen" the image but this then doesn't help much with overfitting.

