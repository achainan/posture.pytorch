# posture.pytorch

A pytorch implementation of posture estimation

Simply run by using: 
```
python train.py --num_epochs 1000
```

## Dataset

Images are not included in the data folder as this is a public facing project.

The dataset should follow this folder structure:

```
  ./B/train_data.csv
  ./B/validation_data.csv
  ./B/train/
      1.jpg
      2.jpg
      ...
      N.jpg
  ./B/val/
      1.jpg
      2.jpg
      ...
      N.jpg
```

## Tensorboard

You can see the progress via tensorboard by running `tensorboard --logdir runs` and going to http://localhost:6006/

## Caution

This code is still under development.
Currently, we have small amount of annotated data.


## Changelog

- New validation data has been added. 
- Removing warning: Please note that currently the validation_data.csv is the same as the train_data.csv. This will change once we gather enough annotated data.
- Face center coordinates have been added to the training and validation dataset.

