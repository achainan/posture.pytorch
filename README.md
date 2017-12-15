# posture.pytorch

A pytorch implementation of posture estimation

Simply run by using: 
```
python train.py
```

## Dataset

Images are not included in the data folder as this is a public facing project.
The dataset should follow this folder structure:

```
  ./B/posture_data.csv
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

#### Under Development
