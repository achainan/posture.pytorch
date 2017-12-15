"""This module trains the pose estimation model."""

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import models
import constants
import cv2
from dataset import load_dataset
from tensorboardX import SummaryWriter
from third_party import AverageMeter

cuda = torch.cuda.is_available()
logger = SummaryWriter()


def main():
    shuffle = True

    dataset = load_dataset()
    train_dataset = dataset["train"]
    val_dataset = dataset["valid"]

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=constants.batch_size,
                                               shuffle=shuffle,
                                               pin_memory=cuda,
                                               num_workers=constants.num_workers,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=constants.batch_size,
                                             shuffle=shuffle,
                                             pin_memory=cuda,
                                             num_workers=constants.num_workers,
                                             drop_last=True)

    cnn = models.CNN()
    if cuda:
        cnn.cuda()

    # cnn.summary()

    # Optimizer and Loss
    optimizer = torch.optim.Adam(cnn.parameters(), lr=constants.learning_rate)
    criterion = nn.MSELoss()
    if cuda:
        criterion.cuda()

    # Train the Model
    for epoch in range(constants.num_epochs):
        train(train_loader, cnn, optimizer, criterion, epoch)

        validate(val_loader, cnn, criterion, epoch)

        if (epoch + 1) % constants.save_interval == 0:
            save_checkpoint(cnn)

    logger.close()

    # Save the Trained Model
    torch.save(cnn.state_dict(), 'result/cnn.pkl')
    torch.save(cnn, 'result/cnn.pth')


def save_checkpoint(model, filename='result/cnn_checkpoint.pth'):
    print "saving checkpoint"
    torch.save(model, filename)


def save_preview(images, outputs, i, name='model/(train)output'):
    """This function logs a preview image to tensorboard"""
    image = images.data[0].cpu().numpy()
    output = outputs.data[0].view(1, -1, 2).cpu().numpy()
    image = np.rollaxis(image, 0, 3)
    image = image.squeeze()

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for coordinates in output[0]:
        x = coordinates[0] * 670
        y = coordinates[1] * 670
        circle_size = 2
        cv2.circle(image, (int(x), int(y)), circle_size, (0, 0, 255), -1)

    logger.add_image(name, image, i + 1)


def validate(loader, model, criterion, epoch):
    model.eval()
    losses = AverageMeter()

    for i, (val_images, val_labels) in enumerate(loader):
        images = Variable(val_images)
        labels = Variable(val_labels)
        if cuda:
            images = images.cuda()
            labels = labels.cuda()

        # compute output
        outputs = model(images)
        loss = criterion(outputs, labels)

        # log the loss
        losses.update(loss.data[0], images.size(0))

        niter = epoch * len(loader) + i
        logger.add_scalar('Val/Loss', loss.data[0], niter)

        if i % constants.print_freq == 0:
            save_preview(images, outputs, i, 'model/(val)output')
            print('[VALID] - EPOCH %d/ %d - BATCH LOSS: %.8f/ %.8f(avg) '
                  % (epoch + 1, constants.num_epochs, losses.val, losses.avg))


def train(loader, model, optimizer, criterion, epoch):
    model.train()
    losses = AverageMeter()

    for i, (train_images, train_labels) in enumerate(loader):
        images = Variable(train_images)
        labels = Variable(train_labels)
        if cuda:
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        losses.update(loss.data[0], images.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        niter = epoch * len(loader) + i
        logger.add_scalar('Train/Loss', loss.data[0], niter)

        if i % constants.print_freq == 0:
            print('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.8f/ %.8f(avg) '
                  % (epoch + 1, constants.num_epochs, losses.val, losses.avg))


if __name__ == '__main__':
    main()
