"""This module trains the pose estimation model."""

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import models
import constants
import preview as P
from dataset import load_dataset
from tensorboardX import SummaryWriter
from third_party import AverageMeter

cuda = torch.cuda.is_available()
logger = SummaryWriter()

images_mean, images_std, labels_mean, labels_std = constants.normalization_values(
    grayscale=constants.grayscale)


def main():
    shuffle = True

    dataset = load_dataset(
        images_mean,
        images_std,
        labels_mean,
        labels_std,
        grayscale=constants.grayscale)
    train_dataset = dataset["train"]
    val_dataset = dataset["valid"]

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=constants.train_batch_size,
                                               shuffle=shuffle,
                                               pin_memory=cuda,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=constants.val_batch_size,
                                             shuffle=shuffle,
                                             pin_memory=cuda,
                                             drop_last=True)

    input_channels = 3
    if constants.grayscale:
        input_channels = 1

    # We use the random input for testing purposes
    # We square our data hence the shape's width is equal to its height the longer side
    random_input = torch.randn(
        1,
        input_channels,
        constants.scaled_height,
        constants.scaled_height)

    cnn = models.CNN(input_channels)
    criterion = nn.MSELoss()
    if cuda:
        criterion.cuda()
        cnn.cuda()
        random_input = random_input.cuda()

    random_input = Variable(random_input, requires_grad=False)

    # cnn.summary(random_input)

    h = cnn(random_input)
    logger.add_graph(cnn, h)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(cnn.parameters(), lr=constants.learning_rate)

    best_val_error = None
    # Train the Model

    assert len(train_loader) != 1, "The train loader length is 1"

    for epoch in range(constants.num_epochs):
        train_error = train(train_loader, cnn, optimizer, criterion, epoch)

        val_error = validate(val_loader, cnn, criterion, epoch)

        scalars = {"test": val_error, "train": train_error}
        logger.add_scalars('Loss', scalars, epoch)

        if (epoch + 1) % constants.save_interval == 0:
            save_checkpoint(cnn)
            if val_error < best_val_error or best_val_error is None:
                best_val_error = val_error
                save_checkpoint(cnn, 'result/best_checkpoint.pth')

    logger.close()

    # Save the Trained Model
    torch.save(cnn.state_dict(), 'result/posture.pkl')
    torch.save(cnn, 'result/posture.pth')


def save_checkpoint(model, filename='result/cnn_checkpoint.pth'):
    print "saving checkpoint"
    torch.save(model, filename)


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
            print('[VALID] - EPOCH %d/ %d - BATCH LOSS: %.8f/ %.8f(avg) '
                  % (epoch + 1, constants.num_epochs, losses.val, losses.avg))

    if epoch % constants.display_freq == 0:
        preview = P.load_preview(
            images,
            outputs,
            labels_std,
            labels_mean,
            images_std,
            images_mean)
        logger.add_image('Val/Output', preview, i + 1)

    return losses.avg


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

    if epoch % constants.display_freq == 0:
        preview = P.load_preview(
            images,
            outputs,
            labels_std,
            labels_mean,
            images_std,
            images_mean)
        logger.add_image('Train/Output', preview, i + 1)
        target = P.load_preview(
            images,
            labels,
            labels_std,
            labels_mean,
            images_std,
            images_mean)
        logger.add_image('Train/Target', target, i + 1)

    return losses.avg


if __name__ == '__main__':
    main()
