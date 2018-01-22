"""This module trains the pose estimation model."""

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import argparse

import models
import constants
import preview as P
from dataset import load_dataset, Normalize
from tensorboardX import SummaryWriter
from third_party import AverageMeter
from normalization import normalization_values

parser = argparse.ArgumentParser(description='Train the pose estimation model.')
parser.add_argument('--num_epochs', type=int, default=3000)
parser.add_argument('--csv_dir', type=str, default='B/')
parser.add_argument('--root_dir', type=str, default='B/')
parser.add_argument('--input_height', type=int, default=64)
parser.add_argument('--model_name', type=str, default='posture')

args = parser.parse_args()

cuda = torch.cuda.is_available()
logger = SummaryWriter()

scale = args.input_height/760.0

images_mean, images_std, labels_mean, labels_std = normalization_values(grayscale=constants.grayscale, root_dir=args.root_dir, csv_file=args.csv_dir+'train_data.csv', scale=scale)

def main():
    shuffle = True

    normalization = Normalize(images_mean, images_std, labels_mean, labels_std)
    dataset = load_dataset(normalization, grayscale=constants.grayscale, root_dir=args.root_dir, csv_dir=args.csv_dir, scale=scale)
    train_dataset = dataset["train"]
    val_dataset = dataset["valid"]

    dataset_sample = train_dataset[0]
    label_sample = dataset_sample[1]
    out_features = torch.numel(label_sample)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=constants.train_batch_size,
                                               shuffle=shuffle,
                                               pin_memory=cuda)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=constants.val_batch_size,
                                             shuffle=shuffle,
                                             pin_memory=cuda)

    input_channels = 3
    if constants.grayscale:
        input_channels = 1

    # We use the random input for testing purposes
    # We square our data hence the shape's width is equal to its height the longer side
    random_input = torch.randn(1, input_channels, int(constants.default_height * scale), int(constants.default_height * scale))

    cnn = models.Posture(input_channels, args.input_height, out_features)
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
    print "The train loader length is ", len(train_loader)

    for epoch in range(args.num_epochs):
        train_error = train(train_loader, cnn, optimizer, criterion, epoch)

        val_error = validate(val_loader, cnn, criterion, epoch)

        scalars = {"test": val_error, "train": train_error}
        logger.add_scalars('Posture/Loss', scalars, epoch)

        if (epoch + 1) % constants.save_interval == 0:
            save_model(cnn, 'checkpoint')
            if val_error < best_val_error or best_val_error is None:
                best_val_error = val_error
                save_model(cnn, 'best_checkpoint')
                
    logger.close()

    # Save the Trained Model
    save_model(cnn, "final")
    torch.save(cnn, 'result/final.pth')

def save_model(model, filename):
    filepath = "result/" + args.model_name + "_" + filename + ".pkl"
    print "saving " + filepath
    torch.save(model.state_dict(), filepath)

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
        logger.add_scalar('Posture/Val/Loss', loss.data[0], niter)

        if i % constants.print_freq == 0:
            print('[VALID] - EPOCH %d/ %d - BATCH LOSS: %.8f/ %.8f(avg) '
                  % (epoch + 1, args.num_epochs, losses.val, losses.avg))

    if epoch % constants.display_freq == 0:
        output = outputs.data[0].cpu().numpy()
        output = output.reshape(-1, 2)
        preview = P.load_preview(images, output, labels_std, labels_mean, images_std, images_mean, 1)
        logger.add_image('Posture/Val/Output', preview, i + 1)

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
        logger.add_scalar('Posture/Train/Loss', loss.data[0], niter)

        if i % constants.print_freq == 0:
            print('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.8f/ %.8f(avg) '
                  % (epoch + 1, args.num_epochs, losses.val, losses.avg))

    if epoch % constants.display_freq == 0:
        output = outputs.data[0].cpu().numpy()
        output = output.reshape(-1, 2)
        preview = P.load_preview(images, output, labels_std, labels_mean, images_std, images_mean,  1)
        logger.add_image('Posture/Train/Output', preview, i + 1)

        label = labels.data[0].cpu().numpy()
        label = label.reshape(-1, 2)
        target = P.load_preview(images, label, labels_std, labels_mean, images_std, images_mean, 1)
        logger.add_image('Posture/Train/Target', target, i + 1)

    return losses.avg


if __name__ == '__main__':
    main()
