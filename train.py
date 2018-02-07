"""This module trains the pose estimation model."""
import faulthandler
faulthandler.enable()

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import models
import constants
import preview as P
from dataset import functional as F
from dataset import load_dataset, Normalize
from tensorboardX import SummaryWriter
from third_party import AverageMeter
from normalization import normalization_values
from config import args

cuda = torch.cuda.is_available()
logger = SummaryWriter()

if cuda:
    print "Using CUDA..."

input_height = args.input_height
# Calculate scale
scale = args.input_height/constants.default_width

# Backout input width
input_width = int(round(constants.default_width * scale))
input_height = int(round(constants.default_height * scale))

csv_file = args.csv_dir+'train_data.csv'

images_mean, images_std, labels_mean, labels_std = 0.0, 1.0, 0.0, 1.0

print scale
if args.normalize:
    images_mean, images_std, labels_mean, labels_std = normalization_values(root_dir=args.root_dir, csv_file=csv_file, scale=scale, cache=args.cached)

# print images_mean, images_std, labels_mean, labels_std
np.set_printoptions(suppress=True)

def main():
    shuffle = True

    normalization = Normalize(images_mean, images_std, labels_mean, labels_std)
    dataset = load_dataset(normalization, root_dir=args.root_dir, csv_dir=args.csv_dir, scale=scale, random=True)
    train_dataset = dataset["train"]
    val_dataset = dataset["valid"]

    dataset_sample = train_dataset[0]
    label_sample = dataset_sample[1]
    out_features = torch.numel(label_sample)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.train_batch_size,
                                               shuffle=shuffle,
                                               num_workers=2,
                                               pin_memory=cuda)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.val_batch_size,
                                             shuffle=shuffle,
                                             num_workers=2,
                                             pin_memory=cuda)

    input_channels = 3

    # We use the random input for testing purposes
    # We square our data hence the shape's width is equal to its height the longer side
    random_input = torch.randn(2, input_channels, input_width, input_height)

    cnn = models.Posture(input_channels, input_width, input_height, out_features)
    criterion = nn.MSELoss()
    if cuda:
        criterion.cuda()
        cnn.cuda()
        random_input = random_input.cuda()

    random_input = Variable(random_input, requires_grad=False)

    # cnn.summary(random_input)

    h = cnn(random_input)
    # logger.add_graph(cnn, h)

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
        logger.add_scalars(args.model_name + '/Loss', scalars, epoch)

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
    filepath = "result/" + args.model_name + "_" + filename + "_" + str(args.input_height) + ".pkl"
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
        logger.add_scalar(args.model_name + '/Val/Loss', loss.data[0], niter)

        if i % constants.print_freq == 0:
            print('[VALID] - EPOCH %d/ %d - BATCH LOSS: %.8f/ %.8f(avg) '
                  % (epoch + 1, args.num_epochs, losses.val, losses.avg))

    if args.display and epoch % args.display_freq == 0:
        output = outputs.data[0].cpu().numpy()
        label = labels.data[0].cpu().numpy()
        
        output = output * labels_std + labels_mean
        label = label * labels_std + labels_mean
        print "FAKE: ", output.astype(int).tolist()
        print "REAL: ", label.astype(int).tolist()
                
        output = output.reshape(-1, 2)
        images = F.denormalize_image_tensor(torch.from_numpy(images_mean), torch.from_numpy(images_std), images, cuda)
        image = images.data[0].cpu().numpy()
        image = np.rollaxis(image, 0, 3)
        
        output *= scale
        preview = P.load_preview(image, output, 1)
        logger.add_image(args.model_name + '/Val/Output', preview, i + 1)

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
        logger.add_scalar(args.model_name + '/Train/Loss', loss.data[0], niter)

        if i % constants.print_freq == 0:
            print('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.8f/ %.8f(avg) '
                  % (epoch + 1, args.num_epochs, losses.val, losses.avg))

    if args.display and epoch % args.display_freq == 0:             
        images = F.denormalize_image_tensor(torch.from_numpy(images_mean), torch.from_numpy(images_std), images, cuda)
        image = images.data[0].cpu().numpy()
        image = np.rollaxis(image, 0, 3)

        outputs = outputs.cpu() * Variable(torch.from_numpy(labels_std)) + Variable(torch.from_numpy(labels_mean))
        output = outputs.data[0].numpy()
        output = output.reshape(-1, 2)
                
        output *= scale
        preview = P.load_preview(image, output,  1)
        logger.add_image(args.model_name + '/Train/Output', preview, i + 1)

        labels = labels.cpu() * Variable(torch.from_numpy(labels_std)) + Variable(torch.from_numpy(labels_mean))
        label = labels.data[0].numpy()
        label = label.reshape(-1, 2)
                
        label *= scale        
        target = P.load_preview(image, label, 1)
        logger.add_image(args.model_name + '/Train/Target', target, i + 1)

    return losses.avg


if __name__ == '__main__':
    main()
