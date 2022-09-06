import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as tf
from torchvision.datasets import MNIST, CIFAR10

from fastprogress import master_bar, progress_bar

from convcnp.dataset import ConHydro2D, CateHydro2D
from convcnp.models.convcnp2d import ConvCNP2d, channel_last
from convcnp.visualize import plot_all_2d, convert_tfboard


def train(model, dataloader, optimizer):
    model.train()
    avg_loss = 0

    for index, (I, _) in enumerate(progress_bar(dataloader, parent=args.mb)):
        I = I.to(args.device)
        # print(I.shape)
        # i = channel_last(I)
        # print(i.shape)
        optimizer.zero_grad()

        pred_dist = model(I)

        loss = - pred_dist.log_prob(channel_last((I))).sum(-1).mean()

        loss.backward()
        optimizer.step()

        avg_loss -= loss.item() * I.size(0)
        print(avg_loss)
        # if index % 5 == 0:
        #     print(5)
        if index % 10 == 0:
            args.mb.child.comment = 'loss={:.3f}'.format(loss.item())
            # print(index)

    return avg_loss / len(dataloader.dataset)


def validate(model, dataloader):
    model.eval()

    I, _ = iter(dataloader).next()
    I = I.to(args.device)

    with torch.no_grad():
        Mc, f, dist = model.complete(I)

    likelihood = dist.log_prob(channel_last(I)).sum(-1).mean()
    rmse = (I - f).pow(2).mean()
    # print(I)
    # print(Mc)
    # print(f)
    image = plot_all_2d(I, Mc, f)
    # image.imshow()
    image = convert_tfboard(image)
    return likelihood, rmse, image


def main():
    if args.dataset == 'conhydro2d':
        trainset = ConHydro2D(data_path="/home/user/data/Con_Hydro_2D/images",
                              txt_path="/home/user/data/Con_Hydro_2D/dataset.txt", train=True)
        testset = ConHydro2D(data_path="/home/user/data/Con_Hydro_2D/images",
                             txt_path="/home/user/data/Con_Hydro_2D/dataset.txt", train=False)
        cnp = ConvCNP2d(channel=1)
    elif args.dataset == 'catehydro2d':
        trainset = CateHydro2D(data_path="/home/user/data/Cate_Hydro_2D3/images",
                              txt_path="/home/user/data/Cate_Hydro_2D3/dataset.txt", train=True)
        testset = CateHydro2D(data_path="/home/user/data/Cate_Hydro_2D3/images",
                             txt_path="/home/user/data/Cate_Hydro_2D3/dataset.txt", train=False)
        cnp = ConvCNP2d(channel=1)
    elif args.dataset == 'conhydro2dlarge':
        trainset = ConHydro2D(data_path="/home/user/data/Con_Hydro_2D_L/images",
                              txt_path="/home/user/data/Con_Hydro_2D_L/dataset.txt", train=True)
        testset = ConHydro2D(data_path="/home/user/data/Con_Hydro_2D_L/images",
                             txt_path="/home/user/data/Con_Hydro_2D_L/dataset.txt", train=False)
        cnp = ConvCNP2d(channel=1)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    print("dataset load finish")

    cnp = cnp.to(args.device)

    optimizer = optim.Adam(cnp.parameters(), lr=args.learning_rate)

    args.mb = master_bar(range(1, args.epochs + 1))

    for epoch in args.mb:
        avg_train_loss = train(cnp, trainloader, optimizer)
        valid_ll, rmse, image = validate(cnp, testloader)

        args.writer.add_scalar('train/likelihood', avg_train_loss, epoch)
        args.writer.add_scalar('validate/likelihood', valid_ll, epoch)
        args.writer.add_scalar('validate/rmse', rmse, epoch)
        args.writer.add_image('validate/image', image, epoch)
        print("avg train loss: " + str(avg_train_loss))
        print("rmse: " + str(rmse))
        if epoch % 10 == 0:
            torch.save(cnp.state_dict(), filename + "_" + str(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-B', type=int, default=16)
    parser.add_argument('--learning-rate', '-LR', type=float, default=5e-4)
    parser.add_argument('--epochs', '-E', type=int, default=100)
    parser.add_argument('--dataset', '-D', type=str,
                        default='conhydro2dlarge',
                        choices=['catehydro2d', 'conhydro2d', 'conhydro2dlarge'])
    parser.add_argument('--logging', default=True, action='store_true')

    args = parser.parse_args()

    filename = 'train/convcnp2d_{}.pth.gz'.format(args.dataset)

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        # args.device = torch.device('cpu')
    else:
        args.device = torch.device('cpu')

    args.writer = SummaryWriter(comment='test_comment', filename_suffix='test_suffix')
    main()
    args.writer.close()
