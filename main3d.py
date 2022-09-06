import argparse

import torch
from fastprogress import progress_bar, master_bar
from torch import optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

import convcnp.utils
from convcnp.dataset.dataset import CateHydro3D
from convcnp.models.convcnp3d import ConvCNP3d
from convcnp.utils import channel_last_3d


def train(model, dataloader, optimizer):
    model.train()
    avg_loss = 0

    for index, (I, _) in enumerate(progress_bar(dataloader, parent=args.mb)):
        I = I.to(args.device)
        optimizer.zero_grad()

        pred_dist = model(I)

        loss = - pred_dist.log_prob(channel_last_3d((I))).sum(-1).mean()
        # loss = - pred_dist.cdf(channel_last_3d(I))
        # print(loss)
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
    # print(I)
    with torch.no_grad():
        Mc, f, dist = model.complete(I)

    likelihood = dist.log_prob(channel_last_3d(I)).sum(-1).mean()
    # likelihood = dist.cdf(channel_last_3d(I))
    rmse = (I - f).pow(2).mean()
    # root = '/home/user/data/Cate_Hydro_3D/output/'
    # convcnp.utils.write_model_2_sgems_file(cast2sgems(I), root + "test_reference.sgems")
    # convcnp.utils.write_model_2_sgems_file(cast2sgems(f), root + "test_output.sgems")
    # print(I)
    # print(Mc)
    # print(f)
    return likelihood, rmse


def cast2sgems(x):
    print(x.shape)
    x = channel_last_3d(x)
    x = torch.squeeze(x)
    x = torch.squeeze(x)
    x = convcnp.utils.cast_numpy(x)
    print(x.shape)
    return x


def main():
    if args.dataset == 'catehydro3d':
        trainset = CateHydro3D(data_path="/home/user/data/Cate_Hydro_3D/images", txt_path="/home/user/data/Cate_Hydro_3D/dataset.txt", train=True)
        testset = CateHydro3D(data_path="/home/user/data/Cate_Hydro_3D/images", txt_path="/home/user/data/Cate_Hydro_3D/dataset.txt", train=False)
        cnp = ConvCNP3d(channel=1)
    elif args.dataset == 'fold3d':
        trainset = CateHydro3D(data_path="/home/user/data/Fold_64/images", txt_path="/home/user/data/Fold_64/dataset.txt", train=True)
        testset = CateHydro3D(data_path="/home/user/data/Fold_64/images", txt_path="/home/user/data/Fold_64/dataset.txt", train=False)
        cnp = ConvCNP3d(channel=1)
    elif args.dataset == 'confold3d':
        trainset = CateHydro3D(data_path="/home/user/data/Fold_Con_64/images", txt_path="/home/user/data/Fold_Con_64/dataset.txt", train=True)
        testset = CateHydro3D(data_path="/home/user/data/Fold_Con_64/images", txt_path="/home/user/data/Fold_Con_64/dataset.txt", train=False)
        cnp = ConvCNP3d(channel=1)
    elif args.dataset == 'Svelocity':
        trainset = CateHydro3D(data_path="/home/user/data/Svelocity/images", txt_path="/home/user/data/Svelocity/dataset.txt", train=True)
        testset = CateHydro3D(data_path="/home/user/data/Svelocity/images", txt_path="/home/user/data/Svelocity/dataset.txt", train=False)
        cnp = ConvCNP3d(channel=1)
    elif args.dataset == 'facies':
        trainset = CateHydro3D(data_path="/home/user/data/facies/images", txt_path="/home/user/data/facies/dataset.txt", train=True)
        testset = CateHydro3D(data_path="/home/user/data/facies/images", txt_path="/home/user/data/facies/dataset.txt", train=False)
        cnp = ConvCNP3d(channel=1)
    elif args.dataset == 'Pvelocity':
        trainset = CateHydro3D(data_path="/home/user/data/Pvelocity/images", txt_path="/home/user/data/Pvelocity/dataset.txt", train=True)
        testset = CateHydro3D(data_path="/home/user/data/Pvelocity/images", txt_path="/home/user/data/Pvelocity/dataset.txt", train=False)
        cnp = ConvCNP3d(channel=1)


    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print("dataset load finish")

    cnp = cnp.to(args.device)

    optimizer = optim.Adam(cnp.parameters(), lr=args.learning_rate)

    args.mb = master_bar(range(1, args.epochs + 1))

    for epoch in args.mb:
        avg_train_loss = train(cnp, trainloader, optimizer)
        valid_ll, rmse = validate(cnp, testloader)

        # args.writer.add_scalar('train/likelihood', avg_train_loss, epoch)
        # args.writer.add_scalar('validate/likelihood', valid_ll, epoch)
        # args.writer.add_scalar('validate/rmse', rmse, epoch)
        # args.writer.add_image('validate/image', image, epoch)
        print("avg train loss: " + str(avg_train_loss))
        print("rmse: " + str(rmse))
        torch.save(cnp.state_dict(), filename + "_" + str(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-B', type=int, default=1)
    parser.add_argument('--learning-rate', '-LR', type=float, default=5e-4)
    parser.add_argument('--epochs', '-E', type=int, default=200)
    parser.add_argument('--dataset', '-D', type=str, default='facies', choices=['catehydro3d', 'fold3d', 'confold3d', 'Svelocity', 'Pvelocity', 'facies'])
    parser.add_argument('--logging', default=True, action='store_true')

    args = parser.parse_args()

    filename = 'train/convcnp3d_{}.pth.gz'.format(args.dataset)

    # if torch.cuda.is_available():
    args.device = torch.device('cuda')
    #     args.device = torch.cuda.set_device(0)
    # else:
    #     args.device = torch.device('cpu')
    # torch.cuda.set_device(0)
    # args.writer = SummaryWriter(comment='test_comment', filename_suffix='test_suffix')
    main()
    # args.writer.close()
