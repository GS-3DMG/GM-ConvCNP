import random

import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal

from convcnp import utils
from convcnp.modules.resblock import Conv3dResBlock
from convcnp.utils import channel_last_3d


def random_sample_3d(batch_size, channel, z_size, y_size, x_size, p):
    conditioning_data = torch.zeros(batch_size, channel, z_size, y_size, x_size)
    section = np.zeros([y_size, x_size])

    return conditioning_data


class ConvCNP3d(nn.Module):
    def __init__(self, channel=1):
        super().__init__()

        self.conv_theta = nn.Conv3d(channel, 128, 9, 1, 4)
        # print(self.conv_theta)
        self.cnn = nn.Sequential(
            nn.Conv3d(128 + 128, 128, 1, 1, 0),
            Conv3dResBlock(128, 128),
            Conv3dResBlock(128, 128),
            Conv3dResBlock(128, 128),
            Conv3dResBlock(128, 128),
            nn.Conv3d(128, 2 * channel, 1, 1, 0)
        )

        self.pos = nn.Softplus()

        self.channel = channel

        self.mr = [0.5, 0.7, 0.9]

    def forward(self, I):
        # print(I.shape)
        n_total = I.size(2) * I.size(3) * I.size(4)
        # print(n_total)
        num_context = int(torch.empty(1).uniform_(n_total / 1000, n_total / 100).item())
        # print(num_context)
        M_c = I.new_empty(I.size(0), 1, I.size(2), I.size(3), I.size(4)).bernoulli_(p=num_context / n_total).repeat(1, self.channel, 1, 1, 1)
        # print(M_c.shape)
        signal = I * M_c
        density = M_c

        # self.conv_theta.abs_constraint()
        density_prime = self.conv_theta(density)
        signal_prime = self.conv_theta(signal)
        # signal_prime = signal_prime.div(density_prime + 1e-8)
        # # self.conv_theta.abs_unconstraint()
        h = torch.cat([signal_prime, density_prime], 1)

        f = self.cnn(h)
        mean, std = f.split(self.channel, 1)
        std = self.pos(std)

        # print(mean.shape)
        # print(std.shape)

        mean, std = channel_last_3d(mean), channel_last_3d(std)
        # utils.write_model_2_sgems_file(mean, "mean.sgems")
        # utils.write_model_2_vtk_file(mean, "mean.vtk")
        # utils.write_model_2_sgems_file(std, "std.sgems")
        # utils.write_model_2_vtk_file(std, "std.vtk")
        mvn = MultivariateNormal(mean, scale_tril=std.diag_embed())
        # print(mvn)
        # print(mvn.mean)
        # print(mvn.batch_shape)
        # print(mvn.event_shape)

        return mvn

    def complete(self, I, M_c=None, missing_rate=None):
        if M_c is None:
            if missing_rate is None:
                missing_rate = random.choice(self.mr)
            M_c = random_sample_3d(I.size(0), self.channel, I.size(2), I.size(3), I.size(4), p=1 - missing_rate)
            M_c = I.new_empty(I.size(0), 1, I.size(2), I.size(3), I.size(4)).bernoulli_(p=1 - missing_rate).repeat(1, self.channel, 1, 1, 1)
        # print(M_c)
        # random.seed(0)
        # cd_xoy = random.sample(range(1, 4096), 200)
        # for idx in cd_xoy:
        #     x = int(idx / 64)
        #     y = idx-(64*x)
        #     for z in range(I.size(4)):
        #         for b in range(I.size(0)):
        #             M_c[b][0][x][y][z] = 1

        signal = I * M_c
        density = M_c

        density_prime = self.conv_theta(density)
        signal_prime = self.conv_theta(signal)
        h = torch.cat([signal_prime, density_prime], 1)

        f = self.cnn(h)
        mean, std = f.split(self.channel, 1)
        std = self.pos(std)

        mean2, std2 = channel_last_3d(mean), channel_last_3d(std)
        # print(mean2.shape)
        # mean3 = torch.squeeze(mean2)
        # std3 = torch.squeeze(std2)
        # print(mean3.shape)
        # utils.write_model_2_sgems_file(utils.cast_numpy(mean3), "mean.sgems")
        # utils.write_model_2_vtk_file(utils.cast_numpy(mean3), "mean.vtk")
        # utils.write_model_2_sgems_file(utils.cast_numpy(std3), "std.sgems")
        # utils.write_model_2_vtk_file(utils.cast_numpy(std3), "std.vtk")
        mvn = MultivariateNormal(mean2, scale_tril=std2.diag_embed())

        return M_c, mean, mvn

