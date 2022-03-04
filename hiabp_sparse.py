import os
import time

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
from torchvision.utils import save_image
from model.utils.data_io import DataSet, saveSampleResults
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(opt.dim * opt.dim * opt.channel, 2000)
        self.fc2 = nn.Linear(2000, 2000)

        self.alpha1 = nn.Linear(2000, opt.z_size)
        self.sigmoid = nn.Sigmoid()

        self.mu1 = nn.Linear(2000, opt.z_size)
        self.logvar1 = nn.Linear(2000, opt.z_size)

        self.mu2 = nn.Linear(2000, opt.z_size)
        self.logvar2 = nn.Linear(2000, opt.z_size)

    def reparameterize(self, alpha1, mu1, logvar1, mu2, logvar2):
        std1 = torch.exp(0.5*logvar1)
        std2 = torch.exp(0.5*logvar2)
        eps1 = torch.randn_like(std1)
        eps2 = torch.randn_like(std2)

        return alpha1 * (std1.mul(eps1).add_(mu1)) + (1 - alpha1) * (std2.mul(eps2).add_(mu2))

    def forward(self, x):
        out = self.relu(self.fc1(x.view(-1, self.opt.dim * self.opt.dim * self.opt.channel)))
        out = self.relu(self.fc2(out))
        mu1 = self.mu1(out)
        alpha1 = self.sigmoid(self.alpha1(out))
        lgvar1 = self.logvar1(out)
        mu2 = self.mu2(out)
        lgvar2 = self.logvar2(out)
        z = self.reparameterize(alpha1, mu1, lgvar1, mu2, lgvar2)

        return z, mu1, alpha1, lgvar1, mu2, lgvar2

class Encoder_Conv(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(opt.channel, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64*2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False)

        self.alpha1 = nn.Linear(2000, opt.z_size)
        self.sigmoid = nn.Sigmoid()

        self.mu1 = nn.Linear(2000, opt.z_size)
        self.logvar1 = nn.Linear(2000, opt.z_size)

        self.mu2 = nn.Linear(2000, opt.z_size)
        self.logvar2 = nn.Linear(2000, opt.z_size)

    def reparameterize(self, alpha1, mu1, logvar1, mu2, logvar2):
        std1 = torch.exp(0.5*logvar1)
        std2 = torch.exp(0.5*logvar2)
        eps1 = torch.randn_like(std1)
        eps2 = torch.randn_like(std2)

        return alpha1 * (std1.mul(eps1).add_(mu1)) + (1 - alpha1) * (std2.mul(eps2).add_(mu2))

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        print(out.shape)
        mu1 = self.mu1(out)
        alpha1 = self.sigmoid(self.alpha1(out))
        lgvar1 = self.logvar1(out)
        mu2 = self.mu2(out)
        lgvar2 = self.logvar2(out)
        z = self.reparameterize(alpha1, mu1, lgvar1, mu2, lgvar2)

        return z, mu1, alpha1, lgvar1, mu2, lgvar2

class Encoder_SVHN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(opt.channel, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64*2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False)

        self.alpha1 = nn.Linear(2000, opt.z_size)
        self.sigmoid = nn.Sigmoid()

        self.mu1 = nn.Linear(2000, opt.z_size)
        self.logvar1 = nn.Linear(2000, opt.z_size)

        self.mu2 = nn.Linear(2000, opt.z_size)
        self.logvar2 = nn.Linear(2000, opt.z_size)

    def reparameterize(self, alpha1, mu1, logvar1, mu2, logvar2):
        std1 = torch.exp(0.5*logvar1)
        std2 = torch.exp(0.5*logvar2)
        eps1 = torch.randn_like(std1)
        eps2 = torch.randn_like(std2)

        return alpha1 * (std1.mul(eps1).add_(mu1)) + (1 - alpha1) * (std2.mul(eps2).add_(mu2))

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        print(out.shape)
        mu1 = self.mu1(out)
        alpha1 = self.sigmoid(self.alpha1(out))
        lgvar1 = self.logvar1(out)
        mu2 = self.mu2(out)
        lgvar2 = self.logvar2(out)
        z = self.reparameterize(alpha1, mu1, lgvar1, mu2, lgvar2)

        return z, mu1, alpha1, lgvar1, mu2, lgvar2

class Encoder_Mnist(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(opt.dim * opt.dim * opt.channel, 400)

        self.alpha1 = nn.Linear(400, opt.z_size)
        self.sigmoid = nn.Sigmoid()

        self.mu1 = nn.Linear(400, opt.z_size)
        self.logvar1 = nn.Linear(400, opt.z_size)

        self.mu2 = nn.Linear(400, opt.z_size)
        self.logvar2 = nn.Linear(400, opt.z_size)

    def reparameterize(self, alpha1, mu1, logvar1, mu2, logvar2):
        std1 = torch.exp(0.5*logvar1)
        std2 = torch.exp(0.5*logvar2)
        eps1 = torch.randn_like(std1)
        eps2 = torch.randn_like(std2)

        if self.opt.fix:
            z = alpha1 * eps1  + (1 - alpha1) * (std1.mul(eps2).add_(mu1))
        else:
            z = alpha1 * (std1.mul(eps1).add_(mu1)) + (1 - alpha1) * (std2.mul(eps2).add_(mu2))

        return z

    def forward(self, x):
        out = self.relu(self.fc1(x.view(-1, self.opt.dim * self.opt.dim * self.opt.channel)))
        mu1 = self.mu1(out)
        alpha1 = self.sigmoid(self.alpha1(out))
        lgvar1 = self.logvar1(out)
        mu2 = self.mu2(out)
        lgvar2 = self.logvar2(out)
        z = self.reparameterize(alpha1, mu1, lgvar1, mu2, lgvar2)



        return z, mu1, alpha1, lgvar1, mu2, lgvar2


class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fc1 = nn.Linear(opt.z_size, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, opt.dim * opt.dim * opt.channel)
        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

    def forward(self, z):
        out = self.relu(self.fc1(z))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))
        return out.view(-1, self.opt.channel,self.opt.dim, self.opt.dim)

class Generator_SVHN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fc1 = nn.Linear(opt.z_size, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, opt.dim * opt.dim * opt.channel)
        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

    def forward(self, z):
        out = self.relu(self.fc1(z))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))
        return out.view(-1, self.opt.channel,self.opt.dim, self.opt.dim)

class Generator_Conv(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.conv1 = nn.ConvTranspose2d(opt.z_size, 64*8, 4, 1, 0, bias=True)
        self.conv2 = nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False)
        self.conv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64*8)
        self.bn2 = nn.BatchNorm2d(64*4)
        self.bn3 = nn.BatchNorm2d(64*2)
        self.bn4 = nn.BatchNorm2d(64*1)
        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

    def forward(self, z):
        '''
        out = self.relu(self.bn1(self.conv1(z.view(-1, self.opt.z_size, 1, 1))))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        '''
        out = self.relu(self.conv1(z.view(-1, self.opt.z_size, 1, 1)))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))

        out = self.tanh(self.conv5(out))
        return out.view(-1, self.opt.channel,self.opt.dim, self.opt.dim)


class Generator_Mnist(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fc1 = nn.Linear(opt.z_size, 400)
        self.fc2 = nn.Linear(400, opt.dim * opt.dim * opt.channel)
        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

    def forward(self, z):
        out = self.relu(self.fc1(z))
        out = self.tanh(self.fc2(out))
        return out.view(-1, self.opt.channel, self.opt.dim, self.opt.dim)


class Generator_Texture(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fc1 = nn.Linear(100, 56*56)
        self.conv1 = nn.ConvTranspose2d(15, 32, 4, 4, 0, bias=True)
        self.conv2 = nn.ConvTranspose2d(32, 3, 4, 4, 0, bias=False)
        self.conv3 = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(4, 3, 4, 2, 1, bias=False)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, z):
        out = self.relu(self.conv1(z.reshape(-1, 15, 4, 4)))
        out = self.sig(self.conv2(out))
    #    out = self.sig(self.conv3(out))
     #   out = self.sig(self.conv4(out))
        return out.view(-1, 3, 64, 64)

class SS_HiABP(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.img_size = opts.img_size
        self.num_chain = opts.nRow * opts.nCol
        self.opts = opts
        if opts.with_noise == True:
            print('Do Langevin with noise')
        else:
            print('Do Langevin without noise')

        self.alpha = 0.01
        self.slab_variance = 0.1


    def latent_gradient(self, z):
        if self.alpha == 1:
            return -z
        ratio1 = (1 - self.alpha) / (self.alpha * self.slab_variance)
        ratio2 = (self.slab_variance * self.slab_variance - 1) / (self.slab_variance * self.slab_variance)
        gradient = -z + ratio1 * ratio2 * z / (torch.exp(-ratio2 * torch.pow(z, 2) / 2) + ratio1)
        del ratio1, ratio2
        return gradient

    def langevin_dynamics_generator(self, z, obs, training=True):
        obs = obs.detach()
        criterian = nn.MSELoss(reduction='sum')
        for i in range(self.opts.langevin_step_num_gen):
            noise = Variable(torch.randn(z.shape[0], 16*16).cuda())
            z = Variable(z, requires_grad=True)
            gen_res = self.generator(z)
            gen_loss = 1.0 / (2.0 * self.opts.sigma_gen * self.opts.sigma_gen) * criterian(gen_res, obs)
            gen_loss.backward()
            grad = z.grad
            sparse_grad = self.latent_gradient(z)
            if self.opts.dense == True:
                z = z - 0.5 * self.opts.langevin_step_size_gen * self.opts.langevin_step_size_gen * (z + grad)
            else:
                z = z - 0.5 * self.opts.langevin_step_size_gen * self.opts.langevin_step_size_gen * (-sparse_grad + grad)
            if self.opts.with_noise == True and training == True:
                z += self.opts.langevin_step_size_gen * noise

        return z

    def init_weight(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight)


    def train(self):
        if self.opts.ckpt_enc != None and self.opts.ckpt_enc != 'None':
            self.encoder = torch.load(self.opts.ckpt_enc)
            print('Loading Encoder from ' + self.opts.ckpt_enc + '...')
        else:
            if self.opts.set == 'celeba' or self.opts.set == 'lsun':
                self.opts.dim = 64
                self.opts.channel = 3
                if self.opts.conv == True:
                    self.encoder = Encoder_Conv(self.opts).cuda()
                else:
                    self.encoder = Encoder(self.opts).cuda()
                print('Loading Encoder without initialization...')
            elif self.opts.set == 'svhn':
                self.opts.dim = 32
                self.opts.channel = 3
                self.encoder = Encoder_SVHN(self.opts).cuda()
                print('Loading Encoder without initialization...')
            elif self.opts.set == 'mnist' or self.opts.set == 'fashion':
                self.opts.dim = 28
                self.opts.channel = 1
                self.encoder = Encoder_Mnist(self.opts).cuda()
                print('Loading Encoder without initialization...')
            elif self.opts.set == 'beehive':
                self.opts.dim = 64
                self.opts.channel = 3
                self.encoder = Encoder_Mnist(self.opts).cuda()
            else:
                raise NotImplementedError('The set should be either scene, lsun, or cifar')

        if self.opts.ckpt_gen != None and self.opts.ckpt_gen != 'None':
            self.generator = torch.load(self.opts.ckpt_gen)
            print('Loading Generator from ' + self.opts.ckpt_gen + '...')
        else:
            if self.opts.set == 'celeba' or self.opts.set == 'lsun':
                if self.opts.conv == True:
                    self.generator = Generator_Conv(self.opts).cuda()
                    self.generator.apply(self.init_weight)
                else:
                    self.generator = Generator(self.opts).cuda()
                print('Loading Generator without initialization...')
            elif self.opts.set == 'mnist' or self.opts.set == 'fashion':
                self.generator = Generator_Mnist(self.opts).cuda()
                print('Loading Generator without initialization')
            elif self.opts.set == 'svhn':
                self.generator = Generator_SVHN(self.opts).cuda()
            elif self.opts.set == 'beehive':
                self.opts.dim = 64
                self.opts.channel = 3
                self.generator = Generator_Texture(self.opts).cuda()
            else:
                raise NotImplementedError('The set should be either scene, lsun or cifar')

        batch_size = self.opts.batch_size
        if self.opts.set == 'scene' or self.opts.set == 'cifar':
            train_data = DataSet(os.path.join('./data/' + self.opts.set, self.opts.category), image_size=self.opts.img_size)
        elif self.opts.set == 'celeba':
            train_data = torch.utils.data.DataLoader(torchvision.datasets.CelebA(root='./data/' + self.opts.set + '_train',
                                                   split='train',
                                                   transform=transforms.Compose([transforms.CenterCrop(128),
                                                                                 transforms.Resize(self.img_size),
                                                                                 transforms.ToTensor()])),
                        batch_size=batch_size, shuffle=False)

        elif self.opts.set == 'svhn':
            train_data = torch.utils.data.DataLoader(
                torchvision.datasets.SVHN(root='./data/' + self.opts.set, split='train', download=True,
                                           transform=transforms.Compose([transforms.ToTensor()])),
                batch_size=batch_size, shuffle=False)

        elif self.opts.set == 'mnist':
            train_data = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(root='./data/' + self.opts.set, train=True, download=True,
                                           transform=transforms.Compose([transforms.ToTensor()])),
                batch_size=batch_size, shuffle=False)

        elif self.opts.set == 'beehive':
            train_data = torchvision.io.read_image('./data/texture/beehive.jpg').float()
            transform = torchvision.transforms.Resize((self.opts.dim, self.opts.dim))
            new = torch.zeros((8, 3, 112, 112))
            new[0] = train_data[:, 0:112, 0:112]
            new[0] -= new[0].min(1, keepdim=True)[0]
            new[0] /= new[0].max(1, keepdim=True)[0]
       #     new[0] = transform(new[0].reshape(1, 3, 112, 112))
            new[1] = train_data[:, 112:224, 0:112]
            new[1] -= new[1].min(1, keepdim=True)[0]
            new[1] /= new[1].max(1, keepdim=True)[0]
      #      new[1] = transform(new[1])
            new[2] = train_data[:, 112:224, 112:224]
            new[2] -= new[2].min(1, keepdim=True)[0]
            new[2] /= new[2].max(1, keepdim=True)[0]
       #     new[2] = transform(new[2])
            new[3] = train_data[:, 0:112, 112:224]
            new[3] -= new[3].min(1, keepdim=True)[0]
            new[3] /= new[3].max(1, keepdim=True)[0]
       #     new[3] = transform(new[3])
            new[4] = train_data[:, 56:112+56, 112:224]
            new[4] -= new[4].min(1, keepdim=True)[0]
            new[4] /= new[4].max(1, keepdim=True)[0]
       #     new[4] = transform(new[4])
            new[5] = train_data[:, 0:112, 56:112+56]
            new[5] -= new[5].min(1, keepdim=True)[0]
            new[5] /= new[5].max(1, keepdim=True)[0]
      #      new[5] = transform(new[5])
            new[6] = train_data[:, 56:112+56, 0:112]
            new[6] -= new[6].min(1, keepdim=True)[0]
            new[6] /= new[6].max(1, keepdim=True)[0]
      #      new[6] = transform(new[6])
            new[7] = train_data[:, 112:224, 56:112+56]
            new[7] -= new[7].min(1, keepdim=True)[0]
            new[7] /= new[7].max(1, keepdim=True)[0]
       #     new[7] = transform(new[7])
            new = transform(new)
          #  transform = torchvision.transforms.Resize((self.opts.dim, self.opts.dim))
      #      train_data = transform(train_data)
            train_data = new

        #    train_data -= train_data.min(1, keepdim=True)[0]
        #    train_data /= train_data.max(1, keepdim=True)[0]

        else:
            train_data = torch.utils.data.DataLoader(
                torchvision.datasets.FashionMNIST(root='./data/' + self.opts.set, train=True, download=True, transform=transforms.Compose(
                    [transforms.ToTensor()])),
                batch_size=batch_size, shuffle=False)


        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.opts.lr_des,
                                         betas=[self.opts.beta1_des, 0.999])
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.opts.lr_gen,
                                         betas=[self.opts.beta1_gen, 0.999])


        if not os.path.exists(self.opts.ckpt_dir):
            os.makedirs(self.opts.ckpt_dir)
        if not os.path.exists(self.opts.output_dir):
            os.makedirs(self.opts.output_dir)
        logfile = open(self.opts.ckpt_dir + '/log', 'w+')

        mse_loss = torch.nn.MSELoss(reduction='sum')

        '''
        if not os.path.exists(self.opts.test_dir):
            os.makedirs(self.opts.test_dir)

        plt.imshow(train_data.permute(1, 2, 0), interpolation='nearest')
        plt.savefig("%s/observed.png" % (self.opts.test_dir))
        plt.close('all')
        '''

        recon_all = []
        enc_all = []
        alpha_all = []
        mu1_all = []
        mu2_all = []
        lgvar1_all = []
        lgvar2_all = []

        if self.opts.abp:
            self.opts.graph = False
        e_loss = 0
   #     z = torch.randn((200000, self.opts.z_size)).cuda()
        for epoch in range(self.opts.num_epoch):
            start_time = time.time()
            gen_loss_epoch, enc_loss_epoch, recon_loss_epoch = [], [], []
            alpha1_epoch, mu1_epoch, mu2_epoch, lgvar1_epoch, lgvar2_epoch = [], [], [], [], []

            if epoch == 90:

                w1 = (self.generator.conv1.weight - self.generator.conv1.weight.min()) / (self.generator.conv1.weight.max() - self.generator.conv1.weight.min())
                w2 = (self.generator.conv2.weight - self.generator.conv2.weight.min()) / (
                            self.generator.conv2.weight.max() - self.generator.conv2.weight.min())
                w3 = (self.generator.conv3.weight - self.generator.conv3.weight.min()) / (
                            self.generator.conv3.weight.max() - self.generator.conv3.weight.min())

                save_image((w1.reshape(-1, 1, 4, 4)), "%s/weights1.png" % (self.opts.output_dir))
                save_image((w2.reshape(-1, 1, 4, 4)),
                           "%s/weights2.png" % (self.opts.output_dir))
                save_image((w3.reshape(-1, 1, 4, 4)),
                           "%s/weights3.png" % (self.opts.output_dir))

            if self.opts.set == 'beehive':
                obs_data = train_data.cuda().reshape(8, 3, 64, 64)

                if epoch <= 20:
                    self.alpha = 1
                else:
                    self.alpha -= 0.033

                if self.alpha < 1:
                    self.alpha = 0.01

                z = torch.zeros((obs_data.shape[0], 15*4*4)).cuda()
                z_tau = self.langevin_dynamics_generator(z, obs_data)
                gen_res = self.generator(z_tau)
                gen_loss = 1.0 / (2.0 * self.opts.sigma_gen * self.opts.sigma_gen) * mse_loss(gen_res,
                                                                                              obs_data.detach())
                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()

                # Compute reconstruction loss
                recon_loss = mse_loss(obs_data, gen_res)
                gen_loss_epoch.append(gen_loss.cpu().data)
                recon_loss_epoch.append(recon_loss.cpu().data)

                if not self.opts.abp:
                    enc_loss_epoch.append(encoder_loss.cpu().data)
                    alpha1_epoch.append(torch.mean(alpha1).cpu().data)
                    mu1_epoch.append(torch.mean(mu1).cpu().data)
                    mu2_epoch.append(torch.mean(mu2).cpu().data)
                    lgvar1_epoch.append(torch.mean(lgvar1).cpu().data)
                    lgvar2_epoch.append(torch.mean(lgvar2).cpu().data)
                else:
                    enc_loss_epoch = 0


            else:
                for index, (data, _) in enumerate(train_data):
                    obs_data = data
                    obs_data = Variable(torch.Tensor(obs_data).cuda())  # ,requires_grad=True
                    # z0
                    # for i in range(self.opts.encoder_steps):

                    if epoch <= 20:
                        self.alpha = 1
                    else:
                        self.alpha -= 0.033

                    if self.alpha < 0.001:
                        self.alpha = 0.001

                    if self.opts.abp:
                        if self.opts.dense == True:
                            z = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
                        else:
                            z = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
                    else:
                        z, mu1, alpha1, lgvar1, mu2, lgvar2 = self.encoder(obs_data)
                    # z tau

                    z_tau = self.langevin_dynamics_generator(z, obs_data)

                    '''
                    if not self.opts.abp:
                        z_tau_clone = z_tau.clone().detach()

                        if self.opts.fix:

                            encoder_loss = (1 / 2 * torch.pow((z_tau_clone), 2) - torch.log(
                                alpha1) \
                                            - torch.log(
                                        1 + (1 - alpha1) / (alpha1 * torch.exp(0.5 * lgvar2)) \
                                        * torch.exp(
                                            1 / 2 * torch.pow((z_tau_clone), 2) - 1 / (
                                                    2 * torch.exp(lgvar2)) * torch.pow((z_tau_clone - mu2), 2)))).sum()

                        else:
                            encoder_loss = (1 / (2 * torch.exp(lgvar1)) * torch.pow((z_tau_clone - mu1),
                                                                                    2) + 0.5 * lgvar1 - torch.log(alpha1) \
                                            - torch.log(
                                        1 + (1 - alpha1) * torch.exp(0.5 * lgvar1) / (alpha1 * torch.exp(0.5 * lgvar2)) \
                                        * torch.exp(1 / (2 * torch.exp(lgvar1)) * torch.pow((z_tau_clone - mu1), 2) - 1 / (
                                                    2 * torch.exp(lgvar2)) * torch.pow((z_tau_clone - mu2), 2)))

                                            ).sum()

                        encoder_optimizer.zero_grad()
                        encoder_loss.backward()
                        encoder_optimizer.step()
                    '''

                    gen_res = self.generator(z_tau)
                    gen_loss = 1.0 / (2.0 * self.opts.sigma_gen * self.opts.sigma_gen) * mse_loss(gen_res,
                                                                                                  obs_data.detach())
                    gen_optimizer.zero_grad()
                    gen_loss.backward()
                    gen_optimizer.step()

                    #     e_loss += encoder_loss

                    # Compute reconstruction loss
                    recon_loss = mse_loss(obs_data, gen_res)
                    gen_loss_epoch.append(gen_loss.cpu().data)
                    recon_loss_epoch.append(recon_loss.cpu().data)

                    if not self.opts.abp:
                        enc_loss_epoch.append(encoder_loss.cpu().data)
                        alpha1_epoch.append(torch.mean(alpha1).cpu().data)
                        mu1_epoch.append(torch.mean(mu1).cpu().data)
                        mu2_epoch.append(torch.mean(mu2).cpu().data)
                        lgvar1_epoch.append(torch.mean(lgvar1).cpu().data)
                        lgvar2_epoch.append(torch.mean(lgvar2).cpu().data)
                    else:
                        enc_loss_epoch = 0

            save_image((obs_data), "%s/observed.png" % (self.opts.output_dir))

            save_image((gen_res).cpu(), "%s/gen_%03d.png" % (self.opts.output_dir, epoch + 1))

            save_image((torch.cat((obs_data[0:8], gen_res[0:8]),0)), "%s/gen_vis_%03d.png" % (self.opts.output_dir, epoch + 1))

          #  print(e_loss / (index+1))

            end_time = time.time()
            print('Epoch #{:d}/{:d}, des_loss: {:.4f}, gen_loss: {:.4f}, recon_loss: {:.4f}, '
                  'time: {:.2f}s'.format(epoch + 1, self.opts.num_epoch, np.mean(enc_loss_epoch),
                                         np.mean(gen_loss_epoch), np.mean(recon_loss_epoch),
                                         end_time - start_time))

            # python 3

            print('Epoch #{:d}/{:d}, des_loss: {:.4f}, gen_loss: {:.4f}, recon_loss: {:.4f}, '
                  'time: {:.2f}s'.format(epoch, self.opts.num_epoch, np.mean(enc_loss_epoch), np.mean(gen_loss_epoch),
                                         np.mean(recon_loss_epoch),
                                         end_time - start_time), file=logfile)



            if self.opts.graph == True:
                recon_all.append(np.mean(recon_loss_epoch))
                x = np.arange(len(recon_all))
                yR = np.asarray(recon_all)

                plt.plot(x, yR, color='red', label='Recon Loss')
                plt.title("Red: Revised Recon")
                plt.savefig("%s/aRecon_plot.png" % (self.opts.output_dir))
                plt.close("all")

                enc_all.append(np.mean(enc_loss_epoch))
                yD = np.asarray(enc_all)
                plt.plot(x, yD, color='red', label='Encoder Loss')
                plt.title("Encoder Loss")
                plt.savefig("%s/aEncoder_plot.png" % (self.opts.output_dir))
                plt.close("all")

                alpha_all.append(np.mean(alpha1_epoch))
                yA1 = np.asarray(alpha_all)
                #   print(yA1)
                yA2 = 1 - yA1
                #  print(yA2)

                plt.plot(x, yA1, color='red', label='Alpha1 value')
                plt.plot(x, yA2, color='blue', label='Alpha2 value')
                plt.title("Red: Alpha1, Blue: Alpha2")
                plt.savefig("%s/aAlpha_plot.png" % (self.opts.output_dir))
                plt.close("all")

                mu1_all.append(np.mean(mu1_epoch))
                mu2_all.append(np.mean(mu2_epoch))
                yM1 = np.asarray(mu1_all)
                yM2 = np.asarray(mu2_all)

                plt.plot(x, yM1, color='red', label='Mu1 value')
                plt.plot(x, yM2, color='blue', label='Mu2 value')
                plt.title("Red: Mu1, Blue: Mu2")
                plt.savefig("%s/aMu_plot.png" % (self.opts.output_dir))
                plt.close("all")

                lgvar1_all.append(np.mean(lgvar1_epoch))
                lgvar2_all.append(np.mean(lgvar2_epoch))
                yV1 = np.asarray(lgvar1_all)
                yV2 = np.asarray(lgvar2_all)

                plt.plot(x, yV1, color='red', label='Lgvar1 value')
                plt.plot(x, yV2, color='blue', label='Lgvar2 value')
                plt.title("Red: Lgvar1, Blue: Lgvar2")
                plt.savefig("%s/aLgvar_plot.png" % (self.opts.output_dir))
                plt.close("all")

                z = z.reshape(-1).cpu().detach().numpy()
                plt.hist(z, color='blue', edgecolor='red', bins=100)
                plt.savefig("%s/aZdist_plot.png" % (self.opts.output_dir))
                plt.close('all')


            if epoch + 10 >= self.opts.num_epoch:
                for i in range(1):
                    if self.opts.abp:
                        if self.opts.dense == 'True':
                            z = torch.randn((1, self.opts.z_size)).cuda()
                        else:
                            z = torch.zeros((1, 15*4*4)).cuda()
                    else:
                        z, _, _, _, _, _ = self.encoder(obs_data[i:i + 1])
            #        z = torch.zeros((1, 1, 8, 8)).cuda()
               #     z = self.langevin_dynamics_generator(z, obs_data[i:i + 1], False)
                    z = self.langevin_dynamics_generator(z, obs_data[0:1], False)
             #       self.plot_latent((obs_data)[i:i + 1], z, epoch, i + 1)
                    self.plot_latent((obs_data[0:1]), z, epoch, i + 1)
        #            self.plot_traversal(obs_data[i:i + 1], z, epoch, i + 1)
        #    self.normal_sample(epoch)



            if epoch % 5 == 0:
                torch.save(self.generator, self.opts.ckpt_dir + '/gen_ckp.pth')
                if not self.opts.abp:
                    torch.save(self.encoder, self.opts.ckpt_dir + '/enc_ckp.pth')

        self.plot_texture()

        torch.save(self.generator, self.opts.ckpt_dir + '/gen_ckp.pth')
        if not self.opts.abp:
            torch.save(self.encoder, self.opts.ckpt_dir + '/enc_ckp.pth')
        logfile.close()

    def plot_texture(self):
        print("Testing Texture")
        if not os.path.exists(self.opts.test_dir):
            os.makedirs(self.opts.test_dir)

        mul = np.arange(-3, 3, 0.5)


        for q in range(16):
            z = torch.zeros((1, 15* 4 * 4)).cuda()
            image = self.generator(z)
            for k in range(15):
                z = torch.zeros((1, 15 * 4 * 4)).cuda()
                image = self.generator(z)
                for i in mul:
                    z = torch.zeros((1, 15*4*4)).cuda().reshape(-1, 15, 4*4)
                    z[0][k][q] = i
                    image = torch.cat((image, self.generator(z)), 0)
                save_image(image, "%s/texture_img%03d_%03d.png" % (self.opts.test_dir, q, k))



    def plot_latent(self, image, z, epoch, index):
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        img = torchvision.utils.make_grid(image.cpu()).detach().numpy()
        ax0.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
        ax0.set_title('Input Image', fontsize=20)

        z = z.reshape(1, -1)
        ax1.bar(np.arange(z.shape[1]), height=z.cpu().detach().numpy()[0], width=1 / 7, align='center')
        ax1.stem(np.arange(z.shape[1]), z.cpu().detach().numpy()[0], markerfmt=' ', use_line_collection=True)
        ax1.axhline(y=0)
        ax1.set_title(r"Latent Dimension %d" % (z.shape[1]), fontsize=20)

        img = self.generator(z.reshape(1, -1))
        img = torchvision.utils.make_grid((img).view(-1, 3, 64, 64)).cpu().detach().numpy()
        ax2.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
        ax2.set_title('Decoded Image', fontsize=20)
        plt.subplots_adjust(hspace=0.5)
        plt.savefig("%s/aLatent_%03d_%03d.png" % (self.opts.output_dir, epoch + 1, index))
        plt.close('all')



    def plot_traversal(self, obs_data, z, epoch, index):


        mul = [0, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]
        for j in range(self.opts.z_size):
            if torch.abs(z[0, j]) > 0.5:
                image = obs_data
                for k in range(12):
                    z1 = z.clone()
                    z1[0, j] = z1[0, j] + mul[k]
                    image = torch.cat((image, self.generator(z1)), 0)

                save_image(image, "%s/traversal_sample_epoch%03d_index%03d_%03d.png" % (self.opts.test_dir, epoch, index, j))
                plt.close('all')

    def normal_sample(self, epoch):
        z = torch.randn((40, self.opts.z_size)).cuda()
        image = self.generator(z)

        save_image(image, "%s/normal_sample_epoch%03d.png" % (self.opts.test_dir, epoch))

    def latent_gradient_testing(self, z, alpha):
        if alpha == 1:
            return -z
        ratio1 = (1 - alpha) / (alpha * self.slab_variance)
        ratio2 = (self.slab_variance * self.slab_variance - 1) / (self.slab_variance * self.slab_variance)
        gradient = -z + ratio1 * ratio2 * z / (torch.exp(-ratio2 * torch.pow(z, 2) / 2) + ratio1)
        del ratio1, ratio2
        return gradient

    def langevin_dynamics_generator_testing(self, z, obs, generator, alpha=0.01, steps=30):
        obs = obs.detach()
        criterian = nn.MSELoss(reduction='sum')
        if steps != self.opts.langevin_step_num_gen:
            self.opts.langevin_step_num_gen = steps
        for i in range(self.opts.langevin_step_num_gen):
            z = Variable(z, requires_grad=True)
            gen_res = generator(z)
            gen_loss = 1.0 / (2.0 * self.opts.sigma_gen * self.opts.sigma_gen) * criterian(gen_res, obs)
            gen_loss.backward()
            grad = z.grad
            sparse_grad = self.latent_gradient_testing(z, alpha)
            z = z - 0.5 * self.opts.langevin_step_size_gen * self.opts.langevin_step_size_gen * (-sparse_grad + grad)

        return z


    def test_generation(self, generator):
        print("Testing Generation")
        test_batch=int(np.ceil(self.opts.test_size/self.opts.nRow/self.opts.nCol))
        print('===Generated images saved to %s ===' % (self.opts.output_dir))

        for i in range(test_batch):
            z = torch.zeros((self.opts.z_size, self.opts.z_size))
            for j in range(self.opts.z_size):
                index = torch.randint(low=0, high=self.opts.z_size, size=(1,))
                z[j, index] = 1
            z = Variable(z.cuda())

            gen_res = generator(z)
            gen_res = gen_res.detach().cpu()
            print('Generating {:05d}/{:05d}'.format(i + 1, test_batch))
            save_image((gen_res*0.5+0.5), "%s/testres_%03d.png" % (test_dir, i + 1))

    def test_recon(self, generator, encoder, test_data):
        print("Testing Reconstruction")


        mse_loss = torch.nn.MSELoss(reduction='sum')
        gen_loss = 0.0
        for index, (data, _) in enumerate(test_data):
            obs_data = data
            obs_data = Variable(torch.Tensor(obs_data).cuda())  # ,requires_grad=True

            # z0
            if self.opts.abp:
                z1 = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
            else:
                z1, mu1, alpha1, lgvar1, mu2, lgvar2 = encoder(obs_data)
            # z tau
            z_tau = self.langevin_dynamics_generator_testing(z1, obs_data, generator, alpha=1)
            # Recon x
            gen_res = generator(z_tau)
            gen_loss += mse_loss(gen_res,obs_data.detach())

        print("Generator Loss Total: ", gen_loss)
        print("Generator Loss Per Pixel", gen_loss / len(test_data.dataset) / (self.opts.dim * self.opts.dim * self.opts.channel) )

    def test_recon_plot(self, generator, test_data):
        print("Test Recon Plot")

        for i in range(0, 50, 10):
            for index, (data, _) in enumerate(test_data):
                obs_data = data[i:i + 10].cuda()

                # z0
                if self.opts.abp:
                    z1 = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
                else:
                    z1, mu1, alpha1, lgvar1, mu2, lgvar2 = encoder(obs_data)
                # z tau
                z_tau = self.langevin_dynamics_generator_testing(z1, obs_data, generator)

                gen_res = generator(z_tau)


                images = torch.cat((obs_data, gen_res), 0)
                save_image(images, "%s/recon_test_%03d.png" % (self.opts.test_dir, index+1), nrow=10)


    def visual_alter_latent(self, generator, test_data):
        for i in range(50):
            for index, (data, _) in enumerate(test_data):
                obs_data = data.cuda()[i:i + 1]
                break

            z = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
            new_z = self.langevin_dynamics_generator_testing(z, obs_data, generator)
            for j in range(self.opts.z_size):
                if torch.abs(new_z[j]) > 0.5:
                    # Modify the highest value in encoding
                    minn = torch.abs(new_z).max().cpu().detach().numpy() * -1
                    maxx = torch.abs(new_z).max().cpu().detach().numpy()
                    max_ind = torch.argmax(torch.abs(new_z)).item()

                    z1 = new_z.clone()
                    z1[0, max_ind] *= 0.01
                    z2 = new_z.clone()
                    z2[0, max_ind] *= -0.5

                    plt.figure()
                    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(21, 14))

                    # Plot encodings
                    ax[0, 0].bar(np.arange(new_z.shape[1]), height=new_z.cpu().detach().numpy()[0], width=1 / 1.5)
                    ax[0, 0].scatter(np.arange(new_z.shape[1]), new_z.cpu().detach().numpy()[0], color='blue')
                    ax[0, 0].scatter(max_ind, new_z[0, max_ind].cpu().detach().numpy(), color='red', s=100)
                    ax[0, 0].set_title(r"$z_{%d}$ = %.2f " % (max_ind, new_z[0, max_ind]), fontsize=20)
                    ax[0, 0].set_ylim(minn - 0.1, maxx + 0.1)

                    ax[0, 1].bar(np.arange(z1.shape[1]), height=z1.cpu().detach().numpy()[0], width=1 / 1.5)
                    ax[0, 1].scatter(np.arange(z1.shape[1]), z1.cpu().detach().numpy()[0], color='blue')
                    ax[0, 1].scatter(max_ind, z1[0, max_ind].cpu().detach().numpy(), color='red', s=100)
                    ax[0, 1].set_title(r"$z_{%d}$ = %.2f " % (max_ind, z1[0, max_ind]), fontsize=20)
                    ax[0, 1].set_ylim(minn - 0.1, maxx + 0.1)

                    ax[0, 2].bar(np.arange(z2.shape[1]), height=z2.cpu().detach().numpy()[0], width=1 / 1.5)
                    ax[0, 2].scatter(np.arange(z2.shape[1]), z2.cpu().detach().numpy()[0], color='blue')
                    ax[0, 2].scatter(max_ind, z2[0, max_ind].cpu().detach().numpy(), color='red', s=100)
                    ax[0, 2].set_title(r"$z_{%d}$ = %.2f " % (max_ind, z2[0, max_ind]), fontsize=20)
                    ax[0, 2].set_ylim(minn - 0.1, maxx + 0.1)

                    # Plot decoded images
                    img = torchvision.utils.make_grid(generator(new_z).view(1, 3, 64, 64))
                    npimg = img.cpu().detach().numpy() * 0.5 + 0.5  # White background
                    ax[1, 0].imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

                    img = torchvision.utils.make_grid(generator(z1).view(1, 3, 64, 64))
                    npimg = img.cpu().detach().numpy() * 0.5 + 0.5  # White background
                    ax[1, 1].imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

                    img = torchvision.utils.make_grid(generator(z2).view(1, 3, 64, 64))
                    npimg = img.cpu().detach().numpy() * 0.5 + 0.5  # White background
                    ax[1, 2].imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

                    plt.savefig("%s/alter_latent_%03d.png" % (self.opts.test_dir, i))
                    plt.close()


    def test_traversal(self, generator, test_data):
        print("Testing Traversal")

        mul = [0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5]

        for index, (data, _) in enumerate(test_data):
            if index == 0:
                data = data.cuda()
                break

        for q in range(100):
            for i in range(q, q + 1):

                obs_data = data[i:i+1].cuda()


                z = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
                z = self.langevin_dynamics_generator_testing(z, obs_data, generator)

                for j in range(self.opts.z_size):
                    if torch.abs(z[0, j]) > 1:
                        image = generator(z)
                        for k in range(7):
                            z1 = z.clone()
                            z1[0, j] = z1[0, j] + mul[k]
                            image = torch.cat((image, generator(z1)), 0)

                        save_image(image, "%s/traversal_sample_img%03d_%03d_1.png" % (self.opts.test_dir, i, j))

            mul = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
            for i in range(q, q + 1):
                obs_data = data[i:i+1].cuda()
                z = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
                z = self.langevin_dynamics_generator_testing(z, obs_data, generator)

                for j in range(self.opts.z_size):
                    if torch.abs(z[0, j]) > 1:
                        image = generator(z)
                        for k in range(7):
                            z1 = z.clone()
                            z1[0, j] = z1[0, j] + mul[k]
                            image = torch.cat((image, generator(z1)), 0)

                        save_image(image, "%s/traversal_sample_img%03d_%03d_2.png" % (self.opts.test_dir, i, j))

    def visual_latent(self, generator, test_data):
        for i in range(1):
            for index, (data, _) in enumerate(test_data):
                obs_data = data.cuda()[i:i + 1]
                break

        z = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
        z = self.langevin_dynamics_generator_testing(z, obs_data, generator)
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        img = torchvision.utils.make_grid((obs_data).cpu()).detach().numpy()
        ax0.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
        ax0.set_title('Input Image', fontsize=20)

        z = z.reshape(1, -1)
        ax1.bar(np.arange(z.shape[1]), height=z.cpu().detach().numpy()[0], width=1 / 7, align='center')
        ax1.stem(np.arange(z.shape[1]), z.cpu().detach().numpy()[0], markerfmt=' ', use_line_collection=True)
        ax1.axhline(y=0)
        ax1.set_title(r"Latent Dimension %d" % (z.shape[1]), fontsize=20)

        img = generator(z.reshape(1, -1))
        img = torchvision.utils.make_grid(
            (img).view(1, self.opts.channel, self.opts.dim, self.opts.dim)).cpu().detach().numpy()
        ax2.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
        ax2.set_title('Decoded Image', fontsize=20)
        plt.subplots_adjust(hspace=0.5)
        plt.savefig("%s/visual_latent.png" % (self.opts.test_dir))
        plt.close('all')

    def test_cond(self, generator, test_data):
        print("Testing Conditional Sampling")
        for i in range(100):
            for index, (data, _) in enumerate(test_data):
                obs_data = data[i:i + 1].cuda()
                break

            z = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()

            z = self.langevin_dynamics_generator_testing(z, obs_data, generator)

            gen_res = generator(z)
            nonzero = []
            for i in range(z.shape[1]):
                if np.abs(z[0, i].cpu().detach().numpy()) > 1:
                    nonzero.append(i)
            print(nonzero)
            ind = np.random.choice(nonzero, int(0.7 * len(nonzero)))

            image = gen_res
            for _ in range(7):
                z1 = z.clone()
                for i in ind:
                    z1[0, i] = np.random.normal()
                image = torch.cat((image, generator(z1)), 0)

            save_image(image, "%s/cond_sample_%03d.png" % (self.opts.test_dir, i))


        print("Conditional Sampling Done!")

    def test_noise(self, generator, test_data):
        from skimage.metrics import structural_similarity as ssim
        print("Test Denoise")
        j = 5
        for i in range(0, 40, 8):
            print(i)
            new_data = torch.zeros((10, 1, 28, 28)).cuda()
            for index, (data, label) in enumerate(test_data):
                if index == j:
                    obs_data = data.cuda()
                    break
            digits = {}
            for ord in range(10):
                digits[ord] = np.where(label==ord)[0]
            for ord in range(10):
                new_data[ord] = obs_data[digits[ord][ord]]
            obs_data = new_data
            obs_data1 = obs_data + torch.empty_like(obs_data).normal_(mean=0, std=0.3).cuda()
            z = torch.randn((obs_data.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, obs_data1, generator, alpha=1)
            gen_res = generator(z)

            ss = 0
            for i in range(10):
                ss += ssim((obs_data1[i].view(28, 28)).cpu().detach().numpy(), (gen_res[i].view(28, 28)).cpu().detach().numpy())
            print("VSC SSIM1:", ss/10)

            images = torch.cat((obs_data1, gen_res), 0)

            obs_data2 = obs_data + torch.empty_like(obs_data).normal_(mean=0, std=0.5).cuda()
            images = torch.cat((images, obs_data2), 0)
            z = torch.randn((obs_data.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, obs_data2, generator, alpha=1)
            gen_res = generator(z)

            ss = 0
            for i in range(10):
                ss += ssim((obs_data2[i].view(28, 28)).cpu().detach().numpy(),
                           (gen_res[i].view(28, 28)).cpu().detach().numpy())
            print("VSC SSIM2:", ss / 10)

            images = torch.cat((images, gen_res), 0)

            obs_data3 = obs_data + torch.empty_like(obs_data).normal_(mean=0, std=0.7).cuda()
            images = torch.cat((images, obs_data3), 0)
            z = torch.randn((obs_data.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, obs_data3, generator, alpha=1)
            gen_res = generator(z)

            ss = 0
            for i in range(10):
                ss += ssim((obs_data3[i].view(28, 28)).cpu().detach().numpy(),
                           (gen_res[i].view(28, 28)).cpu().detach().numpy())
            print("VSC SSIM3:", ss / 10)

            images = torch.cat((images, gen_res), 0)
            save_image(images, "%s/denoise_%03d.png" % (self.opts.test_dir, i), nrow=10)

            generator2 = torch.load('./checkpoint_hiabp_mnist_decay_30_sig_100/gen_ckp.pth').eval()

            self.opts.langevin_step_num_gen = 30

            obs_data5 = obs_data + torch.empty_like(obs_data).normal_(mean=0, std=0.3).cuda()

            z = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, obs_data5, generator2)
            gen_res = generator2(z)

            ss = 0
            for i in range(10):
                ss += ssim((obs_data5[i].view(28, 28)).cpu().detach().numpy(),
                           (gen_res[i].view(28, 28)).cpu().detach().numpy())
            print("VP SSIM1:", ss / 10)

            images = torch.cat((obs_data1, gen_res), 0)
            images = torch.cat((images, obs_data2), 0)

            obs_data6 = obs_data + torch.empty_like(obs_data).normal_(mean=0, std=0.5).cuda()

            z = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, obs_data6, generator2)
            gen_res = generator2(z)

            ss = 0
            for i in range(10):
                ss += ssim((obs_data6[i].view(28, 28)).cpu().detach().numpy(),
                           (gen_res[i].view(28, 28)).cpu().detach().numpy())
            print("VP SSIM2:", ss / 10)

            images = torch.cat((images, gen_res), 0)
            images = torch.cat((images, obs_data3), 0)

            obs_data4 = obs_data + torch.empty_like(obs_data).normal_(mean=0, std=0.7).cuda()

            z = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, obs_data4, generator2)
            gen_res = generator2(z)

            ss = 0
            for i in range(10):
                ss += ssim((obs_data4[i].view(28, 28)).cpu().detach().numpy(),
                           (gen_res[i].view(28, 28)).cpu().detach().numpy())
            print("VP SSIM3:", ss / 10)

            images = torch.cat((images, gen_res), 0)
            save_image(images, "%s/denoise_%03d_sparse.png" % (self.opts.test_dir, i), nrow=10)



    def test_sparsity(self, generator1, test_data):
        print("Testing Sparsity")

        generator2 = torch.load('./checkpoint_hiabp_celeba050_decay_30_sig_100/gen_ckp.pth').eval()
        generator3 = torch.load('./checkpoint_hiabp_celeba100_decay_30_sig_100/gen_ckp.pth').eval()

        j = 0

        for i in range(100):
            for index, (data, _) in enumerate(test_data):
                if index == j:
                    obs_data = data[i:i + 1].cuda()
                    break

            z1 = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
            alpha = 0.01
            z1 = self.langevin_dynamics_generator_testing(z1, obs_data, generator1, alpha)
            gen1 = generator1(z1)

            z2 = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
            alpha = 0.5
            z2 = self.langevin_dynamics_generator_testing(z2, obs_data, generator2, alpha)
            gen2 = generator2(z2)

            z3 = torch.zeros((obs_data.shape[0], self.opts.z_size)).cuda()
            alpha = 1
            z3 = self.langevin_dynamics_generator_testing(z3, obs_data, generator3, alpha)
            gen3 = generator3(z3)

            plt.figure()
            fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))

            # ax0.bar(np.arange(z3.shape[1]), height=z3.cpu().detach().numpy()[0], width=1 / 7, align='center')
            ax0.scatter(np.arange(z3.shape[1]), z3.cpu().detach().numpy()[0], s=3, color='blue')
            ax0.set_title(r"$\alpha$ = 1 ", fontsize=20)
            ax0.set_ylim(ymin=-torch.max(torch.abs(z3)).cpu().detach().numpy() - 0.3,
                         ymax=torch.max(torch.abs(z3)).cpu().detach().numpy() + 0.3)
            ax0.axes.xaxis.set_visible(False)
            ax0.axes.yaxis.set_visible(False)

            #  ax1.bar(np.arange(z2.shape[1]), height=z2.cpu().detach().numpy()[0], width=1 / 7, align='center')
            ax1.scatter(np.arange(z2.shape[1]), z2.cpu().detach().numpy()[0], s=3, color='blue')
            ax1.set_title(r"$\alpha$ = 0.5 ", fontsize=20)
            ax1.set_ylim(ymin=-torch.max(torch.abs(z2)).cpu().detach().numpy() - 0.3,
                         ymax=torch.max(torch.abs(z2)).cpu().detach().numpy() + 0.3)
            ax1.axes.xaxis.set_visible(False)
            ax1.axes.yaxis.set_visible(False)

            # ax2.bar(np.arange(z1.shape[1]), height=z1.cpu().detach().numpy()[0], width=1 / 7, align='center')
            ax2.scatter(np.arange(z1.shape[1]), z1.cpu().detach().numpy()[0], s=3, color='blue')
            ax2.set_title(r"$\alpha$ = 0.01 ", fontsize=20)
            ax2.set_ylim(ymin=-torch.max(torch.abs(z1)).cpu().detach().numpy() - 0.3,
                         ymax=torch.max(torch.abs(z1)).cpu().detach().numpy() + 0.3)
            ax2.axes.xaxis.set_visible(False)
            ax2.axes.yaxis.set_visible(False)

            gen3 = torchvision.utils.make_grid((gen3).cpu()).detach().numpy()
            ax3.imshow(np.transpose(gen3, (1, 2, 0)), interpolation='nearest')
            ax3.axis('off')

            gen2 = torchvision.utils.make_grid((gen2).cpu()).detach().numpy()
            ax4.imshow(np.transpose(gen2, (1, 2, 0)), interpolation='nearest')
            ax4.axis('off')

            gen1 = torchvision.utils.make_grid((gen1).cpu()).detach().numpy()
            ax5.imshow(np.transpose(gen1, (1, 2, 0)), interpolation='nearest')
            ax5.axis('off')

            fig.tight_layout()


            plt.savefig("%s/visual_sparsity_%03d.png" % (self.opts.test_dir, i))
            plt.close('all')



    def test_tsne(self, generator, test_data):
        if self.opts.dense == True:
            z = torch.zeros((len(test_data.dataset), self.opts.z_size)).cuda()
        else:
            z = torch.zeros((len(test_data.dataset), self.opts.z_size)).cuda()
        y = torch.zeros(len(test_data.dataset))
        for index, (data, label) in enumerate(test_data):
            obs_data = data.cuda()
            z[index*test_data.batch_size:(index+1)*test_data.batch_size] = self.langevin_dynamics_generator_testing(z[index*test_data.batch_size:(index+1)*test_data.batch_size], obs_data, generator)
            y[index*test_data.batch_size:(index+1)*test_data.batch_size] = label
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, random_state=123)
        z = tsne.fit_transform(z.cpu().detach().numpy())

        df = pd.DataFrame()
        df["y"] = y
        df["comp-1"] = z[:, 0]
        df["comp-2"] = z[:, 1]

        sns_plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 10),
                        data=df).set(title="Short-run T-SNE on MNIST", xlabel=None, ylabel=None)

        plt.savefig("%s/dense_tsne.png" % (self.opts.test_dir))
        plt.close("all")

    def test_denoise_classify(self, generator, noise):
        print("Test Denoise Classify")
        train_data = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(root='./data/' + self.opts.set, train=True, download=False,
                                                  transform=transforms.Compose(
                                                      [transforms.ToTensor()])),
                batch_size=self.opts.batch_size, shuffle=False)

        test_data = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(root='./data/' + self.opts.set, train=False, download=False,
                                                  transform=transforms.Compose(
                                                      [transforms.ToTensor()])),
                batch_size=self.opts.batch_size, shuffle=False)

        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(784, 10)  # single layer
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.layer(x))  # activation function for hidden layer
                return x

        net = Classifier().cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = torch.nn.CrossEntropyLoss()

        generator2 = torch.load('./checkpoint_denseabp_mnist100/gen_ckp.pth').eval()
        generator3 = torch.load('./checkpoint_denseabp_mnist100_warm/gen_ckp.pth').eval()
        model1 = torch.load('./checkpoint_bvae_mnist100/model.pth').eval()
        model2 = torch.load('./checkpoint_vsc_mnist100/model.pth').eval()

        for epoch in range(20):

            running_loss = 0.0

            for index, train_batch in enumerate(train_data, 0):
                ux, uy = train_batch
                ux = ux.cuda()
                uy = uy.cuda()

                outputs = net(ux.view(-1, 28*28))

                loss = loss_func(outputs, uy)
                optimizer.zero_grad()
                loss.backward()
                running_loss += loss
                optimizer.step()

        correct_sparse = 0
        correct_dense = 0
        correct_bvae = 0
        correct_vsc = 0
        correct_abp = 0
        total = 0

        for label_batch in test_data:
            lx, ly = label_batch
            lx = lx.cuda() + torch.empty_like(lx).normal_(mean=0, std=noise).cuda()
            total += ly.size(0)

            z = torch.zeros((lx.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, lx, generator)
            gen_res = generator(z)
            outputs = net(gen_res.view(-1, 28*28))
            _, predicted = torch.max(outputs.data, 1)
            correct_sparse += (predicted.cpu() == ly).sum().item()


            z = torch.randn((lx.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, lx, generator2, alpha=1)
            gen_res = generator2(z)
            outputs = net(gen_res.view(-1, 28 * 28))
            _, predicted = torch.max(outputs.data, 1)
            correct_dense += (predicted.cpu() == ly).sum().item()

            z = torch.randn((lx.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, lx, generator3, alpha=1, steps=100)
            gen_res = generator3(z)
            outputs = net(gen_res.view(-1, 28 * 28))
            _, predicted = torch.max(outputs.data, 1)
            correct_abp += (predicted.cpu() == ly).sum().item()

            gen_res, _, _ = model1(lx.view(-1, 28*28))
            outputs = net(gen_res.view(-1, 28 * 28))
            _, predicted = torch.max(outputs.data, 1)
            correct_bvae += (predicted.cpu() == ly).sum().item()

            gen_res, _, _, _ = model2(lx.view(-1, 28*28))
            outputs = net(gen_res.view(-1, 28 * 28))
            _, predicted = torch.max(outputs.data, 1)
            correct_vsc += (predicted.cpu() == ly).sum().item()



        print("Noise: ", noise, " Sparse Accuracy", 100 * correct_sparse / total)
        print("Dense: ", 100 * correct_dense / total)
        print("ABP: ", 100 * correct_abp / total)
        print("BVAE: ", 100 * correct_bvae / total)
        print("VSC: ", 100 * correct_vsc / total)

    def test_celeba_noise(self, generator, test_data):
        print("Test Denoise")
        j = 3
        for i in range(0, 100, 8):
            for index, (data, label) in enumerate(test_data):
                if index == j:
                    obs_data = data[i:i+8].cuda()
                    break


            obs_data1 = obs_data + torch.empty_like(obs_data).normal_(mean=0, std=0.1).cuda()

            images = obs_data1


            z = torch.randn((obs_data.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, obs_data1, generator)
            gen_res = generator(z)

            images = torch.cat((images, gen_res), 0)

            obs_data2 = obs_data + torch.empty_like(obs_data).normal_(mean=0, std=0.15).cuda()
            z = torch.randn((obs_data.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, obs_data2, generator)
            gen_res = generator(z)

            images = torch.cat((images, obs_data2), 0)
            images = torch.cat((images, gen_res), 0)

            obs_data3 = obs_data + torch.empty_like(obs_data).normal_(mean=0, std=0.2).cuda()
            z = torch.randn((obs_data.shape[0], self.opts.z_size)).cuda()
            z = self.langevin_dynamics_generator_testing(z, obs_data3, generator)
            gen_res = generator(z)

            images = torch.cat((images, obs_data3), 0)
            images = torch.cat((images, gen_res), 0)

            save_image(images, "%s/denoise_%03d_sparse.png" % (self.opts.test_dir, i), nrow=8)

    def test_fashion(self, generator):
        test_set = torchvision.datasets.FashionMNIST(root='./data/' + self.opts.set, train=False, download=False,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()]))


        data0 = test_set.data[test_set.targets == 0].float().cuda().view(-1, 1, 28, 28) / 255
        data1 = test_set.data[test_set.targets == 1].float().cuda().view(-1, 1, 28, 28) / 255
        data2 = test_set.data[test_set.targets == 2].float().cuda().view(-1, 1, 28, 28) / 255
        data3 = test_set.data[test_set.targets == 3].float().cuda().view(-1, 1, 28, 28) / 255
        data4 = test_set.data[test_set.targets == 4].float().cuda().view(-1, 1, 28, 28) / 255
        data5 = test_set.data[test_set.targets == 5].float().cuda().view(-1, 1, 28, 28) / 255
        data6 = test_set.data[test_set.targets == 6].float().cuda().view(-1, 1, 28, 28) / 255
        data7 = test_set.data[test_set.targets == 7].float().cuda().view(-1, 1, 28, 28) / 255
        data8 = test_set.data[test_set.targets == 8].float().cuda().view(-1, 1, 28, 28) / 255
        data9 = test_set.data[test_set.targets == 9].float().cuda().view(-1, 1, 28, 28) / 255

        z0 = torch.zeros((data0.shape[0], self.opts.z_size)).cuda()
        z0 = self.langevin_dynamics_generator_testing(z0, data0, generator)
        z0 = torch.where(torch.abs(z0)>0.2, torch.tensor(1).cuda(), torch.tensor(0).cuda()).float()
        z0_m = z0.mean(dim=0)

        z1 = torch.zeros((data1.shape[0], self.opts.z_size)).cuda()
        z1 = self.langevin_dynamics_generator_testing(z1, data1, generator)
        z1 = torch.where(torch.abs(z1) > 0.2, torch.tensor(1).cuda(), torch.tensor(0).cuda()).float()
        z1_m = z1.mean(dim=0)

        z2 = torch.zeros((data2.shape[0], self.opts.z_size)).cuda()
        z2 = self.langevin_dynamics_generator_testing(z2, data2, generator)
        z2 = torch.where(torch.abs(z2) > 0.2, torch.tensor(1).cuda(), torch.tensor(0).cuda()).float()
        z2_m = z2.mean(dim=0)

        z3 = torch.zeros((data3.shape[0], self.opts.z_size)).cuda()
        z3 = self.langevin_dynamics_generator_testing(z3, data3, generator)
        z3 = torch.where(torch.abs(z3) > 0.2, torch.tensor(1).cuda(), torch.tensor(0).cuda()).float()
        z3_m = z3.mean(dim=0)

        z4 = torch.zeros((data4.shape[0], self.opts.z_size)).cuda()
        z4 = self.langevin_dynamics_generator_testing(z4, data4, generator)
        z4 = torch.where(torch.abs(z4) > 0.2, torch.tensor(1).cuda(), torch.tensor(0).cuda()).float()
        z4_m = z4.mean(dim=0)

        z5 = torch.zeros((data5.shape[0], self.opts.z_size)).cuda()
        z5 = self.langevin_dynamics_generator_testing(z5, data5, generator)
        z5 = torch.where(torch.abs(z5) > 0.2, torch.tensor(1).cuda(), torch.tensor(0).cuda()).float()
        z5_m = z5.mean(dim=0)

        z6 = torch.zeros((data6.shape[0], self.opts.z_size)).cuda()
        z6 = self.langevin_dynamics_generator_testing(z6, data6, generator)
        z6 = torch.where(torch.abs(z6) > 0.2, torch.tensor(1).cuda(), torch.tensor(0).cuda()).float()
        z6_m = z6.mean(dim=0)

        z7 = torch.zeros((data7.shape[0], self.opts.z_size)).cuda()
        z7 = self.langevin_dynamics_generator_testing(z7, data7, generator)
        z7 = torch.where(torch.abs(z7) > 0.2, torch.tensor(1).cuda(), torch.tensor(0).cuda()).float()
        z7_m = z7.mean(dim=0)

        z8 = torch.zeros((data8.shape[0], self.opts.z_size)).cuda()
        z8 = self.langevin_dynamics_generator_testing(z8, data8, generator)
        z8 = torch.where(torch.abs(z8) > 0.2, torch.tensor(1).cuda(), torch.tensor(0).cuda()).float()
        z8_m = z8.mean(dim=0)

        z9 = torch.zeros((data9.shape[0], self.opts.z_size)).cuda()
        z9 = self.langevin_dynamics_generator_testing(z9, data9, generator)
        z9 = torch.where(torch.abs(z9) > 0.2, torch.tensor(1).cuda(), torch.tensor(0).cuda()).float()
        z9_m = z9.mean(dim=0)



        z_all = torch.stack((z0_m, z2_m, z3_m, z4_m, z6_m, z5_m, z7_m, z9_m, z1_m, z8_m))


        df = pd.DataFrame(z_all.cpu().detach().numpy())
        df.index = ['Top', 'Pullover', 'Dress', 'Coat', 'Shirt', 'Sandal', 'Sneaker',  'Boot', 'Trouser', 'Bag']
        df

        g = sns.heatmap(df, cmap="YlGnBu")
        g.set_title('Latent Heatmap')
        g.set_xticks(range(30))
        plt.xticks([0, 10, 20, 30])
        g.hlines([5,8,9], *g.get_xlim())
        g.set_xticklabels(['0', '10', '20', '30'])
        g.set_yticklabels(['Top', 'Pullover', 'Dress', 'Coat', 'Shirt', 'Sandal',  'Sneaker',  'Boot', 'Trouser', 'Bag'])
        plt.savefig("%s/heatmap.png" % (self.opts.test_dir))
        plt.close("all")



    def test(self):
        assert self.opts.ckpt_gen is not None, 'Please specify the path to the checkpoint of generator.'
        assert self.opts.ckpt_enc is not None, 'Please specify the path to the checkpoint of encoder.'
        print('===Test on ' + self.opts.ckpt_gen + ' and ' + self.opts.ckpt_enc+' ===')
        generator = torch.load(self.opts.ckpt_gen).eval()
        if not self.opts.abp:
            encoder = torch.load(self.opts.ckpt_enc).eval()
        else:
            encoder = torch.load(self.opts.ckpt_gen).eval()

        if not os.path.exists(self.opts.test_dir):
            os.makedirs(self.opts.test_dir)


        if self.opts.set == 'mnist':
            test_data = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(root='./data/' + self.opts.set, train=False, download=False,
                                                  transform=transforms.Compose(
                                                      [transforms.ToTensor()])),
                batch_size=self.opts.batch_size, shuffle=False)
            self.opts.dim = 28
            self.opts.channel = 1
        elif self.opts.set == 'fashion':
            test_data = torch.utils.data.DataLoader(
                torchvision.datasets.FashionMNIST(root='./data/' + self.opts.set, train=False, download=False,
                                                  transform=transforms.Compose(
                                                      [transforms.ToTensor()])),
                batch_size=self.opts.batch_size, shuffle=False)
            self.opts.dim = 28
            self.opts.channel = 1
        elif self.opts.set == 'celeba':
            test_data = torch.utils.data.DataLoader(
                torchvision.datasets.CelebA(root='./data/' + self.opts.set + '_train',
                                            split='test', download=False,
                                            transform=transforms.Compose([transforms.CenterCrop(128),
                                                                          transforms.Resize(self.img_size),
                                                                          transforms.ToTensor()])),
                batch_size=self.opts.batch_size, shuffle=False)
            self.opts.dim = 64
            self.opts.channel = 3

        elif self.opts.set == 'svhn':
            test_data = torch.utils.data.DataLoader(
                torchvision.datasets.SVHN(root='./data/' + self.opts.set + '_train',
                                            split='test', download=True,
                                            transform=transforms.Compose([
                                                                          transforms.ToTensor()])),
                batch_size=self.opts.batch_size, shuffle=False)
            self.opts.dim = 32
            self.opts.channel = 3
        else:
            pass

       # self.test_generation(generator)
      #  self.test_recon(generator, encoder, test_data)
      #  self.test_recon_plot(generator, test_data)
       # self.test_cond(generator, test_data)
      #  self.test_traversal(generator, test_data)
      #  self.visual_latent(generator, test_data)
        self.test_tsne(generator, test_data)

       # self.test_sparsity(generator, test_data)
     #   self.test_noise(generator, test_data)
     #   self.test_celeba_noise(generator, test_data)
      #  self.test_classify(generator)
     #   for i in range(5):
     #       self.test_denoise_classify(generator, noise=0.3)
     #       self.test_denoise_classify(generator, noise=0.5)
      #      self.test_denoise_classify(generator, noise=0.7)
     #   self.test_fashion(generator)

        print ('===Image generation done.===')

