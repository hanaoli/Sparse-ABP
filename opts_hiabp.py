import os
import argparse

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('-num_epoch', type=int, default=100, help='training epochs')
        self.parser.add_argument('-batch_size', type=int, default=100, help='training batch size')
        self.parser.add_argument('-nRow', type=int, default=10, help='how many rows of images in the output')
        self.parser.add_argument('-nCol', type=int, default=10, help='how many columns of images in the output')
        self.parser.add_argument('-img_size', type=int, default=64, help='output image size')
        self.parser.add_argument('-abp', type=bool, default=True, help='whether training with iabp or abp')
        self.parser.add_argument('-fix', type=bool, default=True, help='whether fix standard normal')
        self.parser.add_argument('-conv', type=bool, default=False, help='whether use conv layer')
        self.parser.add_argument('-dense', type=bool, default=False, help='whether use dense or sparse abp')


        #test setting
        self.parser.add_argument('-test_size', type=int, default=1, help='How many images to generate during testing')
        self.parser.add_argument('-test', default=False, action = 'store_true', help='add `-test` for testing')
        self.parser.add_argument('-score', action = 'store_true', help='add `-score` for reporting scores')

        self.parser.add_argument('-z_size', type=int, default=800, help='dimension of latent variable sample from latent space')
        self.parser.add_argument('-category', default='alp', help='training category')
        self.parser.add_argument('-set', default='beehive', help='which dataset, celeba/scene/cifar')
        self.parser.add_argument('-test_dir', default='./test_beehive_m4',help='directory for testing')
        self.parser.add_argument('-output_dir', default='./sparse_beehive_m4', help='directory to save output synthesized images')
        self.parser.add_argument('-ckpt_dir', default='./checkpoint_beehive_m4', help='directory to save checkpoints')
        self.parser.add_argument('-log_epoch', type=int, default=5, help='save checkpoint each `log_epoch` epochs')
        self.parser.add_argument('-graph', type=bool, default=False, help='whether plot during training')

        self.parser.add_argument('-dim', type=int, default=64, help='length and width of data')
        self.parser.add_argument('-channel', type=int, default=3, help='channels of data')
        self.parser.add_argument('-with_noise', type=bool, default=False, help='add noise during the langevin or not')

        #Generator Parameters
        self.parser.add_argument('-ckpt_gen', default=None, help='load checkpoint for generator')
       # self.parser.add_argument('-ckpt_gen', default='./checkpoint_vae_svhn100/model.pth', help='load checkpoint for generator')
        self.parser.add_argument('-sigma_gen', type=float, default=0.3,help='sigma of reference distribution')
        self.parser.add_argument('-langevin_step_num_gen', type=int, default=30, help='langevin step number for generator')
        self.parser.add_argument('-langevin_step_size_gen', type=float, default=0.05, help='langevin step size for generator')
        self.parser.add_argument('-lr_gen', type=float, default=0.01,help='learning rate of generator')
        self.parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')

        #Encoder Parameters
        self.parser.add_argument('-ckpt_enc', default=None, help='load checkpoint for descriptor')
     #   self.parser.add_argument('-ckpt_enc', default='./checkpoint_hiabp_mnist_decay_30_sig_100/enc_ckp.pth', help='load checkpoint for descriptor')
        self.parser.add_argument('-lr_des', type=float, default=0.0001,help='learning rate of descriptor')
        self.parser.add_argument('-beta1_des', type=float, default=0.5,help='beta of Adam for descriptor')
        self.parser.add_argument('-encoder_steps', type=int, default=1, help='number of times updating encoder')

        # 0 100
        # 1 800 sv=0.1
        # 2 800 sv=0.01
        # 3 800 sv=0.1 alpha=0.5
        # 4 800 sv=0.1 alpha=1
        # conv_0001_2 LRG=0.0001
        # conv_0001_21 Grad_Mul = 0.9
        # conv_0001_2_20 LS = 20
        # conv_0001_n Noise
        # conv_0001_20n Noise LS=20
        # den/den1 800 sv=0.1 alpha=0.01

        # svhn 0.0001 10 0.1 Noise
        # svhn_30 LS = 30 Noise
        # svhn_nn no noise
        # svhn_30_nn LS=30 no noise

        # celeba0001 alpha=0.001



    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        if not os.path.exists(self.opt.ckpt_dir):
            os.makedirs(self.opt.ckpt_dir)
        file_name = os.path.join(self.opt.ckpt_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
        return self.opt