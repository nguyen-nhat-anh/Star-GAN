import argparse
import os
from model import StarGAN


IMG_DIR = os.path.join('data', 'img_align_celeba', 'img_align_celeba')
LABEL_PATH = os.path.join('data', 'list_attr_celeba.csv')
SELECTED_ATTRIBUTES = ['Male', 'Young']
BUFFER_SIZE = 5000
BATCH_SIZE = 16

G_CONV_DIM = 64
D_CONV_DIM = 64

LAMBDA_CLS = 1.0
LAMBDA_REC = 10.0
LAMBDA_GP = 10.0

N_CRITICS = 5
G_INIT_LR = 1e-4
D_INIT_LR = 1e-4

DECAY_ITER = 100000
DECAY_FREQ = 1000

CKPT_DIR = 'model'
SAMPLE_DIR = 'samples'

START_ITER = 0
END_ITER = 200000
DISPLAY_FREQ = 1000
SAVE_FREQ = 10000


argparser = argparse.ArgumentParser(description="Star GAN")

argparser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
# for training
argparser.add_argument('--img_dir', type=str, default=IMG_DIR)
argparser.add_argument('--label_path', type=str, default=LABEL_PATH)
argparser.add_argument('--selected_attributes', type=str, nargs='+', default=SELECTED_ATTRIBUTES)
argparser.add_argument('--buffer_size', type=int, default=BUFFER_SIZE)
argparser.add_argument('--batch_size', type=int, default=BATCH_SIZE)

argparser.add_argument('--g_conv_dim', type=int, default=G_CONV_DIM)
argparser.add_argument('--d_conv_dim', type=int, default=D_CONV_DIM)

argparser.add_argument('--lambda_cls', type=float, default=LAMBDA_CLS)
argparser.add_argument('--lambda_rec', type=float, default=LAMBDA_REC)
argparser.add_argument('--lambda_gp', type=float, default=LAMBDA_GP)

argparser.add_argument('--n_critics', type=int, default=N_CRITICS)
argparser.add_argument('--g_init_lr', type=float, default=G_INIT_LR)
argparser.add_argument('--d_init_lr', type=float, default=D_INIT_LR)

argparser.add_argument('--decay_iter', type=int, default=DECAY_ITER)
argparser.add_argument('--decay_freq', type=int, default=DECAY_FREQ)

argparser.add_argument('--ckpt_dir', type=str, default=CKPT_DIR)
argparser.add_argument('--sample_dir', type=str, default=SAMPLE_DIR)

argparser.add_argument('--start_iter', type=int, default=START_ITER)
argparser.add_argument('--end_iter', type=int, default=END_ITER)
argparser.add_argument('--display_freq', type=int, default=DISPLAY_FREQ)
argparser.add_argument('--save_freq', type=int, default=SAVE_FREQ)

# for testing
argparser.add_argument('--test_img_path', type=str)  # e.g 'test/test_img.png'
argparser.add_argument('--attr_values', type=int, nargs='+')  # e.g [1, 0]


def _main(args):
    # Create directories
    assert os.path.exists(args.img_dir)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    stargan_model = StarGAN(config=args)
    if args.mode == 'train':
        stargan_model.train()
    elif args.mode == 'test':
        stargan_model.test()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)