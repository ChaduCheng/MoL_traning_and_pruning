import os
from argparse import ArgumentParser
#from testing import validation
#from trained_model_expert import train
from train_mod_adv import train
from train_mod_adv import test, test_deepfool


def get_args():
    parser = ArgumentParser(description='Mixture of Experts')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset', type=float, default='mnist')
    parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--train_split', type=float, default=0.8)
    # parser.add_argument('--lr', type=float, default=.001)
    # parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--checkpoint_loc', type=str, default='./checkpoint/latest_model.ckpt')
    parser.add_argument('--num_experts', type=int, default=16)
    parser.add_argument('--training', type=bool, default=True)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--method', type=str, default='more')
    parser.add_argument('--usemodel', type=str, default='more')
    args = parser.parse_args()
    # args.epochs = 100
    # args.dataset = 'tinyimagenet'
    args.dataset = 'cifar'
    args.batch_size = 128
    args.train_split = 0.02
    args.lr = 0.1
    args.gpu_ids = '0'

    args.num_experts = 7
    args.training = False
    args.testing = True
    args.norm_linf = 'inf'
    args.epsilon_linf = 8/255
    args.norm_l2 = '2'
    args.epsilon_l2 = 1.0
    # args.usemodel = 'more'
    args.seed = 1
    args.resume = False
    # args.method = 'base'  ## more or base
    return args

def main():
    args = get_args()
    
    if args.training:
        print(args.checkpoint_loc)
        train(args)
    if args.testing:
        print(args.checkpoint_loc)
        test_deepfool(args)



if __name__ == '__main__':
    main()
