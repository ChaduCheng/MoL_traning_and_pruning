import argparse
import json
import numpy as np
import torch


from dataset import load_mnist_test_data, load_cifar10_test_data, load_imagenet_test_data, load_tinyimagenet_test_data
from general_torch_model import GeneralTorchModel
# from general_tf_model import GeneralTFModel
from preact_resnet import PreActResNet18
from model_resnet import *
from tqdm import tqdm

from RayS import RayS

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

np.random.seed(1234)


def main():
    parser = argparse.ArgumentParser(description='RayS Attacks')
    # parser.add_argument('--dataset', default='rob_cifar_trades', type=str,
    #                     help='robust model / dataset')
    # parser.add_argument('--targeted', default='0', type=str,
    #                     help='targeted or untargeted')
    # parser.add_argument('--norm', default='linf', type=str,
    #                     help='Norm for attack, linf only')
    # parser.add_argument('--num', default=10000, type=int,
    #                     help='Number of samples to be attacked from test dataset.')
    # parser.add_argument('--query', default=10000, type=int,
    #                     help='Maximum queries for the attack')
    # parser.add_argument('--batch', default=10, type=int,
    #                     help='attack batch size.')
    # parser.add_argument('--epsilon', default=0.05, type=float,
    #                     help='attack strength')
    args = parser.parse_args()
    args.dataset = 'cifar'
    args.targeted = '0'
    args.num = 200
    args.query = 10000
    args.batch = 10
    # args.norm = 'l2'
    # args.epsilon = 1.0
    args.norm = 'linf'
    args.epsilon = 8/255
    args.early = '1'

    targeted = True if args.targeted == '1' else False
    order = 2 if args.norm == 'l2' else np.inf

    print(args)
    summary_all = ''

    if args.dataset == 'cifar':
        # model = MoE_ResNet18_adv(7, 200).cuda()
        model = MoE_ResNet18_base(4, 200).cuda()

        # model = ResNet18(200).cuda()
        # model = PreActResNet18().cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_linf_8_255.pth')
        # checkpoint = torch.load('../checkpoint/linf_8.pt')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_clean_at.pth')
        # checkpoint = torch.load('../checkpoint/ckptnat_resnet18_cifar_ensemble_max_all_at.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_snow_bright_2.5_trans.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_fog_t_0.15_light_0.6_trans.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_l2_1_at.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_linf_8_255_at.pth')

        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_ensemble_aver_adv_at.pth')
        # checkpoint = torch.load('/home/haocheng/MoRE_final_ensure/cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv+4nat_final_ensure_at.pth')

        # checkpoint = torch.load('/home/haocheng/MoRE_final_ensure/cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv_expertloss_at_change.pth')


        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_linf_8_255.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_l2_255_255.pth')
        # model_loc = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_linf_8_255.pth'
        # model_loc = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_l2_255_255.pth'
        # checkpoint = torch.load('../checkpoint/ckptnat_resnet18_cifar_ensemble_max_all_at_change.pth')

        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_ensemble_aver_all_at_change.pth')

        # checkpoint = torch.load('/home/haocheng/MoRE_final_ensure/cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+3adv_l1_2_inf_expertloss.pth')

        # checkpoint = torch.load('/home/chenghao/MoRE_final_ensure/cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+3adv_l1_2_inf_baseline.pth')

        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_l1_16_at_change_26.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_l2_1_at_change.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_linf_8_255_at_change.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_fog_t_2.0_light_1.0_at_change.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_snow_bright_20_change.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_cifar_rota_180_at.pth')

        # checkpoint = torch.load('../checkpoint/tinyimagenet_clean.pth')
        # checkpoint = torch.load('../checkpoint/tinyimagenet_l1_12.0.pt')
        # checkpoint = torch.load('../checkpoint/tinyimagenet_l2_1.0.pt')
        # checkpoint = torch.load('../checkpoint/tinyimagenet_linf_8.pt')
        # checkpoint = torch.load('../checkpoint/tinyimagenet_fog_2.0_1.0.pt')
        # checkpoint = torch.load('../checkpoint/tinyimagenet_snow_20.pt')
        # checkpoint = torch.load('../checkpoint/tinyimagenet_rotate_180.pt')

        # checkpoint = torch.load('../checkpoint/iter_55.pt')
        # checkpoint = torch.load('../checkpoint/ckptnat_resnet18_tinyimagenet_ensemble_max_adv_at_change.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_tinyimagenet_ensemble_aver_adv_at_change.pth')
        # checkpoint = torch.load('../checkpoint/ckptnat_resnet18_tinyimagenet_ensemble_max_all_at_change.pth')
        # checkpoint = torch.load('../checkpoint/ckpt_resnet18_tinyimagenet_ensemble_aver_all_at_change.pth')

##  MoRE
        # checkpoint = torch.load('/home/chenan/haotest/cifar_16_resenet/checkpoint/ckptMoE_resnet_timage_clean+3adv_l1_2_inf_expertloss.pth')
        # checkpoint = torch.load('/home/chenan/haotest/cifar_16_resenet/checkpoint/ckptMoE_resnet_tiny_clean+3adv+3nat_expertloss_at_final.pth')
        checkpoint = torch.load('/home/chenghao/MoRE_final_ensure/cifar_16_resenet/checkpoint/ckptMoE_resnet_timage_clean+3adv_l1_2_inf_base.pth')



        # checkpoint = torch.load('../MSD/Final/MSD_V0/all_iter_30.pt')


        # checkpoint = torch.load('../checkpoint/ckptnat_resnet18_cifar_ensemble_max_adv.pth')
        # checkpoint = torch.load('../checkpoint/ckptnat_resnet18_cifar_ensemble_max_all.pth')
        # checkpoint = torch.load('../MSD/Final/MSD_V0/iter_25.pt')
        # checkpoint = torch.load('/home/chenghao/MoRE_final_ensure/cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv_averexperts_new_at.pth')
        # checkpoint = torch.load('/proj/usr/hao.cheng/test/cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv_final_ensure_noclean.pth')
        # checkpoint = torch.load('/proj/usr/hao.cheng/test/cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv+4nat_final_ensure.pth')
        model.load_state_dict(checkpoint['net'])
        # model.load_state_dict(checkpoint)
        # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
        # test_loader = load_cifar10_test_data(args.batch)
        # test_loader = load_cifar10_test_data(args.batch)
        test_loader = load_tinyimagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model, im_mean=None, im_std=None)
    else:
        print("Invalid dataset")
        exit(1)

    attack = RayS(torch_model, epsilon=args.epsilon, order=order)

    adbd = []
    queries = []
    succ = []
    i = 0

    count = 0
    for data, label in tqdm(test_loader):
        data, label = data.cuda(), label.cuda()

        if count >= args.num:
            break

        if targeted:
            target = np.random.randint(torch_model.n_class) * torch.ones(
                label.shape, dtype=torch.long).cuda() if targeted else None
            while target and torch.sum(target == label) > 0:
                print('re-generate target label')
                target = np.random.randint(
                    torch_model.n_class) * torch.ones(len(data), dtype=torch.long).cuda()
        else:
            target = None

        _, queries_b, adbd_b, succ_b = attack(
            data, label, target=target, query_limit=args.query)

        queries.append(queries_b)
        adbd.append(adbd_b)
        succ.append(succ_b)

        count += data.shape[0]

        summary_batch = "Batch: {:4d} Avg Queries (when found adversarial examples): {:.4f} ADBD: {:.4f} Robust Acc: {:.4f}\n" \
            .format(
                i + 1,
                torch.stack(queries).flatten().float().mean(),
                torch.stack(adbd).flatten().mean(),
                1 - torch.stack(succ).flatten().float().mean()
            )
        print(summary_batch)
        summary_all += summary_batch
        i = i+1

    name = args.dataset + '_query_' + str(args.query) + '_batch'
    with open(name + '_summary' + '.txt', 'w') as fileopen:
        json.dump(summary_all, fileopen)


if __name__ == "__main__":
    main()
