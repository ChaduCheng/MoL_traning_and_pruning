

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
import torch.backends.cudnn as cudnn

import utils
# from model_adv import AlexNet, MoE_alexnet
from model_adv_att import AttackPGD
from model_deepfool_att import deepfool
from model_resnet import *
from preact_resnet import PreActResNet18

import string


def train_clean (train_loader, device,optimizer,model,CE_loss, lr_schedule, epoch_i):
    a = 0
    j = 0
    for j in range(1,2):
        print('Doing clean images training No. ' + str(j))
        for images, labels in tqdm(train_loader):
        #for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                # optimizer.zero_grad()
                # prediction, weights = model(images)
                prediction = model(images)

                #print('prediction value is :', prediction )

                loss = CE_loss(prediction, labels)
                # print('loss value is :', loss )

                lr = lr_schedule(epoch_i + (a+1)/len(train_loader))
                optimizer.param_groups[0].update(lr=lr)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                a = a+1

    # return weights
def train_adv(train_loader, device,optimizer, basic_model, model, AttackPGD ,CE_loss,config,attack, lr_schedule, epoch_i):
    b = 0
    j = 0
    all_loss = 0
    aver_loss = 0
    all_exp_loss = 0
    aver_exp_loss = 0
    for i in config:
        j = j+1
        print('Adv Training ' + str(config[i]['_type']) + '  epsilon:' + str(j))

        for images_adv, labels in tqdm(train_loader):
        #for images_adv, labels in train_loader:

                model.eval()
                net_attack = AttackPGD(basic_model,config[i])

                #net_attack = AttackPGD(model, config[i])

                #print(net_attack)

                net_attack = net_attack.to(device)
                images_adv = images_adv.to(device)
                labels = labels.to(device)
                #print(images.device)
                # images.cuda(args.gpu_ids[0])
                # labels.cuda(args.gpu_ids[0])
                #print(images)
                #print(images.shape)
                #print(type(images))

                images_att = net_attack(images_adv,labels, attack)



                model.train()
                #optimizer.zero_grad()
                # prediction, weights = model(images_att)
                prediction, predicts = model(images_att)



                #print('prediction value is :', prediction )
                losses = []

                # loss = CE_loss(prediction, labels)

                losses.append(CE_loss(predicts[0], labels))

                losses.append(CE_loss(predicts[1], labels))

                losses.append(CE_loss(predicts[2], labels))

                losses.append(CE_loss(predicts[3], labels))


                loss = max(losses)
                loss_exp_aver = sum(losses)/4


                lr = lr_schedule(epoch_i + (b+1)/len(train_loader))
                optimizer.param_groups[0].update(lr=lr)

                optimizer.zero_grad()
                # print('loss value is :', loss )
                loss.backward()

                optimizer.step()

                b = b+1

                all_loss = all_loss + loss
                all_exp_loss = all_exp_loss + loss_exp_aver

        aver_loss = all_loss / len(train_loader.dataset)
        aver_exp_loss = all_exp_loss / len(train_loader.dataset)

    return aver_loss, aver_exp_loss


def train_adv_base(train_loader, device,optimizer, basic_model, model, AttackPGD ,CE_loss,config,attack, lr_schedule, epoch_i):
    b = 0
    j = 0
    all_loss = 0
    aver_loss = 0
    for i in config:
        j = j+1
        print('Adv Training ' + str(config[i]['_type']) + '  epsilon:' + str(j))

        for images_adv, labels in tqdm(train_loader):
        #for images_adv, labels in train_loader:

                model.eval()
                net_attack = AttackPGD(basic_model,config[i])

                #net_attack = AttackPGD(model, config[i])

                #print(net_attack)

                net_attack = net_attack.to(device)
                images_adv = images_adv.to(device)
                labels = labels.to(device)
                #print(images.device)
                # images.cuda(args.gpu_ids[0])
                # labels.cuda(args.gpu_ids[0])
                #print(images)
                #print(images.shape)
                #print(type(images))

                images_att = net_attack(images_adv,labels, attack)


                model.train()
                #optimizer.zero_grad()
                # prediction, weights = model(images_att)
                prediction, predicts = model(images_att)

                loss = CE_loss(prediction, labels)

                # losses = []
                #
                # losses.append(CE_loss(predicts[0], labels))
                #
                # losses.append(CE_loss(predicts[1], labels))
                #
                # losses.append(CE_loss(predicts[2], labels))
                #
                # losses.append(CE_loss(predicts[3], labels))
                #
                # loss = max(losses)
                # loss = sum(losses)/4

                #print('prediction value is :', prediction )


                lr = lr_schedule(epoch_i + (b+1)/len(train_loader))
                optimizer.param_groups[0].update(lr=lr)

                optimizer.zero_grad()
                # print('loss value is :', loss )
                loss.backward()

                optimizer.step()

                b = b+1

                all_loss = all_loss + loss

        aver_loss = all_loss / len(train_loader.dataset)

    return aver_loss



def val(val_loader, device, model,  basic_model, AttackPGD, config_l1, config_l2, config_linf, attack,\
        correct_final_nat, best_acc_nat, correct_final_l1, best_acc_l1, correct_final_l2, best_acc_l2, correct_final_linf,\
            best_acc_linf, best_acc_aver, checkpoint_loc):

    acc_linf = []
    acc_l2 = []
    acc_l1 =[]

    print('Valuation clean images')

    for images_1, labels in tqdm(val_loader):


            images_1 = images_1.to(device)
            labels = labels.to(device)

            #images_att = net_attack(images,labels, attack)

            # prediction, weights_nat = model(images_1)
            prediction, _ = model(images_1)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_1 = pred.eq(labels.view_as(pred)).sum().item()
            #correct_final.append(correct)
            correct_final_nat = correct_final_nat + correct_1

    acc_nat = correct_final_nat / len(val_loader.dataset)

    j=0

    for i in config_l1:
        j = j + 1
        # print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
        print('Valuation l_1  epsilon: 50+ ' + str(j) + '*10 / 255')

        correct_1 = 0
        correct_final_l1 = 0

        for images_0, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD(basic_model, config_l1[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_0= images_0.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_0, labels, attack)

            # prediction,weights_l2 = model(images_att)
            prediction, _ = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_0 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_l1 = correct_final_l1 + correct_0

        acc_l1.append(correct_final_l1 / len(val_loader.dataset))

    for i in config_l2:
        j = j + 1
        #print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
        print('Valuation l_2  epsilon: 50+ ' + str(j) + '*10 / 255')

        correct_2 = 0
        correct_final_l2 = 0

        for images_2, labels in tqdm(val_loader):

                #a = a+1
                net_attack = AttackPGD(basic_model, config_l2[i])

                #print(net_attack)

                net_attack = net_attack.to(device)
                #print('processing testing image:', a)
                #images, labels = utils.cuda([images, labels], args.gpu_ids)
                images_2 = images_2.to(device)
                labels = labels.to(device)

                images_att = net_attack(images_2,labels, attack)

                # prediction,weights_l2 = model(images_att)
                prediction, _ = model(images_att)
                pred = prediction.argmax(dim=1, keepdim=True)
                correct_2 = pred.eq(labels.view_as(pred)).sum().item()
                #correct_final.append(correct)
                correct_final_l2 = correct_final_l2 + correct_2

        acc_l2.append(correct_final_l2 / len(val_loader.dataset))

    j = 0
    for i in config_linf:

        j = j+1

        #print('Valuation' + str(config_linf[i]['_type']) + '  epsilon:' + str(config_linf[i]['epsilon']))
        print('Valuation l_inf  epsilon: 5+ ' + str(j) + '*10 / 255')
        correct_2 = 0
        correct_final_linf = 0
        for images_3, labels in tqdm(val_loader):

                #a = a+1
                net_attack = AttackPGD(basic_model,config_linf[i])

                #print(net_attack)

                net_attack = net_attack.to(device)
                #print('processing testing image:', a)
                #images, labels = utils.cuda([images, labels], args.gpu_ids)
                images_3 = images_3.to(device)
                labels = labels.to(device)

                images_att = net_attack(images_3,labels, attack)

                prediction, _ = model(images_att)
                # prediction, weights_linf = model(images_att)
                pred = prediction.argmax(dim=1, keepdim=True)
                correct_3 = pred.eq(labels.view_as(pred)).sum().item()
                #correct_final.append(correct)
                correct_final_linf = correct_final_linf + correct_3

        acc_linf.append(correct_final_linf / len(val_loader.dataset))




    if (acc_nat + sum(acc_l2) + sum(acc_l2) + sum(acc_linf))/4 > best_acc_aver:
            print('saving..')

            state = {
            'net': model.state_dict(),
            'acc_clean': acc_nat,
            'acc_l1': acc_l1,
            'acc_l2': acc_l2,
            'acc_linf': acc_linf,
            #'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_loc)
            best_acc_nat = acc_nat
            best_acc_l1 = acc_l1
            best_acc_l2 = acc_l2
            best_acc_linf = acc_linf
            best_acc_aver = (best_acc_nat + sum(best_acc_l1) + sum(best_acc_l2) + sum(best_acc_linf))/4

    return acc_nat, best_acc_nat, acc_l1, best_acc_l1, acc_l2, best_acc_l2, acc_linf, best_acc_linf, best_acc_aver


def testing(val_loader, device, model, basic_model, AttackPGD, config_l2, config_linf, attack, \
        correct_final_nat, best_acc_nat, correct_final_l2, best_acc_l2, correct_final_linf, \
        best_acc_linf, best_acc_aver, checkpoint_loc, transform_rt):
    acc_linf = []
    acc_l2 = []

    print('Valuation clean images')

    for images_1, labels in tqdm(val_loader):
        images_1 = images_1.to(device)
        labels = labels.to(device)

        # images_att = net_attack(images,labels, attack)

        images_rt = transform_rt['trans'](images_1)

        prediction = model(images_rt)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct_1 = pred.eq(labels.view_as(pred)).sum().item()
        # correct_final.append(correct)
        correct_final_nat = correct_final_nat + correct_1

    acc_nat = correct_final_nat / len(val_loader.dataset)

    j = 0

    for i in config_l2:
        j = j + 1
        # print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
        print('Valuation l_2  epsilon: 50+ ' + str(j) + '*10 / 255')

        correct_2 = 0
        correct_final_l2 = 0

        for images_2, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD(basic_model, config_l2[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_2 = images_2.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_2, labels, attack)

            prediction = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_2 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_l2 = correct_final_l2 + correct_2

        acc_l2.append(correct_final_l2 / len(val_loader.dataset))

    j = 0
    for i in config_linf:

        j = j + 1

        # print('Valuation' + str(config_linf[i]['_type']) + '  epsilon:' + str(config_linf[i]['epsilon']))
        print('Valuation l_inf  epsilon: 5+ ' + str(j) + '*10 / 255')
        correct_2 = 0
        correct_final_linf = 0
        for images_3, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD(basic_model, config_linf[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_3 = images_3.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_3, labels, attack)

            prediction = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_3 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_linf = correct_final_linf + correct_3

        acc_linf.append(correct_final_linf / len(val_loader.dataset))

    if (acc_nat * 2 + sum(acc_l2) + sum(acc_linf)) / 6 > best_acc_aver:
        print('updating..')
        best_acc_nat = acc_nat
        best_acc_l2 = acc_l2
        best_acc_linf = acc_linf
        best_acc_aver = (best_acc_nat * 2 + sum(best_acc_l2) + sum(best_acc_linf)) / 6

    return acc_nat, best_acc_nat, acc_l2, best_acc_l2, acc_linf, best_acc_linf, best_acc_aver


def testing_deepfool(val_loader, device, model, basic_model, use_norm, use_epsilon):

    acc = []

    j = 0

    # for i in config_l2:
    #     j = j + 1
        # print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
    print('Valuation deepfool')

    correct = 0
    correct_final = 0

    for images_2, labels in tqdm(val_loader):
        # a = a+1
        _, _, _, _, pert_image = deepfool(images_2, basic_model, norm=use_norm, eps=use_epsilon)

        # print(net_attack)

        # net_attack = net_attack.to(device)
        # print('processing testing image:', a)
        # images, labels = utils.cuda([images, labels], args.gpu_ids)
        pert_image = pert_image.to(device)
        labels = labels.to(device)

        # images_att = net_attack(pert_image, labels, attack)

        prediction = model(pert_image)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        # correct_final.append(correct)
        correct_final = correct_final + correct

    acc = correct_final / len(val_loader.dataset)



    # if (acc_nat * 2 + sum(acc_l2) + sum(acc_linf)) / 6 > best_acc_aver:
    #     print('updating..')
    #     best_acc_nat = acc_nat
    #     best_acc_l2 = acc_l2
    #     best_acc_linf = acc_linf
    #     best_acc_aver = (best_acc_nat * 2 + sum(best_acc_l2) + sum(best_acc_linf)) / 6

    return acc

def train(args):



    # config_linf_6 = {
    #     'epsilon': 6.0 / 255,
    #     'num_steps': 7,
    #     # 'step_size': 2.0 / 255,
    #     'step_size': 0.01,
    #     'random_start': True,
    #     'loss_func': 'xent',
    #     '_type': 'linf'
    # }
    # config_linf_8 = {
    #     'epsilon': 8.0 / 255,
    #     'num_steps': 7,
    #     # 'step_size': 2.0 / 255,
    #     'step_size': 0.01,
    #     'random_start': True,
    #     'loss_func': 'xent',
    #     '_type': 'linf'
    # }
    #
    # config_linf = dict(config_linf_6=config_linf_6, config_linf_8=config_linf_8)
    #
    # config_l2_1_2 = {
    #     'epsilon': 0.5,
    #     'num_steps': 7,
    #     # 'step_size': 2.0 / 255,
    #     'step_size': 0.5 / 5,
    #     'random_start': True,
    #     'loss_func': 'xent',
    #     '_type': 'l2'
    # }
    # config_l2_1 = {
    #     'epsilon': 1.0,
    #     'num_steps': 7,
    #     # 'step_size': 2.0 / 255,
    #     'step_size': 1.0 / 5,
    #     'random_start': True,
    #     'loss_func': 'xent',
    #     '_type': 'l2'
    # }
    #
    # config_l2 = dict(config_l2_1_2=config_l2_1_2, config_l2_60=config_l2_1)

# single train

    config_linf_6 = {
        'epsilon': 6.0 / 255,
        'num_steps': 20,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }
    # config_linf_8 = {
    #     'epsilon': 8.0 / 255,
    #     'num_steps': 20,
    #     # 'step_size': 2.0 / 255,
    #     'step_size': 0.01,
    #     'random_start': True,
    #     'loss_func': 'xent',
    #     '_type': 'linf'
    # }

    config_linf_8 = {
        'epsilon': 8.0 / 255,
        'num_steps': 7,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }

    # config_linf = dict(config_linf_6=config_linf_6, config_linf_8=config_linf_8)

    config_linf = dict(config_linf_8=config_linf_8)


    config_l2_1_2 = {
        'epsilon': 0.5,
        'num_steps': 20,
        # 'step_size': 2.0 / 255,
        'step_size': 0.5 / 5,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l2'
    }
    # config_l2_1 = {
    #     'epsilon': 1.0,
    #     'num_steps': 20,
    #     # 'step_size': 2.0 / 255,
    #     'step_size': 1.0 / 5,
    #     'random_start': True,
    #     'loss_func': 'xent',
    #     '_type': 'l2'
    # }

    config_l2_1 = {
        'epsilon': 1.0,
        'num_steps': 7,
        # 'step_size': 2.0 / 255,
        'step_size': 2.5 * 1.0 / 7,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l2'
    }

    # config_l2 = dict(config_l2_1_2=config_l2_1_2, config_l2_1=config_l2_1)

    config_l2 = dict(config_l2_1=config_l2_1)

    # config_l1_16 = {
    #     'epsilon': 16.0,
    #     'num_steps': 20,
    #     # 'step_size': 2.0 / 255,
    #     'step_size': 2.5 * 16.0 / 20,
    #     'random_start': True,
    #     'loss_func': 'xent',
    #     '_type': 'l1'
    # }
    config_l1_16 = {
        'epsilon': 16.0,
        'num_steps': 7,
        # 'step_size': 2.0 / 255,
        'step_size': 2.5 * 16.0 / 7,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l1'
    }

    config_l1 = dict(config_l1_16=config_l1_16)
    attack = 'true'

    if args.dataset == 'cifar':
        output_classes = 10
    if args.dataset == 'tinyimagenet':
        output_classes = 200


    global best_acc_nat, best_acc_l2, best_acc_linf
    best_acc_nat = 0
    best_acc_l1 = []
    best_acc_l2 = []
    best_acc_linf = []
    best_acc_aver = 0


    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)

    # operate this train_loader to generate new loader

    train_loader = DataLoader(dataset['train_data'], batch_size = args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset['test_data'], batch_size = args.batch_size, shuffle=True)

    # l2 attack and linf attack to generate new data

    # for images, labels in train_loader:
    #     print(images)
    #     print(images.type)
    #     print(images.shape)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    CE_loss = nn.CrossEntropyLoss()
    # jiang LeNet dan du chan fen le chu lai
    #model =  LeNet(output_classes)
    basic_model =  ResNet18(output_classes)
    basic_model =  basic_model.to(device)
    #print(output_classes.device)
    #model = MoE(basic_model, output_classes, output_classes)
    # model = MoE_ResNet18(args.num_experts, output_classes)
    if args.method == 'more':
       model = MoE_ResNet18_adv(args.num_experts, output_classes)
       print('tiny more model check')
    elif args.method == 'base':
       model = MoE_ResNet18_base(args.num_experts, output_classes)
       print('tiny base model check')




    model = model.to(device)




    if args.resume == False:
        if args.dataset == 'cifar':
            # model_loc0 = '../cifar_clean_train_resnet/checkpoint/ckptnat_resnet18_cifar_tsave.pth'
            # model_loc0 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_clean.pth'
            model_loc0 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_clean_at.pth'


            # model_loc1 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_linf_6_255_at_change.pth'
            model_loc1 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_linf_8_255_at_change.pth'
            # model_loc3 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_l2_1_2_at_change.pth'
            model_loc2 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_l2_1_at_change.pth'
            model_loc3 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_l1_16_at_change_26.pth'
        elif args.dataset == 'tinyimagenet':
            # model_loc0 = '../cifar_clean_train_resnet/checkpoint/ckptnat_resnet18_cifar_tsave.pth'
            # model_loc0 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_clean.pth'
            model_loc0 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_clean.pth'

            # model_loc1 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_linf_6_255_at_change.pth'
            model_loc1 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_linf_8.pt'
            # model_loc3 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_l2_1_2_at_change.pth'
            model_loc2 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_l2_1.0.pt'
            model_loc3 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_l1_12.0.pt'

        utils.load_model(model_loc0, model.experts[0])
        utils.load_model(model_loc1, model.experts[1])
        utils.load_model(model_loc2, model.experts[2])
        utils.load_model(model_loc3, model.experts[3])
    else:
        print('load model here')
        utils.load_model(args.checkpoint_loc, model)
        print(args.checkpoint_loc)
    # model_loc1 = '../cifar_clean_train_resnet/checkpoint/linf_6.pt'
    # model_loc2 = '../cifar_clean_train_resnet/checkpoint/linf_8.pt'
    # model_loc3 = '../cifar_clean_train_resnet/checkpoint/l2_0.5.pt'
    print(model_loc0)
    print(model_loc1)
    print(model_loc3)



    #utils.load_model(model_loc1, model)
    # key_word = 'experts'

    # for name,value in model.named_parameters():
    #     if key_word in name:
    #         value.requires_grad = False

    #print(model.device)

    #print(basic_model.device)

    #model = MoE(args.num_experts, output_classes)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True


    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [0, 0.1, 0.005, 0])[0]

    print(config_l1)
    print(config_l2)
    print(config_linf)


    for i in range(args.epochs):
        model.train()
        #j = 0
        print('The epoch number is: ' + str(i))




        if args.method == 'more':
            print('update loss value')

            # loss_linf, loss_linf_aver = train_adv(train_loader, device, optimizer, model, model, AttackPGD, CE_loss, config_linf, attack,
            #                          lr_schedule, i)
            #
            # print('linf loss is:', loss_linf, 'avg linf loss is:', loss_linf_aver)
            #
            # loss_l2, loss_l2_aver = train_adv(train_loader, device, optimizer, model, model, AttackPGD, CE_loss, config_l2, attack,
            #                        lr_schedule, i)
            #
            # print('l2 loss is:', loss_l2, 'avg l2 loss is:', loss_l2_aver)
            #
            # loss_l1, loss_l1_aver = train_adv(train_loader, device, optimizer, model, model, AttackPGD, CE_loss, config_l1, attack,
            #                        lr_schedule, i)
            #
            # print('l1 loss is:', loss_l1, 'avg l1 loss is:', loss_l1_aver)

            loss_linf = train_adv_base(train_loader, device, optimizer, model, model, AttackPGD, CE_loss,
                                                  config_linf, attack,
                                                  lr_schedule, i)

            print('linf loss is:', loss_linf)

            loss_l2 = train_adv_base(train_loader, device, optimizer, model, model, AttackPGD, CE_loss, config_l2,
                                              attack,
                                              lr_schedule, i)

            print('l2 loss is:', loss_l2)

            loss_l1 = train_adv_base(train_loader, device, optimizer, model, model, AttackPGD, CE_loss, config_l1,
                                              attack,
                                              lr_schedule, i)

            print('l1 loss is:', loss_l1)
        elif args.method == 'base':

            print('update base loss value')

            loss_linf = train_adv_base(train_loader, device, optimizer, model, model, AttackPGD, CE_loss,
                                                  config_linf, attack,
                                                  lr_schedule, i)

            print('linf loss is:', loss_linf)

            loss_l2 = train_adv_base(train_loader, device, optimizer, model, model, AttackPGD, CE_loss, config_l2,
                                              attack,
                                              lr_schedule, i)

            print('l2 loss is:', loss_l2)

            loss_l1 = train_adv_base(train_loader, device, optimizer, model, model, AttackPGD, CE_loss, config_l1,
                                              attack,
                                              lr_schedule, i)

            print('l1 loss is:', loss_l1)


        

        model.eval()
        correct_final_nat = 0
        correct_final_l1 = 0
        correct_final_l2 = 0
        correct_final_linf = 0



        acc_nat, best_acc_nat, acc_l1, best_acc_l1, acc_l2, best_acc_l2, acc_linf, best_acc_linf, best_acc_aver = val(test_loader, device, model,  model, AttackPGD, config_l1, config_l2, config_linf, attack,\
        correct_final_nat, best_acc_nat, correct_final_l1, best_acc_l1, correct_final_l2, best_acc_l2, correct_final_linf,\
            best_acc_linf, best_acc_aver, args.checkpoint_loc)

        #acc_1, best_acc_nat = val_clean(val_loader, device, model, correct_final_1, best_acc_nat, args.checkpoint_loc)

        print('Epoch: ', i+1, ' Done!!  Natural  Accuracy: ', acc_nat)
        print('Epoch: ', i+1, '  Best Natural  Accuracy: ', best_acc_nat)

        print('Epoch: ', i+1, ' Done!!  l1(50, ..., 110)  Accuracy: ', acc_l1)
        print('Epoch: ', i+1, '  Best l1  Accuracy: ', best_acc_l1)


        #acc_2, best_acc_l2 = val_adv(val_loader, device, model,  model, AttackPGD, config_l2, attack, correct_final_2, best_acc_l2 ,args.checkpoint_loc)

        print('Epoch: ', i+1, ' Done!!  l2(50, ..., 110)  Accuracy: ', acc_l2)
        print('Epoch: ', i+1, '  Best l2  Accuracy: ', best_acc_l2)

        #acc_3, best_acc_linf = val_adv(val_loader, device, model,  basic_model, AttackPGD, config_linf, attack, correct_final_3, best_acc_linf ,args.checkpoint_loc)



        print('Epoch: ', i+1, ' Done!!  l_inf(5, ..., 11)  Accuracy: ', acc_linf)
        print('Epoch: ', i+1, '  Best l_inf  Accuracy: ', best_acc_linf)

        #print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

        print('Epoch: ', i+1, ' Done!!  average Accuracy: ', (acc_nat + sum(acc_l1) + sum(acc_l2) + sum(acc_linf))/4 )
        print('Epoch: ', i+1, '  Best average  Accuracy: ', best_acc_aver)

        print('./tiny_base_adv_true.txt')

        with open('./tiny_base_adv_true.txt', 'at') as f:
            if args.method == 'more':
                # print('epoch', i + 1, 'linf loss is:', loss_linf, 'avg linf loss is:', loss_linf_aver, file=f)
                #
                # print('epoch', i + 1, 'l2 loss is:', loss_l2, 'avg l2 loss is:', loss_l2_aver, file=f)
                #
                # print('epoch', i + 1, 'l1 loss is:', loss_l1, 'avg l1 loss is:', loss_l1_aver, file=f)
                print('epoch', i + 1, 'linf loss is:', loss_linf, file=f)

                print('epoch', i + 1, 'l2 loss is:', loss_l2, file=f)

                print('epoch', i + 1, 'l1 loss is:', loss_l1, file=f)

            elif args.method == 'base':
                print('epoch', i + 1, 'linf loss is:', loss_linf, file=f)

                print('epoch', i + 1, 'l2 loss is:', loss_l2, file=f)

                print('epoch', i + 1, 'l1 loss is:', loss_l1, file=f)

            print('Epoch: ', i + 1, ' Done!!  Natural  Accuracy: ', acc_nat, file=f)
            print('Epoch: ', i + 1, '  Best Natural  Accuracy: ', best_acc_nat, file=f)


            print('Epoch: ', i + 1, ' Done!!  l1(50, ..., 110)  Accuracy: ', acc_l1, file=f)
            print('Epoch: ', i + 1, '  Best l1  Accuracy: ', best_acc_l1, file=f)

            print('Epoch: ', i + 1, ' Done!!  l2(50, ..., 110)  Accuracy: ', acc_l2, file=f)
            print('Epoch: ', i + 1, '  Best l2  Accuracy: ', best_acc_l2, file=f)

            print('Epoch: ', i + 1, ' Done!!  l_inf(5, ..., 11)  Accuracy: ', acc_linf, file=f)
            print('Epoch: ', i + 1, '  Best l_inf  Accuracy: ', best_acc_linf, file=f)


            # print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

            # print('Epoch: ', i+1, ' Done!!  average Accuracy: ', (acc_nat_1 + acc_nat_2 + sum(acc_l2) + sum(acc_linf) + sum(acc_fog) + sum(acc_snow))/10 )
            print('Epoch: ', i + 1, '  Best average  Accuracy: ', best_acc_aver, file=f)

      

def test(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    config_linf_6 = {
        'epsilon': 6.0 / 255,
        'num_steps': 20,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }
    config_linf_8 = {
        'epsilon': 8.0 / 255,
        'num_steps': 20,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }

    config_linf = dict( config_linf_8 = config_linf_8)


    config_l2_1_2 = {
        'epsilon': 0.5,
        'num_steps': 20,
        # 'step_size': 2.0 / 255,
        'step_size': 0.5 / 5,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l2'
    }
    config_l2_1 = {
        'epsilon': 1.0,
        'num_steps': 20,
        # 'step_size': 2.0 / 255,
        # 'step_size': 1.0 / 5,
        'step_size': 1.0 / 5,
        'random_start': False,
        'loss_func': 'xent',
        '_type': 'l2'
    }


    config_l2 = dict(config_l2_1 = config_l2_1)


    config_l1 = {
    'epsilon': 8,
    'num_steps': 20,
    # 'step_size': 2.0 / 255,
    'step_size': 2.5 * 8 / 20,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'l1'
    }

    # config_l1 = {
    # 'epsilon': 12,
    # 'num_steps': 20,
    # # 'step_size': 2.0 / 255,
    # 'step_size': 0.05,
    # 'random_start': True,
    # 'loss_func': 'xent',
    # '_type': 'l1'
    # }

    config_l1 = dict(config_l1 = config_l1)

    print(config_l1)
    attack = 'true'

    if args.dataset == 'cifar':
        output_classes = 10

    global best_acc_nat, best_acc_l2, best_acc_linf
    best_acc_nat = 0
    best_acc_l2 = []
    best_acc_linf = []
    best_acc_aver = 0

    transform = utils.get_transformation(args.dataset)
    transform_rt = utils.get_transformation_rotation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)

    # operate this train_loader to generate new loader

    test_loader = DataLoader(dataset['test_data'], batch_size=args.batch_size, shuffle=True)

    # l2 attack and linf attack to generate new data

    # for images, labels in train_loader:
    #     print(images)
    #     print(images.type)
    #     print(images.shape)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    CE_loss = nn.CrossEntropyLoss()
    

    if args.usemodel == 'more':
        model = MoE_ResNet18_adv(args.num_experts, output_classes)
    elif args.usemodel == 'expert':
        model = ResNet18(output_classes)
    # model_loc = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_linf_8_255.pth'
    # model_loc = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_l2_1_2.pth'

    model = model.to(device)
    utils.load_model(args.checkpoint_loc, model)
    # utils.load_model(model_loc, model)




    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    
    lr_schedule = lambda t: \
    np.interp([t], [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs], [0, 0.1, 0.005, 0])[0]

    for i in range(args.epochs):
        # model.train()
        # j = 0
        print('The epoch number is: ' + str(i))
        print(config_l1)

        model.eval()
        correct_final_nat = 0
        correct_final_l1 = 0
        correct_final_l2 = 0
        correct_final_linf = 0



        acc_nat, acc_l1, acc_l2, acc_linf  = val_adv_expert(test_loader, device, model, model, AttackPGD, config_l1, config_l2, config_linf, attack, correct_final_nat, correct_final_l1, correct_final_l2,  correct_final_linf,  args.checkpoint_loc)


        # acc_1, best_acc_nat = val_clean(val_loader, device, model, correct_final_1, best_acc_nat, args.checkpoint_loc)

        print('Epoch: ', i + 1, ' Done!!  Natural  Accuracy: ', acc_nat)
        # print('Epoch: ', i + 1, '  Best Natural  Accuracy: ', best_acc_nat)

        # acc_2, best_acc_l2 = val_adv(val_loader, device, model,  model, AttackPGD, config_l2, attack, correct_final_2, best_acc_l2 ,args.checkpoint_loc)

        print('Epoch: ', i + 1, ' Done!!  l1(50, ..., 110)  Accuracy: ', acc_l1)
        # print('Epoch: ', i + 1, '  Best l2  Accuracy: ', best_acc_l2)

        print('Epoch: ', i + 1, ' Done!!  l1(50, ..., 110)  Accuracy: ', acc_l2)
        # print('Epoch: ', i + 1, '  Best l1  Accuracy: ', best_acc_l2)

        # acc_3, best_acc_linf = val_adv(val_loader, device, model,  basic_model, AttackPGD, config_linf, attack, correct_final_3, best_acc_linf ,args.checkpoint_loc)

        print('Epoch: ', i + 1, ' Done!!  l_inf(5, ..., 11)  Accuracy: ', acc_linf)
        # print('Epoch: ', i + 1, '  Best l_inf  Accuracy: ', best_acc_linf)

        # print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

        print('Epoch: ', i + 1, ' Done!!  average Accuracy: ', (acc_nat * 2 + sum(acc_l2) + sum(acc_linf)) / 6)
        print('Epoch: ', i + 1, '  Best average  Accuracy: ', best_acc_aver)

def test_deepfool(args):

    # attack = 'true'

    # if args.dataset == 'cifar':
    #     output_classes = 10

    output_classes = 200
    global best_acc_nat, best_acc_l2, best_acc_linf
    best_acc_nat = 0
    best_acc_l2 = []
    best_acc_linf = []
    best_acc_aver = 0

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)

    # operate this train_loader to generate new loader

    test_loader = DataLoader(dataset['test_data'], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset['val_data'], batch_size=args.batch_size, shuffle=True)

    # l2 attack and linf attack to generate new data

    # for images, labels in train_loader:
    #     print(images)
    #     print(images.type)
    #     print(images.shape)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    CE_loss = nn.CrossEntropyLoss()
    # jiang LeNet dan du chan fen le chu lai
    # model =  LeNet(output_classes)
    # model = ResNet18(output_classes)
    # model = PreActResNet18()
    # basic_model = basic_model.to(device)
    # print(output_classes.device)
    # model = MoE_ResNet18_adv(args.num_experts, output_classes)
    if args.usemodel == 'more':
        model = MoE_ResNet18_adv(args.num_experts, output_classes)
    elif args.usemodel == 'expert':
        model = ResNet18(200)

    # model_adv = ResNet18(output_classes)

    # model_loc = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_clean.pth'
    # model_loc = '../cifar_clean_train_resnet/checkpoint/linf_8.pt'

    # model_loc = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_ensemble_aver_adv.pth'
    # model_loc = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_ensemble_aver_adv.pth'
    # model_loc = '../cifar_clean_train_resnet/checkpoint/ckptnat_resnet18_cifar_ensemble_max_adv.pth'
    # model_loc = '../cifar_clean_train_resnet/MSD/Final/MSD_V0/iter_25.pt'

    # model_loc = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_linf_8_255.pth'
    # model_loc = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_clean.pth'

    model = model.to(device)
    # utils.load_model(model_loc, model)




    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True


    utils.load_model(args.checkpoint_loc, model)

    # model.load_state_dict(torch.load(args.checkpoint_loc))


    # model = utils.cuda(model, args.gpu_ids)
    # model.cuda(args.gpu_ids[0])
    # model = model.to(device)
    # model = torch.nn.DataParallel(model, device_ids = [0]).cuda()
    # print(model.device)
    # model = MoE(basic_model, args.num_experts, output_classes)
    # print(model)
    # print(type(model))
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_schedule = lambda t: \
    np.interp([t], [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs], [0, 0.1, 0.005, 0])[0]

    for i in range(args.epochs):
        # model.train()
        # j = 0
        print('The epoch number is: ' + str(i))

        model.eval()
        correct_final_nat = 0
        correct_final_l2 = 0
        correct_final_linf = 0

        acc_deepfool_linf = testing_deepfool(val_loader, device, model, model, args.norm_linf, args.epsilon_linf)

        print('Epoch: ', i + 1, ' Done!!  linf deepfool  Accuracy: ', acc_deepfool_linf)

        acc_deepfool_l2 = testing_deepfool(val_loader, device, model, model, args.norm_l2, args.epsilon_l2)

        print('Epoch: ', i + 1, ' Done!!  l2 deepfool  Accuracy: ', acc_deepfool_l2)



def val_adv(val_loader, device, model, basic_model, AttackPGD, config_l1, config_l2, config_linf, attack, \
            correct_final_nat, correct_final_l1, correct_final_l2, correct_final_linf, \
            checkpoint_loc):
    acc_linf = []
    acc_l2 = []
    acc_l1 = []

    print('Valuation clean images')

    for images_0, labels in tqdm(val_loader):
        images_0 = images_0.to(device)
        labels = labels.to(device)

        # images_att = net_attack(images,labels, attack)

        prediction, _ = model(images_0)
        # prediction = model(images_1)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct_0 = pred.eq(labels.view_as(pred)).sum().item()
        # correct_final.append(correct)
        correct_final_nat = correct_final_nat + correct_0

    acc_nat = correct_final_nat / len(val_loader.dataset)

    j = 0

    for i in config_l1:
        j = j + 1
        # print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
        print('Valuation l_1  epsilon: 50+ ' + str(j) + '*10 / 255')

        correct_1 = 0
        correct_final_l1 = 0

        for images_1, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD(basic_model, config_l1[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_1 = images_1.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_1, labels, attack)

            prediction, _ = model(images_att)
            # prediction = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_1 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_l1 = correct_final_l1 + correct_1

        acc_l1.append(correct_final_l1 / len(val_loader.dataset))

    for i in config_l2:
        j = j + 1
        # print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
        print('Valuation l_2  epsilon: 50+ ' + str(j) + '*10 / 255')

        correct_2 = 0
        correct_final_l2 = 0

        for images_2, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD(basic_model, config_l2[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_2 = images_2.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_2, labels, attack)

            prediction, _ = model(images_att)
            # prediction = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_2 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_l2 = correct_final_l2 + correct_2

        acc_l2.append(correct_final_l2 / len(val_loader.dataset))

    j = 0
    for i in config_linf:

        j = j + 1

        # print('Valuation' + str(config_linf[i]['_type']) + '  epsilon:' + str(config_linf[i]['epsilon']))
        print('Valuation l_inf  epsilon: 5+ ' + str(j) + '*10 / 255')
        correct_2 = 0
        correct_final_linf = 0
        for images_3, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD(basic_model, config_linf[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_3 = images_3.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_3, labels, attack)

            prediction, _ = model(images_att)
            # prediction = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_3 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_linf = correct_final_linf + correct_3

        acc_linf.append(correct_final_linf / len(val_loader.dataset))

  

    return acc_nat, acc_l1, acc_l2, acc_linf

def val_adv_expert(val_loader, device, model, basic_model, AttackPGD, config_l1, config_l2, config_linf, attack, \
            correct_final_nat, correct_final_l1, correct_final_l2, correct_final_linf, \
            checkpoint_loc):
    acc_linf = []
    acc_l2 = []
    acc_l1 = []

    print('Valuation clean images')

    for images_0, labels in tqdm(val_loader):
        images_0 = images_0.to(device)
        labels = labels.to(device)

        # images_att = net_attack(images,labels, attack)

        prediction = model(images_0)
        # prediction = model(images_1)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct_0 = pred.eq(labels.view_as(pred)).sum().item()
        # correct_final.append(correct)
        correct_final_nat = correct_final_nat + correct_0

    acc_nat = correct_final_nat / len(val_loader.dataset)

    j = 0

    for i in config_l1:
        j = j + 1
        # print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
        print('Valuation l_1  epsilon: 50+ ' + str(j) + '*10 / 255')

        correct_1 = 0
        correct_final_l1 = 0

        for images_1, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD_expert(basic_model, config_l1[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_1 = images_1.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_1, labels, attack)

            prediction = model(images_att)
            # prediction = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_1 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_l1 = correct_final_l1 + correct_1

        acc_l1.append(correct_final_l1 / len(val_loader.dataset))

    for i in config_l2:
        j = j + 1
        # print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
        print('Valuation l_2  epsilon: 50+ ' + str(j) + '*10 / 255')

        correct_2 = 0
        correct_final_l2 = 0

        for images_2, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD_expert(basic_model, config_l2[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_2 = images_2.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_2, labels, attack)

            prediction = model(images_att)
            # prediction = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_2 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_l2 = correct_final_l2 + correct_2

        acc_l2.append(correct_final_l2 / len(val_loader.dataset))

    j = 0
    for i in config_linf:

        j = j + 1

        # print('Valuation' + str(config_linf[i]['_type']) + '  epsilon:' + str(config_linf[i]['epsilon']))
        print('Valuation l_inf  epsilon: 5+ ' + str(j) + '*10 / 255')
        correct_2 = 0
        correct_final_linf = 0
        for images_3, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD_expert(basic_model, config_linf[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_3 = images_3.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_3, labels, attack)

            prediction = model(images_att)
            # prediction = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_3 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_linf = correct_final_linf + correct_3

        acc_linf.append(correct_final_linf / len(val_loader.dataset))

   

    return acc_nat, acc_l1, acc_l2, acc_linf