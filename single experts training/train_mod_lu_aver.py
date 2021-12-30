

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
import torch.backends.cudnn as cudnn

import utils
from model_adv import AlexNet, MoE_alexnet
from model_adv_att import AttackPGD
from model_resnet import *
import string
from weather_generation import *


def train_clean(images, labels, device, optimizer, model, CE_loss, lr_schedule, epoch_i):
    # a = 0
    j = 0
    loss_all = []
    for j in range(1, 2):
        #print('Doing clean images training No. ' + str(j))
        #for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # optimizer.zero_grad()
        prediction = model(images)

        # print('prediction value is :', prediction )

        loss = CE_loss(prediction, labels)
        # print('loss value is :', loss )
        loss_all.append(loss)

        # lr = lr_schedule(epoch_i + (a + 1) / len(train_loader))
        # optimizer.param_groups[0].update(lr=lr)
        #
        # optimizer.zero_grad()
        #
        # loss.backward()
        #
        # optimizer.step()

        # a = a + 1

    return loss_all


def train_fog(images, labels, device, optimizer, model, CE_loss, config_fog, lr_schedule, epoch_i):
    c = 0
    loss_all = []
    #print('training snow step')

    j = 0

    for i in config_fog:
        j = j + 1

        # b = 0

        #print('fog Training ' + str(config_fog[i]['t']) + ' t:' + str(j))

        #for images, labels in tqdm(train_loader):

        # b = b+1

        # print('training ' + str(b) + ' batch')

        images = images.to(device)
        labels = labels.to(device)

        for a in range(0, images.shape[0]):
            # print(images[i])
            images_fog = add_fog(images[a], config_fog[i]['t'], config_fog[i]['light'])
            # print(images_fog)
            images[a] = images_fog
            # add fog to images

        prediction = model(images)

        # print('prediction value is :', prediction )

        loss = CE_loss(prediction, labels)

        loss_all.append(loss)

        #
        # lr = lr_schedule(epoch_i + (c + 1) / len(train_loader))
        # optimizer.param_groups[0].update(lr=lr)
        #
        # # print('loss value is :', loss )
        # optimizer.zero_grad()
        # loss.backward()
        #
        # optimizer.step()

        c = c + 1

    return loss_all


def train_snow(images, labels, device, optimizer, model, CE_loss, config_snow, lr_schedule, epoch_i):
    b = 0
    loss_all = []
    #print('training snow step')

    j = 0

    for i in config_snow:

        j = j + 1
        #print('snow Training ' + str(config_snow[i]) + ' No.' + str(j))

        b = 0

        #for images, labels in tqdm(train_loader):

        b = b + 1

        # print('training ' + str(b) + ' batch')

        # for a in range(0,images.shape[0]):

        #        # print(images[i])
        #         images_snow = add_snow(images[a], config_snow[i])
        #         #print(images_fog)
        #         images[a] = images_snow

        images = images.to(device)
        labels = labels.to(device)

        for a in range(0, images.shape[0]):
            # print(images[i])
            images_snow = add_snow(images[a], config_snow[i])
            # print(images_fog)
            images[a] = images_snow
        # add fog to images

        prediction = model(images)

        # print('prediction value is :', prediction )

        loss = CE_loss(prediction, labels)

        loss_all.append(loss)
        # print('loss value is :', loss )

        # lr = lr_schedule(epoch_i + (b + 1) / len(train_loader))
        # optimizer.param_groups[0].update(lr=lr)
        #
        # optimizer.zero_grad()
        #
        # loss.backward()
        #
        # optimizer.step()

        b = b + 1

    return loss_all

def train_rt(images, labels, device, optimizer, model, CE_loss, lr_schedule, transform_rt):
    # a = 0
    j = 0
    loss_all = []
    for j in range(1, 2):
        #print('Doing clean images training No. ' + str(j))
        #for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # optimizer.zero_grad()

        images_rt = transform_rt['trans'](images)


        prediction = model(images_rt)

        # print('prediction value is :', prediction )

        loss = CE_loss(prediction, labels)
        # print('loss value is :', loss )
        loss_all.append(loss)

        # lr = lr_schedule(epoch_i + (a + 1) / len(train_loader))
        # optimizer.param_groups[0].update(lr=lr)
        #
        # optimizer.zero_grad()
        #
        # loss.backward()
        #
        # optimizer.step()

        # a = a + 1

    return loss_all

def train_adv(images_adv, labels, device, optimizer, basic_model, model, AttackPGD, CE_loss, config, attack, lr_schedule,
              epoch_i):
    b = 0
    loss_all = []
    j = 0
    for i in config:
        j = j + 1
        #print('Adv Training ' + str(config[i]['_type']) + '  epsilon:' + str(j))

        #for images_adv, labels in tqdm(train_loader):
        # for images_adv, labels in train_loader:

        model.eval()

        net_attack = AttackPGD(basic_model, config[i])

        # print(net_attack)

        net_attack = net_attack.to(device)
        images_adv = images_adv.to(device)
        labels = labels.to(device)
        # print(images.device)
        # images.cuda(args.gpu_ids[0])
        # labels.cuda(args.gpu_ids[0])
        # print(images)
        # print(images.shape)
        # print(type(images))

        images_att = net_attack(images_adv, labels, attack)

        # optimizer.zero_grad()

        model.train()
        prediction = model(images_att)

        # print('prediction value is :', prediction )

        loss = CE_loss(prediction, labels)

        loss_all.append(loss)

        # lr = lr_schedule(epoch_i + (b + 1) / len(train_loader))
        # optimizer.param_groups[0].update(lr=lr)
        #
        # optimizer.zero_grad()
        # # print('loss value is :', loss )
        # loss.backward()
        #
        # optimizer.step()

        b = b + 1

    return loss_all


def ensemble_train(loss_l1, loss_l2, loss_linf, loss_fog, loss_snow, loss_rt, size_train, b, lr_schedule, optimizer, epoch_i):

    all_loss = []

    l_l1_max = sum(loss_l1)
    all_loss.append(l_l1_max)
    l_l2_max = sum(loss_l2)
    all_loss.append(l_l2_max)
    l_linf_max = sum(loss_linf)
    all_loss.append(l_linf_max)
    l_fog_max = sum(loss_fog)
    all_loss.append(l_fog_max)
    l_snow_max = sum(loss_snow)
    all_loss.append(l_snow_max)
    l_rt_max = sum(loss_rt)
    all_loss.append(l_rt_max)

    #print(all_loss)
    # loss = sum(all_loss)/10
    loss = sum(all_loss)/6


    #print(loss)

    lr = lr_schedule(epoch_i + (b + 1) / size_train)
    optimizer.param_groups[0].update(lr=lr)

    optimizer.zero_grad()
    # print('loss value is :', loss )
    loss.backward()

    optimizer.step()

def ensemble_train_adv(loss_l1, loss_l2, loss_linf, size_train, b, lr_schedule, optimizer, epoch_i):

    all_loss = []

    l_l1_max = sum(loss_l1)
    all_loss.append(l_l1_max)
    l_l2_max = sum(loss_l2)
    all_loss.append(l_l2_max)
    l_linf_max = sum(loss_linf)
    all_loss.append(l_linf_max)
    # l_fog_max = sum(loss_fog)
    # all_loss.append(l_fog_max)
    # l_snow_max = sum(loss_snow)
    # all_loss.append(l_snow_max)
    # l_rt_max = sum(loss_rt)
    # all_loss.append(l_rt_max)

    #print(all_loss)
    # loss = sum(all_loss)/10
    loss = sum(all_loss)/4


    #print(loss)

    lr = lr_schedule(epoch_i + (b + 1) / size_train)
    optimizer.param_groups[0].update(lr=lr)

    optimizer.zero_grad()
    # print('loss value is :', loss )
    loss.backward()

    optimizer.step()


def train(args):
    # config_linf_6 = {
    #     'epsilon': 6.0 / 255,
    #     'num_steps': 20,
    #     # 'step_size': 2.0 / 255,
    #     'step_size': 0.01,
    #     'random_start': True,
    #     'loss_func': 'xent',
    #     '_type': 'linf'
    # }
    config_linf_8 = {
        'epsilon': 8.0 / 255,
        'num_steps': 20,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }

    # config_linf = dict(config_linf_6=config_linf_6, config_linf_8=config_linf_8)
    config_linf = dict(config_linf_8=config_linf_8)

    # config_l2_1_2 = {
    #     'epsilon': 0.5,
    #     'num_steps': 20,
    #     # 'step_size': 2.0 / 255,
    #     'step_size': 0.5 / 5,
    #     'random_start': True,
    #     'loss_func': 'xent',
    #     '_type': 'l2'
    # }
    config_l2_1 = {
        'epsilon': 1.0,
        'num_steps': 20,
        # 'step_size': 2.0 / 255,
        'step_size': 1.0 / 5,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l2'
    }

    # config_l2 = dict(config_l2_1_2=config_l2_1_2, config_l2_1=config_l2_1)
    config_l2 = dict(config_l2_1=config_l2_1)

    config_l1_16 = {
        'epsilon': 16.0,
        'num_steps': 20,
        # 'step_size': 2.0 / 255,
        'step_size': 2.5 * 16.0 / 20,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l1'
    }

    config_l1 = dict(config_l1_16=config_l1_16)

    # config_fog_1 = {
    #     't': 0.12,
    #     'light': 0.8
    # }

    config_fog_2 = {
        't': 1.5,
        'light': 1.2
    }

    # config_fog = dict(config_fog_1=config_fog_1, config_fog_2=config_fog_2)

    config_fog = dict(config_fog_2=config_fog_2)


    # brightness_1 = 2.0
    # brightness_2 = 2.5
    brightness_2 = 10


    config_snow = dict(bbrightness_2=brightness_2)


    attack = 'true'

    if args.dataset == 'cifar':
        output_classes = 10
        
    global best_acc_nat, best_acc_l2, best_acc_linf
    best_acc_nat_1 = 0
    best_acc_nat_2 = 0
    best_acc_l1 =[]
    best_acc_l2 = []
    best_acc_linf = []
    best_acc_fog = []
    best_acc_snow = []
    best_acc_rt = []
    best_acc_aver = 0
    

    transform = utils.get_transformation(args.dataset)
    transform_rt = utils.get_transformation_rotation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)
    
    # operate this train_loader to generate new loader
    
    train_loader = DataLoader(dataset['train_data'], batch_size = args.batch_size, shuffle=True)
    #val_loader = DataLoader(dataset['val_data'], batch_size = args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset['test_data'], batch_size = args.batch_size, shuffle=True)


    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    CE_loss = nn.CrossEntropyLoss()
    # jiang LeNet dan du chan fen le chu lai
    #model =  LeNet(output_classes)
    #model =  AlexNet(output_classes)
    model = ResNet18(output_classes)

    model = model.to(device)
    

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True


    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [0, 0.1, 0.005, 0])[0]

    for i in range(args.epochs):
        model.train()
        #j = 0
        print('The epoch number is: ' + str(i))
        
        lr = lr_schedule(i + (i+1)/args.batch_size)

        size_train = len(train_loader)

        b = 0

        print(config_l1)
        print(config_l2)
        print(config_linf)
        print(config_fog)
        print(config_snow)
        print(transform_rt)

        for images, labels in tqdm(train_loader):


            # loss_clean = train_clean(images, labels, device, optimizer, model, CE_loss, lr_schedule, i)


            #print('nat training weights', weights_nat, 'using time:', (etime_nat - stime_nat))



            loss_l1 = train_adv(images, labels, device, optimizer, model, model, AttackPGD, CE_loss, config_l1,
                                   attack, lr_schedule, i)



            loss_l2 = train_adv(images, labels, device, optimizer, model, model, AttackPGD, CE_loss, config_l2,
                                   attack, lr_schedule, i)

            #print('l2 training weights', weights_l2, 'using time:', (etime_l2 - stime_l2))



            loss_linf = train_adv(images, labels, device, optimizer, model, model, AttackPGD, CE_loss, config_linf,
                                     attack, lr_schedule, i)



            #print('after linf training weights(final):', weights_linf, 'using time:', (etime_linf - stime_linf))



            # weights_linf = train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_linf, attack)



            loss_fog = train_fog(images, labels, device, optimizer, model, CE_loss, config_fog, lr_schedule, i)
            # #print('after fog training weights(final):', weights_fog, 'using time:', (etime_fog - stime_fog))
            #
            #
            #
            loss_snow = train_snow(images, labels, device, optimizer, model, CE_loss, config_snow, lr_schedule, i)
            #
            #
            #
            loss_rt = train_rt(images, labels, device, optimizer, model, CE_loss, lr_schedule, transform_rt)


            # ensemble_train(loss_clean, loss_l2, loss_linf, loss_fog, loss_snow, size_train, b, lr_schedule, optimizer, i)
            # ensemble_train_adv(loss_l1, loss_l2, loss_linf, size_train, b, lr_schedule, optimizer, i)
            ensemble_train(loss_l1, loss_l2, loss_linf, loss_fog, loss_snow, loss_rt, size_train, b, lr_schedule, optimizer, i)

            b = b+1
            # if b == 1:
            #     break

        model.eval()
        correct_final_nat_1 = 0
        correct_final_l1 = 0
        correct_final_l2 = 0
        correct_final_linf = 0

        correct_final_nat_2 = 0
        correct_final_fog = 0
        correct_final_snow = 0
        correct_final_rt = 0


        #
        # acc_nat, best_acc_nat, acc_l2, best_acc_l2, acc_linf, best_acc_linf = val_adv(test_loader, device, model,  basic_model, AttackPGD, config_l2, config_linf, attack,\
        # correct_final_nat, best_acc_nat, correct_final_l2, best_acc_l2, correct_final_linf,\
        #     best_acc_linf, args.checkpoint_loc)

        # acc_nat, best_acc_nat = val_clean(val_loader, device, model, correct_final_nat, best_acc_nat, args.checkpoint_loc)

        acc_nat_1, acc_l1, acc_l2, acc_linf  = val_adv(test_loader, device, model, model, AttackPGD, config_l1, config_l2, config_linf, attack, correct_final_nat_1, correct_final_l1, correct_final_l2,  correct_final_linf,  args.checkpoint_loc)

        acc_nat_2, acc_fog, acc_snow, acc_rt = val_nat(test_loader, device, model, correct_final_nat_2, correct_final_fog, correct_final_snow, correct_final_rt, args.checkpoint_loc, config_fog, config_snow, transform_rt)

        if (acc_nat_1 + acc_nat_2 + sum(acc_l2) + sum(acc_l2) + sum(acc_linf) + sum(acc_fog) + sum(acc_snow) + sum(acc_rt)) / 8 > best_acc_aver:
            print('saving..')

            state = {
                'net': model.state_dict(),
                'acc_clean_1': acc_nat_1,
                'acc_clean_2': acc_nat_2,
                'acc_l1': acc_l1,
                'acc_l2': acc_l2,
                'acc_linf': acc_linf,
                'acc_fog': acc_fog,
                'acc_snow': acc_snow,
                'acc_rt': acc_rt,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, args.checkpoint_loc)
            best_acc_nat_1 = acc_nat_1
            best_acc_nat_2 = acc_nat_2
            best_acc_l1 = acc_l1
            best_acc_l2 = acc_l2
            best_acc_linf = acc_linf
            best_acc_fog = acc_fog
            best_acc_snow = acc_snow
            best_acc_rt = acc_rt
            best_acc_aver = (best_acc_nat_1 + best_acc_nat_2 + sum(best_acc_l1) + sum(best_acc_l2) + sum(best_acc_linf) \
                             + sum(best_acc_fog) + sum(best_acc_snow) + sum(best_acc_rt)) / 8

        # if (acc_nat_1 + sum(acc_l1) + sum(acc_l2) + sum(acc_linf) ) / 4 > best_acc_aver:
        #     print('saving..')
        #
        #     state = {
        #         'net': model.state_dict(),
        #         'acc_clean_1': acc_nat_1,
        #         'acc_l1': acc_l1,
        #         'acc_l2': acc_l2,
        #         'acc_linf': acc_linf,
        #         # 'epoch': epoch,
        #     }
        #
        #     if not os.path.isdir('checkpoint'):
        #         os.mkdir('checkpoint')
        #     torch.save(state, args.checkpoint_loc)
        #     best_acc_nat_1 = acc_nat_1
        #     best_acc_l1 = acc_l1
        #     best_acc_l2 = acc_l2
        #     best_acc_linf = acc_linf
        #     best_acc_aver = (best_acc_nat_1 + sum(best_acc_l1) +  sum(best_acc_l2) + sum(best_acc_linf)) / 4

       

        print('Epoch: ', i + 1, ' Done!!  Natural  Accuracy: ', acc_nat_1)
        print('Epoch: ', i + 1, '  Best Natural  Accuracy: ', best_acc_nat_1)

        # print('Epoch: ', i + 1, ' Done!!  Natural  Accuracy: ', acc_nat_2)
        # print('Epoch: ', i + 1, '  Best Natural  Accuracy: ', best_acc_nat_2)

        print('Epoch: ', i + 1, ' Done!!  l1(50, ..., 110)  Accuracy: ', acc_l1)
        print('Epoch: ', i + 1, '  Best l1  Accuracy: ', best_acc_l1)

        print('Epoch: ', i + 1, ' Done!!  l2(50, ..., 110)  Accuracy: ', acc_l2)
        print('Epoch: ', i + 1, '  Best l2  Accuracy: ', best_acc_l2)

        print('Epoch: ', i + 1, ' Done!!  l_inf(5, ..., 11)  Accuracy: ', acc_linf)
        print('Epoch: ', i + 1, '  Best l_inf  Accuracy: ', best_acc_linf)

        print('Epoch: ', i + 1, ' Done!!  fog  Accuracy: ', acc_fog)
        print('Epoch: ', i + 1, '  Best fog  Accuracy: ', best_acc_fog)

        print('Epoch: ', i + 1, ' Done!!  snow  Accuracy: ', acc_snow)
        print('Epoch: ', i + 1, '  Best snow  Accuracy: ', best_acc_snow)


        print('Epoch: ', i + 1, ' Done!!  rt  Accuracy: ', acc_rt)
        print('Epoch: ', i + 1, '  Best rt  Accuracy: ', best_acc_rt)

        # print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

        # print('Epoch: ', i + 1, ' Done!!  average Accuracy: ',
        #       (acc_nat_1 + acc_nat_2 + sum(acc_l2) + sum(acc_linf) + sum(acc_fog) + sum(acc_snow)) / 10)
        print('Epoch: ', i + 1, '  Best average  Accuracy: ', best_acc_aver)

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
            net_attack = AttackPGD(basic_model, config_l1[i])

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
            net_attack = AttackPGD(basic_model, config_l2[i])

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
            net_attack = AttackPGD(basic_model, config_linf[i])

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

        # if (acc_nat*2 + sum(acc_l2) + sum(acc_linf))/6 > best_acc_aver:
    #         print('saving..')

    #         state = {
    #         'net': model.state_dict(),
    #         'acc_clean': acc_nat,
    #         'acc_l2': acc_l2,
    #         'acc_linf': acc_linf,
    #         #'epoch': epoch,
    #         }

    #         if not os.path.isdir('checkpoint'):
    #             os.mkdir('checkpoint')
    #         torch.save(state, checkpoint_loc)
    #         best_acc_nat = acc_nat
    #         best_acc_l2 = acc_l2
    #         best_acc_linf = acc_linf
    #         best_acc_aver = (best_acc_nat*2 + sum(best_acc_l2) + sum(best_acc_linf))/6

    return acc_nat, acc_l1, acc_l2, acc_linf

def val_nat(val_loader, device, model, \
            correct_final_nat, correct_final_fog, \
            correct_final_snow, correct_final_rt, checkpoint_loc, \
            config_fog, config_snow, transform_rt):
    print('testing clean step')

    acc_fog = []
    acc_snow = []
    acc_rt = []

    for images_1, labels in tqdm(val_loader):
        images_1 = images_1.to(device)
        labels = labels.to(device)

        # images_att = net_attack(images,labels, attack)

        # prediction, weights_clean = model(images_1)
        prediction = model(images_1)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct_1 = pred.eq(labels.view_as(pred)).sum().item()
        # correct_final.append(correct)
        correct_final_nat = correct_final_nat + correct_1

    acc_nat = correct_final_nat / len(val_loader.dataset)

    print('testing fog step')

    j = 0

    for i in config_fog:

        j = j + 1

        print('Valuation fog No. ' + str(j))
        correct_2 = 0
        correct_final_fog = 0

        for images_2, labels in tqdm(val_loader):

            # a = a+1
            # net_attack = AttackPGD(basic_model,config_l2)

            # print(net_attack)

            # net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_2 = images_2.to(device)
            labels = labels.to(device)

            for a in range(0, images_2.shape[0]):
                # print(images[i])
                images_fog = add_fog(images_2[a], config_fog[i]['t'], config_fog[i]['light'])
                # print(images_fog)
                images_2[a] = images_fog

                # images_att = net_attack(images,labels, attack)

            prediction = model(images_2)
            # prediction = model(images_2)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_2 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_fog = correct_final_fog + correct_2

        acc_fog.append(correct_final_fog / len(val_loader.dataset))

    print('testing snow step')

    j = 0

    for i in config_snow:

        j = j + 1

        #print('Valuation snow No. ' + str(j))
        correct_3 = 0
        correct_final_snow = 0

        for images_3, labels in tqdm(val_loader):

            # a = a+1
            # net_attack = AttackPGD(basic_model,config_linf)

            # #print(net_attack)

            # net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_3 = images_3.to(device)
            labels = labels.to(device)

            for a in range(0, images_3.shape[0]):
                # print(images[i])
                images_snow = add_snow(images_3[a], config_snow[i])
                # print(images_fog)
                images_3[a] = images_snow

                # images_att = net_attack(images,labels, attack)

            prediction = model(images_3)
            # prediction = model(images_3)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_3 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_snow = correct_final_snow + correct_3

        acc_snow.append(correct_final_snow / len(val_loader.dataset))


    print('testing rotation step')

    for images_4, labels in tqdm(val_loader):
        images_4 = images_4.to(device)
        labels = labels.to(device)

        # images_att = net_attack(images,labels, attack)

        # prediction, weights_clean = model(images_1)
        images_rt = transform_rt['trans'](images_4)
        prediction = model(images_rt)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct_4 = pred.eq(labels.view_as(pred)).sum().item()
        # correct_final.append(correct)
        correct_final_rt = correct_final_rt + correct_4

    acc_rt.append(correct_final_rt / len(val_loader.dataset))

    return acc_nat, acc_fog, acc_snow, acc_rt

