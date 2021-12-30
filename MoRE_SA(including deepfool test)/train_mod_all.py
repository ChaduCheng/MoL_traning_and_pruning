

import os
import torch
from tqdm import tqdm
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
from image_trans import *
import matplotlib.pyplot as plt
from PIL import Image
import time


def train_clean (train_loader, device,optimizer,model,CE_loss, lr_schedule, epoch_i):
    a = 0
    j = 0
    for j in range(1,3):    
        print('Doing clean images training No. ' + str(j))
        for images, labels in tqdm(train_loader):
    
                images = images.to(device)
                labels = labels.to(device)
    
                # optimizer.zero_grad()
                prediction, = model(images)
                
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

def train_fog (train_loader, device,optimizer,model,CE_loss, config_fog, lr_schedule, epoch_i):
    
    c = 0
    
    print('training snow step')
    
    j = 0
    
    for i in config_fog:
        j = j+1 
        
        #b = 0
        
        print('fog Training ' + str(config_fog[i]['t']) + ' t:' + str(j))

        for images, labels in tqdm(train_loader):
            
            #b = b+1
            
            #print('training ' + str(b) + ' batch')    
    
           
    
            images = images.to(device)
            labels = labels.to(device)
            
            for a in range(0,images.shape[0]):
                    
                   # print(images[i])
                    images_fog = add_fog(images[a], config_fog[i]['t'], config_fog[i]['light'])
                    #print(images_fog)
                    images[a] = images_fog 
    # add fog to images
    
            prediction = model(images)
            
            #print('prediction value is :', prediction )
    
            loss = CE_loss(prediction, labels)

            lr = lr_schedule(epoch_i + (c+1)/len(train_loader))
            optimizer.param_groups[0].update(lr=lr)

            # print('loss value is :', loss )
            optimizer.zero_grad()
            loss.backward()
    
            optimizer.step()
            
            c = c+1


    # return weights_fog

def train_snow (train_loader, device,optimizer,model,CE_loss, config_snow, lr_schedule, epoch_i):
    b = 0
    print('training snow step')
    
    j = 0

    for i in config_snow:

        j = j+1
        print('snow Training ' + str(config_snow[i]) + ' No.' + str(j))    
        
        b = 0
        
        for images, labels in tqdm(train_loader):
            
            # b = b+1
            
           # print('training ' + str(b) + ' batch')    
            
            # for a in range(0,images.shape[0]):
                    
            #        # print(images[i])
            #         images_snow = add_snow(images[a], config_snow[i])
            #         #print(images_fog)
            #         images[a] = images_snow       
    
            images = images.to(device)
            labels = labels.to(device)
            
            for a in range(0,images.shape[0]):
                    
                   # print(images[i])
                    images_snow = add_snow(images[a], config_snow[i])
                    #print(images_fog)
                    images[a] = images_snow
    # add fog to images
    
    

            prediction = model(images)
            
            #print('prediction value is :', prediction )
    
            loss = CE_loss(prediction, labels)
            # print('loss value is :', loss )
            
            
            lr = lr_schedule(epoch_i + (b+1)/len(train_loader))
            optimizer.param_groups[0].update(lr=lr)            
            
            optimizer.zero_grad()            

            loss.backward()
    
            optimizer.step()
            
            b = b+1

    # return weights_snow

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
        prediction, _ = model(images_1)
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

            prediction, _ = model(images_2)
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

            prediction, _ = model(images_3)
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
        # [transform_rt['trans'](transforms.ToPILImage()(image)) for image in images_4]
        prediction, _ = model(images_rt)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct_4 = pred.eq(labels.view_as(pred)).sum().item()
        # correct_final.append(correct)
        correct_final_rt = correct_final_rt + correct_4

    acc_rt.append(correct_final_rt / len(val_loader.dataset))

    return acc_nat, acc_fog, acc_snow, acc_rt





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
        'num_steps': 7,
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

    config_l1_16 = {
        'epsilon': 12.0,
        'num_steps': 7,
        # 'step_size': 2.0 / 255,
        'step_size': 2.5 * 12.0 / 7,
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
        't': 2.0,
        'light': 1.0
    }

    # config_fog = dict(config_fog_1=config_fog_1, config_fog_2=config_fog_2)

    config_fog = dict(config_fog_2=config_fog_2)

    # brightness_1 = 2.0
    # brightness_2 = 2.5
    brightness_2 = 20

    config_snow = dict(brightness_2=brightness_2)
    
    

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
    best_acc_fog = []
    best_acc_snow = []
    best_acc_rt = []
    best_acc_aver = 0
    

    transform = utils.get_transformation(args.dataset)
    transform_rt = utils.get_transformation_rotation(args.dataset)
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
    # model = MoE_ResNet18_adv(args.num_experts, output_classes)
    model = MoE_ResNet18_base(args.num_experts, output_classes)
    print('tiny base check here')

    model = model.to(device)
    
    ### load pretrained model
    
   

    if args.resume == False:
        if args.dataset == 'cifar':
            model_loc0 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_clean_at.pth'

            # model_loc1 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_linf_6_255_at_change.pth'
            model_loc1 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_linf_8_255_at_change.pth'
            # model_loc3 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_l2_1_2_at_change.pth'
            model_loc2 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_l2_1_at_change.pth'
            model_loc3 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_l1_16_at_change_26.pth'
            model_loc4 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_fog_t_2.0_light_1.0_at_change.pth'
            model_loc5 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_snow_bright_20_change.pth'
            model_loc6 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_rota_180_at.pth'
        elif args.dataset == 'tinyimagenet':
            # model_loc0 = '../cifar_clean_train_resnet/checkpoint/ckptnat_resnet18_cifar_tsave.pth'
            # model_loc0 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_clean.pth'
            model_loc0 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_clean.pth'

            # model_loc1 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_linf_6_255_at_change.pth'
            model_loc1 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_linf_8.pt'
            # model_loc3 = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_l2_1_2_at_change.pth'
            model_loc2 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_l2_1.0.pt'
            model_loc3 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_l1_12.0.pt'
            model_loc4 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_fog_2.0_1.0.pt'
            model_loc5 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_snow_20.pt'
            model_loc6 = '../cifar_clean_train_resnet/checkpoint/tinyimagenet_rotate_180.pt'
        utils.load_model(model_loc0, model.experts[0])
        utils.load_model(model_loc1, model.experts[1])
        utils.load_model(model_loc2, model.experts[2])
        utils.load_model(model_loc3, model.experts[3])
        utils.load_model(model_loc4, model.experts[4])
        utils.load_model(model_loc5, model.experts[5])
        utils.load_model(model_loc6, model.experts[6])
    else:
        utils.load_model(args.checkpoint_loc, model)
        print(args.checkpoint_loc)

    print(model_loc0)
    print(model_loc3)
    print(model_loc6)

    
    
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    print(config_l1)
    print(config_l2)
    print(config_linf)

    #model = utils.cuda(model, args.gpu_ids)
    #model.cuda(args.gpu_ids[0])
    #model = model.to(device)
    #model = torch.nn.DataParallel(model, device_ids = [0]).cuda()
    #print(model.device)
    #model = MoE(basic_model, args.num_experts, output_classes)
    #print(model)
    #print(type(model))
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [0, 0.1, 0.005, 0])[0]

    for i in range(args.epochs):
        model.train()
        #j = 0
        print('The epoch number is: ' + str(i))
        


        print('loss base value check')
        #train_clean (train_loader, device,optimizer,model,CE_loss)
        
        #train_clean (train_loader, device,optimizer,model,CE_loss)
        
        loss_linf = train_adv_base(train_loader, device, optimizer, model, model, AttackPGD, CE_loss, config_linf, attack,
                                 lr_schedule, i)

        print('linf loss is:', loss_linf)

        loss_l2 = train_adv_base(train_loader, device, optimizer, model, model, AttackPGD, CE_loss, config_l2, attack,
                               lr_schedule, i)

        print('l2 loss is:', loss_l2)

        loss_l1 = train_adv_base(train_loader, device, optimizer, model, model, AttackPGD, CE_loss, config_l1, attack,
                               lr_schedule, i)

        # loss_l1 =0
        # loss_l2=0
        # loss_linf =0

        print('l1 loss is:', loss_l1)




        #

       

        model.eval()
        correct_final_nat_1 = 0
        correct_final_l1 = 0
        correct_final_l2 = 0
        correct_final_linf = 0

        correct_final_nat_2 = 0
        correct_final_fog = 0
        correct_final_snow = 0
        correct_final_rt = 0

        
        
      

        acc_nat_1, acc_l1, acc_l2, acc_linf  = val_adv(test_loader, device, model, model, AttackPGD, config_l1, config_l2, config_linf, attack, correct_final_nat_1, correct_final_l1, correct_final_l2,  correct_final_linf,  args.checkpoint_loc)

        acc_nat_2, acc_fog, acc_snow, acc_rt = val_nat(test_loader, device, model, correct_final_nat_2, correct_final_fog, correct_final_snow, correct_final_rt, args.checkpoint_loc, config_fog, config_snow, transform_rt)
        
        

        if (acc_nat_1 + acc_nat_2 + sum(acc_l1) + sum(acc_l2) + sum(acc_linf) + sum(acc_fog) + sum(acc_snow) + sum(acc_rt)) / 8 > best_acc_aver:
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

        print('Epoch: ', i + 1, ' Done!!  Natural  Accuracy: ', acc_nat_1)
        print('Epoch: ', i + 1, '  Best Natural  Accuracy: ', best_acc_nat_1)

        print('Epoch: ', i + 1, ' Done!!  Natural  Accuracy: ', acc_nat_2)
        print('Epoch: ', i + 1, '  Best Natural  Accuracy: ', best_acc_nat_2)

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

        #print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

        # print('Epoch: ', i+1, ' Done!!  average Accuracy: ', (acc_nat_1 + acc_nat_2 + sum(acc_l2) + sum(acc_linf) + sum(acc_fog) + sum(acc_snow))/10 )
        print('Epoch: ', i+1, '  Best average  Accuracy: ', best_acc_aver)

        print('./tiny_base_all_true.txt')

        with open('./tiny_base_all_true.txt', 'at') as f:

            print('epoch', i+1 ,'linf loss is:', loss_linf, file=f)

            print('epoch', i+1 ,'l2 loss is:', loss_l2, file=f)

            print('epoch', i+1 ,'l1 loss is:', loss_l1, file=f)

            print('Epoch: ', i + 1, ' Done!!  Natural  Accuracy: ', acc_nat_1, file=f)
            print('Epoch: ', i + 1, '  Best Natural  Accuracy: ', best_acc_nat_1, file=f)

            print('Epoch: ', i + 1, ' Done!!  Natural  Accuracy: ', acc_nat_2, file=f)
            print('Epoch: ', i + 1, '  Best Natural  Accuracy: ', best_acc_nat_2, file=f)

            print('Epoch: ', i + 1, ' Done!!  l1(50, ..., 110)  Accuracy: ', acc_l1, file=f)
            print('Epoch: ', i + 1, '  Best l1  Accuracy: ', best_acc_l1, file=f)

            print('Epoch: ', i + 1, ' Done!!  l2(50, ..., 110)  Accuracy: ', acc_l2, file=f)
            print('Epoch: ', i + 1, '  Best l2  Accuracy: ', best_acc_l2, file=f)

            print('Epoch: ', i + 1, ' Done!!  l_inf(5, ..., 11)  Accuracy: ', acc_linf, file=f)
            print('Epoch: ', i + 1, '  Best l_inf  Accuracy: ', best_acc_linf, file=f)

            print('Epoch: ', i + 1, ' Done!!  fog  Accuracy: ', acc_fog, file=f)
            print('Epoch: ', i + 1, '  Best fog  Accuracy: ', best_acc_fog, file=f)

            print('Epoch: ', i + 1, ' Done!!  snow  Accuracy: ', acc_snow, file=f)
            print('Epoch: ', i + 1, '  Best snow  Accuracy: ', best_acc_snow, file=f)

            print('Epoch: ', i + 1, ' Done!!  rt  Accuracy: ', acc_rt, file=f)
            print('Epoch: ', i + 1, '  Best rt  Accuracy: ', best_acc_rt, file=f)

            # print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

            # print('Epoch: ', i+1, ' Done!!  average Accuracy: ', (acc_nat_1 + acc_nat_2 + sum(acc_l2) + sum(acc_linf) + sum(acc_fog) + sum(acc_snow))/10 )
            print('Epoch: ', i + 1, '  Best average  Accuracy: ', best_acc_aver, file=f)
      
        

        
        # #print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

        # print('Epoch: ', i+1, ' Done!!  average Accuracy: ', (acc_nat*2 + sum(acc_l2) + sum(acc_linf))/6 )
        # print('Epoch: ', i+1, '  Best average  Accuracy: ', best_acc_aver)

    

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



                #optimizer.zero_grad()
                # prediction, weights = model(images_att)
                model.train()
                prediction, predicts = model(images_att)



                # loss = CE_loss(prediction, labels)
                losses = []
                losses.append(CE_loss(predicts[0], labels))

                losses.append(CE_loss(predicts[1], labels))

                losses.append(CE_loss(predicts[2], labels))

                losses.append(CE_loss(predicts[3], labels))

                losses.append(CE_loss(predicts[4], labels))

                losses.append(CE_loss(predicts[5], labels))

                losses.append(CE_loss(predicts[6], labels))

                loss = max(losses)
                loss_exp_aver = sum(losses)/7

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


def train_adv_base(train_loader, device, optimizer, basic_model, model, AttackPGD, CE_loss, config, attack, lr_schedule,
              epoch_i):
    b = 0
    j = 0
    all_loss = 0
    aver_loss = 0
    for i in config:
        j = j + 1
        print('Adv Training ' + str(config[i]['_type']) + '  epsilon:' + str(j))

        for images_adv, labels in tqdm(train_loader):
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
            # prediction, weights = model(images_att)
            model.train()
            prediction, predicts = model(images_att)

            loss = CE_loss(prediction, labels)

            # losses = []
            # losses.append(CE_loss(predicts[0], labels))
            #
            # losses.append(CE_loss(predicts[1], labels))
            #
            # losses.append(CE_loss(predicts[2], labels))
            #
            # losses.append(CE_loss(predicts[3], labels))
            #
            # losses.append(CE_loss(predicts[4], labels))
            #
            # losses.append(CE_loss(predicts[5], labels))
            #
            # losses.append(CE_loss(predicts[6], labels))
            #
            # loss = max(losses)

            lr = lr_schedule(epoch_i + (b + 1) / len(train_loader))
            optimizer.param_groups[0].update(lr=lr)

            optimizer.zero_grad()
            # print('loss value is :', loss )
            loss.backward()

            optimizer.step()

            b = b + 1
            all_loss = all_loss + loss


        aver_loss = all_loss / len(train_loader.dataset)

    return aver_loss
        
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