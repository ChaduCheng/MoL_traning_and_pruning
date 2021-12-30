
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

def train_snow (train_loader, device,optimizer,model,CE_loss,brightness,lr_schedule, epoch_i):
    
    print('training snow step')
    b = 0
    j = 0
    
    for images, labels in tqdm(train_loader):
        
        j = j+1
        
        #print('training ' + str(j) + ' batch')    
        # for i in range(0,images.shape[0]):
                
        #        # print(images[i])
        #         images_snow = add_snow(images[i], brightness)
        #         #print(images_fog)
        #         images[i] = images_snow       

        images = images.to(device)
        labels = labels.to(device)
        
        for i in range(0,images.shape[0]):
                
               # print(images[i])
                images_snow = add_snow(images[i], brightness)
                #print(images_fog)
                images[i] = images_snow
# add fog to images



        prediction = model(images)
        loss = CE_loss(prediction, labels)        

        lr = lr_schedule(epoch_i + (b+1)/len(train_loader))
        optimizer.param_groups[0].update(lr=lr)

        #print('prediction value is :', prediction )
        optimizer.zero_grad()

        # print('loss value is :', loss )
        loss.backward()

        optimizer.step()

        b = b+1




def val(val_loader, device, model,  basic_model,\
        correct_final_nat, best_acc_nat, correct_final_fog, \
            best_acc_fog, correct_final_snow, best_acc_snow, checkpoint_loc,\
                t, light, brightness):
    
    print('testing clean step')
    
    acc_nat = 0
    acc_fog = 0
    acc_snow = 0


    for images_1, labels in tqdm(val_loader):
            
 
            images_1 = images_1.to(device)
            labels = labels.to(device)
            
            #images_att = net_attack(images,labels, attack)

                  
            prediction = model(images_1)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_1 = pred.eq(labels.view_as(pred)).sum().item()
            #correct_final.append(correct)
            correct_final_nat = correct_final_nat + correct_1
            
    acc_nat = correct_final_nat / len(val_loader.dataset)
    
    # print('testing fog step')
    
    # for images_2, labels in tqdm(val_loader):
            
    #         #a = a+1
    #         #net_attack = AttackPGD(basic_model,config_l2)
           
    #         #print(net_attack)
            
    #         #net_attack = net_attack.to(device)
    #         #print('processing testing image:', a)
    #         #images, labels = utils.cuda([images, labels], args.gpu_ids)
    #         images_2 = images_2.to(device)
    #         labels = labels.to(device)
            
    #         for i in range(0,images_2.shape[0]):
    #              # print(images[i])
    #               images_fog = add_fog(images_2[i], t, light)
    #               #print(images_fog)
    #               images_2[i] = images_fog 
            
    #         #images_att = net_attack(images,labels, attack)



    #         prediction = model(images_2)
    #         pred = prediction.argmax(dim=1, keepdim=True)
    #         correct_2 = pred.eq(labels.view_as(pred)).sum().item()
    #         #correct_final.append(correct)
    #         correct_final_fog = correct_final_fog + correct_2
        
    # acc_fog = correct_final_fog / len(val_loader.dataset)

    print('testing snow step')

    for images_3, labels in tqdm(val_loader):
            
            #a = a+1
            # net_attack = AttackPGD(basic_model,config_linf)
            
            # #print(net_attack)
           
            # net_attack = net_attack.to(device)
            #print('processing testing image:', a)
            #images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_3 = images_3.to(device)
            labels = labels.to(device)
            
            for i in range(0,images_3.shape[0]):
                
                  
                
                 # print(images[i])
                  images_snow = add_snow(images_3[i], brightness)
                  #print(images_fog)
                  images_3[i] = images_snow 
            
           # images_att = net_attack(images,labels, attack)

            prediction = model(images_3)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_2 = pred.eq(labels.view_as(pred)).sum().item()
            #correct_final.append(correct)
            correct_final_snow = correct_final_snow + correct_2
        
    acc_snow = correct_final_snow / len(val_loader.dataset)    
    
    
    
    
    if (acc_snow + acc_nat)/2 > best_acc_fog:
            print('saving..')
            
            state = {
            'net': model.state_dict(),
            'acc_clean': acc_nat,
            'acc_fog': acc_fog,
            'acc_snow': acc_snow,
            #'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_loc)
            best_acc_nat = acc_nat
            best_acc_fog = (acc_snow + acc_nat)/2
            best_acc_snow = acc_snow
    
    return acc_nat, best_acc_nat, acc_fog, best_acc_fog, acc_snow, best_acc_snow



def train(args):
    
    # config_linf = {
    # 'epsilon': 8.0 / 255,
    # #'epsilon': 0.314,
    # 'num_steps': 10,
    # 'step_size': 2.0 / 255,
    # 'random_start': True,
    # 'loss_func': 'xent',
    # '_type': 'linf'
    #  }
    
    # config_l2 = {
    # #'epsilon': 8.0 / 255,
    # 'epsilon': 0.314,
    # 'num_steps': 10,
    # 'step_size': 2.0 / 255,
    # 'random_start': True,
    # 'loss_func': 'xent',
    # '_type': 'l2'
    #  }
    
    # attack = 'true'

    if args.dataset == 'cifar':
        output_classes = 10
        
    global best_acc_nat, best_acc_l2, best_acc_linf
    best_acc_nat = 0
    best_acc_fog = 0
    best_acc_snow = 0
    

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)
    
    # operate this train_loader to generate new loader
    
    train_loader = DataLoader(dataset['train_data'], batch_size = args.batch_size, shuffle=True)
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
        
        t = 1.5
        light = 1.2
        brightness = 15
        print('the brightness is:', brightness)
        train_snow (train_loader, device,optimizer,model,CE_loss, brightness, lr_schedule, i)
        
       # train_clean (train_loader, device,optimizer,model,CE_loss)
        
       # train_clean (train_loader, device,optimizer,model,CE_loss)
        
        #train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_l2, attack)
                
        #train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_linf, attack)
        
        


        model.eval()
        correct_final_nat = 0
        correct_final_fog = 0
        correct_final_snow = 0

        
        
        acc_nat, best_acc_nat, acc_fog, best_acc_fog, acc_snow, best_acc_snow = val(test_loader, device, model,  model,\
        correct_final_nat, best_acc_nat, correct_final_fog, best_acc_fog, correct_final_snow,\
            best_acc_snow, args.checkpoint_loc, t, light, brightness)
        
       # acc_nat, best_acc_nat = val_clean(val_loader, device, model, correct_final_1, best_acc_nat, args.checkpoint_loc)
        
        print('Epoch: ', i+1, ' Done!!  Natural  Accuracy: ', acc_nat)
        print('Epoch: ', i+1, '  Best Natural  Accuracy: ', best_acc_nat) 
          
        
        
        
        #acc_2, best_acc_l2 = val_adv(val_loader, device, model,  model, AttackPGD, config_l2, attack, correct_final_2, best_acc_l2 ,args.checkpoint_loc)
        
        print('Epoch: ', i+1, ' Done!!  fog  Accuracy: ', acc_fog)
        print('Epoch: ', i+1, '  Best fog  Accuracy: ', best_acc_fog)         
        
        #acc_3, best_acc_linf = val_adv(val_loader, device, model,  model, AttackPGD, config_linf, attack, correct_final_3, best_acc_linf ,args.checkpoint_loc)


        
        print('Epoch: ', i+1, ' Done!!  snow  Accuracy: ', acc_snow)
        print('Epoch: ', i+1, '  Best snow  Accuracy: ', best_acc_snow)
        



    

            

















def val_clean(val_loader, device, model, correct_final, best_acc, checkpoint_loc):
    
    for images, labels in val_loader:
            
 
            images = images.to(device)
            labels = labels.to(device)
            
            #images_att = net_attack(images,labels, attack)
    
            prediction = model(images)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_1 = pred.eq(labels.view_as(pred)).sum().item()
            #correct_final.append(correct)
            correct_final = correct_final + correct_1
            
    acc = correct_final / len(val_loader.dataset)
    if acc > best_acc:
            print('saving..')
            
            state = {
            'net': model.state_dict(),
            'acc': acc,
            #'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_loc)
            best_acc = acc
    
    return acc, best_acc

def val_adv(val_loader, device, model,  basic_model, AttackPGD, config, attack, correct_final, best_acc, checkpoint_loc):
    for images, labels in val_loader:
            
            #a = a+1
            net_attack = AttackPGD(basic_model,config)
            
            #print(net_attack)
            
            net_attack = net_attack.to(device)
            #print('processing testing image:', a)
            #images, labels = utils.cuda([images, labels], args.gpu_ids)
            images = images.to(device)
            labels = labels.to(device)
            
            images_att = net_attack(images,labels, attack)

            prediction = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_2 = pred.eq(labels.view_as(pred)).sum().item()
            #correct_final.append(correct)
            correct_final = correct_final + correct_2
        
    acc = correct_final / len(val_loader.dataset)
    
    if acc > best_acc:
            print('saving..')
            
            state = {
            'net': model.state_dict(),
            'acc': acc,    #zhu ming type
            #'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_loc)
            best_acc = acc
    
    return acc, best_acc
