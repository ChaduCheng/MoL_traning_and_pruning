import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms, models

def cuda(xs, gpu_id):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda(int(gpu_id[0]))
        else:
            return [x.cuda(int(gpu_id[0])) for x in xs]
    return xs

def load_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


# def get_dataset(dataset ,transform = None, train_split=0.8):
#
#     if dataset == 'cifar':
#         if transform == None:
#             print('Specify a transformation function')
#
#         train_data = datasets.CIFAR10('./cifar', train=True, transform = transform['train_transform'], download=True)
#         # num_training_imgs = int(len(training_data) * train_split)
#         # torch.manual_seed(0)
#         # train_data, val_data = torch.utils.data.random_split(training_data, [num_training_imgs, len(training_data) - num_training_imgs])
#
#         test_data = datasets.CIFAR10('./cifar', train=False, transform = transform['test_transform'], download=True)
#
#         return {'train_data': train_data,
#                 #'val_data': val_data,
#                 'test_data': test_data}


def get_dataset(dataset, transform=None, test_split=0.8):
    if dataset == 'cifar':
        if transform == None:
            print('Specify a transformation function')

        train_data = datasets.CIFAR10('./cifar', train=True, transform=transform['train_transform'], download=True)


        test_data = datasets.CIFAR10('./cifar', train=False, transform=transform['test_transform'], download=True)
        num_testing_imgs = int(len(test_data) * test_split)
        torch.manual_seed(0)
        val_data, _ = torch.utils.data.random_split(test_data, [num_testing_imgs,
                                                                               len(test_data) - num_testing_imgs])

        return {'train_data': train_data,
                # 'val_data': val_data,
                'test_data': test_data,
                'val_data': val_data}


    if dataset == 'tinyimagenet':
            import tiny_imagenet
            trainset = tiny_imagenet.TinyImageNet200(root='./data', train= True, transform=transform['train_transform'],download=True)
            # trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)

            testset = tiny_imagenet.TinyImageNet200(root='./data', train=False, transform=transform['test_transform'])
            # testloader = torch.utils.data.DataLoader(testset, batch_size=25, shuffle=False, num_workers=2)
            num_testing_imgs = int(len(testset) * test_split)
            torch.manual_seed(0)
            val_data, _ = torch.utils.data.random_split(testset, [num_testing_imgs,
                                                                    len(testset) - num_testing_imgs])


            print( "trainset num = {}".format(len(trainset)) )
            print( "trainloader num = {}".format(len(testset)) )
            return {'train_data': trainset,
                    'val_data': val_data,
                    'test_data': testset}

def get_transformation(dataset):
    
    if dataset == 'cifar':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # true value
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # true value
        ])


        transform = {'train_transform': train_transform,
                     'test_transform': test_transform}

    elif dataset == 'tinyimagenet':

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=64),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            # transforms.RandomRotation(45),
            transforms.ToTensor(),
        ])
        transform = {'train_transform': train_transform,
                     'test_transform': test_transform}
    else:
        print('Please choose the right dataset!!!')

    return transform



def get_transformation_rotation(dataset):
    # if dataset == 'cifar':
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(.25, .25, .25),
        # transforms.ToPILImage(),
        transforms.RandomRotation((180)),
        # transforms.ToTensor(),
    ])

    # test_transform = transforms.Compose([
    #     transforms.RandomRotation((180)),
    #     # transforms.RandomHorizontalFlip(p=1.0),
    #     # transforms.RandomVerticalFlip(p=1.0),
    #     # transforms.ToTensor(),
    # ])

    transform_rotation = {'trans': train_transform}

    # else:
    #     print('Please choose the right dataset!!!')

    return transform_rotation

# def load_model(model_path, basic_net):
#     checkpoint = torch.load(model_path)
#     basic_net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
    
#     return best_acc, start_epoch

def load_model(model_path, basic_net):
    checkpoint = torch.load(model_path)

    # print(checkpoint)

    basic_net.load_state_dict(checkpoint['net'])
    # basic_net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    # basic_net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})



def load_model_test(model_path, basic_net):
    checkpoint = torch.load(model_path)
    
    #print(checkpoint)
    
    #basic_net.load_state_dict(checkpoint['net'])
    basic_net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()})
    # best_acc_nat1 = checkpoint['acc_clean_1']
    # best_acc_nat2 = checkpoint['acc_clean_2']
    best_acc_nat = checkpoint['acc_clean']
    best_acc_l2 = checkpoint['acc_l2']
    best_acc_linf = checkpoint['acc_linf']
    # best_acc_fog = checkpoint['acc_fog']
    # best_acc_snow = checkpoint['acc_snow']

    # 'net': model.state_dict(),
    # 'acc_clean_1': acc_nat_1,
    # 'acc_clean_2': acc_nat_2,
    # 'acc_l2': acc_l2,
    # 'acc_linf': acc_linf,
    # 'acc_fog': acc_fog,
    # 'acc_snow': acc_snow,

    # return best_acc_nat1, best_acc_nat2, best_acc_l2, best_acc_linf, best_acc_fog, best_acc_snow
    return best_acc_nat, best_acc_l2, best_acc_linf


def load_model_all(model_path, basic_net):
    checkpoint = torch.load(model_path)

    # print(checkpoint)

    # basic_net.load_state_dict(checkpoint['net'])
    basic_net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    best_acc_nat1 = checkpoint['acc_clean_1']
    best_acc_nat2 = checkpoint['acc_clean_2']
    # best_acc_nat = checkpoint['acc_clean']
    best_acc_l2 = checkpoint['acc_l2']
    best_acc_linf = checkpoint['acc_linf']
    best_acc_fog = checkpoint['acc_fog']
    best_acc_snow = checkpoint['acc_snow']

    # 'net': model.state_dict(),
    # 'acc_clean_1': acc_nat_1,
    # 'acc_clean_2': acc_nat_2,
    # 'acc_l2': acc_l2,
    # 'acc_linf': acc_linf,
    # 'acc_fog': acc_fog,
    # 'acc_snow': acc_snow,

    return best_acc_nat1, best_acc_nat2, best_acc_l2, best_acc_linf, best_acc_fog, best_acc_snow
    # return best_acc_nat, best_acc_l2, best_acc_linf

def load_model_nat(model_path, basic_net):
    checkpoint = torch.load(model_path)
    
    #print(checkpoint)
    
    #basic_net.load_state_dict(checkpoint['net'])
    basic_net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()})
    best_acc_nat = checkpoint['acc_clean']
    best_acc_fog = checkpoint['acc_fog']
    best_acc_snow = checkpoint['acc_snow']    
    
    return best_acc_nat, best_acc_fog, best_acc_snow
    #return best_acc_nat, best_acc_linf

def load_all_model(model_path, basic_net):
    checkpoint = torch.load(model_path)

    # print(checkpoint)

    # basic_net.load_state_dict(checkpoint['net'])
    basic_net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    acc_clean_1 = checkpoint['acc_clean_1']
    acc_clean_2 = checkpoint['acc_clean_1']
    best_acc_l2 = checkpoint['acc_l2']
    best_acc_linf = checkpoint['acc_linf']
    best_acc_fog = checkpoint['acc_l2']
    best_acc_snow = checkpoint['acc_linf']

    return acc_clean_1, acc_clean_2, best_acc_l2, best_acc_linf, best_acc_fog, best_acc_snow

def EM_loss():
    '''
    I am still not able to get the notation specified in the paper for calculating EM loss as it is a bit confusing how they have
    calculated the posterior probabilities in equation (5).
    Also the updation step for the expert and gating weights they have used is just simply iteration based rather than gradient based which
    is also pretty weird.

    Also Cross Entropy would also work in this particular case as the model is a classification model and
    I think that the gradient based method also solves the MLE optimization problem(for which EM algorithm is suggested) as specified in the paper.
    '''
