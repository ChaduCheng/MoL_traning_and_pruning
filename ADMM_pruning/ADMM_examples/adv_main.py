'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import torchvision
import torchvision.transforms as transforms
import sys
import os
import argparse
from utils import *
from models import *
from config import Config
from attack_steps import L2Step, LinfStep, L1Step
from weather_generation import *




sys.path.append('../../') # append root directory

from admm.warmup_scheduler import GradualWarmupScheduler
from admm.cross_entropy import CrossEntropyLossMaybeSmooth
from admm.utils import mixup_data, mixup_criterion

from admm.init_func import Init_Func

import admm


model_names = ['vgg16','resnet18','vgg16_1by8','vgg16_1by16','vgg16_1by32']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AttackPGD(nn.Module):
    def __init__(self, basic_model, config):
        super(AttackPGD, self).__init__()
        self.basic_model = basic_model
        self.rand = config.random_start
        self.step_size = config.step_size/255
        self.epsilon = config.epsilon/255
        self.num_steps = config.num_steps


    def forward(self,input, target):    # do forward in the module.py
        #if not args.attack :
        #    return self.basic_model(input), input



        x = input.detach()

        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_model(x)
                loss = F.cross_entropy(logits, target, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, input - self.epsilon), input + self.epsilon)

            x = torch.clamp(x, 0, 1)

        return self.basic_model(input), self.basic_model(x) , x


class AttackPGD_each(nn.Module):
    def __init__(self, basic_model, config):
        super(AttackPGD_each, self).__init__()
        self.basic_model = basic_model
        self.rand = config.random_start
        self.step_size = config.step_size
        self.epsilon = config.epsilon
        self.num_steps = config.num_steps
        self._type = config.adv_type
        # assert config['loss_func'] == 'xent', 'Only xent supported for now.'


    def forward(self,input, target):    # do forward in the module.py
        #if not args.attack :
        #    return self.basic_model(input), input

        # print('adv type is:', self._type)
        # print(self.epsilon)
        # print(self.num_steps)
        # print(self.step_size)

        if self._type == 'l2':
            step = L2Step(eps=self.epsilon, orig_input=input, step_size=self.step_size)
        elif self._type == 'linf':
            step = LinfStep(eps=self.epsilon, orig_input=input, step_size=self.step_size)
        elif self._type == 'l1':
            step = L1Step(eps=self.epsilon, orig_input=input, step_size=self.step_size)
        else:
            NotImplementedError

        x = input.clone().detach().requires_grad_(True)

        if self.rand:
            # x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            x = step.random_perturb(x)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_model(x)
                loss = F.cross_entropy(logits, target, size_average=False, reduction='none')
            grad = torch.autograd.grad(loss, [x])[0]
            # x = x.detach() + self.step_size*torch.sign(grad.detach())
            # x = torch.min(torch.max(x, input - self.epsilon), input + self.epsilon)
            #
            # x = torch.clamp(x, 0, 1)
            with torch.no_grad():
                x = step.step(x, grad)
                x = step.project(x)

        return self.basic_model(input), self.basic_model(x) , x

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config_file', type=str, default='', help ="config file")
parser.add_argument('--stage', type=str, default='', help ="select the pruning stage")



#init = Init_Func(config.init_func)
#torch.manual_seed(config.random_seed)
args = parser.parse_args()

# args.config_file = './config.yaml.example'
# args.stage = 'admm'

config = Config(args)



best_acc = 0  # best test accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_mean_loss = 100.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if config.logging:
    log_dir = config.log_dir
    logger = getLogger(log_dir)
    logger.info(json.dumps(config.__dict__, indent=4))
else:
    logger = None


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor()
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.25, .25, .25),
    transforms.RandomRotation(2),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

transform_rt = transforms.Compose([
    transforms.RandomRotation((180, 180))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=config.workers)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Model
print('==> Building model..')
model = None
if config.arch == "vgg16":
    model = VGG('vgg16')
elif config.arch =='vgg16_adv':
    model = VGG_adv('vgg16',w = config.width_multiplier)
elif config.arch =='vgg16_ori_adv':
    model = VGG_ori_adv('vgg11',w = config.width_multiplier)
elif config.arch =="resnet18":
    model = ResNet18(10)
elif config.arch == "googlenet":
    model = GoogLeNet()
elif config.arch == "densenet121":
    model = DenseNet121()
elif config.arch == "vgg16_1by8":
    model = VGG('vgg16_1by8')
elif config.arch == "vgg16_1by16":
    model = VGG('vgg16_1by16')
elif config.arch == "vgg16_1by32":
    model = VGG('vgg16_1by32')
elif config.arch == "resnet18_1by16":
    model = ResNet18_1by16()
elif config.arch == 'resnet18_adv':
    model = ResNet18_adv(w=config.width_multiplier)
elif config.arch == 'lenet_adv':
    model = LeNet_adv(w = config.width_multiplier)
elif config.arch == 'lenet':
    model = LeNet(w= config.width_multiplier)
elif config.arch == 'resnet18_adv_wide':
    model = ResNet18_adv_wide()
# model = PreActResNet18()
# model = GoogLeNet()
# model = DenseNet121()
# model = ResNeXt29_2x64d()
# model = MobileNet()
# model = MobileNetV2()
# model = DPN92()
# model = ShuffleNetG2()
# model = SENet18()
# model = ShuffleNetV2(1)
# print (model)





# model = AttackPGD(model,config)
config.model = model

# config.model = config.model.to(device)

# if device == 'cuda':
#     if config.gpu is not None:
#         torch.cuda.set_device(config.gpu)
#         config.model = torch.nn.DataParallel(model,device_ids = [config.gpu])
#     else:
#         config.model.cuda()
#         config.model = torch.nn.DataParallel(model)
#     cudnn.benchmark = True


if device == 'cuda':
    config.model = torch.nn.DataParallel(config.model)
    cudnn.benchmark = True

if config.load_model:
    # unlike resume, load model does not care optimizer status or start_epoch
    print('==> Loading from {}'.format(config.load_model))

    # config.model.load_state_dict(torch.load(config.load_model)['net']) # i call 'net' "model"
    checkpoint = torch.load(config.load_model)



    config.model.load_state_dict(checkpoint)

    #print(checkpoint)
    # if args.stage == 'admm':
    #     print('admm loading')
    #     if config.per_type == 'weather':
    #         config.model.load_state_dict(checkpoint['net'])
    #     elif config.per_type == 'adv':
    #         config.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    #     elif config.per_type == 'clean':
    #         config.model.load_state_dict(checkpoint['net'])
    # elif args.stage == 'retrain':
    #     print('retrain loading')
    #     config.model.load_state_dict(checkpoint)






config.prepare_pruning() # take the model and prepare the pruning

ADMM = None

if config.admm:
    ADMM = admm.ADMM(config)



if config.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    config.model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    ADMM.ADMM_U = checkpoint['admm']['ADMM_U']
    ADMM.ADMM_Z = checkpoint['admm']['ADMM_Z']

criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config.smooth_eps).cuda(config.gpu)
config.smooth = config.smooth_eps > 0.0
config.mixup = config.alpha > 0.0


config.warmup = (not config.admm) and config.warmup_epochs > 0
optimizer_init_lr = config.warmup_lr if config.warmup else config.lr

optimizer = None
if (config.optimizer == 'sgd'):
    optimizer = torch.optim.SGD(config.model.parameters(), optimizer_init_lr,
                            momentum=0.9,
                                weight_decay=1e-4)
elif (config.optimizer =='adam'):
    optimizer = torch.optim.Adam(config.model.parameters(), optimizer_init_lr)



scheduler = None
if config.lr_scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs*len(trainloader), eta_min=4e-08)
elif config.lr_scheduler == 'default':
    # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
    #epoch_milestones = [150, 250, 350]
    epoch_milestones = [80,150] # for adv training
    """Set the learning rate of each parameter group to the initial lr decayed
        by gamma once the number of epoch reaches one of the milestones
    """
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i*len(trainloader) for i in epoch_milestones], gamma=0.1)
else:
    raise Exception("unknown lr scheduler")

if config.warmup:
    scheduler = GradualWarmupScheduler(optimizer, multiplier=config.lr/config.warmup_lr, total_iter=config.warmup_epochs*len(trainloader), after_scheduler=scheduler)


def train(train_loader,criterion, optimizer, epoch, config, config_pertype, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    nat_losses = AverageMeter()
    adv_losses = AverageMeter()
    nat_loss = 0
    adv_loss = 0
    nat_top1 = AverageMeter()
    adv_top1 = AverageMeter()    


    # switch to train mode
    config.model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if config.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, config)
        else:
            scheduler.step()

        # if config.gpu is not None:
        #     input = input.cuda(config.gpu, non_blocking=True)
        # target = target.cuda(config.gpu, non_blocking=True)



        if config.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, config.alpha)

        input = input.to(device)
        target = target.to(device)

        # compute output
        # nat_output,adv_output,pert_inputs = config.model(input,target)
        config.model.eval()
        # nat_output = config.model(input)
        # print(nat_output)
        # print(input.size())
        net_attack = AttackPGD_each(config.model, config)
        net_attack = net_attack.to(device)
        # pert_inputs = net_attack(input, target)
        nat_output, adv_output, pert_inputs = net_attack(input, target)
        config.model.train()

        if config.mixup:
            adv_loss = mixup_criterion(criterion, adv_output, target_a, target_b, lam, config.smooth)
            nat_loss = mixup_criterion(criterion, nat_output, target_a, target_b, lam, config.smooth)
        else:
            adv_loss = criterion(adv_output, target, smooth=config.smooth)
            nat_loss = criterion(nat_output, target, smooth=config.smooth)
        if config.admm:
            admm.admm_update(config,ADMM,device,train_loader,optimizer,epoch,input,i)   # update Z and U
            adv_loss,admm_loss,mixed_loss = admm.append_admm_loss(config,ADMM,adv_loss) # append admm losss

        # measure accuracy and record loss
        nat_acc1,_ = accuracy(nat_output, target, topk=(1,5))
        adv_acc1,_ = accuracy(adv_output, target, topk=(1,5))
        
        nat_losses.update(nat_loss.item(), input.size(0))
        adv_losses.update(adv_loss.item(), input.size(0))
        adv_top1.update(adv_acc1[0], input.size(0))
        nat_top1.update(nat_acc1[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        if config.admm:
            mixed_loss.backward()
        else:
            adv_loss.backward()

        if config.masked_progressive:
            with torch.no_grad():
                for name,W in config.model.named_parameters():
                    if name in config.zero_masks:
                            W.grad *=config.zero_masks[name]


        if config.masked_retrain:
            with torch.no_grad():
                for name,W in config.model.named_parameters():
                    if name in config.masks:
                            W.grad *= config.masks[name] #returns boolean array called mask when weights are above treshhold

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Nat_Loss {nat_loss.val:.4f} ({nat_loss.avg:.4f})\t'
                  'Nat_Acc@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                  'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                  'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, nat_loss=nat_losses, nat_top1=nat_top1,adv_loss=adv_losses,adv_top1=adv_top1))



def validate(val_loader,criterion, config, config_pertype, device):
    batch_time = AverageMeter()
    nat_losses = AverageMeter()
    adv_losses = AverageMeter()    
    nat_top1 = AverageMeter()
    adv_top1 = AverageMeter()    
    nat_loss = 0
    adv_loss = 0

    # switch to evaluate mode
    config.model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # if config.gpu is not None:
            #     input = input.cuda(config.gpu, non_blocking=True)
            # target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            # nat_output,adv_output,pert_inputs = config.model(input,target)

            input = input.to(device)
            target = target.to(device)

            config.model.eval()
            # nat_output = config.model(input)
            # print(nat_output)
            # print(input.size())
            net_attack = AttackPGD_each(config.model, config)
            net_attack = net_attack.to(device)
            # pert_inputs = net_attack(input, target)
            nat_output, adv_output, pert_inputs = net_attack(input,target)
            config.model.train()

            nat_loss = criterion(nat_output, target)
            adv_loss = criterion(adv_output, target)            

            # measure accuracy and record loss
            nat_acc1, nat_acc5 = accuracy(nat_output, target, topk=(1, 5))
            adv_acc1, adv_acc5 = accuracy(adv_output, target, topk=(1, 5))
            nat_losses.update(nat_loss.item(), input.size(0))
            adv_losses.update(adv_loss.item(), input.size(0))            
            nat_top1.update(nat_acc1[0], input.size(0))
            adv_top1.update(adv_acc1[0], input.size(0))            


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Nat_Loss {nat_loss.val:.4f} ({nat_loss.avg:.4f})\t'
                      'Nat_Acc@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                      'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                      'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'                      
                      .format(
                       i, len(val_loader), batch_time=batch_time, nat_loss=nat_losses,
                          nat_top1=nat_top1,adv_loss=adv_losses,adv_top1=adv_top1))


        print(' * Nat_Acc@1 {nat_top1.avg:.3f} *Adv_Acc@1 {adv_top1.avg:.3f}'
              .format(nat_top1=nat_top1,adv_top1=adv_top1))

        global best_mean_loss
        mean_loss = (adv_losses.avg+nat_losses.avg)/2
        if mean_loss<best_mean_loss and not config.admm:
            best_mean_loss = mean_loss
            print ('new best_mean_loss is {}'.format(mean_loss))
            print ('saving model {}'.format(config.save_model))
            torch.save(config.model.state_dict(),config.save_model)

    return adv_top1.avg

def train_nat(train_loader, criterion, optimizer, epoch, config, config_pertype, device,config_fog, config_snow, transform_rt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    nat_losses = AverageMeter()
    adv_losses = AverageMeter()
    nat_loss = 0
    adv_loss = 0
    nat_top1 = AverageMeter()
    adv_top1 = AverageMeter()

    # switch to train mode
    config.model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if config.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, config)
        else:
            scheduler.step()

        # if config.gpu is not None:
        #     input = input.cuda(config.gpu, non_blocking=True)
        # target = target.cuda(config.gpu, non_blocking=True)

        input = input.to(device)
        target = target.to(device)

        if config.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, config.alpha)



        # compute output
        # nat_output,adv_output,pert_inputs = config.model(input,target)
        config.model = config.model.to(device)

        nat_output = config.model(input)

        if config.nat_type == 'fog':
            for a in range(0, input.shape[0]):
                # print(images[i])
                images_fog = add_fog(input[a], config_fog['t'], config_fog['light'])
                # print(images_fog)
                input[a] = images_fog
        elif config.nat_type == 'snow':
            for a in range(0, input.shape[0]):
                # print(images[i])
                images_snow = add_snow(input[a], config_snow['brightness'])
                # print(images_fog)
                input[a] = images_snow
        elif config.nat_type == 'rt':
            input = transform_rt(input)

        adv_output = config.model(input)





        if config.mixup:
            adv_loss = mixup_criterion(criterion, adv_output, target_a, target_b, lam, config.smooth)
            nat_loss = mixup_criterion(criterion, nat_output, target_a, target_b, lam, config.smooth)
        else:
            adv_loss = criterion(adv_output, target, smooth=config.smooth)
            nat_loss = criterion(nat_output, target, smooth=config.smooth)
        if config.admm:
            admm.admm_update(config, ADMM, device, train_loader, optimizer, epoch, input, i)  # update Z and U
            adv_loss, admm_loss, mixed_loss = admm.append_admm_loss(config, ADMM, adv_loss)  # append admm losss

        # measure accuracy and record loss
        nat_acc1, _ = accuracy(nat_output, target, topk=(1, 5))
        adv_acc1, _ = accuracy(adv_output, target, topk=(1, 5))

        nat_losses.update(nat_loss.item(), input.size(0))
        adv_losses.update(adv_loss.item(), input.size(0))
        adv_top1.update(adv_acc1[0], input.size(0))
        nat_top1.update(nat_acc1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if config.admm:
            mixed_loss.backward()
        else:
            adv_loss.backward()

        if config.masked_progressive:
            with torch.no_grad():
                for name, W in config.model.named_parameters():
                    if name in config.zero_masks:
                        W.grad *= config.zero_masks[name]

        if config.masked_retrain:
            with torch.no_grad():
                for name, W in config.model.named_parameters():
                    if name in config.masks:
                        W.grad *= config.masks[
                            name]  # returns boolean array called mask when weights are above treshhold

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Nat_Loss {nat_loss.val:.4f} ({nat_loss.avg:.4f})\t'
                  'Nat_Acc@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                  'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                  'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, nat_loss=nat_losses, nat_top1=nat_top1, adv_loss=adv_losses, adv_top1=adv_top1))


def validate_nat(val_loader, criterion, config, config_pertype, device, config_fog, config_snow, transform_rt):
    batch_time = AverageMeter()
    nat_losses = AverageMeter()
    adv_losses = AverageMeter()
    nat_top1 = AverageMeter()
    adv_top1 = AverageMeter()
    nat_loss = 0
    adv_loss = 0

    # switch to evaluate mode
    config.model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # if config.gpu is not None:
            #     input = input.cuda(config.gpu, non_blocking=True)
            # target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            # nat_output,adv_output,pert_inputs = config.model(input,target)

            input = input.to(device)
            target = target.to(device)
            config.model = config.model.to(device)

            nat_output = config.model(input)

            if config.nat_type == 'fog':
                for a in range(0, input.shape[0]):
                    # print(images[i])
                    images_fog = add_fog(input[a], config_fog['t'], config_fog['light'])
                    # print(images_fog)
                    input[a] = images_fog
            elif config.nat_type == 'snow':
                for a in range(0, input.shape[0]):
                    # print(images[i])
                    images_snow = add_snow(input[a], config_snow['brightness'])
                    # print(images_fog)
                    input[a] = images_snow
            elif config.nat_type == 'rt':
                input = transform_rt(input)

            adv_output = config.model(input)

            nat_loss = criterion(nat_output, target)
            adv_loss = criterion(adv_output, target)

            # measure accuracy and record loss
            nat_acc1, nat_acc5 = accuracy(nat_output, target, topk=(1, 5))
            adv_acc1, adv_acc5 = accuracy(adv_output, target, topk=(1, 5))
            nat_losses.update(nat_loss.item(), input.size(0))
            adv_losses.update(adv_loss.item(), input.size(0))
            nat_top1.update(nat_acc1[0], input.size(0))
            adv_top1.update(adv_acc1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Nat_Loss {nat_loss.val:.4f} ({nat_loss.avg:.4f})\t'
                      'Nat_Acc@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                      'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                      'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'
                    .format(
                    i, len(val_loader), batch_time=batch_time, nat_loss=nat_losses,
                    nat_top1=nat_top1, adv_loss=adv_losses, adv_top1=adv_top1))

        print(' * Nat_Acc@1 {nat_top1.avg:.3f} *Adv_Acc@1 {adv_top1.avg:.3f}'
              .format(nat_top1=nat_top1, adv_top1=adv_top1))

        global best_mean_loss
        mean_loss = (adv_losses.avg + nat_losses.avg) / 2
        if mean_loss < best_mean_loss and not config.admm:
            best_mean_loss = mean_loss
            print('new best_mean_loss is {}'.format(mean_loss))
            print('saving model {}'.format(config.save_model))
            torch.save(config.model.state_dict(), config.save_model)

    return adv_top1.avg



def train_clean(train_loader,criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to train mode
    config.model.train()

    end = time.time()
    config.model = config.model.to(device)

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if config.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, config)
        else:
            scheduler.step()

        if config.gpu is not None:
            input = input.cuda(config.gpu, non_blocking=True)
        target = target.cuda(config.gpu, non_blocking=True)

        if config.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, config.alpha)


        input = input.to(device)
        target = target.to(device)

        # compute output
        output = config.model(input)

        if config.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, config.smooth)
        else:
            ce_loss = criterion(output, target, smooth=config.smooth)

        if config.admm:
            admm.admm_update(config,ADMM,device,train_loader,optimizer,epoch,input,i)   # update Z and U
            ce_loss,admm_loss,mixed_loss = admm.append_admm_loss(config,ADMM,ce_loss) # append admm losss

        # measure accuracy and record loss
        acc1,_ = accuracy(output, target, topk=(1,5))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        if config.admm:
            mixed_loss.backward()
        else:
            ce_loss.backward()

        if config.masked_progressive:
            with torch.no_grad():
                for name,W in config.model.named_parameters():
                    if name in config.zero_masks:
                            W.grad *=config.zero_masks[name]


        if config.masked_retrain:
            with torch.no_grad():
                for name,W in config.model.named_parameters():
                    if name in config.masks:
                            W.grad *= config.masks[name] #returns boolean array called mask when weights are above treshhold

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))



def validate_clean(val_loader,criterion, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to evaluate mode
    config.model.eval()
    config.model = config.model.to(device)

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if config.gpu is not None:
                input = input.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)

            # compute output

            input = input.to(device)
            target = target.to(device)

            output = config.model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      .format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

        print(' * Acc@1 {top1.avg:.3f} '
              .format(top1=top1))
        global best_acc
        if top1.avg.item()>best_acc and not config.admm:
            best_acc = top1.avg.item()
            print ('new best_acc is {top1.avg:.3f}'.format(top1=top1))
            print ('saving model {}'.format(config.save_model))
            if config.stage == 'admm':
                if not os.path.exists(config.save_model[:-17]):  # 如果路径不存在
                    os.makedirs(args.checkpoint_loc[:-17])
            torch.save(config.model.state_dict(),config.save_model)

    return top1.avg





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


config_fog = {
    't': 0.4,
    'light': 1.2
}

# config_fog = dict(config_fog=config_fog_2)

brightness_2 = 8

config_snow = dict(brightness= brightness_2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if config.admm:
    print('sparsity type:', config.sparsity_type)
    print('per type is:', config.per_type)
    print('adv type is:', config.adv_type)
    print('nat type is:', config.nat_type)
    print('all epochs:', config.epochs)
    print('==> Loading from {}'.format(config.load_model))
    print('==> Saving to {}'.format(config.save_model))
    print(config.epsilon)
    print(config.num_steps)
    print(config.step_size)
    if config.per_type == 'adv':
        validate(testloader, criterion, config, config_l1_16, device)
    elif config.per_type == 'weather':
        validate_nat(testloader,criterion,config, config_l1_16, device, config_fog, config_snow, transform_rt)
    elif config.per_type == 'clean':
        validate_clean(testloader,criterion, config)

if config.masked_retrain:
    print('sparsity type:', config.sparsity_type)
    print('per type is:', config.per_type)
    print('adv type is:', config.adv_type)
    print('nat type is:', config.nat_type)
    print('all epochs:', config.epochs)
    print('==> Loading from {}'.format(config.load_model))
    print('==> Saving to {}'.format(config.save_model))
    print(config.epsilon)
    print(config.num_steps)
    print(config.step_size)
    # make sure small weights are pruned and confirm the acc
    print ("<============masking both weights and gradients for retrain")
    admm.masking(config)
    print ("<============testing sparsity before retrain")
    admm.test_sparsity(config)
    if config.per_type == 'adv':
        validate(testloader, criterion, config, config_l1_16, device)
    elif config.per_type == 'weather':
        validate_nat(testloader,criterion,config, config_l1_16, device, config_fog, config_snow, transform_rt)
    elif config.per_type == 'clean':
        validate_clean(testloader,criterion, config)
if config.masked_progressive:
    admm.zero_masking(config)



for epoch in range(start_epoch, start_epoch+config.epochs):
    print('sparsity type:', config.sparsity_type)
    print('per type is:', config.per_type)
    print('adv type is:', config.adv_type)
    print('nat type is:', config.nat_type)
    print('all epochs:', config.epochs)
    print('==> Loading from {}'.format(config.load_model))
    print('==> Saving to {}'.format(config.save_model))
    print(config.epsilon)
    print(config.num_steps)
    print(config.step_size)
    if config.per_type == 'adv':
        train(trainloader, criterion, optimizer, epoch, config, config_l1_16, device)
        validate(testloader, criterion, config, config_l1_16, device)
    elif config.per_type == 'weather':
        train_nat(trainloader, criterion, optimizer, epoch, config, config_l1_16, device, config_fog, config_snow, transform_rt)
        validate_nat(testloader, criterion, config, config_l1_16, device, config_fog, config_snow, transform_rt)
    elif config.per_type == 'clean':
        train_clean(trainloader,criterion,optimizer,epoch,config)
        validate_clean(testloader, criterion, config)

####LOG HERE###
if config.logging:
    logger.info(f'---Final Results---')
    logger.info(f'overall best_acc is {best_acc}')

print ('overall  best_mean_loss is {}'.format(best_mean_loss))


if config.masked_retrain:
    print ("<=====confirm sparsity")
    admm.test_sparsity(config)


if config.save_model and config.admm:
    print ('saving model {}'.format(config.save_model))
    torch.save(config.model.state_dict(),config.save_model)



