import random
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def load_mnist_test_data(test_batch_size=1):
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    # MNIST Dataset
    test_dataset = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    return test_loader


def load_cifar10_test_data(test_batch_size=1):
    # CIFAR10 Dataset
    test_dataset = dsets.CIFAR10('./data/cifar10-py', download=True, train=False, transform=transforms.ToTensor())
    # num_training_imgs = int(len(test_dataset) * 0.05)
    # torch.manual_seed(0)
    # test_data_small, test_data_big = torch.utils.data.random_split(test_dataset, [num_training_imgs, len(test_dataset) - num_training_imgs])
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    return test_loader

def load_tinyimagenet_test_data(test_batch_size=1):
    import tiny_imagenet
    # # CIFAR10 Dataset
    # test_dataset = dsets.CIFAR10('./data/cifar10-py', download=True, train=False, transform=transforms.ToTensor())
    # # num_training_imgs = int(len(test_dataset) * 0.05)
    # # torch.manual_seed(0)
    # # test_data_small, test_data_big = torch.utils.data.random_split(test_dataset, [num_training_imgs, len(test_dataset) - num_training_imgs])
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)



# 133
#     testset = tiny_imagenet.TinyImageNet200(root='/home/chenghao/MoRE_final_ensure/cifar_16_resenet/data', download=True, train=False, transform=transforms.ToTensor())
# 148
    testset = tiny_imagenet.TinyImageNet200(root='/home/chenan/haotest/cifar_16_resenet/data', download=True, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)



    return test_loader

def load_imagenet_test_data(test_batch_size=1, folder='../val/'):
    val_dataset = dsets.ImageFolder(
        folder,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    rand_seed = 42

    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=True)

    return val_loader
