import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

normalize_cifar10 = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.247, 0.2435, 0.2616],
)

normalize_cifar100 = transforms.Normalize(
    mean=[0.5071, 0.4865, 0.4409],
    std=[0.2673, 0.2564, 0.2762],
)

aug_transforms = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
]

transform_cifar = {
    10: {
        'train': transforms.Compose([
            *aug_transforms,
            transforms.ToTensor(),
            normalize_cifar10,
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            normalize_cifar10,
        ]),
    },
    100: {
        'train': transforms.Compose([
            *aug_transforms,
            transforms.ToTensor(),
            normalize_cifar100,
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            normalize_cifar100,
        ]),
    },
}

cifar_data = {10: CIFAR10, 100: CIFAR100}

def train_val_samplers(n, split_shuffle=True, split_seed=0, val_size=0.1):
    if split_shuffle:
        inds = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed))
    else:
        inds = torch.arange(n)
    split_idx = int((1.0 - val_size) * n)
    train_sampler = SubsetRandomSampler(inds[:split_idx])
    val_sampler = SubsetRandomSampler(inds[split_idx:])
    return train_sampler, val_sampler

def cifar_train_val(root_dir='./cifar_data', version=10,
                    split_shuffle=True, split_seed=0, val_size=0.1,
                    batch_size=128, num_workers=2, pin_memory=False):
    train_dataset = cifar_data[version](root=root_dir, train=True, download=True,
                                        transform=transform_cifar[version]['train'])
    val_dataset = cifar_data[version](root=root_dir, train=True, download=True,
                                        transform=transform_cifar[version]['test'])
    train_sampler, val_sampler = train_val_samplers(len(train_dataset), split_shuffle,
                                                    split_seed, val_size)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler,
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return train_dataloader, val_dataloader

def cifar_test(root_dir='./cifar_data', version=10,
               batch_size=128, num_workers=2, pin_memory=False):
    test_dataset = cifar_data[version](root=root_dir, train=False, download=True,
                                       transform=transform_cifar[version]['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=pin_memory)
    return test_dataloader
