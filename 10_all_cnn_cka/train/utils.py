try:
    from cutout import Cutout
except:
    from train.cutout import Cutout


def get_dataloaders(batch_size, use_augment=False, use_cutout=False):
    """Get PyTorch Data Loaders for CIFAR-10."""
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    # Add additional data augmentations
    if use_augment:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if use_cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

    train_transform.transforms.append(transforms.Lambda(lambda x: x.permute(1, 2, 0)))

    trainset = torchvision.datasets.CIFAR10(
        root="../data", train=True, download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=10
    )

    testset = torchvision.datasets.CIFAR10(
        root="../data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=5000, shuffle=False, num_workers=10
    )
    return trainloader, testloader
