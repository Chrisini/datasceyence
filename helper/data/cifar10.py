import torch
from torchvision import datasets, transforms


class Dataset:
    def __init__(self, dataset, _batch_size):
        super(Dataset, self).__init__()
       
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(
            'examples/example_data/cifar', train=True, download=True, transform=data_transform)
        test_dataset = datasets.CIFAR10(
            'examples/example_data/cifar', train=False, download=True, transform=data_transform)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=_batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=_batch_size, shuffle=False)
        