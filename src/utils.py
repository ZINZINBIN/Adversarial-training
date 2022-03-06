import torch
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader
from typing import Optional
import matplotlib.pyplot as plt

mnist_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485,), (0.229,))
])

download_root = "./MNIST_DATSET"

def load_mnist_data(transform = mnist_transform):
    train_dataset = MNIST(download_root, train = True, transform = transform, download=True)
    valid_dataset = MNIST(download_root, train = True, transform = transform, download=True)
    test_dataset = MNIST(download_root, train = False, transform = transform, download = True)
    return train_dataset, valid_dataset, test_dataset

def load_dataloader(train_dataset = None, valid_dataset = None, test_dataset = None, batch_size : Optional[int] = None):
    if train_dataset is not None:
        train_dataloader = DataLoader(train_dataset, batch_size, True)
    else:
        train_dataloader = None

    if valid_dataset is not None:
        valid_dataloader = DataLoader(valid_dataset, batch_size, True)
    else:
        valid_dataloader = None

    if test_dataset is not None:
        test_dataloader = DataLoader(test_dataset, batch_size, True)
    else:
        test_dataloader = None

    return train_dataloader, valid_dataloader, test_dataloader

def generate_dataloader(batch_size):
    train_dataset, valid_dataset, test_dataset = load_mnist_data(mnist_transform)
    train_dataloader, valid_dataloader, test_dataloader = load_dataloader(train_dataset, valid_dataset, test_dataset, batch_size = batch_size)
    return train_dataloader, valid_dataloader, test_dataloader

from sklearn.datasets import fetch_openml

def get_mnist_from_sklearn():
    mnist = fetch_openml("mnist_784", data_home = "mnist_784")
    img = mnist.data.reshape(-1,1, 28,28)
    label = mnist.target.reshape(-1,)
    label = np.array([int(x) for x in label])
    return img, label

def plot_results(model : torch.nn.Module, img:np.ndarray, label:np.array, batch_size = 32, save_dir = "./weights/best_auto.pt", device : Optional[str] = "cpu"):
    if device is None:
        device = 'cpu'
    model.load_state_dict(torch.load(save_dir, map_location=device))

    img = torch.FloatTensor(img).to(device)
    z = model.encode(img).detach().numpy()
    label = label.reshape(-1,)

    color = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    plt.figure(figsize = (12,10))
    plt.scatter(z[:,0], z[:,1], c = color[label])
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig("./results/auto-encoder-z.png")