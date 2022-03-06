import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import numpy as np
from typing import Optional

def train_per_epoch(model : nn.Module, train_dataloader : DataLoader, device : str = 'cpu', loss_fn = None, optimizer : Optimizer = None, scheduler = None, max_grad_norm : float = 1.0):

    train_loss = 0

    model.to(device)
    model.train()
    for idx, (batch_x, batch_y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch_output = model.forward(batch_x.to(device))
        loss = loss_fn(batch_output, batch_x.to(device))

        batch_num = batch_x.size(0)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        train_loss += loss.cpu().data / batch_num

    if scheduler is not None:
        scheduler.step()

    train_loss /= (idx + 1)

    return train_loss

def valid_per_epoch(model : nn.Module, valid_dataloader : DataLoader, device : str = 'cpu', loss_fn = None, optimizer : Optimizer = None, scheduler = None, max_grad_norm : float = 1.0):

    valid_loss = 0

    model.to(device)
    model.eval()
    for idx, (batch_x, batch_y) in enumerate(valid_dataloader):
        with torch.no_grad():
            optimizer.zero_grad()
            batch_output = model.forward(batch_x.to(device))
            loss = loss_fn(batch_output, batch_x.to(device))
            batch_num = batch_x.size(0)
            valid_loss += loss.cpu().data / batch_num

    valid_loss /= (idx + 1)

    return valid_loss

def train(
    model : nn.Module, 
    num_epochs : int, 
    train_dataloader = None,
    valid_dataloader = None,
    save_best_only = False,
    save_path = "./weights/best_auto.pt",
    loss_fn = None, 
    optimizer = None, 
    scheduler = None,
    device : str = "cpu",
    max_grad_norm = 1.0,
    verbose : Optional[int]= None
    ):

    train_losses = []
    valid_losses = []
    best_loss = np.inf

    model.train()
    model.to(device)
    for n_iter in tqdm(range(num_epochs)):
        # train process

        train_loss = train_per_epoch(model, train_dataloader, device, loss_fn, optimizer, scheduler, max_grad_norm)
        valid_loss = valid_per_epoch(model, valid_dataloader, device, loss_fn, optimizer, scheduler, max_grad_norm)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if(save_best_only):
            if(valid_loss <= best_loss):
                best_loss = valid_loss
                torch.save(model.state_dict(), save_path)

        if verbose is not None and n_iter % verbose == 0:
            print("# iter : {:3d} train_loss : {:.3f}, valid_loss : {:.3f}, best_loss : {:.3f}".format(n_iter + 1, train_loss, valid_loss, best_loss))
    
    return train_losses, valid_losses