# import wandb
import gc
from torch import long
import torch
import matplotlib.pyplot as plt
import argparse
from src.model import AutoEncoder
from src.train import train
from src.utils import generate_dataloader

parser = argparse.ArgumentParser(description="training auto encoder for mnist")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--latent_dims", type = int, default = 64)
parser.add_argument("--beta_1", type = float, default = 0.99)
parser.add_argument("--beta_2", type = float, default = 0.9)
parser.add_argument("--num_epochs", type = int, default = 128)
parser.add_argument("--max_grad_norm", type = float, default = 1.0)
parser.add_argument("--verbose", type = int, default = 8)
parser.add_argument("--step_size", type = int, default = 32)

args = vars(parser.parse_args())

lr = args["lr"]
batch_size = args["batch_size"]
latent_dims = args["latent_dims"]
layer_filters = [16, 32]
gamma = args["gamma"]
beta_1 = args["beta_1"]
beta_2 = args["beta_2"]
num_epochs = args["num_epochs"]
max_grad_norm = args["max_grad_norm"]
verbose = args["verbose"]
step_size = args["step_size"]

# cuda check
if torch.cuda.is_available():
    print("cuda available : ", torch.cuda.is_available())
    print("cuda device count : ", torch.cuda.device_count())
    device = "cuda:" + str(args["gpu_num"])
else:
    device = "cpu" 

# Network loaded
auto_encoder = AutoEncoder(layer_filters = layer_filters, input_shape = (1,28,28), kernel_size = 3, latent_dims = latent_dims)

# device allocation
auto_encoder.to(device)

# opimizer, loss and scheduler
optimizer = torch.optim.AdamW(auto_encoder.parameters(), lr = lr, betas = [beta_1, beta_2], weight_decay=gamma)
loss_fn = torch.nn.MSELoss(reduction = 'sum')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma, last_epoch = -1)

if __name__ == "__main__":

    train_dataloader, valid_dataloader, test_dataloader = generate_dataloader(batch_size)
    train_loss, valid_loss = train(
        auto_encoder,
        num_epochs,
        train_dataloader,
        valid_dataloader,
        True,
        "./weights/best_auto.pt",
        loss_fn,
        optimizer,
        scheduler,
        device = device,
        max_grad_norm = max_grad_norm,
        verbose = verbose
    )
    
    print("training auto encoder process is done....!")