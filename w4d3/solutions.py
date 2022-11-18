# %% 

import torch as t
from torch import nn, optim
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import plotly.express as px
import torchinfo
import time
import wandb

MAIN = __name__ == "__main__"

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
assert str(device) == "cuda:0"

# %%
# ============================================ Data ============================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = datasets.MNIST(root="./data/mnist/", train=True, transform=transform, download=True)
testset = datasets.MNIST(root="./data/mnist/", train=False, transform=transform, download=True)

data_to_plot = dict()
for data, target in DataLoader(testset, batch_size=1):
    if target.item() not in data_to_plot:
        data_to_plot[target.item()] = data.squeeze()
        if len(data_to_plot) == 10:
            break
data_to_plot = t.stack([data_to_plot[i] for i in range(10)]).to(t.float).unsqueeze(1)

loss_fn = nn.MSELoss()

print_output_interval = 10

epochs = 10

# %%
# ============================================ Autoencoders ============================================


class Autoencoder(nn.Module):

    def __init__(self, latent_dim_size):
        super().__init__()

        in_features_list = (28*28, 100)
        out_features_list = (100, latent_dim_size)

        encoder = [("rearrange", Rearrange("batch 1 height width -> batch (height width)"))]
        for i, (ic, oc) in enumerate(zip(in_features_list, out_features_list), 1):
            encoder.append((f"fc{i}", nn.Linear(ic, oc)))
            if i < len(in_features_list):
                encoder.append((f"relu{i}", nn.ReLU()))
        self.encoder = nn.Sequential(OrderedDict(encoder))

        decoder = []
        for i, (ic, oc) in enumerate(zip(out_features_list[::-1], in_features_list[::-1]), 1):
            decoder.append((f"fc{i}", nn.Linear(ic, oc)))
            if i < len(in_features_list):
                decoder.append((f"relu{i}", nn.ReLU()))
        decoder.append(("rearrange", Rearrange("batch (height width) -> batch 1 height width", height=28)))
        self.decoder = nn.Sequential(OrderedDict(decoder))

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if MAIN:
    latent_dim_size = 5
    batch_size = 128

    model = Autoencoder(latent_dim_size).to(device)

    optimizer = optim.Adam(model.parameters())
    torchinfo.summary(model, input_data=trainset[0][0].unsqueeze(0))

# %%

class AutoencoderLarge(nn.Module):

    def __init__(self, latent_dim_size):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, 128),
            nn.ReLU(),
            nn.Linear(128, 7 * 7 * 32),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c w h", c=32, h=7, w=7),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if MAIN:
    img_shape = (28, 28)
    batch_size = 128

    model = AutoencoderLarge(latent_dim_size=5)

    optimizer = optim.Adam(model.parameters())

    torchinfo.summary(model, input_data=trainset[0][0].unsqueeze(0))


# %%

def show_images(model, data_to_plot, return_arr=False):

    device = next(model.parameters()).device
    data_to_plot = data_to_plot.to(device)
    output = model(data_to_plot)
    if isinstance(output, tuple):
        output = output[0]

    both = t.concat((data_to_plot.squeeze(), output.squeeze()), dim=0).cpu().detach().numpy()
    both = np.clip((both * 0.3081) + 0.1307, 0, 1)

    if return_arr:
        arr = rearrange(both, "(b1 b2) h w -> (b1 h) (b2 w) 1", b1=2)
        return arr

    fig = px.imshow(both, facet_col=0, facet_col_wrap=10, color_continuous_scale="greys_r")
    fig.update_layout(coloraxis_showscale=False).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    for i in range(10):
        fig.layout.annotations[i]["text"] = ""
        fig.layout.annotations[i+10]["text"] = str(i)
    fig.show()


# %%

def train_autoencoder(model, optimizer, loss_fn, trainset, data_to_plot, epochs, batch_size, print_output_interval=15, use_wandb=True):

    t_last = time.time()
    examples_seen = 0

    model.to(device).train()
    data_to_plot = data_to_plot.to(device)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    if use_wandb:
        wandb.init()
        wandb.watch(model, log="all", log_freq=15)

    for epoch in range(epochs):

        progress_bar = tqdm(trainloader)

        for img, label in progress_bar:

            examples_seen += img.size(0)

            img = img.to(device)
            img_reconstructed = model(img)

            loss = loss_fn(img, img_reconstructed)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch+1}, Loss = {loss.item():>10.3f}")

            if use_wandb:
                wandb.log({"loss": loss.item()}, step=examples_seen)
            
            if time.time() - t_last > print_output_interval:
                t_last += print_output_interval
                with t.inference_mode():
                    if use_wandb:
                        arr = show_images(model, data_to_plot, return_arr=True)
                        images = wandb.Image(arr, caption="Top: original, Bottom: reconstructed")
                        wandb.log({"images": [images, images]}, step=examples_seen)
                    else:
                        show_images(model, data_to_plot, return_arr=False)

    wandb.run.save()
    wandb.finish()

    return model

# %%
if MAIN:
    model = train_autoencoder(model, optimizer, loss_fn, trainset, data_to_plot, epochs, batch_size, print_output_interval=10)

# %%
if MAIN:
    # Choose number of interpolation points
    n_points = 11

    # Constructing latent dim data by making two of the dimensions vary independently between 0 and 1
    latent_dim_data = t.zeros((n_points, n_points, latent_dim_size), device=device)
    x = t.linspace(-1, 1, n_points)
    latent_dim_data[:, :, 0] = x.unsqueeze(0)
    latent_dim_data[:, :, 1] = x.unsqueeze(1)
    # Rearranging so we have a single batch dimension
    latent_dim_data = rearrange(latent_dim_data, "b1 b2 latent_dim -> (b1 b2) latent_dim")

    # Getting model output, and normalising & truncating it in the range [0, 1]
    output = model.decoder(latent_dim_data).detach().cpu().numpy()
    output_truncated = np.clip((output * 0.3081) + 0.1307, 0, 1)
    output_single_image = rearrange(output_truncated, "(b1 b2) 1 height width -> (b1 height) (b2 width)", b1=n_points)

    # Plotting results
    fig = px.imshow(output_single_image, color_continuous_scale="greys_r")
    fig.update_layout(
        title_text="Decoder output from varying first two latent space dims", title_x=0.5,
        coloraxis_showscale=False, 
        xaxis=dict(title_text="x0", tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x]),
        yaxis=dict(title_text="x1", tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x])
    )
    fig.show()

    # def write_to_html(fig, filename):
    #     with open(f"{filename}.html", "w") as f:
    #         f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
    # write_to_html(fig, 'autoencoder_interpolation.html')

# %%

import pandas as pd

def make_scatter_plot(trainset, n_examples=1000):
    trainloader = DataLoader(trainset, batch_size=64)
    df_list = []
    device = next(model.parameters()).device
    for img, label in trainloader:
        output = model.encoder(img.to(device)).detach().cpu().numpy()
        for label_single, output_single in zip(label, output):
            df_list.append({
                "x0": output_single[0],
                "x1": output_single[1],
                "label": str(label_single.item()),
            })
        if (n_examples is not None) and (len(df_list) >= n_examples):
            break
    df = pd.DataFrame(df_list)
    fig = px.scatter(df, x="x0", y="x1", color="label")
    fig.show()

if MAIN:
    make_scatter_plot(trainset, n_examples=2000)
    # Result: better than I expected, density is pretty uniform and most of the space is utilised, 
    # although this is only a cross section of 2 dimensions so is a small subset of total space

# %%

# ============================================ VAEs ============================================

class VAE(nn.Module):

    def __init__(self, latent_dim_size):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim_size*2),
            Rearrange("b (n latent_dim) -> n b latent_dim", n=2) # makes it easier to separate mu and sigma
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, 128),
            nn.ReLU(),
            nn.Linear(128, 7 * 7 * 32),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c w h", c=32, h=7, w=7),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        mu, logsigma = self.encoder(x)
        sigma = t.exp(logsigma)
        # z = t.randn(self.latent_dim_size).to(device)
        z = t.randn_like(mu)
        z = mu + sigma * z
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logsigma

if MAIN:
    batch_size = 128

    model = VAE(latent_dim_size=5).to(device).train()

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

    loss_fn = nn.MSELoss()

    torchinfo.summary(model, input_data=trainset[0][0].unsqueeze(0).to(device))

# %%

def plot_loss(loss_fns_dict):
    df = pd.DataFrame(loss_fns_dict)
    fig = px.line(df, template="simple_white")
    fig.show()

def train_vae(model, optimizer, loss_fn, trainset, data_to_plot, epochs, batch_size, beta=0.1, print_output_interval=15):

    t_last = time.time()
    loss_fns = {"reconstruction": [],"kl_div": []}
    model.to(device).train()
    data_to_plot = data_to_plot.to(device)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):

        progress_bar = tqdm(trainloader)

        for img, label in progress_bar:

            img = img.to(device)
            img_reconstructed, mu, logsigma = model(img)

            reconstruction_loss = loss_fn(img, img_reconstructed)
            kl_div_loss = ( 0.5 * (mu ** 2 + t.exp(2 * logsigma) - 1) - logsigma ).mean() * beta

            loss = reconstruction_loss + kl_div_loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch+1}, reconstruction_loss = {reconstruction_loss.item():>10.3f}, kl_div_loss = {kl_div_loss.item():>10.3f}, mean={mu.mean():>10.3f}, std={t.exp(logsigma).mean():>10.3f}")
            loss_fns["reconstruction"].append(reconstruction_loss.item())
            loss_fns["kl_div"].append(kl_div_loss.item())

            if time.time() - t_last > print_output_interval:
                t_last += print_output_interval
                with t.inference_mode():
                    show_images(model, data_to_plot)
                    plot_loss(loss_fns)
    return model

if MAIN:
    model = train_vae(model, optimizer, loss_fn, trainset, data_to_plot, epochs, batch_size, print_output_interval)

# %%

if MAIN:
    # Choose number of interpolation points
    n_points = 11

    # Constructing latent dim data by making two of the dimensions vary independently between 0 and 1
    latent_dim_data = t.zeros((n_points, n_points, latent_dim_size), device=device)
    x = t.linspace(-1, 1, n_points)
    latent_dim_data[:, :, 4] = x.unsqueeze(0)
    latent_dim_data[:, :, 3] = x.unsqueeze(1)
    # Rearranging so we have a single batch dimension
    latent_dim_data = rearrange(latent_dim_data, "b1 b2 latent_dim -> (b1 b2) latent_dim")

    # Getting model output, and normalising & truncating it in the range [0, 1]
    output = model.decoder(latent_dim_data).detach().cpu().numpy()
    output_truncated = np.clip((output * 0.3081) + 0.1307, 0, 1)
    output_single_image = rearrange(output_truncated, "(b1 b2) 1 height width -> (b1 height) (b2 width)", b1=n_points)

    # Plotting results
    fig = px.imshow(output_single_image, color_continuous_scale="greys_r")
    fig.update_layout(
        title_text="Decoder output from varying first two latent space dims", title_x=0.5,
        coloraxis_showscale=False, 
        xaxis=dict(title_text="x0", tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x]),
        yaxis=dict(title_text="x1", tickmode="array", tickvals=list(range(14, 14+28*n_points, 28)), ticktext=[f"{i:.2f}" for i in x])
    )
    fig.show()
