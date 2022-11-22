# %%
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Optional, Union
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import wandb

import plotly.express as px
import plotly.graph_objects as go

MAIN = __name__ == "__main__"

device = t.device("cuda:0") if t.cuda.is_available() else t.device("cpu")

# %%

def gradient_images(n_images: int, img_size: tuple[int, int, int]) -> t.Tensor:
    '''
    Generate n_images of img_size, each a color gradient
    '''
    (C, H, W) = img_size
    corners = t.randint(0, 255, (2, n_images, C))
    xs = t.linspace(0, W / (W + H), W)
    ys = t.linspace(0, H / (W + H), H)
    (x, y) = t.meshgrid(xs, ys, indexing="xy")
    grid = x + y
    grid = grid / grid[-1, -1]
    grid = repeat(grid, "h w -> b c h w", b=n_images, c=C)
    base = repeat(corners[0], "n c -> n c h w", h=H, w=W)
    ranges = repeat(corners[1] - corners[0], "n c -> n c h w", h=H, w=W)
    gradients = base + grid * ranges
    assert gradients.shape == (n_images, C, H, W)
    return gradients / 255


def plot_img(img: t.Tensor, title: Optional[str] = None) -> None:
    img = rearrange(img, "c h w -> h w c")
    img = (255 * img).to(t.uint8)
    fig = px.imshow(img, title=title)
    fig.update_layout(margin=dict(t=60, l=40, r=40, b=40))
    fig.show()

def plot_img_slideshow(imgs: t.Tensor, title: Optional[str] = None) -> None:
    imgs = rearrange(imgs, "b c h w -> b h w c")
    imgs = (255 * imgs).to(t.uint8)
    fig = px.imshow(imgs, animation_frame=0, title=title)
    fig.show()

if MAIN:
    print("A few samples from the input distribution: ")
    img_shape = (3, 16, 16)
    n_images = 5
    imgs = gradient_images(n_images, img_shape)
    for i in range(n_images):
        plot_img(imgs[i])


# %%

def normalize_img(img: t.Tensor) -> t.Tensor:
    return img * 2 - 1

def denormalize_img(img: t.Tensor) -> t.Tensor:
    return ((img + 1) / 2).clamp(0, 1)

if MAIN:
    plot_img(imgs[0], "Original")
    plot_img(normalize_img(imgs[0]), "Normalized")
    plot_img(denormalize_img(normalize_img(imgs[0])), "Denormalized")

# %%

def linear_schedule(max_steps: int, min_noise: float = 0.0001, max_noise: float = 0.02) -> t.Tensor:
    '''
    Return the forward process variances as in the paper.

    max_steps: total number of steps of noise addition
    out: shape (step=max_steps, ) the amount of noise at each step
    '''
    return t.linspace(min_noise, max_noise, max_steps)

if MAIN:
    betas = linear_schedule(max_steps=200)

# %%

def q_forward_simple(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    '''Return the input image with num_steps iterations of noise added according to schedule.
    x: shape (channels, height, width)
    betas: shape (T, ) with T >= num_steps

    out: shape (channels, height, width)
    '''
    # zips have length equal to the minimum of the zipped elements
    for step, beta in zip(range(num_steps), betas):
        x *= (1 - beta) ** 0.5
        x += (beta ** 0.5) * t.randn_like(x)
    
    return x


if MAIN:
    noise_steps = [1, 5, 10, 20, 50, 100, 200]
    x = normalize_img(gradient_images(1, (3, 16, 16))[0])
    arr = t.zeros((len(noise_steps)+1, *x.shape))
    print(x.shape, arr.shape)
    arr[0] = denormalize_img(x)
    for i, n in enumerate(noise_steps, 1):
        xt = q_forward_simple(x, n, betas)
        arr[i] = denormalize_img(xt)
    plot_img_slideshow(arr, f"Noise steps {noise_steps}")
# %%

def q_forward_fast(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    '''Equivalent to Equation 2 but without a for loop.'''
    alphas = 1 - betas
    alpha_bar = t.prod(alphas)
    xt = ((1 - betas[-1]) ** 0.5) * (betas[-1] ** 0.5) * t.rand_like(x)
    return xt


if MAIN:
    x = normalize_img(gradient_images(1, (3, 16, 16))[0])
    arr = t.zeros((len(noise_steps)+1, *x.shape))
    print(x.shape, arr.shape)
    arr[0] = denormalize_img(x)
    for i, n in enumerate(noise_steps, 1):
        xt = q_forward_simple(x, n, betas)
        arr[i] = denormalize_img(xt)
    plot_img_slideshow(arr, f"Noise steps {noise_steps}")

# %%

class NoiseSchedule(nn.Module):
    betas: t.Tensor
    alphas: t.Tensor
    alpha_bars: t.Tensor

    def __init__(self, max_steps: int, device: Union[t.device, str]) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.device = device

        betas = linear_schedule(max_steps)
        self.register_buffer("betas", betas)
        alphas = 1 - betas
        self.register_buffer("alphas", alphas)
        alpha_bars = t.cumprod(alphas, dim=-1)
        self.register_buffer("alpha_bars", alpha_bars)

        self.to(device)

    @t.inference_mode()
    def beta(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        '''
        Returns the beta(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        '''
        return self.betas[num_steps]

    @t.inference_mode()
    def alpha(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        '''
        Returns the alphas(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        '''
        return self.alphas[num_steps]

    @t.inference_mode()
    def alpha_bar(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        '''
        Returns the alpha_bar(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        '''
        return self.alpha_bars[num_steps]

    def __len__(self) -> int:
        return self.max_steps

# %%

def noise_img(
    img: t.Tensor, noise_schedule: NoiseSchedule, max_steps: Optional[int] = None
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    '''
    Adds a uniform random number of steps of noise to each image in img.

    img: An image tensor of shape (B, C, H, W)
    noise_schedule: The NoiseSchedule to follow
    max_steps: if provided, only perform the first max_steps of the schedule

    Returns a tuple composed of:
    num_steps: an int tensor of shape (B,) of the number of steps of noise added to each image
    noise: the unscaled, standard Gaussian noise added to each image, a tensor of shape (B, C, H, W)
    noised: the final noised image, a tensor of shape (B, C, H, W)
    '''
    # Find the actual number of max steps
    if max_steps is None:
        max_steps = t.inf
    max_steps = min(max_steps, len(noise_schedule))
    num_steps = t.randint(low=1, high=max_steps, size=(img.shape[0],), device=img.device)

    alpha_bars = noise_schedule.alpha_bar(num_steps).to(img.device)
    noise = t.randn_like(img)

    img_sf = rearrange(alpha_bars ** 0.5, "b -> b 1 1 1")
    noise_sf = rearrange((1 - alpha_bars) ** 0.5, "b -> b 1 1 1")
    noised = img_sf * img + noise_sf * noise

    return num_steps, noise, noised


if MAIN:
    noise_schedule = NoiseSchedule(max_steps=200, device="cpu")
    img = gradient_images(1, (3, 16, 16))
    (num_steps, noise, noised) = noise_img(normalize_img(img), noise_schedule, max_steps=10)
    plot_img(img[0], "Gradient")
    plot_img(noise[0], "Applied Unscaled Noise")
    plot_img(denormalize_img(noised[0]), "Gradient with Noise Applied")

# %%

def reconstruct(noisy_img: t.Tensor, noise: t.Tensor, num_steps: t.Tensor, noise_schedule: NoiseSchedule) -> t.Tensor:
    '''
    Subtract the scaled noise from noisy_img to recover the original image. We'll later use this with the model's output to log reconstructions during training. We'll use a different method to sample images once the model is trained.

    Returns img, a tensor with shape (B, C, H, W)
    '''
    alpha_bars = rearrange(noise_schedule.alpha_bar(num_steps), "b -> b 1 1 1")
    reconstructed = noisy_img / alpha_bars.sqrt() - noise * ((1 - alpha_bars) / alpha_bars).sqrt()
    return reconstructed

if MAIN:
    reconstructed = reconstruct(noised, noise, num_steps, noise_schedule)
    denorm = denormalize_img(reconstructed)
    plot_img(img[0], "Original Gradient")
    plot_img(denorm[0], "Reconstruction")
    t.testing.assert_close(denorm, img)

# %%

class DiffusionModel(nn.Module, ABC):
    img_shape: tuple[int, ...]
    noise_schedule: Optional[NoiseSchedule]

    @abstractmethod
    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        ...


@dataclass(frozen=True)
class TinyDiffuserConfig:
    img_shape: tuple[int, ...]
    hidden_size: int
    max_steps: int


class TinyDiffuser(DiffusionModel):
    def __init__(self, config: TinyDiffuserConfig):
        '''
        A toy diffusion model composed of an MLP (Linear, ReLU, Linear)
        '''
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.img_shape = config.img_shape
        self.noise_schedule = None
        self.max_steps = config.max_steps

        c, h, w = img_shape
        img_numel = c*h*w
        
        self.mlp = nn.Sequential(*[
            nn.Linear(img_numel + 1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, img_numel),
            Rearrange("b (c w h) -> b c h w", c=c, w=w, h=h)
        ])

    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        '''
        Given a batch of images and noise steps applied, attempt to predict the noise that was applied.
        images: tensor of shape (B, C, H, W)
        num_steps: tensor of shape (B,)

        Returns
        noise_pred: tensor of shape (B, C, H, W)
        '''
        input = t.concat([
            images.flatten(1),
            num_steps.unsqueeze(1)
        ], dim=1)

        return self.mlp(input)

if MAIN:
    img_shape = (3, 4, 5)
    n_images = 5
    imgs = gradient_images(n_images, img_shape)
    n_steps = t.zeros(imgs.size(0))
    model_config = TinyDiffuserConfig(img_shape, 16, 100)
    model = TinyDiffuser(model_config)
    out = model(imgs, n_steps)
    plot_img(out[0].detach(), "Noise prediction of untrained model")

# %%

def log_images(
    img: t.Tensor, noised: t.Tensor, noise: t.Tensor, noise_pred: t.Tensor, reconstructed: t.Tensor, num_images: int = 3
) -> list[wandb.Image]:
    """
    Convert tensors to a format suitable for logging to Weights and Biases. Returns an image with the ground truth in the upper row, and model reconstruction on the bottom row. Left is the noised image, middle is noise, and reconstructed image is in the rightmost column.
    """
    actual = t.cat((noised, noise, img), dim=-1)
    pred = t.cat((noised, noise_pred, reconstructed), dim=-1)
    log_img = t.cat((actual, pred), dim=-2)
    images = [wandb.Image(i) for i in log_img[:num_images]]
    return images

# TODO: use testset
def train(
    model: DiffusionModel, config_dict: dict[str, Any], trainset: TensorDataset, testset: Optional[TensorDataset] = None
) -> DiffusionModel:

    wandb.init(project="diffusion_models", config=config_dict)
    config = wandb.config
    print(f"Training with config: {config}")

    t_last = time.time()

    model.to(device).train()
    wandb.watch(model, log="all", log_freq=15)
    
    optimizer = t.optim.Adam(model.parameters())

    noise_schedule = NoiseSchedule(max_steps=200, device=device)

    trainloader = DataLoader(trainset, batch_size=config_dict["batch_size"], shuffle=True)

    n_examples_seen = 0

    for epoch in range(config_dict["epochs"]):

        progress_bar = tqdm(trainloader)

        for (img,) in progress_bar:

            img = img.to(device)
            num_steps, noise, noised = noise_img(img, noise_schedule, config_dict["max_steps"])

            noise_pred = model(noised, num_steps)

            loss = F.mse_loss(noise, noise_pred)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch+1}, Loss = {loss.item():>10.3f}")

            n_examples_seen += img.shape[0]
            wandb.log({"loss": loss.item()}, step=n_examples_seen)

            if time.time() - t_last > 5:
                t_last += 5
                with t.inference_mode():
                    reconstructed = reconstruct(noised, noise, num_steps, noise_schedule)
                    images = log_images(img, noised, noise, noise_pred, reconstructed)
                    wandb.log({"images": images}, step=n_examples_seen)
    
    wandb.run.save()
    wandb.finish()

    return model    

if MAIN:
    config: dict[str, Any] = dict(
        lr=0.001,
        image_shape=(3, 4, 5),
        hidden_size=128,
        epochs=20,
        max_steps=100,
        batch_size=128,
        img_log_interval=200,
        n_images_to_log=3,
        n_images=50000,
        n_eval_images=1000,
        device=device,
    )
    images = normalize_img(gradient_images(config["n_images"], config["image_shape"]))
    train_set = TensorDataset(images)
    images = normalize_img(gradient_images(config["n_eval_images"], config["image_shape"]))
    test_set = TensorDataset(images)
    model_config = TinyDiffuserConfig(config["image_shape"], config["hidden_size"], config["max_steps"])
    model = TinyDiffuser(model_config).to(config["device"])
    model = train(model, config, train_set, test_set)

# %%