# %%
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
from typing import Any, Optional, Union, Tuple, List, Dict
import torch as t
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import DataLoader
import wandb
from torchvision import transforms
import torchinfo
from torch import nn
import plotly.express as px
from einops.layers.torch import Rearrange
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Lambda
from pathlib import Path
from fancy_einsum import einsum

MAIN = __name__ == "__main__"

from w0d2.solutions import Linear, conv2d, force_pair, IntOrPair
from w0d3.solutions import Sequential
from w2d2.solutions_own_transformer import GELU, PositionalEncoding
from w4d1.solutions import ConvTranspose2d
from w5d1 import utils

device = t.device("cuda:0") if t.cuda.is_available() else t.device("cpu")


# ===============================================================
# =========================== PART 2 ============================
# ===============================================================
# %%

def gradient_images(n_images: int, img_size: Tuple[int, int, int]) -> t.Tensor:
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

def plot_img_grid(imgs: t.Tensor, title: Optional[str] = None, cols: Optional[int] = None) -> None:
    b = imgs.shape[0]
    imgs = rearrange(imgs, "b c h w -> b h w c")
    imgs = (255 * imgs).to(t.uint8)
    if cols is None:
        cols = int(b**0.5) + 1
    fig = px.imshow(imgs, facet_col=0, facet_col_wrap=cols, title=title)
    for annotation in fig.layout.annotations:
        annotation["text"] = ""
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

def q_forward_slow(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
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
        xt = q_forward_slow(x, n, betas)
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
        xt = q_forward_slow(x, n, betas)
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
) -> IntOrPair:
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
    img_shape: Tuple[int, ...]
    noise_schedule: Optional[NoiseSchedule]

    @abstractmethod
    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        ...

@dataclass(frozen=True)
class TinyDiffuserConfig:
    img_shape: Tuple[int, ...]
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
) -> List[wandb.Image]:
    """
    Convert tensors to a format suitable for logging to Weights and Biases. Returns an image with the ground truth in the upper row, and model reconstruction on the bottom row. Left is the noised image, middle is noise, and reconstructed image is in the rightmost column.
    """
    actual = t.cat((noised, noise, img), dim=-1)
    pred = t.cat((noised, noise_pred, reconstructed), dim=-1)
    log_img = t.cat((actual, pred), dim=-2)
    caption = "Top: noised, true noise, original\nBottom: noised, predicted noise, reconstructed"
    images = [wandb.Image(i, caption=caption) for i in log_img[:num_images]]
    return images

def train(
    model: DiffusionModel, config_dict: Dict[str, Any], trainset: TensorDataset, testset: Optional[TensorDataset] = None
) -> DiffusionModel:

    wandb.init(project="diffusion_models", config=config_dict)
    config = wandb.config
    print(f"Training with config: {config}")

    t_last = time.time()

    model.to(device).train()
    # wandb.watch(model, log="all", log_freq=15)
    
    optimizer = t.optim.Adam(model.parameters())

    schedule = NoiseSchedule(max_steps=200, device=device)
    model.noise_schedule = schedule

    trainloader = DataLoader(trainset, batch_size=config_dict["batch_size"], shuffle=True)
    testloader = DataLoader(testset, batch_size=config_dict["batch_size"], shuffle=True)

    n_examples_seen = 0
    n_steps = 0

    for epoch in range(config_dict["epochs"]):

        progress_bar = tqdm(trainloader)

        for (img,) in progress_bar:

            img = img.to(device)
            num_steps, noise, noised = noise_img(img, schedule, config_dict["max_steps"])

            noise_pred = model(noised, num_steps)

            loss = F.mse_loss(noise, noise_pred)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch+1}, Loss = {loss.item():>10.3f}")

            n_examples_seen += img.shape[0]
            n_steps += 1
            wandb.log({"loss": loss.item()}, step=n_examples_seen)

            if (n_steps + 1) % config_dict["img_log_interval"] == 0:
                with t.inference_mode():
                    reconstructed = reconstruct(noised, noise, num_steps, schedule)
                    images = log_images(img, noised, noise, noise_pred, reconstructed, num_images=config_dict["n_images_to_log"])
                    wandb.log({"images": images}, step=n_examples_seen)

        if testset is not None:
            total_loss = 0
            for (img,) in tqdm(testloader, desc=f"Epoch {epoch+1} eval"):
                img = img.to(device)
                num_steps, noise, noised = noise_img(img, schedule)
                with t.inference_mode():
                    noise_pred = model(noised, num_steps)
                    loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()
            wandb.log({"test_loss": total_loss/len(testloader)}, step=n_examples_seen)
    
    wandb.save("./wandb/gradients.h5")
    wandb.finish()

    return model    

if MAIN:
    config: Dict[str, Any] = dict(
        lr=0.001,
        image_shape=(3, 4, 5),
        hidden_size=128,
        epochs=20,
        max_steps=100,
        batch_size=128,
        img_log_interval_seconds=10,
        n_images_to_log=3,
        n_images=50000,
        n_eval_images=1000,
        device=device,
    )
    images = normalize_img(gradient_images(config["n_images"], config["image_shape"]))
    trainset = TensorDataset(images)
    images = normalize_img(gradient_images(config["n_eval_images"], config["image_shape"]))
    testset = TensorDataset(images)
    model_config = TinyDiffuserConfig(config["image_shape"], config["hidden_size"], config["max_steps"])
    model = TinyDiffuser(model_config).to(config["device"])
    model = train(model, config, trainset, testset)

# %%

def sample(model, n_samples: int, return_all_steps: bool = False) -> t.Tensor:
    """
    Sample, following Algorithm 2 in the DDPM paper

    model: The trained noise-predictor
    n_samples: The number of samples to generate
    return_all_steps: if true, return a list of the reconstructed tensors generated at each step, rather than just the final reconstructed image tensor.

    out: shape (B, C, H, W), the denoised images
            or (T, B, C, H, W), if return_all_steps=True (where [i,:,:,:,:]th element is result of (i+1) steps of sampling)
    """
    schedule = model.noise_schedule
    assert schedule is not None
    
    # Creating list of arrays of shape (max_steps, B, C, H, W), to store all the results
    T = len(schedule)
    out = t.zeros(T, n_samples, *model.img_shape)
    model.eval()

    # Algorithm:
    # STEP (1)
    x = t.randn(size=(n_samples, *model.img_shape)).to(device)
    # STEP (2)
    for t_ in tqdm(range(T, 0, -1)):
        # STEP (3)
        z = t.randn_like(x) if t_ > 1 else 0
        # STEP (4)
        alpha = schedule.alpha(t_-1)
        alpha_bar = schedule.alpha_bar(t_-1)
        beta = schedule.beta(t_-1)
        sigma = beta ** 0.5
        t_full = t.full((n_samples,), fill_value=t_, device=schedule.device)
        eps = model(x, t_full)
        sf_1 = 1 / alpha.sqrt()
        sf_2 = (1 - alpha) / ((1 - alpha_bar).sqrt())
        x = sf_1 * (x - sf_2 * eps) + sigma * z
        out[-t_] = x
        # STEP (5)

    # STEP (6)
    if return_all_steps:
        return out
    else:
        return out[-1]


if MAIN:
    print("Generating multiple images")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 6)
        samples_denormalized = denormalize_img(samples).cpu()
    plot_img_grid(samples_denormalized, title="Sample denoised images", cols=3)
    # for s in samples:
    #     plot_img(denormalize_img(s).cpu())
if MAIN:
    print("Printing sequential denoising")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 1, return_all_steps=True)[::5, 0, :]
        samples_denormalized = denormalize_img(samples).cpu()
    plot_img_slideshow(samples_denormalized, title="Sample denoised image slideshow")


# %%
# ===============================================================
# =========================== PART 3 ============================
# ===============================================================

# %%

def get_fashion_mnist(train_transform, test_transform) -> Tuple[TensorDataset, TensorDataset]:
    """Return MNIST data using the provided Tensor class."""
    mnist_train = datasets.FashionMNIST("../data", train=True, download=True)
    mnist_test = datasets.FashionMNIST("../data", train=False)
    print("Preprocessing data...")
    train_tensors = TensorDataset(
        t.stack([train_transform(img) for (img, label) in tqdm(mnist_train, desc="Training data")])
    )
    test_tensors = TensorDataset(t.stack([test_transform(img) for (img, label) in tqdm(mnist_test, desc="Test data")]))
    return (train_tensors, test_tensors)


if MAIN:
    train_transform = Compose([ToTensor(), RandomHorizontalFlip(), Lambda(lambda t: t * 2 - 1)])
    data_folder = Path("data/w3d4")
    data_folder.mkdir(exist_ok=True, parents=True)
    DATASET_FILENAME = data_folder / "generative_models_dataset_fashion.pt"
    if DATASET_FILENAME.exists():
        (train_dataset, test_dataset) = t.load(str(DATASET_FILENAME))
    else:
        (train_dataset, test_dataset) = get_fashion_mnist(train_transform, train_transform)
        t.save((train_dataset, test_dataset), str(DATASET_FILENAME))

# %%

class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0, bias: bool = True
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        kernel_height, kernel_width = force_pair(kernel_size)
        sf = 1 / (in_channels * kernel_width * kernel_height) ** 0.5
        weight = sf * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1)
        self.weight = nn.Parameter(weight)
        if bias:
            bias = sf * (2 * t.rand(out_channels,) - 1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d you wrote earlier."""
        x = conv2d(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            x += rearrange(self.bias, "o_c -> o_c 1 1")
        return x

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])

# %%

class GroupNorm(nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        device: Optional[Union[t.device, str]] = None,
        dtype: Optional[t.dtype] = None,
    ) -> None:
        super().__init__()
        assert num_channels % num_groups == 0

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.device = device
        self.dtype = dtype

        if affine:
            self.weight = nn.Parameter(t.empty(num_channels, device=device, dtype=dtype))
            self.bias = nn.Parameter(t.empty(num_channels, device=device, dtype=dtype))
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weight and bias, if applicable."""
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply normalization to each group of channels.

        x: shape (batch, channels, height, width)
        out: shape (batch, channels, height, width)
        """
        assert x.shape[1] == self.num_channels

        x = rearrange(x, "b (c1 c2) h w -> b c1 c2 h w", c1=self.num_groups)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x_shifted = (x - mean) / (var + self.eps).sqrt()
        x = rearrange(x_shifted, "b c1 c2 h w -> b (c1 c2) h w")
        if self.affine:
            weight = rearrange(self.weight, "c -> c 1 1")
            bias = rearrange(self.bias, "c -> c 1 1")
            x = x * weight + bias
        return x

if MAIN:
    utils.test_groupnorm(GroupNorm, affine=False)
    utils.test_groupnorm(GroupNorm, affine=True)

# %%

class PositionalEncoding(nn.Module):

    def __init__(self, max_steps: int, embedding_dim: int):
        super().__init__()
        angles = t.outer(t.arange(max_steps), 1 / 10000 ** (2 * t.arange(embedding_dim//2) / embedding_dim))
        pe = t.zeros((max_steps, embedding_dim))
        pe[:, ::2] = t.sin(angles)
        pe[:, 1::2] = t.cos(angles)
        self.register_buffer('pe', pe)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, ) - for each batch element, the number of noise steps
        Out: shape (batch, embedding_dim)
        """
        return self.pe[x]

# %%

def swish(x: t.Tensor) -> t.Tensor:
    return x * x.sigmoid()

class SiLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return swish(x)

if MAIN:
    x = t.linspace(-5, 5, 100)
    px.line(x=x, y=swish(x), title="Swish").show()

# %%

class SelfAttention(nn.Module):
    W_QKV: Linear
    W_O: Linear

    def __init__(self, channels: int, num_heads: int = 4):
        """
        Self-Attention with two spatial dimensions.

        channels: the number of channels. Should be divisible by the number of heads.
        """
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.headsize = channels // num_heads
        
        self.W_QKV = Linear(channels, 3*channels)
        self.W_O = Linear(channels, channels)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        out: shape (batch, channels, height, width)
        """
        b, c, h, w = x.shape
        assert c == self.channels

        x_flat = rearrange(x, "b c h w -> b (h w) c")
        QKV = self.W_QKV(x_flat)
        Q, K, V = t.split(QKV, self.channels, dim=-1)

        attention_values = self.multihead_attention_2d(Q, K, V)

        output = self.W_O(attention_values)

        output = rearrange(output, "b (h w) c -> b c h w", h=h)

        return output

    def multihead_attention_2d(self, Q, K, V):

        q = rearrange(Q, "b s (n h) -> b s n h", n=self.num_heads)
        k = rearrange(K, "b s (n h) -> b s n h", n=self.num_heads)
        v = rearrange(V, "b s (n h) -> b s n h", n=self.num_heads)

        attention_scores = einsum("b sQ n h, b sK n h -> b n sQ sK", q, k) / (self.headsize ** 0.5)
        attention_probabilities = attention_scores.softmax(dim=-1)
        attention_values = einsum("b n sQ sK, b sK n h -> b sQ n h", attention_probabilities, v)

        return rearrange(attention_values, "b sQ n h -> b sQ (n h)")


if MAIN:
    utils.test_self_attention(SelfAttention)

# %%

class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = GroupNorm(1, channels)
        self.attn = SelfAttention(channels)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x + self.attn(self.groupnorm(x))

if MAIN:
    utils.test_attention_block(SelfAttention)

# %%

class Identity(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, step_dim: int, groups: int):
        """
        input_channels: number of channels in the input to foward
        output_channels: number of channels in the returned output
        step_dim: embedding dimension size for the number of steps
        groups: number of groups in the GroupNorms

        Note that the conv in the left branch is needed if c_in != c_out.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.step_dim = step_dim
        self.groups = groups

        if input_channels != output_channels:
            self.res_conv = Conv2d(input_channels, output_channels, 1)
        else:
            self.res_conv = Identity()

        self.image_block = Sequential(
            Conv2d(input_channels, output_channels, 3, 1, 1),
            GroupNorm(groups, output_channels),
            SiLU(),
        )
        self.num_steps_block = Sequential(
            SiLU(),
            Linear(step_dim, output_channels),
            # Next line is to add width and height dimensions
            Rearrange("batch c_out -> batch c_out 1 1")
        )
        self.out_block = Sequential(*[
            Conv2d(output_channels, output_channels, 3, 1, 1),
            GroupNorm(groups, output_channels),
            SiLU()
        ])

    def forward(self, x: t.Tensor, time_emb: t.Tensor) -> t.Tensor:
        """
        Note that the output of the (silu, linear) block should be of shape (batch, c_out). Since we would like to add this to the output of the first (conv, norm, silu) block, which will have a different shape, we need to first add extra dimensions to the output of the (silu, linear) block.
        """
        x_skip = self.res_conv(x)

        x = self.image_block(x)
        time_emb = self.num_steps_block(time_emb)
        x = self.out_block(x + time_emb)

        return x_skip + x

if MAIN:
    utils.test_residual_block(ResidualBlock)

# %%

class DownBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, time_emb_dim: int, groups: int, downsample: bool):
        super().__init__()
        self.downsample = downsample

        self.resblock1 = ResidualBlock(channels_in, channels_out, time_emb_dim, groups)
        self.resblock2 = ResidualBlock(channels_out, channels_out, time_emb_dim, groups)
        self.attn = AttentionBlock(channels_out)
        if downsample:
            self.downsample = Conv2d(channels_out, channels_out, 4, 2, 1)
        else:
            self.downsample = Identity()

    def forward(self, x: t.Tensor, step_emb: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """
        x: shape (batch, channels, height, width)
        step_emb: shape (batch, emb)
        Return: (downsampled output, full size output to skip to matching UpBlock)
        """
        b, c, h, w = x.shape
        if isinstance(self.downsample, Conv2d):
            assert h % 2 == 0

        x = self.resblock1(x, step_emb)
        x = self.resblock2(x, step_emb)
        x = self.attn(x)
        
        return self.downsample(x), x

if MAIN:
    utils.test_downblock(DownBlock, downsample=True)
    utils.test_downblock(DownBlock, downsample=False)

# %%

class UpBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, time_emb_dim: int, groups: int, upsample: bool):
        super().__init__()

        self.resblock1 = ResidualBlock(2*dim_in, dim_out, time_emb_dim, groups)
        self.resblock2 = ResidualBlock(dim_out, dim_out, time_emb_dim, groups)
        self.attn = AttentionBlock(dim_out)
        if upsample:
            self.upsample = ConvTranspose2d(dim_out, dim_out, 4, 2, 1)
        else:
            self.upsample = Identity()

    def forward(self, x: t.Tensor, step_emb: t.Tensor, skip: t.Tensor) -> t.Tensor:
        x = t.concat([skip, x], dim=1)
        x = self.resblock1(x, step_emb)
        x = self.resblock2(x, step_emb)
        x = self.attn(x)
        if isinstance(self.upsample, ConvTranspose2d):
            x = self.upsample(x)
        return x

# %%

class MidBlock(nn.Module):
    def __init__(self, mid_dim: int, time_emb_dim: int, groups: int):
        super().__init__()
        self.resblock1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim, groups)
        self.attn = AttentionBlock(mid_dim)
        self.resblock2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim, groups)

    def forward(self, x: t.Tensor, step_emb: t.Tensor):
        x = self.resblock1(x, step_emb)
        x = self.attn(x)
        x = self.resblock2(x, step_emb)
        return x

if MAIN:
    utils.test_midblock(MidBlock)

# %%

class Unet(DiffusionModel):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        channels: int = 128,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        groups: int = 4,
        max_steps: int = 1000,
    ):
        """
        image_shape: the input and output image shape, a tuple of (C, H, W)
        channels: the number of channels after the first convolution.
        dim_mults: the number of output channels for downblock i is dim_mults[i] * channels. Note that the default arg of (1, 2, 4, 8) will contain one more DownBlock and UpBlock than the DDPM image above.
        groups: number of groups in the group normalization of each ResnetBlock (doesn't apply to attention block)
        max_steps: the max number of (de)noising steps. We also use this value as the sinusoidal positional embedding dimension (although in general these do not need to be related).
        """
        self.noise_schedule = None
        self.img_shape = image_shape
        self.channels = channels
        self.dim_mults = dim_mults
        self.groups = groups
        self.max_steps = max_steps

        super().__init__()

        self.n_downblocks = len(dim_mults)
        self.n_upblocks = len(dim_mults) - 1

        C, H, W = image_shape
        time_emb_dim = 4 * channels
        channels_list = tuple([channels * d for d in dim_mults])

        self.first_conv = Conv2d(C, channels, 7, 1, 3)
        
        downblock_in_channels = (channels,) + channels_list[:-1]
        downblock_out_channels = channels_list
        is_downsample = [True for _ in range(self.n_downblocks - 1)] + [False]
        for i, (channels_in, channels_out, downsample) in enumerate(zip(downblock_in_channels, downblock_out_channels, is_downsample)):
            downblock = DownBlock(channels_in, channels_out, time_emb_dim, groups, downsample)
            self.add_module(f"DownBlock{i}", downblock)
        
        self.mid = MidBlock(channels_out, time_emb_dim, groups)

        upblock_in_channels = channels_list[-1:0:-1]
        upblock_out_channels = channels_list[-2::-1]
        is_upsample = [True for _ in range(self.n_upblocks)]
        for i, (channels_in, channels_out, upsample) in enumerate(zip(upblock_in_channels, upblock_out_channels, is_upsample)):
            self.add_module(f"UpBlock{i}", UpBlock(channels_in, channels_out, time_emb_dim, groups, upsample))
        
        self.time_emb_block = Sequential(*[
            PositionalEncoding(1000, max_steps),
            Linear(max_steps, time_emb_dim),
            GELU(),
            Linear(time_emb_dim, time_emb_dim)
        ])

        self.resblock = ResidualBlock(channels, channels, time_emb_dim, groups)
        self.last_conv = Conv2d(channels, C, 1)

    def forward(self, x: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        num_steps: shape (batch, )

        out: shape (batch, channels, height, width)
        """

        x = self.first_conv(x)
        step_emb = self.time_emb_block(num_steps)

        skip_list = []
        for n in range(self.n_downblocks):
            x, skip = self._modules[f"DownBlock{n}"](x, step_emb)
            if n > 0:
                skip_list.append(skip)

        x = self.mid(x, step_emb)
        
        for n in range(self.n_upblocks):
            skip = skip_list.pop()
            x = self._modules[f"UpBlock{n}"](x, step_emb, skip)

        x = self.resblock(x, step_emb)
        x = self.last_conv(x)
        return x

if MAIN:
    utils.test_unet(Unet)


# %%
# ===============================================================
# =========================== PART 4 ============================
# ===============================================================


# %%

def get_fashion_mnist(train_transform, test_transform) -> Tuple[TensorDataset, TensorDataset]:
    """Return MNIST data using the provided Tensor class."""
    mnist_train = datasets.FashionMNIST("../data", train=True, download=True)
    mnist_test = datasets.FashionMNIST("../data", train=False)
    print("Preprocessing data...")
    train_tensors = TensorDataset(
        t.stack([train_transform(img) for (img, label) in tqdm(mnist_train, desc="Training data")])
    )
    test_tensors = TensorDataset(t.stack([test_transform(img) for (img, label) in tqdm(mnist_test, desc="Test data")]))
    return (train_tensors, test_tensors)


if MAIN:
    train_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.RandomHorizontalFlip(), 
        transforms.Lambda(lambda t: t * 2 - 1)
    ])
    data_folder = Path("./data/fashion_mnist")
    data_folder.mkdir(exist_ok=True, parents=True)
    DATASET_FILENAME = data_folder / "generative_models_dataset_fashion.pt"
    if DATASET_FILENAME.exists():
        (trainset, testset) = t.load(str(DATASET_FILENAME))
    else:
        (trainset, testset) = get_fashion_mnist(train_transform, train_transform)
        t.save((trainset, testset), str(DATASET_FILENAME))

# %%

if MAIN:
    config_dict: Dict[str, Any] = dict(
        model_channels=28,
        model_dim_mults=(1, 2, 4),
        image_shape=(1, 28, 28),
        max_steps=200,
        epochs=10,
        lr=0.001,
        batch_size=256,
        img_log_interval=200,
        n_images_to_log=3,
        device=device,
    )
    model = Unet(
        image_shape=config_dict["image_shape"], 
        channels=config_dict["model_channels"],
        dim_mults=config_dict["model_dim_mults"],
        max_steps=config_dict["max_steps"]
    )
    model.noise_schedule = NoiseSchedule(
        config_dict["max_steps"], config_dict["device"]
    )
    batch_size_mini = 8
    x = trainset.tensors[0][:batch_size_mini]
    num_steps = t.randint(low=0, high=config_dict["max_steps"], size=(batch_size_mini,))
    summary = torchinfo.summary(model, input_data=(x, num_steps))
    print(summary)

# %%

if MAIN:
    model = train(model, config_dict, trainset, testset)

# %%

def plot_img_grid_fashionMNIST(imgs: t.Tensor, cols: int, title: Optional[str] = None) -> None:
    imgs = (255 * imgs).to(t.uint8)
    fig = px.imshow(imgs, facet_col=0, facet_col_wrap=cols, title=title, color_continuous_scale="gray")
    for annotation in fig.layout.annotations: annotation["text"] = ""
    fig.show()

def plot_img_slideshow_fashionMNIST(imgs: t.Tensor, title: Optional[str] = None) -> None:
    imgs = (255 * imgs).to(t.uint8)
    fig = px.imshow(imgs, animation_frame=0, title=title, color_continuous_scale="gray")
    fig.show()

if MAIN:
    print("Generating multiple images")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 6)
        samples = denormalize_img(samples).cpu().squeeze()
    plot_img_grid_fashionMNIST(samples, cols=3)

if MAIN:
    print("Printing sequential denoising")
    with t.inference_mode():
        samples = sample(model, 1, return_all_steps=True).squeeze()
        samples = denormalize_img(samples).cpu()[::5]
    plot_img_slideshow_fashionMNIST(samples)
