# %%
import torch as t
from typing import Union, Optional
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import os
import torchinfo
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
import time

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
assert str(device) == "cuda:0"

import utils
import w0d2_solutions
import w0d3_solutions

MAIN = (__name__ == "__main__")

# %%

def conv_transpose1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """

    batch, in_channels, width = x.shape
    in_channels_2, out_channels, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"

    x_mod = w0d2_solutions.pad1d(x, left=kernel_width-1, right=kernel_width-1, pad_value=0)
    weights_mod = rearrange(weights.flip(-1), "i o w -> o i w")

    return w0d2_solutions.conv1d_minimal(x_mod, weights_mod)


def fractional_stride_1d(x, stride: int = 1):
    '''Returns a version of x suitable for transposed convolutions, i.e. "spaced out" with zeros between its values.
    This spacing only happens along the last dimension.

    x: shape (batch, in_channels, width)

    Example: 
        x = [[[1, 2, 3], [4, 5, 6]]]
        stride = 2
        output = [[[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]]
    '''
    batch, in_channels, width = x.shape
    width_new = width + (stride - 1) * (width - 1) # the RHS of this sum is the number of zeros we need to add between elements
    x_new_shape = (batch, in_channels, width_new)

    # Create an empty array to store the spaced version of x in.
    x_new = t.zeros(size=x_new_shape, dtype=x.dtype, device=x.device)

    x_new[..., ::stride] = x
    
    return x_new


def conv_transpose1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """

    batch, ic, width = x.shape
    ic_2, oc, kernel_width = weights.shape
    assert ic == ic_2, f"in_channels for x and weights don't match up. Shapes are {x.shape}, {weights.shape}."

    # Apply spacing
    x_spaced_out = fractional_stride_1d(x, stride)

    # Apply modification (which is controlled by the padding parameter)
    padding_amount = kernel_width - 1 - padding
    assert padding_amount >= 0, "total amount padded should be positive"
    x_mod = w0d2_solutions.pad1d(x_spaced_out, left=padding_amount, right=padding_amount, pad_value=0)

    # Modify weights, then return the convolution
    weights_mod = rearrange(weights.flip(-1), "i o w -> o i w")

    return w0d2_solutions.conv1d_minimal(x_mod, weights_mod)

# %%

IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

def fractional_stride_2d(x, stride_h: int, stride_w: int):
    """
    Same as fractional_stride_1d, except we apply it along the last 2 dims of x (width and height).
    """
    batch, in_channels, height, width = x.shape
    width_new = width + (stride_w - 1) * (width - 1)
    height_new = height + (stride_h - 1) * (height - 1)
    x_new_shape = (batch, in_channels, height_new, width_new)

    # Create an empty array to store the spaced version of x in.
    x_new = t.zeros(size=x_new_shape, dtype=x.dtype, device=x.device)

    x_new[..., ::stride_h, ::stride_w] = x
    
    return x_new

def conv_transpose2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    """Like torch's conv_transpose2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    """

    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)

    batch, ic, height, width = x.shape
    ic_2, oc, kernel_height, kernel_width = weights.shape
    assert ic == ic_2, f"in_channels for x and weights don't match up. Shapes are {x.shape}, {weights.shape}."

    # Apply spacing
    x_spaced_out = fractional_stride_2d(x, stride_h, stride_w)

    # Apply modification (which is controlled by the padding parameter)
    pad_h_actual = kernel_height - 1 - padding_h
    pad_w_actual = kernel_width - 1 - padding_w
    assert min(pad_h_actual, pad_w_actual) >= 0, "total amount padded should be positive"
    x_mod = w0d2_solutions.pad2d(x_spaced_out, left=pad_w_actual, right=pad_w_actual, top=pad_h_actual, bottom=pad_h_actual, pad_value=0)

    # Modify weights
    weights_mod = rearrange(weights.flip(-1, -2), "i o h w -> o i h w")

    # Return the convolution
    return w0d2_solutions.conv2d_minimal(x_mod, weights_mod)

# %%

class ConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        """
        Same as torch.nn.ConvTranspose2d with bias=False.

        Name your weight field `self.weight` for compatibility with the tests.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_size = force_pair(kernel_size)
        sf = 1 / (self.out_channels * kernel_size[0] * kernel_size[1]) ** 0.5

        self.weight = nn.Parameter(sf * (2 * t.rand(in_channels, out_channels, *kernel_size) - 1))

    def forward(self, x: t.Tensor) -> t.Tensor:

        return conv_transpose2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join([
            f"{key}={getattr(self, key)}"
            for key in ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        ])


class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return (t.exp(x) - t.exp(-x)) / (t.exp(x) + t.exp(-x))

class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.where(x > 0, x, self.negative_slope * x)
    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"

class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return 1 / (1 + t.exp(-x))

# %%

# Create a `Sequential` which can accept an ordered dict, and which can be cycled through
class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        if isinstance(modules[0], OrderedDict):
            for name, mod in modules[0].items():
                self.add_module(name, mod)
        else:
            for i, mod in enumerate(modules):
                self.add_module(str(i), mod)

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            if mod is not None:
                x = mod(x)
        return x

class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_size = 100,
        img_size = 64,
        img_channels = 3,
        generator_num_features = 1024,
        n_layers = 4,
        **kwargs
    ):
        super().__init__()

        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.generator_num_features = generator_num_features
        self.n_layers = n_layers
        self.img_channels = img_channels

        # Define the first layer, i.e. latent dim -> (1024, 4, 4) and reshape
        assert img_size % (2 ** n_layers) == 0
        first_height = img_size // (2 ** n_layers)
        first_size = generator_num_features * first_height * first_height
        self.project_and_reshape = Sequential(OrderedDict([
            ("fc", w0d2_solutions.Linear(latent_dim_size, first_size, bias=False)),
            ("rearrange", Rearrange("b (ic h w) -> b ic h w", h=first_height, w=first_height)),
            ("bn", w0d3_solutions.BatchNorm2d(generator_num_features)),
            ("activation_fn", w0d2_solutions.ReLU())
        ]))

        # Get the list of parameters for feeding into the conv layers
        # note that the last out_channels is 3, for the colors of an RGB image
        in_channels_list = (generator_num_features / 2 ** t.arange(n_layers)).to(int).tolist()
        out_channels_list = in_channels_list[1:] + [self.img_channels,]

        # Define all the convolutional blocks (conv_transposed -> batchnorm -> activation)
        conv_layer_list = []
        for i, (ci, co) in enumerate(zip(in_channels_list, out_channels_list)):
            conv_layer = [
                ("convT", ConvTranspose2d(ci, co, 4, 2, 1)),
                ("activation_fn", w0d2_solutions.ReLU() if i < n_layers - 1 else Tanh())
            ]
            if i < n_layers - 1:
                conv_layer.insert(1, ("bn", w0d3_solutions.BatchNorm2d(co)))
            conv_layer_list.append(Sequential(OrderedDict(conv_layer)))
        
        self.layers = Sequential(*conv_layer_list)

    def forward(self, x: t.Tensor) -> t.Tensor:
        
        x = self.project_and_reshape(x)
        x = self.layers(x)

        return x


class Discriminator(nn.Module):

    def __init__(
        self,
        img_size = 64,
        img_channels = 3,
        generator_num_features = 1024,
        n_layers = 4,
        **kwargs
    ):
        super().__init__()

        self.img_size = img_size
        self.generator_num_features = generator_num_features
        self.n_layers = n_layers
        self.img_channels = img_channels

        # Get the list of parameters for feeding into the conv layers
        # note that the last out_channels is 3, for the colors of an RGB image
        out_channels_list = (generator_num_features / 2 ** t.arange(n_layers)).to(int).tolist()[::-1]
        in_channels_list = [self.img_channels,] + out_channels_list

        # Define all the convolutional blocks (conv_transposed -> batchnorm -> activation)
        conv_layer_list = []
        for i, (ci, co) in enumerate(zip(in_channels_list, out_channels_list)):
            conv_layer = [
                ("conv", w0d2_solutions.Conv2d(ci, co, 4, 2, 1)),
                ("activation_fn", LeakyReLU(negative_slope = 0.2))
            ]
            if i > 0:
                conv_layer.insert(1, ("bn", w0d3_solutions.BatchNorm2d(co)))
            conv_layer_list.append(Sequential(OrderedDict(conv_layer)))
        
        self.layers = Sequential(*conv_layer_list)

        # Define the last layer, i.e. reshape and (1024, 4, 4) -> real/fake classification
        assert img_size % (2 ** n_layers) == 0
        first_height = img_size // (2 ** n_layers)
        final_size = generator_num_features * first_height * first_height
        self.classifier = Sequential(OrderedDict([
            ("rearrange", Rearrange("b c h w -> b (c h w)")),
            ("fc", w0d2_solutions.Linear(final_size, 1, bias=False)),
            ("sigmoid", Sigmoid())
        ]))

    def forward(self, x: t.Tensor) -> t.Tensor:
        
        x = self.layers(x)
        x = self.classifier(x)

        return x


celeba_config = dict(
    latent_dim_size = 100,
    img_size = 64,
    img_channels = 3,
    generator_num_features = 1024,
    n_layers = 4,
)
celeba_mini_config = dict(
    latent_dim_size = 100,
    img_size = 64,
    img_channels = 3,
    generator_num_features = 512,
    n_layers = 4,
)

netG_celeb_mini = Generator(**celeba_mini_config).to(device).train()
netD_celeb_mini = Discriminator(**celeba_mini_config).to(device).train()

netG_celeb = Generator(**celeba_config).to(device).train()
netD_celeb = Discriminator(**celeba_config).to(device).train()

def initialize_weights(model) -> None:
    for name, param in model.named_parameters():
        if "weight" in name and "conv" in name:
            nn.init.normal_(param.data, 0.0, 0.02)
        elif "weight" in name and "batchnorm" in name:
            nn.init.normal_(param.data, 1.0, 0.02)
        elif "bias" in name and "batchnorm" in name:
            nn.init.constant_(param.data, 0.0)

# %%

# ======================== CELEB_A ========================

image_size = 64
batch_size = 8

from torchvision import transforms, datasets

transform = transforms.Compose([
    transforms.Resize((image_size)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = datasets.ImageFolder(
    root=r"C:\Users\calsm\Documents\AI Alignment\ARENA\in_progress\transposed_convolutions\data",
    # root=r"./data",
    transform=transform
)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True) # num_workers=2

utils.show_images(trainset, rows=3, cols=5)

# ======================== MNIST ========================

# batch_size = 64
# img_size = 24

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([
#     transforms.Resize(img_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# importlib.reload(utils)
# utils.show_images(trainset.data)


# %%
def train_generator_discriminator(
    netG: Generator, 
    netD: Discriminator, 
    optG,
    optD,
    trainloader,
    epochs: int,
    max_epoch_duration: Optional[Union[int, float]] = None,
    log_netG_output_interval: Optional[Union[int, float]] = None,
    use_wandb: bool = True
):

    # This code controls how we print output from our model at specified times, and controls early epoch termination.
    t0 = time.time()
    n_examples_seen = 0
    if max_epoch_duration is None: max_epoch_duration = t.inf
    if log_netG_output_interval is None: log_netG_output_interval = t.inf
    last_interval = 0 

    netG.train().to(device)
    netD.train().to(device)

    if use_wandb:
        wandb.init()

    for epoch in range(epochs):
        
        t0_epoch = time.time()
        n_examples_seen_this_epoch = 0
        progress_bar = tqdm(trainloader)

        for img_real, label in progress_bar:

            img_real = img_real.to(device)
            label = label.to(device)
            current_batch_size = img_real.size(0)
            noise = t.randn(current_batch_size, netG.latent_dim_size).to(device)

            # ====== DISCRIMINIATOR TRAINING LOOP: maximise log(D(x)) + log(1-D(G(z))) ======

            # Zero gradients
            optD.zero_grad()
            # Calculate the two different components of the objective function
            D_x = netD(img_real)
            img_fake = netG(noise)
            D_G_z = netD(img_fake.detach())
            # Add them to get the objective function
            lossD = - (t.log(D_x).mean() + t.log(1 - D_G_z).mean())
            # Gradient descent step
            lossD.backward()
            optD.step()

            # ====== GENERATOR TRAINING LOOP: maximise log(D(G(z))) ======
            
            # Zero gradients
            optG.zero_grad()
            # Calculate the objective function
            D_G_z = netD(img_fake)
            lossG = - (t.log(D_G_z).mean())
            # Gradient descent step
            lossG.backward()
            optG.step()

            # Update progress bar
            progress_bar.set_description(f"epoch={epoch}, steps={n_examples_seen_this_epoch}/{len(trainloader)}, lossD={lossD.item():.4f}, lossG={lossG.item():.4f}")
            n_examples_seen += current_batch_size
            n_examples_seen_this_epoch += current_batch_size
            if use_wandb:
                wandb.log(dict(lossD=lossD, lossG=lossG), step=n_examples_seen)

            # Log output, if required
            if time.time() - t0 > log_netG_output_interval * (last_interval + 1):
                last_interval += 1
                if use_wandb:
                    arrays = get_generator_output(netG) # shape (8, 64, 64, 3)
                    images = [wandb.Image(arr) for arr in arrays]
                    wandb.log({"images": images}, step=n_examples_seen)
            if time.time() - t0_epoch > max_epoch_duration:
                break

    for model in [netG, netD]:
        name = model.__class__.__name__
        dirname = str(wandb.run.dir) if use_wandb else "models"
        filename = f"{dirname}/{name}.pt"
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if use_wandb:
            wandb.save(filename)
        print(f"Saving {name!r} to: {filename!r}")
        t.save(model.state_dict(), filename)

    if use_wandb:
        wandb.finish()
                
    return netG, netD

@t.inference_mode()
def get_generator_output(netG, n_examples=8, rand_seed=0):
    netG.eval()
    device = next(netG.parameters()).device
    t.manual_seed(rand_seed)
    noise = t.randn(n_examples, netG.latent_dim_size).to(device)
    arrays = rearrange(netG(noise), "b c h w -> b h w c").detach().cpu().numpy()
    netG.train()
    return arrays


# %%

netG = Generator(**celeba_mini_config).to(device).train()
# print_param_count(netG)
x = t.randn(3, 100).to(device)
statsG = torchinfo.summary(netG, input_data=x)
print(statsG, "\n\n")

netD = Discriminator(**celeba_mini_config).to(device).train()
# print_param_count(netD)
statsD = torchinfo.summary(netD, input_data=netG(x))
print(statsD)

initialize_weights(netG)
initialize_weights(netD)

lr = 0.0002
betas = (0.5, 0.999)
optG = t.optim.Adam(netG.parameters(), lr=lr, betas=betas)
optD = t.optim.Adam(netD.parameters(), lr=lr, betas=betas)

epochs = 3
max_epoch_duration = 240
log_netG_output_interval = 10

# %%

netG, netD = train_generator_discriminator(
    netG, 
    netD, 
    optG, 
    optD, 
    trainloader,
    epochs, 
    max_epoch_duration, 
    log_netG_output_interval
)
# %%
