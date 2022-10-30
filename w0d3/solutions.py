from w0d2.solutions import *
from einops import rearrange
import torch as t
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm_notebook
import PIL

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.flatten = Flatten()
        self.fc1 = Linear(in_features=3136, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=10)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.fc2(self.fc1(self.flatten(x)))
        return x

# model = ConvNet()
# print(model)

class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(str(i), mod)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            if mod is not None:
                x = mod(x)
        return x

class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        
        # Calculating mean and var over all dims except for the channel dim
        if self.training:
            # Using keepdim=True so we don't have to worry about broadasting them with x at the end
            mean = t.mean(x, dim=(0, 2, 3), keepdim=True)
            var = t.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            # Updating running mean and variance, in line with PyTorch documentation
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            mean = rearrange(self.running_mean, "channels -> 1 channels 1 1")
            var = rearrange(self.running_var, "channels -> 1 channels 1 1")
        
        # Rearranging these so they can be broadcasted (although there are other ways you could do this)
        weight = rearrange(self.weight, "channels -> 1 channels 1 1")
        bias = rearrange(self.bias, "channels -> 1 channels 1 1")
        
        return ((x - mean) / t.sqrt(var + self.eps)) * weight + bias

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])


class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return t.mean(x, dim=(2, 3))


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        """
        super().__init__()
        
        self.left = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats)
        )
        
        if first_stride > 1:
            self.right = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
                BatchNorm2d(out_feats)
            )
        else:
            self.right = nn.Identity()
            
        self.relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        """
        x_left = self.left(x)
        x_right = self.right(x)
        return self.relu(x_left + x_right)


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride."""
        super().__init__()
        
        blocks = [ResidualBlock(in_feats, out_feats, first_stride)] + [
            ResidualBlock(out_feats, out_feats) for n in range(n_blocks - 1)
        ]
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.blocks(x)



class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        in_feats0 = 64

        self.in_layers = Sequential(
            Conv2d(3, in_feats0, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(in_feats0),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        all_in_feats = [in_feats0] + out_features_per_group[:-1]
        self.residual_layers = Sequential(
            *(
                BlockGroup(*args)
                for args in zip(
                    n_blocks_per_group,
                    all_in_feats,
                    out_features_per_group,
                    first_strides_per_group,
                )
            )
        )
        # Alternative that uses `add_module`, in a way which makes the layer names line up:
        # for idx, (n_blocks, in_feats, out_feats, first_stride) in enumerate(zip(
        #     n_blocks_per_group, all_in_feats, out_features_per_group, strides_per_group
        # )):
        #     self.add_module(f"layer{idx+1}", BlockGroup(n_blocks, in_feats, out_feats, first_stride))

        self.out_layers = Sequential(
            AveragePool(),
            Flatten(),
            Linear(out_features_per_group[-1], n_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        x = self.in_layers(x)
        x = self.residual_layers(x)
        x = self.out_layers(x)
        return x


# ImageNet transforms:
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def prepare_data(images: list[PIL.Image.Image]) -> t.Tensor:
    """
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    """
    x = t.stack([transform(img) for img in images], dim=0)  # type: ignore
    return x




# ================================= ConvNet training & testing =================================

epochs = 3
loss_fn = nn.CrossEntropyLoss()
batch_size = 128

MODEL_FILENAME = "./w0d3_convnet_mnist.pt"
device = "cuda" if t.cuda.is_available() else "cpu"

trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

def train_convnet(trainloader: DataLoader, testloader: DataLoader, epochs: int, loss_fn: Callable) -> list:
    """
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    
    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    """
    
    model = ConvNet().to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []
    accuracy_list = []
    
    for epoch in tqdm_notebook(range(epochs)):
        
        for (x, y) in tqdm_notebook(trainloader, leave=False):
            
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
        
        with t.inference_mode():
            
            accuracy = 0
            total = 0
            
            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            accuracy_list.append(accuracy/total)
            
        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")
    
    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)
    return loss_list, accuracy_list

loss_list, accuracy_list = train_convnet(trainloader, testloader, epochs, loss_fn)