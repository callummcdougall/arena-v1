import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display, HTML
from typing import Union, Optional, Callable
import torchvision

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fancy_einsum import einsum
from einops import reduce, rearrange, repeat
import itertools


def display_array_as_img(img_array):
    """
    Displays a numpy array as an image
    
    Two options:
        img_array.shape = (height, width) -> interpreted as monochrome
        img_array.shape = (3, height, width) -> interpreted as RGB
    """
    shape = img_array.shape
    assert len(shape) == 2 or (shape[0] == 3 and len(shape) == 3), "Incorrect format (see docstring)"
    
    if len(shape) == 3:
        img_array = rearrange(img_array, "c h w -> h w c")
    height, width = img_array.shape[:2]
    
    fig = px.imshow(img_array, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(coloraxis_showscale=False, margin=dict.fromkeys("tblr", 0), height=height, width=width)
    fig.show(config=dict(displayModeBar=False))

def test_einsum_trace(einsum_trace):
    mat = np.random.randn(3, 3)
    np.testing.assert_almost_equal(einsum_trace(mat), np.trace(mat))

def test_einsum_mv(einsum_mv):
    mat = np.random.randn(2, 3)
    vec = np.random.randn(3)
    np.testing.assert_almost_equal(einsum_mv(mat, vec), mat @ vec)

def test_einsum_mm(einsum_mm):
    mat1 = np.random.randn(2, 3)
    mat2 = np.random.randn(3, 4)
    np.testing.assert_almost_equal(einsum_mm(mat1, mat2), mat1 @ mat2)

def test_einsum_inner(einsum_inner):
    vec1 = np.random.randn(3)
    vec2 = np.random.randn(3)
    np.testing.assert_almost_equal(einsum_inner(vec1, vec2), np.dot(vec1, vec2))

def test_einsum_outer(einsum_outer):
    vec1 = np.random.randn(3)
    vec2 = np.random.randn(4)
    np.testing.assert_almost_equal(einsum_outer(vec1, vec2), np.outer(vec1, vec2))



def test_trace(trace_fn):
    for n in range(10):
        assert trace_fn(t.zeros((n, n), dtype=t.long)) == 0
        assert trace_fn(t.eye(n, dtype=t.long)) == n
        x = t.randint(0, 10, (n, n))
        expected = t.trace(x)
        actual = trace_fn(x)
        assert actual == expected

def test_mv(mv_fn):
    mat = t.randn(3, 4)
    vec = t.randn(4)
    mv_actual = mv_fn(mat, vec)
    mv_expected = mat @ vec
    t.testing.assert_close(mv_actual, mv_expected)
    
def test_mv2(mv_fn):
    big = t.randn(30)
    mat = big.as_strided(size=(3, 4), stride=(2, 4), storage_offset=8)
    vec = big.as_strided(size=(4,), stride=(3,), storage_offset=8)
    mv_actual = mv_fn(mat, vec)
    mv_expected = mat @ vec
    t.testing.assert_close(mv_actual, mv_expected)
        
def test_mm(mm_fn):
    matA = t.randn(3, 4)
    matB = t.randn(4, 5)
    mm_actual = mm_fn(matA, matB)
    mm_expected = matA @ matB
    t.testing.assert_close(mm_actual, mm_expected)

def test_mm2(mm_fn):
    big = t.randn(30)
    matA = big.as_strided(size=(3, 4), stride=(2, 4), storage_offset=8)
    matB = big.as_strided(size=(4, 5), stride=(3, 2), storage_offset=8)
    mm_actual = mm_fn(matA, matB)
    mm_expected = matA @ matB
    t.testing.assert_close(mm_actual, mm_expected)
    
def test_conv1d_minimal(conv1d_minimal, n_tests=20):
    import numpy as np
    for _ in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 30)
        ci = np.random.randint(1, 5)
        co = np.random.randint(1, 5)
        kernel_size = np.random.randint(1, 10)
        x = t.randn((b, ci, h))
        weights = t.randn((co, ci, kernel_size))
        my_output = conv1d_minimal(x, weights)
        torch_output = t.conv1d(x, weights, stride=1, padding=0)
        t.testing.assert_close(my_output, torch_output)

def test_conv2d_minimal(conv2d_minimal, n_tests=4):
    """
    Compare against torch.conv2d.
    Due to floating point rounding, they can be quite different in float32 but should be nearly identical in float64.
    """
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))

        x = t.randn((b, ci, h, w), dtype=t.float64)
        weights = t.randn((co, ci, *kernel_size), dtype=t.float64)
        my_output = conv2d_minimal(x, weights)
        torch_output = t.conv2d(x, weights)
        t.testing.assert_close(my_output, torch_output)

def test_conv1d(my_conv, n_tests=10):
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = np.random.randint(1, 5)
        padding = np.random.randint(0, 5)
        kernel_size = np.random.randint(1, 10)

        x = t.randn((b, ci, h))
        weights = t.randn((co, ci, kernel_size))

        my_output = my_conv(x, weights, stride=stride, padding=padding)

        torch_output = t.conv1d(x, weights, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)

def test_pad1d(pad1d):
    """Should work with one channel of width 4."""
    x = t.arange(4).float().view((1, 1, 4))
    actual = pad1d(x, 1, 3, -2.0)
    expected = t.tensor([[[-2.0, 0.0, 1.0, 2.0, 3.0, -2.0, -2.0, -2.0]]])
    t.testing.assert_close(actual, expected)


def test_pad1d_multi_channel(pad1d):
    """Should work with two channels of width 2."""
    x = t.arange(4).float().view((1, 2, 2))
    actual = pad1d(x, 0, 2, -3.0)
    expected = t.tensor([[[0.0, 1.0, -3.0, -3.0], [2.0, 3.0, -3.0, -3.0]]])
    t.testing.assert_close(actual, expected)

def test_pad2d(pad):
    """Should work with one channel of 2x2."""
    x = t.arange(4).float().view((1, 1, 2, 2))
    expected = t.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ]
    )
    actual = pad(x, 0, 1, 2, 3, 0.0)
    t.testing.assert_close(actual, expected)

def test_pad2d_multi_channel(pad):
    """Should work with two channels of 2x1."""
    x = t.arange(4).float().view((1, 2, 2, 1))
    expected = t.tensor([[[[-1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]], [[-1.0, 2.0], [-1.0, 3.0], [-1.0, -1.0]]]])
    actual = pad(x, 1, 0, 0, 1, -1.0)
    t.testing.assert_close(actual, expected)

def test_conv2d(my_conv, n_tests=5):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = tuple(np.random.randint(1, 5, size=(2,)))
        padding = tuple(np.random.randint(0, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w), dtype=t.float64)
        weights = t.randn((co, ci, *kernel_size), dtype=t.float64)
        my_output = my_conv(x, weights, stride=stride, padding=padding)
        torch_output = t.conv2d(x, weights, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)

def test_maxpool2d(my_maxpool2d, n_tests=20):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)
        stride = None if np.random.random() < 0.5 else tuple(np.random.randint(1, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        kH, kW = kernel_size
        padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)
        x = t.randn((b, ci, h, w))
        
        my_output = my_maxpool2d(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        torch_output = t.max_pool2d(
            x,
            kernel_size,
            stride=stride,  # type: ignore (None actually is allowed)
            padding=padding,
        )
        t.testing.assert_close(my_output, torch_output)

def test_maxpool2d_module(MaxPool2d, n_tests=20):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)
        stride = None if np.random.random() < 0.5 else tuple(np.random.randint(1, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        kH, kW = kernel_size
        padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)
        x = t.randn((b, ci, h, w))

        my_output = MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
        )(x)

        torch_output = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
        )(x)
        t.testing.assert_close(my_output, torch_output)

def test_conv2d_module(Conv2d, n_tests=5):
    """
    Your weight should be called 'weight' and have an appropriate number of elements.
    """
    m = Conv2d(4, 5, (3, 3))
    assert isinstance(m.weight, t.nn.parameter.Parameter), "Weight should be registered a parameter!"
    assert m.weight.nelement() == 4 * 5 * 3 * 3
    
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = tuple(np.random.randint(1, 5, size=(2,)))
        padding = tuple(np.random.randint(0, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w))
        my_conv = Conv2d(in_channels=ci, out_channels=co, kernel_size=kernel_size, stride=stride, padding=padding)
        my_output = my_conv(x)
        torch_output = t.conv2d(x, my_conv.weight, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)

def test_batchnorm2d_module(BatchNorm2d):
    """The public API of the module should be the same as the real PyTorch version."""
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.num_features == num_features
    assert isinstance(bn.weight, t.nn.parameter.Parameter), f"weight has wrong type: {type(bn.weight)}"
    assert isinstance(bn.bias, t.nn.parameter.Parameter), f"bias has wrong type: {type(bn.bias)}"
    assert isinstance(bn.running_mean, t.Tensor), f"running_mean has wrong type: {type(bn.running_mean)}"
    assert isinstance(bn.running_var, t.Tensor), f"running_var has wrong type: {type(bn.running_var)}"
    assert isinstance(bn.num_batches_tracked, t.Tensor), f"num_batches_tracked has wrong type: {type(bn.num_batches_tracked)}"

def test_batchnorm2d_forward(BatchNorm2d):
    """For each channel, mean should be very close to 0 and std kinda close to 1 (because of eps)."""
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.training
    x = t.randn((100, num_features, 3, 4))
    out = bn(x)
    assert x.shape == out.shape
    t.testing.assert_close(out.mean(dim=(0, 2, 3)), t.zeros(num_features))
    t.testing.assert_close(out.std(dim=(0, 2, 3)), t.ones(num_features), atol=1e-3, rtol=1e-3)

def test_batchnorm2d_running_mean(BatchNorm2d):
    """Over repeated forward calls with the same data in train mode, the running mean should converge to the actual mean."""
    bn = BatchNorm2d(3, momentum=0.6)
    assert bn.training
    x = t.arange(12).float().view((2, 3, 2, 1))
    mean = t.tensor([3.5000, 5.5000, 7.5000])
    num_batches = 30
    for i in range(num_batches):
        bn(x)
        expected_mean = (1 - ((bn.momentum ** (i + 1)))) * mean
        t.testing.assert_close(bn.running_mean, expected_mean)
    assert bn.num_batches_tracked.item() == num_batches

    # Large enough momentum and num_batches -> running_mean should be very close to actual mean
    bn.eval()
    actual_eval_mean = bn(x).mean((0, 2, 3))
    t.testing.assert_close(actual_eval_mean, t.zeros(3))

def test_relu(ReLU):
    x = t.randn(10) - 0.5
    actual = ReLU()(x)
    expected = F.relu(x)
    t.testing.assert_close(actual, expected)

def test_flatten(Flatten):
    x = t.arange(24).reshape((2, 3, 4))
    assert Flatten(start_dim=0)(x).shape == (24,)
    assert Flatten(start_dim=1)(x).shape == (2, 12)
    assert Flatten(start_dim=0, end_dim=1)(x).shape == (6, 4)
    assert Flatten(start_dim=0, end_dim=-2)(x).shape == (6, 4)

def test_linear_forward(Linear):
    """Your Linear should produce identical results to torch.nn given identical parameters."""
    x = t.rand((10, 512))
    yours = Linear(512, 64)

    assert yours.weight.shape == (64, 512)
    assert yours.bias.shape == (64,)

    official = t.nn.Linear(512, 64)
    yours.weight = official.weight
    yours.bias = official.bias
    actual = yours(x)
    expected = official(x)
    t.testing.assert_close(actual, expected)

def test_linear_parameters(Linear):
    m = Linear(2, 3)
    params = dict(m.named_parameters())
    assert len(params) == 2, f"Your model has {len(params)} recognized Parameters"
    assert list(params.keys()) == [
        "weight",
        "bias",
    ], "For compatibility with PyTorch, your fields should be named weight and bias."

def test_linear_no_bias(Linear):
    
    x = t.rand((10, 512))
    yours = Linear(512, 64, bias=False)

    assert yours.bias is None, "Bias should be None when not enabled."
    assert len(list(yours.parameters())) == 1

    official = nn.Linear(512, 64, bias=False)
    yours.weight = official.weight
    actual = yours(x)
    expected = official(x)
    t.testing.assert_close(actual, expected)

def test_sequential(Sequential):
    from torch.nn import Linear, ReLU

    modules = [Linear(1, 2), ReLU(), Linear(2, 1)]
    s = Sequential(*modules)

    assert list(s.modules()) == [s, *modules], "The sequential and its submodules should be registered Modules."
    assert len(list(s.parameters())) == 4, "Submodules's parameters should be registered."

def test_sequential_forward(Sequential):
    from torch.nn import Linear, ReLU

    modules = [Linear(1, 2), ReLU(), Linear(2, 1)]
    x = t.tensor([5.0])
    s = Sequential(*modules)
    actual_out = s(x)
    expected_out = modules[-1](modules[-2](modules[-3](x)))
    t.testing.assert_close(actual_out, expected_out)

def test_same_predictions(your_model_predictions: list[int]):
    assert your_model_predictions == [367, 207, 103, 604, 865, 562, 628, 39, 980, 447]