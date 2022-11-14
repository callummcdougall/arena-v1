import numpy as np
import plotly.express as px
import torch as t
from einops import rearrange
import pandas as pd
from IPython.display import display

def pad_width_height(data, padding_amt=4):
    new_shape = list(data.shape)
    new_shape[-1] += padding_amt * 2
    new_shape[-2] += padding_amt * 2
    data_padded = t.ones(tuple(new_shape))
    data_padded[..., padding_amt:-padding_amt, padding_amt:-padding_amt] = data
    return data_padded

def show_images(data, rows=3, cols=5):

    if isinstance(data[0], t.Tensor):
        img = t.stack([data[i] for i in range(rows*cols)])
    else:
        img = t.stack([data[i][0] for i in range(rows*cols)])
    img_min = img.min(-1, True).values.min(-2, True).values
    img_max = img.max(-1, True).values.max(-2, True).values
    img = (img - img_min) / (img_max - img_min)
    img = pad_width_height(img)
    if len(img.shape) == 4:
        img = rearrange(img, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=rows)
    else:
        img = rearrange(img, "(b1 b2) h w -> (b1 h) (b2 w)", b1=rows)
    (px.imshow(img, color_continuous_scale="greys_r")
     .update_layout(margin=dict.fromkeys("tblr", 20), coloraxis_showscale=False)
     .update_xaxes(showticklabels=False)
     .update_yaxes(showticklabels=False)
    ).show()
    # img = rearrange(img, "b c h w -> b h w c")
    # px.imshow(img, facet_col=0, facet_col_wrap=cols).show()

def display_generator_output(netG, latent_dim_size, rows=2, cols=5):

    with t.inference_mode():
        netG.eval()
        device = next(netG.parameters()).device
        t.manual_seed(0)
        with t.inference_mode():
            noise = t.randn(rows*cols, latent_dim_size).to(device)
            img = netG(noise)
            print(noise.shape, img.shape)
            img_min = img.min(-1, True).values.min(-2, True).values
            img_max = img.max(-1, True).values.max(-2, True).values
            img = (img - img_min) / (img_max - img_min)
            img = pad_width_height(img)
            img = rearrange(img, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=rows)
        if len(img.shape) == 3:
            img = img.squeeze()
        (px.imshow(img, color_continuous_scale="greys_r")
        .update_layout(margin=dict.fromkeys("tblr", 20))
        .update_xaxes(showticklabels=False)
        .update_yaxes(showticklabels=False)
        ).show()
        netG.train()

def test_conv_transpose1d_minimal(conv_transpose1d_minimal, n_tests=20):
    import numpy as np
    for _ in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 30)
        ci = np.random.randint(1, 5)
        co = np.random.randint(1, 5)
        kernel_size = np.random.randint(1, 10)
        x = t.randn((b, ci, h))
        weights = t.randn((ci, co, kernel_size))
        my_output = conv_transpose1d_minimal(x, weights)
        torch_output = t.conv_transpose1d(x, weights, stride=1, padding=0)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv1d_minimal` passed!")

def test_conv_transpose1d(conv_transpose1d, n_tests=10):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 8)
        h = np.random.randint(10, 50)
        ci = np.random.randint(1, 12)
        co = np.random.randint(1, 6)
        stride = np.random.randint(1, 5)
        kernel_size = np.random.randint(1, 8)
        padding = padding = np.random.randint(0, min(kernel_size, 5))
        x = t.randn((b, ci, h))
        weights = t.randn((ci, co, kernel_size))
        my_output = conv_transpose1d(x, weights, stride=stride, padding=padding)
        torch_output = t.conv_transpose1d(x, weights, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv_transpose1d` passed!")

def test_fractional_stride_1d(fractional_stride_1d):
    x = t.tensor([[[1, 2, 3], [4, 5, 6]]])
    
    actual = fractional_stride_1d(x, stride=1)
    expected = x
    t.testing.assert_close(actual, expected)

    actual = fractional_stride_1d(x, stride=2)
    expected = t.tensor([[[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]])
    t.testing.assert_close(actual, expected)

    print("All tests in `test_fractional_stride_1d` passed!")

def test_conv_transpose2d(conv_transpose2d, n_tests=5):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 8)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 12)
        co = np.random.randint(1, 6)
        stride = tuple(np.random.randint(1, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 8, size=(2,)))
        padding = (np.random.randint(0, min(kernel_size[0], 5)), np.random.randint(0, min(kernel_size[1], 5)))
        x = t.randn((b, ci, h, w))
        weights = t.randn((ci, co, *kernel_size))
        my_output = conv_transpose2d(x, weights, stride=stride, padding=padding)
        torch_output = t.conv_transpose2d(x, weights, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output, atol=1e-4, rtol=0)
    print("All tests in `test_conv_transpose2d` passed!")

def test_ConvTranspose2d(ConvTranspose2d, n_tests=5):
    
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 8)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 12)
        co = np.random.randint(1, 6)
        stride = tuple(np.random.randint(1, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 8, size=(2,)))
        padding = (np.random.randint(0, min(kernel_size[0], 5)), np.random.randint(0, min(kernel_size[1], 5)))
        x = t.randn((b, ci, h, w))
        my_conv_module = ConvTranspose2d(ci, co, kernel_size, stride, padding)
        assert "weight" in my_conv_module._parameters, "You should name your weights 'weight' in your conv module."
        my_output = my_conv_module(x)
        torch_output = t.conv_transpose2d(x, my_conv_module.weight, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output, atol=1e-4, rtol=0)
    
    my_conv_module = ConvTranspose2d(20, 8, (3, 2), stride, padding)
    expected_sf = 1 / (8 * 3 * 2) ** 0.5
    error_msg = "Incorrect weight initialisation - check the PyTorch documentation."
    assert abs(my_conv_module.weight.mean().item()) < 0.01, error_msg
    assert 0.9 * expected_sf < my_conv_module.weight.max().item() < expected_sf, error_msg
    assert -0.9 * expected_sf > my_conv_module.weight.min().item() > -expected_sf, error_msg
    print("All tests in `test_ConvTranspose2d` passed!")

def test_Tanh(Tanh):
    x = t.randn(size=(4, 5, 6))
    expected = t.tanh(x)
    actual = Tanh()(x)
    t.testing.assert_close(expected, actual)
    print("All tests in `test_Tanh` passed.")

def test_LeakyReLU(LeakyReLU):
    x = t.randn(size=(4, 5, 6))
    a = t.randn(1).item()
    expected = t.where(x > 0, x, a * x)
    actual = LeakyReLU(negative_slope=a)(x)
    t.testing.assert_close(expected, actual)
    print("All tests in `test_LeakyReLU` passed.")

def test_Sigmoid(Sigmoid):
    x = t.randn(size=(4, 5, 6))
    expected = t.sigmoid(x)
    actual = Sigmoid()(x)
    t.testing.assert_close(expected, actual)
    print("All tests in `test_Sigmoid` passed.")

def print_param_count(*models, display_df=True, use_state_dict=False):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe

    use_state_dict: bool
        If true, uses model.state_dict() to construct dataframe
            This will include buffers, not just params
        If false, uses model.named_parameters() to construct dataframe
            This misses out buffers (more useful for GPT)
    """
    df_list = []
    gmap_list = []
    for i, model in enumerate(models, start=1):
        print(f"Model {i}, total params = {sum([param.numel() for name, param in model.named_parameters()])}")
        iterator = model.state_dict().items() if use_state_dict else model.named_parameters()
        df = pd.DataFrame([
            {f"name_{i}": name, f"shape_{i}": tuple(param.shape), f"num_params_{i}": param.numel()}
            for name, param in iterator
        ]) if (i == 1) else pd.DataFrame([
            {f"num_params_{i}": param.numel(), f"shape_{i}": tuple(param.shape), f"name_{i}": name}
            for name, param in iterator
        ])
        display(df)
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(models) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(models) > 1:
        param_counts = [df[f"num_params_{i}"].values.tolist() for i in range(1, len(models) + 1)]
        if all([param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(models) + 1):
            s = s.background_gradient(cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i-1])
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df