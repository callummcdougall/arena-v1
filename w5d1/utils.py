import torch as t
from torch import nn

import w5d1.solutions

@t.inference_mode()
def test_groupnorm(GroupNorm, affine: bool):
    if not affine:
        x = t.arange(72, dtype=t.float32).view(3, 6, 2, 2)
        ref = t.nn.GroupNorm(num_groups=3, num_channels=6, affine=False)
        expected = ref(x)
        gn = GroupNorm(num_groups=3, num_channels=6, affine=False)
        actual = gn(x)
        t.testing.assert_close(actual, expected)
        print("All tests in `test_groupnorm(affine=False)` passed.")

    else:
        t.manual_seed(776)
        x = t.randn((3, 6, 8, 10), dtype=t.float32)
        ref = t.nn.GroupNorm(num_groups=3, num_channels=6, affine=True)
        ref.weight = nn.Parameter(t.randn_like(ref.weight))
        ref.bias = nn.Parameter(t.randn_like(ref.bias))
        expected = ref(x)
        gn = GroupNorm(num_groups=3, num_channels=6, affine=True)
        gn.weight.copy_(ref.weight)
        gn.bias.copy_(ref.bias)
        actual = gn(x)
        t.testing.assert_close(actual, expected)
        print("All tests in `test_groupnorm(affine=True)` passed.")

@t.inference_mode()
def test_self_attention(SelfAttention):
    channels = 16
    img = t.randn(1, channels, 64, 64)
    sa = SelfAttention(channels=channels, num_heads=4)
    out = sa(img)
    print("Testing shapes of output...")
    assert out.shape == img.shape
    print("Shape test in `test_self_attention` passed.")
    print("Testing values of output...")
    sa_solns = w5d1.solutions.SelfAttention(channels=channels, num_heads=4)
    try:
        sa.W_QKV = sa_solns.in_proj
        sa.W_O = sa_solns.out_proj
        out_actual = sa(img)
        out_expected = sa_solns(img)
        t.testing.assert_close(out_actual, out_expected)
        print("All tests in `test_self_attention` passed.")
    except:
        print("Warning: you need linear layers called `W_QKV` and `W_O` with biases, otherwise the values test can't be performed.")

@t.inference_mode()
def test_attention_block(AttentionBlock):
    ab = AttentionBlock(channels=16)
    img = t.randn(1, 16, 64, 64)
    out = ab(img)
    assert out.shape == img.shape
    print("Shape test in `test_attention_block` passed.")

@t.inference_mode()
def test_residual_block(ResidualBlock):
    in_channels = 6
    out_channels = 10
    step_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, in_channels, 32, 32)
    rb = ResidualBlock(in_channels, out_channels, step_dim, groups)
    out = rb(img, time_emb)
    print("Testing shapes of output...")
    assert out.shape == (1, out_channels, 32, 32)
    print("Shape test in `test_residual_block` passed.")
    print("Testing parameter count...")
    rb_soln = w5d1.solutions.ResidualBlock(in_channels, out_channels, step_dim, groups)
    param_list = sorted([tuple(p.shape) for p in rb.parameters()], key=lambda x: -t.prod(t.tensor(x)).item())
    param_list_expected = sorted([tuple(p.shape) for p in rb_soln.parameters()], key=lambda x: -t.prod(t.tensor(x)).item())
    if param_list == param_list_expected:
        print("Parameter count test in `test_residual_block` passed.")
    else:
        param_count = sum([p.numel() for p in rb.parameters()])
        param_count_expected = sum([p.numel() for p in rb_soln.parameters()])
        if param_count_expected - param_count == 3 * out_channels:
            print("Your parameter count is off by 3 * out_channels. This is probably because your conv layers have no biases. You can rewrite Conv2d to include biases if you want, otherwise you can proceed to the next section (this won't stop your model working).\nAfter this test, no errors will be raised for missing biases.")
        else:
            error_msg = "\n".join(["Parameter count test failed", f"Your parameter shapes are {param_list}", f"Expected param shapes are {param_list_expected}"])
            raise Exception(error_msg)

@t.inference_mode()
def test_downblock(DownBlock, downsample: bool):
    in_channels = 8
    out_channels = 12
    time_emb_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, in_channels, 32, 32)
    db = DownBlock(in_channels, out_channels, time_emb_dim, groups, downsample)
    out, skip = db(img, time_emb)
    print("Testing shapes of output...")
    assert skip.shape == (1, out_channels, 32, 32)
    if downsample:
        assert out.shape == (1, out_channels, 16, 16)
    else:
        assert out.shape == (1, out_channels, 32, 32)
    print("Shape test in `test_downblock` passed.")
    print("Testing parameter count...")
    db_soln = w5d1.solutions.DownBlock(in_channels, out_channels, time_emb_dim, groups, downsample)
    param_count = sum([p.numel() for p in db.parameters() if p.ndim > 1])
    param_count_expected = sum([p.numel() for p in db_soln.parameters() if p.ndim > 1])
    error_msg = f"Total number of (non-bias) parameters don't match: you have {param_count}, expected number is {param_count_expected}."
    if downsample==False:
        error_msg += "\nNote that downsample=False, so you don't need to define the conv layer."
    assert param_count == param_count_expected, error_msg
    print("Parameter count test in `test_residual_block` passed.\n")

@t.inference_mode()
def test_midblock(MidBlock):
    mid_channels = 8
    time_emb_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, mid_channels, 32, 32)
    mid = MidBlock(mid_channels, time_emb_dim, groups)
    print("Testing shapes of output...")
    out = mid(img, time_emb)
    assert out.shape == (1, mid_channels, 32, 32)
    print("Shape test in `test_midblock` passed.")
    print("Testing parameter count...")
    mid_soln = w5d1.solutions.MidBlock(mid_channels, time_emb_dim, groups)
    param_count = sum([p.numel() for p in mid.parameters() if p.ndim > 1])
    param_count_expected = sum([p.numel() for p in mid_soln.parameters() if p.ndim > 1])
    assert param_count == param_count_expected, f"Total number of (non-bias) parameters don't match: you have {param_count}, expected number is {param_count_expected}."
    print("Parameter count test in `test_midblock` passed.\n")

@t.inference_mode()
def test_upblock(UpBlock, upsample):
    in_channels = 8
    out_channels = 12
    time_emb_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, out_channels, 16, 16)
    skip = t.rand_like(img)
    up = UpBlock(in_channels, out_channels, time_emb_dim, groups, upsample)
    out = up(img, time_emb, skip)
    print("Testing shapes of output...")
    if upsample:
        assert out.shape == (1, in_channels, 32, 32)
    else:
        assert out.shape == (1, in_channels, 16, 16)
    print("Shape test in `test_upblock` passed.")
    print("Testing parameter count...")
    up_soln = w5d1.solutions.UpBlock(in_channels, out_channels, time_emb_dim, groups, upsample)
    param_count = sum([p.numel() for p in up.parameters() if p.ndim > 1])
    param_count_expected = sum([p.numel() for p in up_soln.parameters() if p.ndim > 1])
    error_msg = f"Total number of (non-bias) parameters don't match: you have {param_count}, expected number is {param_count_expected}."
    if upsample==False:
        error_msg += "\nNote that upsample=False, so you don't need to define the convtranspose layer."
    assert param_count == param_count_expected, error_msg
    print("Parameter count test in `test_upblock` passed.\n")

@t.inference_mode()
def test_unet(Unet):
    # dim mults is limited by number of multiples of 2 in the image
    # 28 -> 14 -> 7 is ok but can't half again without having to deal with padding
    image_size = 28
    channels = 8
    batch_size = 8
    model = Unet(
        image_shape=(8, 28, 28),
        channels=channels,
        dim_mults=(1, 2, 4),
    )
    x = t.randn((batch_size, channels, image_size, image_size))
    num_steps = t.randint(0, 1000, (batch_size,))
    out = model(x, num_steps)
    assert out.shape == x.shape