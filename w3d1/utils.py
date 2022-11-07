import torch as t
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from einops import repeat
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Callable

class Net(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)

def format_name(name):
    return name.replace("(", "<br>   ").replace(")", "").replace(", ", "<br>   ")

def format_config(config, line_breaks=False):
    if isinstance(config, dict):
        if line_breaks:
            s = "<br>   " + "<br>   ".join([f"{key}={value}" for key, value in config.items()])
        else:
            s = ", ".join([f"{key}={value}" for key, value in config.items()])
    else:
        param_config, args_config = config
        s = "[" + ", ".join(["{" + format_config(param_group_config) + "}" for param_group_config in param_config]) + "], " + format_config(args_config)
    return s

def plot_fn(fn: Callable, x_range=[-2, 2], y_range=[-1, 3], n_points=100, log_scale=True, show_min=False):
    """Plot the specified function over the specified domain.

    If log_scale is True, take the logarithm of the output before plotting.
    """
    x = t.linspace(*x_range, n_points)
    xx = repeat(x, "w -> h w", h=n_points)
    y = t.linspace(*y_range, n_points)
    yy = repeat(y, "h -> h w", w=n_points)

    z = fn(xx, yy)

    max_contour_label = int(z.log().max().item()) + 1
    contour_range = list(range(max_contour_label))

    fig = make_subplots(
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        rows=1, cols=2,
        subplot_titles=["3D plot", "2D log plot"]
    ).update_layout(height=700, width=1600, title_font_size=40).update_annotations(font_size=20)

    fig.add_trace(
        go.Surface(
            x=x, y=y, z=z,
            colorscale="greys",
            showscale=False,
            hovertemplate = '<b>x</b> = %{x:.2f}<br><b>y</b> = %{y:.2f}<br><b>z</b> = %{z:.2f}</b>',
            contours = dict(
                x = dict(show=True, color="grey", start=x_range[0], end=x_range[1], size=0.2),
                y = dict(show=True, color="grey", start=y_range[0], end=y_range[1], size=0.2),
                # z = dict(show=True, color="red", size=0.001)
            )
        ), row=1, col=1
    )
    fig.add_trace(
        go.Contour(
            x=x, y=y, z=t.log(z) if log_scale else z,
            customdata=z,
            hovertemplate = '<b>x</b> = %{x:.2f}<br><b>y</b> = %{y:.2f}<br><b>z</b> = %{customdata:.2f}</b>',
            colorscale="greys",
            # colorbar=dict(tickmode="array", tickvals=contour_range, ticktext=[f"{math.exp(i):.0f}" for i in contour_range])
        ),
        row=1, col=2
    )
    fig.update_traces(showscale=False, col=2)
    if show_min:
        fig.add_trace(
            go.Scatter(
                mode="markers", x=[1.0], y=[1.0], marker_symbol="x", marker_line_color="midnightblue", marker_color="lightskyblue",
                marker_line_width=2, marker_size=12, name="Global minimum"
            ),
            row=1, col=2
        )

    return fig

def plot_optimization_sgd(opt_fn_with_sgd: Callable, fn: Callable, xy: t.Tensor, x_range=[-2, 2], y_range=[-1, 3], lr=0.001, momentum=0.98, n_iters=100, log_scale=True, n_points=100, show_min=False):

    fig = plot_fn(fn, x_range, y_range, n_points, log_scale, show_min)

    xys = opt_fn_with_sgd(fn, xy, lr, momentum, n_iters)
    x, y = xys.T
    z = fn(x, y)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=6, color="red"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=6, color="red"), line=dict(width=1, color="red")), row=1, col=2)

    fig.update_layout(showlegend=False)
    fig.data = fig.data[::-1]

    return fig

def plot_optimization(opt_fn: Callable, fn: Callable, xy: t.Tensor, optimizers: list, x_range=[-2, 2], y_range=[-1, 3], n_iters: int = 100, log_scale: bool = True, n_points: int = 100, show_min=False):

    fig = plot_fn(fn, x_range, y_range, n_points, log_scale, show_min)

    for i, (color, optimizer) in enumerate(zip(px.colors.qualitative.Set1, optimizers)):
        xys = opt_fn(fn, xy.clone().detach().requires_grad_(True), *optimizer, n_iters).numpy()
        x, y = xys.T
        z = fn(x, y)
        optimizer_active = optimizer[0]([xy.clone().detach().requires_grad_(True)], **optimizer[1])
        name = format_name(str(optimizer_active))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=6, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=6, color=color), line=dict(width=1, color=color), name=name), row=1, col=2)

    fig.data = fig.data[::-1]

    return fig

def plot_optimization_with_schedulers(opt_fn_with_scheduler: Callable, fn: Callable, xy: t.Tensor, optimizers: list, schedulers: list, x_range=[-2, 2], y_range=[-1, 3], n_iters: int = 100, log_scale: bool = True, n_points: int = 100, show_min=False):

    fig = plot_fn(fn, x_range, y_range, n_points, log_scale, show_min)

    for i, (color, optimizer, scheduler) in enumerate(zip(px.colors.qualitative.Set1, optimizers, schedulers)):
        optimizer_active = optimizer[0]([xy.clone().detach().requires_grad_(True)], **optimizer[1])
        name_opt = format_name(str(optimizer_active))
        if len(scheduler) == 0:
            scheduler = (None, dict())
            name = name_opt + "<br>(no scheduler)"
        else:
            scheduler_active = scheduler[0](optimizer_active, **scheduler[1])
            name_sch = format_name(str(scheduler_active))
            name = name_opt + "<br>" + name_sch
        xys = opt_fn_with_scheduler(fn, xy.clone().detach().requires_grad_(True), *optimizer, *scheduler, n_iters).numpy()
        x, y = xys.T
        z = fn(x, y)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=6, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=6, color=color), line=dict(width=1, color=color), name=name), row=1, col=2)

    fig.data = fig.data[::-1]

    return fig


def _get_moon_data(unsqueeze_y=False):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=354)
    X = t.tensor(X, dtype=t.float32)
    y = t.tensor(y, dtype=t.int64)
    if unsqueeze_y:
        y = y.unsqueeze(-1)
    return DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

def _train_with_opt(model, opt):
    dl = _get_moon_data()
    for i, (X, y) in enumerate(dl):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()

def _train_with_scheduler(model, opt, scheduler):
    dl = _get_moon_data()
    for epoch in range(20):
        for i, (X, y) in enumerate(dl):
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            opt.step()
        scheduler.step()

def test_sgd(SGD):

    test_cases = [
        dict(lr=0.1, momentum=0.0, weight_decay=0.0),
        dict(lr=0.1, momentum=0.7, weight_decay=0.0),
        dict(lr=0.1, momentum=0.5, weight_decay=0.0),
        dict(lr=0.1, momentum=0.5, weight_decay=0.05),
        dict(lr=0.2, momentum=0.8, weight_decay=0.05),
    ]
    for opt_config in test_cases:
        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)


def test_rmsprop(RMSprop):

    test_cases = [
        dict(lr=0.1, alpha=0.9, eps=0.001, weight_decay=0.0, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.5),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
    ]
    for opt_config in test_cases:
        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)

def test_adam(Adam):

    test_cases = [
        dict(lr=0.1, betas=(0.8, 0.95), eps=0.001, weight_decay=0.0),
        dict(lr=0.1, betas=(0.8, 0.9), eps=0.001, weight_decay=0.05),
        dict(lr=0.2, betas=(0.9, 0.95), eps=0.01, weight_decay=0.08),
    ]
    for opt_config in test_cases:
        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)




def get_sgd_optimizer(model, opt_config, SGD):
    if isinstance(opt_config, dict):
        return SGD(model.parameters(), **opt_config)
    else:
        opt_params = [d.copy() for d in opt_config[0]]
        _opt_config = opt_config[1]
        weight_params = [param for name, param in model.named_parameters() if "weight" in name]
        bias_params = [param for name, param in model.named_parameters() if "bias" in name]
        for param_group in opt_params:
            param_group["params"] = weight_params if param_group["params"] == "weights" else bias_params
        return SGD(opt_params, **_opt_config)



def test_ExponentialLR(ExponentialLR, SGD):

    print("Testing ExponentialLR, training loop has 30 epochs, 4 batches per epoch")


    test_cases = [
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(gamma=1.0)),
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(gamma=0.5)),
        dict(opt_config=dict(lr=0.01, momentum=0.9, weight_decay=0.1), scheduler_config=dict(gamma=0.5)),
    ]
    for config in test_cases:
        opt_config = config["opt_config"].copy()
        scheduler_config = config["scheduler_config"]

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        scheduler = ExponentialLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_correct = model.layers[0].weight
        b0_correct = model.layers[0].bias

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.SGD(model.parameters(), **opt_config)
        scheduler = optim.lr_scheduler.ExponentialLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_submitted = model.layers[0].weight
        b0_submitted = model.layers[0].bias

        print("\nTesting configuration:\n\toptimizer: ", format_config(opt_config), "\n\tscheduler: ", format_config(scheduler_config))
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        assert isinstance(b0_correct, t.Tensor)
        assert isinstance(b0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)
        t.testing.assert_close(b0_correct, b0_submitted, rtol=0, atol=1e-5)
    print("\nAll tests in `test_ExponentialLR` passed!")


def test_StepLR(StepLR, SGD):

    print("Testing StepLR, training loop has 30 epochs, 4 batches per epoch")


    test_cases = [
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(step_size=30, gamma=1.0)),
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(step_size=3, gamma=1.0)),
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(step_size=1, gamma=0.5)),
        dict(opt_config=dict(lr=0.01, momentum=0.9, weight_decay=0.1), scheduler_config=dict(step_size=3, gamma=0.5)),
    ]
    for config in test_cases:
        opt_config = config["opt_config"].copy()
        scheduler_config = config["scheduler_config"]

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        scheduler = StepLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_correct = model.layers[0].weight
        b0_correct = model.layers[0].bias

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.SGD(model.parameters(), **opt_config)
        scheduler = optim.lr_scheduler.StepLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_submitted = model.layers[0].weight
        b0_submitted = model.layers[0].bias

        print("\nTesting configuration:\n\toptimizer: ", format_config(opt_config), "\n\tscheduler: ", format_config(scheduler_config))
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        assert isinstance(b0_correct, t.Tensor)
        assert isinstance(b0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)
        t.testing.assert_close(b0_correct, b0_submitted, rtol=0, atol=1e-5)
    print("\nAll tests in `test_StepLR` passed!")


def test_MultiStepLR(MultiStepLR, SGD):

    print("Testing MultiStepLR, training loop has 30 epochs, 4 batches per epoch")


    test_cases = [
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(milestones=[40], gamma=1.0)),
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(milestones=[10], gamma=0.5)),
        dict(opt_config=dict(lr=0.01, momentum=0.9, weight_decay=0.1), scheduler_config=dict(milestones=[10, 15], gamma=0.5)),
    ]
    for config in test_cases:
        opt_config = config["opt_config"].copy()
        scheduler_config = config["scheduler_config"]

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        scheduler = MultiStepLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_correct = model.layers[0].weight
        b0_correct = model.layers[0].bias

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.SGD(model.parameters(), **opt_config)
        scheduler = optim.lr_scheduler.MultiStepLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_submitted = model.layers[0].weight
        b0_submitted = model.layers[0].bias

        print("\nTesting configuration:\n\toptimizer: ", format_config(opt_config), "\n\tscheduler: ", format_config(scheduler_config))
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        assert isinstance(b0_correct, t.Tensor)
        assert isinstance(b0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)
        t.testing.assert_close(b0_correct, b0_submitted, rtol=0, atol=1e-5)
    print("\nAll tests in `test_MultiStepLR` passed!")


