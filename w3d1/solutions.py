from typing import Iterable, Callable
import torch as t
from torch import optim


# %%

def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    """
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum, 

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    """
    assert xy.requires_grad

    xys = t.zeros((n_iters, 2))
    optimizer = optim.SGD([xy], lr=lr, momentum=momentum)

    for i in range(n_iters):
        xys[i] = xy.detach()
        out = fn(xy[0], xy[1])
        out.backward()
        optimizer.step()
        optimizer.zero_grad()
    return xys

# %%

class SGD:
    def __init__(
        self, 
        params: Iterable[t.nn.parameter.Parameter], 
        lr: float, 
        momentum: float = 0.0, 
        weight_decay: float = 0.0
    ):
        """Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        """
        self.params = list(params)
        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay
        self.t = 0

        self.gs = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (g, param) in enumerate(zip(self.gs, self.params)):
            # Implement the algorithm from the pseudocode to get new values of params and g
            new_g = param.grad
            if self.lmda != 0:
                new_g = new_g + (self.lmda * param)
            if self.mu != 0 and self.t > 0:
                new_g = (self.mu * g) + new_g
            # Update params (remember, this must be inplace)
            self.params[i] -= self.lr * new_g
            # Update g
            self.gs[i] = new_g
        self.t += 1

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"


class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        """Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop

        """
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.mu = momentum
        self.lmda = weight_decay
        self.alpha = alpha

        self.gs = [t.zeros_like(p) for p in self.params]
        self.bs = [t.zeros_like(p) for p in self.params]
        self.vs = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (p, g, b, v) in enumerate(zip(self.params, self.gs, self.bs, self.vs)):
            new_g = p.grad
            if self.lmda != 0:
                new_g = new_g + self.lmda * p
            self.gs[i] = new_g
            new_v = self.alpha * v + (1 - self.alpha) * new_g.pow(2)
            self.vs[i] = new_v
            if self.mu > 0:
                new_b = self.mu * b + new_g / (new_v.sqrt() + self.eps)
                p -= self.lr * new_b
                self.bs[i] = new_b
            else:
                p -= self.lr * new_g / (new_v.sqrt() + self.eps)

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"


class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lmda = weight_decay
        self.t = 1

        self.gs = [t.zeros_like(p) for p in self.params]
        self.ms = [t.zeros_like(p) for p in self.params]
        self.vs = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (p, g, m, v) in enumerate(zip(self.params, self.gs, self.ms, self.vs)):
            new_g = p.grad
            if self.lmda != 0:
                new_g = new_g + self.lmda * p
            self.gs[i] = new_g
            new_m = self.beta1 * m + (1 - self.beta1) * new_g
            new_v = self.beta2 * v + (1 - self.beta2) * new_g.pow(2)
            self.ms[i] = new_m
            self.vs[i] = new_v
            m_hat = new_m / (1 - self.beta1 ** self.t)
            v_hat = new_v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
        self.t += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"
    


def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_kwargs: dict, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    assert xy.requires_grad

    xys = t.zeros((n_iters, 2))
    optimizer = optimizer_class([xy], **optimizer_kwargs)

    for i in range(n_iters):
        xys[i] = xy.detach()
        out = fn(xy[0], xy[1])
        out.backward()
        optimizer.step()
        optimizer.zero_grad()
    return xys


def opt_fn_with_scheduler(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_kwargs, scheduler_class=None, scheduler_kwargs=dict(), n_iters: int = 100):
    """
    Optimize the a given function starting from the specified point.

    scheduler_class: one of the schedulers you've defined, either ExponentialLR, StepLR or MultiStepLR
    scheduler_kwargs: keyword arguments passed to your optimiser (e.g. gamma)
    """
    assert xy.requires_grad

    xys = t.zeros((n_iters, 2))
    optimizer = optimizer_class([xy], **optimizer_kwargs)
    if scheduler_class is not None:
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)

    for i in range(n_iters):
        xys[i] = xy.detach()
        out = fn(xy[0], xy[1])
        out.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler_class is not None:
            scheduler.step()
    return xys


# %%

class StepLR():
    def __init__(self, optimizer, step_size, gamma=0.1):
        """Implements StepLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.n_steps = 0

    def step(self):
        
        self.n_steps += 1
        if self.n_steps % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    def __repr__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma})"

class MultiStepLR():
    def __init__(self, optimizer, milestones, gamma=0.1):
        """Implements MultiStepLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
        """
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.n_steps = 0

    def step(self):
        self.n_steps += 1
        if self.n_steps in self.milestones:
            self.optimizer.lr *= self.gamma

    def __repr__(self):
        return f"MultiStepLR(milestones={self.milestones}, gamma={self.gamma})"

class ExponentialLR():
    def __init__(self, optimizer, gamma):
        """Implements ExponentialLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
        """
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self):
        self.optimizer.lr *= self.gamma

    def __repr__(self):
        return f"ExponentialLR(gamma={self.gamma})"

