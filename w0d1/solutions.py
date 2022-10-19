# %%
import numpy as np

import plotly.express as px
import plotly.graph_objs as go

from typing import Optional, Callable

import ipywidgets as wg

from fancy_einsum import einsum

def DFT_1d(arr : np.ndarray) -> np.ndarray:
    """
    Takes the DFT of the array `arr`.
    """
    
    N = len(arr)
    
    exponents = - np.outer(np.arange(N), np.arange(N)) * (2j * np.pi / N)
    
    left_matrix = np.exp(exponents)
    
    dft = left_matrix @ arr
    
    return dft


def DFT_1d(arr: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Same as function above, but has additional option of taking the inverse DFT rather than regular DFT.
    """
    
    N = len(arr)
    
    exponents = - np.outer(np.arange(N), np.arange(N)) * (2j * np.pi / N)
    if inverse:
        exponents *= -1
    # Note, we could also take complex conjugate of left_matrix, which has the same effect - but this makes future funcs
    # that we'll be writing a bit messier
    
    left_matrix = np.exp(exponents)
    if inverse:
        left_matrix /= N
    
    dft = left_matrix @ arr
    
    return dft

def test_DFT_func(DFT_1d, x=np.linspace(-1, 1), function=np.square) -> None:

    y = function(x)

    y_DFT_actual = DFT_1d(y)
    y_reconstructed_actual = DFT_1d(y_DFT_actual, inverse=True)

    y_DFT_expected = np.fft.fft(y)

    np.testing.assert_allclose(y_DFT_actual, y_DFT_expected, atol=1e-10, err_msg="DFT failed")
    np.testing.assert_allclose(y_reconstructed_actual, y, atol=1e-10, err_msg="Inverse DFT failed")

def test_DFT_func_bad(DFT_1d, x=np.linspace(-1, 1), function=np.square) -> None:

    y = function(x)
    y_DFT = DFT_1d(y)
    y_reconstructed = DFT_1d(y_DFT, inverse=True)

    np.testing.assert_allclose(y, y_reconstructed, atol=1e-10)


def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Calculates the approximation of the Riemann integral of the function `func`, between the limits x0 and x1.
    
    You should use the Left Rectangular Approximation Method (LRAM).
    """
    x = np.linspace(x0, x1, n_samples, endpoint=False)
    step_size = (x1 - x0) / n_samples
    
    y = func(x)
    
    return y.sum() * step_size

    
def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float):
    """
    Computes the integral of the function x -> func1(x) * func2(x).

    For more, see this page: https://mathworld.wolfram.com/L2-InnerProduct.html
    """
    return integrate_function(lambda x: func1(x) * func2(x), x0, x1)


def calculate_fourier_series(func: Callable, max_freq: int = 50):
    """
    Calculates the fourier coefficients of a function (assumed periodic between -pi and pi).
    
    You should get the constant term, and cosine & sine terms up to max_freq.
    
    This function should return the tuple ((a_0, A_n, B_n), func_approx), where
    a_0 is a float, A_n, B_n are lists of floats (i.e. the coefficients), and func_approx is a function
    representing the finite approximation of the function given by formula above
    
    Hint: you can vectorise a function using `np.vectorize`.
    """
    
    a_0 = (1/np.pi) * integrate_function(func, -np.pi, np.pi)
    
    A_n = [(1/np.pi) * integrate_product(func, lambda x: np.cos(n*x), -np.pi, np.pi) for n in range(1, max_freq+1)]
    
    B_n = [(1/np.pi) * integrate_product(func, lambda x: np.sin(n*x), -np.pi, np.pi) for n in range(1, max_freq+1)]
    
    def func_approx(x):
        y = 0.5 * a_0
        y += (np.array(A_n) * [np.cos(n*x) for n in range(1, max_freq+1)]).sum()
        y += (np.array(B_n) * [np.sin(n*x) for n in range(1, max_freq+1)]).sum()
        return y
    func_approx = np.vectorize(func_approx)
    
    return ((a_0, A_n, B_n), func_approx)








# ==================================== PART 2A ====================================

TARGET_FUNC = lambda x: 1 * (x > 0)
NUM_FREQUENCIES = 2
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

import numpy as np
import math

x = np.linspace(-math.pi, math.pi, 2000)
y = TARGET_FUNC(x)

x_cos = np.array([np.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = np.array([np.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = np.random.randn()
A_n = np.random.randn(NUM_FREQUENCIES)
B_n = np.random.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    
    y_pred = 0.5 * a_0 + x_cos.T @ A_n + x_sin.T @ B_n
    # or with einsum:
    # y_pred = 0.5 * a_0 + einsum("freq x, freq -> x", x_cos, A_n) + einsum("freq x, freq -> x", x_sin, B_n)
    
    # Compute and print loss
    loss = np.square(y - y_pred).sum()
    
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)
    
    # Backprop to compute gradients of coeffs with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a_0 = 0.5 * grad_y_pred.sum()
    grad_A_n = x_cos @ grad_y_pred
    grad_B_n = x_sin @ grad_y_pred
    # or with einsum:
    # grad_A_n = einsum("freq x, x -> freq", x_cos, grad_y_pred)
    # grad_B_n = einsum("freq x, x -> freq", x_sin, grad_y_pred)
    
    # Update weights using gradient descent
    a_0 -= LEARNING_RATE * grad_a_0
    A_n -= LEARNING_RATE * grad_A_n
    B_n -= LEARNING_RATE * grad_B_n



# ==================================== PART 2B ====================================

import torch
from torch import nn, optim
import math

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = torch.randn((), device=device, dtype=dtype)
A_n = torch.randn((NUM_FREQUENCIES), device=device, dtype=dtype)
B_n = torch.randn((NUM_FREQUENCIES), device=device, dtype=dtype)

LEARNING_RATE = 1e-6
TOTAL_STEPS = 4000

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    
    # Forward pass: compute predicted y
    y_pred = 0.5 * a_0 + x_cos.T @ A_n + x_sin.T @ B_n
    # or with einsum:
    # y_pred = 0.5 * a_0 + einsum("freq x, freq -> x", x_cos, A_n) + einsum("freq x, freq -> x", x_sin, B_n)
    
    # Compute and print loss
    loss = np.square(y - y_pred).sum()
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        y_pred_list.append(y_pred)
        coeffs_list.append([a_0.item(), A_n.to("cpu").numpy().copy(), B_n.to("cpu").numpy().copy()])
    
    # Backprop to compute gradients of coeffs with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a_0 = 0.5 * grad_y_pred.sum()
    grad_A_n = x_cos @ grad_y_pred
    grad_B_n = x_sin @ grad_y_pred
    # or with einsum:
    # grad_A_n = einsum("freq x, x -> freq", x_cos, grad_y_pred)
    # grad_B_n = einsum("freq x, x -> freq", x_sin, grad_y_pred)
    
    # Update weights using gradient descent
    a_0 -= LEARNING_RATE * grad_a_0
    A_n -= LEARNING_RATE * grad_A_n
    B_n -= LEARNING_RATE * grad_B_n




# ==================================== PART 2C ====================================

import torch
import math

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = torch.randn((), device=device, dtype=dtype, requires_grad=True)
A_n = torch.randn((NUM_FREQUENCIES), device=device, dtype=dtype, requires_grad=True)
B_n = torch.randn((NUM_FREQUENCIES), device=device, dtype=dtype, requires_grad=True)

LEARNING_RATE = 1e-6
TOTAL_STEPS = 4000

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    
    # Forward pass: compute predicted y
    y_pred = 0.5 * a_0 + einsum("freq x, freq -> x", x_cos, A_n) + einsum("freq x, freq -> x", x_sin, B_n)
    
    # Compute and print loss
    loss = torch.square(y - y_pred).sum()
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        y_pred_list.append(y_pred.detach())
        coeffs_list.append([a_0.item(), A_n.to("cpu").detach().numpy().copy(), B_n.to("cpu").detach().numpy().copy()])
    
    # Backprop to compute gradients of coeffs with respect to loss
    loss.backward()
    
    # Update weights using gradient descent
    with torch.no_grad():
        for coeff in [a_0, A_n, B_n]:
            coeff -= LEARNING_RATE * coeff.grad
            coeff.grad = None



# ==================================== PART 2D ====================================

import torch
import math

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

x_all = torch.concat([x_cos, x_sin], dim=0).T # we use .T so that it the 0th axis is batch dim

LEARNING_RATE = 1e-6
TOTAL_STEPS = 4000

y_pred_list = []
coeffs_list = []

model = torch.nn.Sequential(torch.nn.Linear(2 * NUM_FREQUENCIES, 1), torch.nn.Flatten(0, 1))

for step in range(TOTAL_STEPS):
    
    # Forward pass: compute predicted y
    y_pred = model(x_all)
    
    # Compute and print loss
    loss = torch.square(y - y_pred).sum()
    if step % 100 == 0:
        print(f"loss = {loss:.2f}")
        A_n = list(model.parameters())[0].detach().numpy().squeeze()[:NUM_FREQUENCIES]
        B_n = list(model.parameters())[0].detach().numpy().squeeze()[NUM_FREQUENCIES:]
        a_0 = list(model.parameters())[1].item()
        y_pred_list.append(y_pred.cpu().detach().numpy())
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
    
    # Backprop to compute gradients of coeffs with respect to loss
    loss.backward()
    
    # Update weights using gradient descent
    with torch.no_grad():
        for param in model.parameters():
            param -= LEARNING_RATE * param.grad
    model.zero_grad()



# ==================================== PART 2E ====================================

import torch
import math

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

x_all = torch.concat([x_cos, x_sin], dim=0).T # we use .T so that it the 0th axis is batch dim

LEARNING_RATE = 1e-6
TOTAL_STEPS = 4000

y_pred_list = [] 
coeffs_list = []

model = nn.Sequential(torch.nn.Linear(2 * NUM_FREQUENCIES, 1), torch.nn.Flatten(0, 1))

optimiser = optim.SGD(model.parameters(), lr=1e-6)

for step in range(TOTAL_STEPS):
    
    # Forward pass: compute predicted y
    y_pred = model(x_all)
    
    # Compute and print loss
    loss = nn.MSELoss(reduction='sum')(y_pred, y)
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        A_n = list(model.parameters())[0].detach().numpy()[:3].squeeze()
        B_n = list(model.parameters())[0].detach().numpy()[:6].squeeze()
        a_0 = list(model.parameters())[1].item()
        y_pred_list.append(y_pred.cpu().detach().numpy())
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
    
    # Backprop to compute gradients of coeffs with respect to loss
    # Update weights using gradient descent
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# %%

from tqdm.notebook import tqdm_notebook
import time

for j in tqdm_notebook(range(5)):
    for i in tqdm_notebook(range(100), leave=False):
        time.sleep(0.01)