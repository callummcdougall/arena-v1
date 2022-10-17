import numpy as np

import plotly.express as px
import plotly.graph_objs as go

from typing import Optional, Callable

import ipywidgets as wg

from fancy_einsum import einsum

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
    
def test_integrate_function(integrate_function):
    
    func = lambda x: np.sin(x) ** 2
    x0 = -np.pi
    x1 = np.pi
    
    integral_approx = integrate_function(func, x0, x1)
    integral_true = 0.5 * (x1 - x0)
    
    np.testing.assert_allclose(integral_true, integral_approx, atol=1e-10)

def test_integrate_product(integrate_product):
    
    func1 = np.sin
    func2 = np.cos 
    x0 = -np.pi
    x1 = np.pi
    
    integrate_product_approx = integrate_product(func1, func2, x0, x1)
    integrate_product_true = 0
    np.testing.assert_allclose(integrate_product_true, integrate_product_approx, atol=1e-10)
    
    integrate_product_approx = integrate_product(func1, func1, x0, x1)
    integrate_product_true = 0.5 * (x1 - x0)
    np.testing.assert_allclose(integrate_product_true, integrate_product_approx, atol=1e-10)

def create_interactive_fourier_graph_widgets(calculate_fourier_series: Callable, func: Callable):

    label = wg.Label("Number of terms in Fourier series: ")

    slider = wg.IntSlider(min=0, max=50, value=0)

    x = np.linspace(-np.pi, np.pi, 1000)
    y = func(x)

    fig = go.FigureWidget(
        data = [
            go.Scatter(x=x, y=y, name="Original function", mode="lines"),
            go.Scatter(x=x, y=y, name="Reconstructed function", mode="lines")
        ],
        layout = go.Layout(title_text=r"Original vs reconstructed", template="simple_white", margin_t=100)
    )

    def respond_to_slider(change):
        max_freq = slider.value
        coeffs, func_approx = calculate_fourier_series(func, max_freq)
        fig.data[1].y = np.vectorize(func_approx)(x)

    slider.observe(respond_to_slider)

    respond_to_slider("unimportant text to trigger first response")

    box_layout = wg.Layout(border="solid 1px black", padding="20px", margin="20px", width="80%")

    return wg.VBox([wg.HBox([label, slider], layout=box_layout), fig])

def bool_list(i, m):
    l = [True] + [False for j in range(m)]
    l[i] = True
    return l

def create_interactive_fourier_graph(calculate_fourier_series: Callable, func: Callable):

    x = np.linspace(-np.pi, np.pi, 300)
    y = func(x)
    
    sliders = [dict(
        active=0,
        currentvalue_prefix="Max frequency: ",
        pad_t=40,
        steps=[
            dict(method="update", args=[{"visible": bool_list(i, 30)}])
            for i in range(30)
        ]
    )]
    
    data = [go.Scatter(x=x, y=y, name="Original function", mode="lines")]
    
    for max_freq in range(30):
        data.append(
            go.Scatter(
                x=x, 
                y=calculate_fourier_series(func, max_freq)[1](x),
                name="Reconstructed function", 
                mode="lines",
                visible=(max_freq==0)
            )
        )

    fig = go.Figure(
        data=data,
        layout = go.Layout(
            title_text=r"Original vs reconstructed", 
            template="simple_white", 
            margin_t=100,
            sliders=sliders
        )
    )

    return fig

TARGET_FUNC = np.sin
NUM_FREQUENCIES = 4
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6
    
def get_title_from_coeffs(a_0, A_n, B_n):
    A_n_coeffs = " + ".join([f"{a_n:.2f}" + r"\cos{" + (str(n) if n>1 else "") + " x}" for (n, a_n) in enumerate(A_n, 1)])
    B_n_coeffs = " + ".join([f"{b_n:.2f}" + r"\sin{" + (str(n) if n>1 else "") + " x}" for (n, b_n) in enumerate(B_n, 1)])
    return r"$y = " + f"{0.5*a_0:.2f}" + " + " + A_n_coeffs + " + " + B_n_coeffs + "$"

def visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list):

    label = wg.Label("Number of steps: ")

    slider = wg.IntSlider(min=0, max=TOTAL_STEPS-100, value=0, step=100)

    fig = go.FigureWidget(
        data = [go.Scatter(x=x, y=y, mode="lines", marker_color="blue")] + [
            go.Scatter(x=x, y=y_pred_list[i], mode="lines", marker_color="rgba(100, 100, 100, 0.1)")
            for i in range(len(y_pred_list))
        ],
        layout = go.Layout(title_text=r"Original vs reconstructed", template="simple_white", margin_t=100, showlegend=False)
    )

    def respond_to_slider(change):
        idx = slider.value // 100
        with fig.batch_update():
            fig.update_layout(title_text = get_title_from_coeffs(*coeffs_list[idx]))
            for i in range(len(list(fig.data))-1):
                fig.data[i+1]["marker"]["color"] = "red" if i == idx else "rgba(100, 100, 100, 0.1)"

    slider.observe(respond_to_slider)

    respond_to_slider("unimportant text to trigger first response")

    box_layout = wg.Layout(border="solid 1px black", padding="20px", margin="20px", width="80%")

    return wg.VBox([wg.HBox([label, slider], layout=box_layout), fig])