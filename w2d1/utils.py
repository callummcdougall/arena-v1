from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def plot_results(loss_list, accuracy_list):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=loss_list, name="Training loss"))
    fig.update_xaxes(title_text="Num batches observed")
    fig.update_yaxes(title_text="Training loss", secondary_y=False)
    # This next bit of code plots vertical lines corresponding to the epochs
    if len(accuracy_list) > 1:
        for idx, epoch_start in enumerate(np.linspace(0, len(loss_list), len(accuracy_list), endpoint=False)):
            fig.add_vline(x=epoch_start, line_width=3, line_dash="dash", annotation_text=f"Epoch {idx}", annotation_position="top right")
        fig.add_trace(
            go.Scatter(y=accuracy_list, x=np.linspace(0, len(loss_list), len(accuracy_list)), mode="lines", name="Accuracy"),
            secondary_y=True
        )
    fig.update_layout(template="simple_white", title_text="Training loss & accuracy on CIFAR10")
    fig.show()

def show_cifar_images(trainset, rows=3, cols=5):
    
    img = trainset.data[:rows*cols]
    fig = px.imshow(img, facet_col=0, facet_col_wrap=cols)
    for i, j in enumerate(np.arange(rows*cols).reshape(rows, cols)[::-1].flatten()):
            fig.layout.annotations[i].text = trainset.classes[trainset.targets[j]]
    fig.show()