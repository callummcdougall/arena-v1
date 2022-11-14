# %%

import os
os.chdir(r"C:\Users\calsm\Documents\AI Alignment\ARENA\in_progress\w2d2")

# %%

from solutions_build_bert import *

# %%
import os
import re
import tarfile
from dataclasses import dataclass
import requests
import torch as t
import transformers
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
import plotly.express as px
import pandas as pd
from typing import Callable, Optional, List
import time

# %%

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_FOLDER = "./data/imdb/"
IMDB_PATH = os.path.join(DATA_FOLDER, "acllmdb_v1.tar.gz")
SAVED_TOKENS_PATH = os.path.join(DATA_FOLDER, "tokens.pt")
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%
# ================== Data loading ==================

def maybe_download(url: str, path: str) -> None:
    """Download the file from url and save it to path. If path already exists, do nothing."""
    if not os.path.exists(IMDB_PATH):
        with open(IMDB_PATH, "wb") as file:
            data = requests.get(url).content
            file.write(data)

os.makedirs(DATA_FOLDER, exist_ok=True)
expected_hexdigest = "d41d8cd98f00b204e9800998ecf8427e"
maybe_download(IMDB_URL, IMDB_PATH)

@dataclass(frozen=True)
class Review:
    split: str
    is_positive: bool
    stars: int
    text: str

def load_reviews(path: str) -> list[Review]:
    reviews = []
    tar = tarfile.open(path, "r:gz")
    for member in tqdm(tar.getmembers()):
        m = re.match(r"aclImdb/(train|test)/(pos|neg)/\d+_(\d+)\.txt", member.name)
        if m is not None:
            split, posneg, stars = m.groups()
            buf = tar.extractfile(member)
            assert buf is not None
            text = buf.read().decode("utf-8")
            reviews.append(Review(split, posneg == "pos", int(stars), text))
    return reviews
        
reviews = load_reviews(IMDB_PATH)
assert sum((r.split == "train" for r in reviews)) == 25000
assert sum((r.split == "test" for r in reviews)) == 25000






# %%
# ================== INSPECTION ==================




df = pd.DataFrame(reviews)
df["length"] = [len(text) for text in df["text"]]

# %%

# Stars plot
fig = px.histogram(x=df["stars"]).update_layout(bargap=0.1)

# Lengths plot
fig = px.histogram(x=df["length"])

# Lengths plot, positive and negative compared
fig = px.histogram(df, x="length", color="is_positive", barmode="overlay")



# NEED pip install lingua-language-detector

from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
# We don't use all languages because this takes much too long
detector = LanguageDetectorBuilder.from_languages(*Language).build()

# Sample 200 datapoints, because it takes a while to run
# Result: all 2000 datapoints are english
languages_detected = df.sample(500)["text"].apply(detector.detect_language_of).value_counts()
languages_detected


# %%

# Note about tokenizer - you can't call encode on a sequence, but you can call the tokenizer on a sequence

def to_dataset(tokenizer, reviews: list[Review]) -> TensorDataset:
    """Tokenize the reviews (which should all belong to the same split) and bundle into a TensorDataset.

    The tensors in the TensorDataset should be (in this exact order):

    input_ids: shape (batch, sequence length), dtype int64
    attention_mask: shape (batch, sequence_length), dtype int
    sentiment_labels: shape (batch, ), dtype int
    star_labels: shape (batch, ), dtype int
    """
    df = pd.DataFrame(reviews).iloc[:1000, :]

    encoded = tokenizer(df["text"].values.tolist(), padding="max_length", truncation=True, return_tensors="pt")
    sentiment_labels = t.tensor(df["is_positive"].values).to(t.int64)
    star_labels = t.tensor(df["stars"].values)
    
    return TensorDataset(encoded.input_ids, encoded.attention_mask, sentiment_labels, star_labels)

# TODO: truncate data at the end, not at the start
def to_dataset_clean(tokenizer, reviews: list[Review]) -> TensorDataset:
    """
    Tokenize the reviews, stripping out line breaks.
    """
    df = pd.DataFrame(reviews).iloc[:1000, :]
    df["text"] = df["text"].apply(lambda x: re.sub("<br /><br />", " ", x)).apply(lambda x: re.sub("< br / > < br / >", " ", x))

    encoded = tokenizer(df["text"].values.tolist(), padding="max_length", truncation=True, return_tensors="pt")
    sentiment_labels = t.tensor(df["is_positive"].values).to(t.int64)
    star_labels = t.tensor(df["stars"].values)
    
    return TensorDataset(encoded.input_ids, encoded.attention_mask, sentiment_labels, star_labels) 


tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
train_data = to_dataset_clean(tokenizer, [r for r in reviews if r.split == "train"]) # or to_dataset_clean
test_data = to_dataset_clean(tokenizer, [r for r in reviews if r.split == "test"])

t.save((train_data, test_data), SAVED_TOKENS_PATH)
(train_data, test_data) = t.load(SAVED_TOKENS_PATH)





# %%
# ================== Define BertClassifier, and copy over weights ==================





def copy_weights_from_bert_common(my_bert_common: BertCommon, pretrained_bert_common) -> BertCommon:

    my_bert_common_list = list(my_bert_common.named_parameters())
    pretrained_bert_common_list = list(pretrained_bert_common.named_parameters())
    
    # Initialise an empty dictionary to store the correct key-value pairs
    state_dict = {}

    # Check the number of params/buffers is correct
    assert len(my_bert_common_list) == len(pretrained_bert_common_list), "Number of layers is wrong."
    
    for (my_param_name, my_param), (name, param) in zip(my_bert_common_list, pretrained_bert_common_list):
        state_dict[my_param_name] = param

    if set(state_dict.keys()) != set(my_bert_common.state_dict().keys()):
        raise Exception("State dicts don't match.")
    
    my_bert_common.load_state_dict(state_dict)
    
    return my_bert_common

class BertClassifier(nn.Module):

    def __init__(self, pretrained_bert_common, config: TransformerConfig):
        super().__init__()
        self.bertcommon = copy_weights_from_bert_common(BertCommon(config), pretrained_bert_common)
        self.dropout = nn.Dropout(config.dropout)
        self.sentiment_fc = nn.Linear(config.hidden_size, 2)
        self.star_fc = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """Compute logits for each token in the vocabulary.

        Return: shape (batch, seq, vocab_size)
        """
        x = self.bertcommon(input_ids, one_zero_attention_mask, token_type_ids)

        x = self.dropout(x[:, 0, :])

        x_sentiment = self.sentiment_fc(x)
        x_star = 5 + 5 * self.star_fc(x)

        return {"sentiment": x_sentiment, "star": x_star}

pretrained_bert_common = transformers.BertForMaskedLM.from_pretrained("bert-base-cased").bert
my_bert_classifier = BertClassifier(pretrained_bert_common, config).train().to(device)

# %%
# ================== Training loop ==================

def train(
    model, 
    trainset, 
    testset, 
    epochs: int, 
    loss_fns: List[Callable], # list of two loss functions, for the sentiment and star-classification tasks respectively
    loss_fns_weighting: List[float], # linear combination of the loss_fns to produce the loss we train wrt
    batch_size: int, 
    lr: float, 
    weight_decay: float
) -> None:
    """
    Loss is given by:
        L = w0 * f0(sentiment_output, sentiment_labels) + w1 * f1(star_output, star_labels)
    where [w0, w1] are loss_fns_weighting, and [f0, f1] are loss_fns
    """

    # config_dict = {
    #     "batch_size": batch_size,
    #     "loss_fns_weighting": loss_fns_weighting,
    #     "lr": lr,
    # }
    # wandb.init(project="bert_finetuning_imdb", config=config_dict)

    loss_lists = {"star": [], "sentiment": []}
    w0, w1 = loss_fns_weighting
    f0, f1 = loss_fns
    
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    # wandb.watch(model, log="all", log_freq=20, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for i, (input_ids, attention_mask, sentiment_labels, star_labels) in enumerate(progress_bar):

            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            sentiment_labels, star_labels = sentiment_labels.to(device), star_labels.to(device, t.float32)
            
            output = model(input_ids, attention_mask)

            sentiment_loss = f0(output["sentiment"], sentiment_labels)
            star_loss = f1(output["star"].squeeze(), star_labels)
            loss = w0 * sentiment_loss + w1 * star_loss
            loss_lists["sentiment"].append(sentiment_loss.detach().cpu().sum().item())
            loss_lists["star"].append(star_loss.detach().cpu().sum().item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch = {epoch+1}, Loss = {loss.item():.4f}")
            
            examples_seen += input_ids.size(0)
            
        
        with t.inference_mode():
            
            sentiment_accuracy = 0
            star_accuracy = 0
            star_mae = 0
            total = 0
            
            for (input_ids, attention_mask, sentiment_labels, star_labels) in testloader:

                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                sentiment_labels, star_labels = sentiment_labels.to(device), star_labels.to(device, t.float32)

                output = model(input_ids, attention_mask)
                sentiment_predictions = output["sentiment"].argmax(1)
                star_predictions = t.round(output["star"])
                
                sentiment_accuracy += (sentiment_predictions == sentiment_labels).sum().item()
                star_accuracy += (star_predictions == star_labels).sum().item()
                star_mae += (output["star"] - star_labels).abs().sum()
                total += output["sentiment"].size(0)

                # wandb.log({
                #     "test_sentiment_accuracy": sentiment_accuracy/total, 
                #     "test_star_accuracy": star_accuracy/total,
                #     "test_star_mae": star_mae/total
                # }, step=examples_seen)
        
        print("\n".join([
            f"Epoch {epoch+1}/{epochs}",
            f"Test sentiment accuracy is {sentiment_accuracy}/{total}",
            f"Test star accuracy is {star_accuracy}/{total}",
            f"Test star mean absolute error is {star_mae/total:.4f}"
        ]))

    fig = px.line(
        y=[loss_lists["sentiment"], loss_lists["star"]],
        template="simple_white", 
        labels={
            "x": "No. batches seen", 
            "y": "Loss"
        }, 
        title="Loss on ReversedDigits dataset"
    )
    # This next bit of code plots vertical lines corresponding to the epochs
    if epochs > 1:
        for idx, epoch_start in enumerate(np.linspace(0, len(loss_lists["star"]), epochs, endpoint=False)):
            fig.add_vline(x=epoch_start, line_width=3, line_dash="dash", annotation_text=f"Epoch {idx}", annotation_position="top right")
    fig.show()
    
    # filename = f"{wandb.run.dir}/model_state_dict.h5"
    # wandb.save(filename)
    # wandb.finish()

    return model

# %%

epochs = 3
loss_fns = [nn.CrossEntropyLoss(), nn.L1Loss()]
loss_fns_weighting = [1.0, 0.01]
batch_size = 2
lr = 3e-5
weight_decay = 0.01

my_bert_classifier = train(my_bert_classifier, train_data, test_data, epochs, loss_fns, loss_fns_weighting, batch_size, lr, weight_decay)
# %%

# Turns out all sentiment analysis is correct (I'm surprised there weren't more sarcastic reviews!)
# So this function finds the largest mistakes in star prediction

# t.save(my_bert_classifier.state_dict(), "mymodel.pt")
# my_state_dict = t.load("mymodel.pt")
# my_bert_classifier2 = BertClassifier()

# from copy import copy, deepcopy
# my_bert_classifier3 = deepcopy(my_bert_classifier)

# %%

def find_worst_mistakes(model, testset, batch_size):

    model.eval()
    biggest_errors = dict(
        max=dict(text="", prediction=-t.inf, actual=0),
        min=dict(text="", prediction=t.inf, actual=0)
    )

    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    for (input_ids, attention_mask, sentiment_labels, star_labels) in testloader:

        outputs = model(input_ids.to(device), attention_mask.to(device))["star"].squeeze()
        labels = star_labels.to(device)

        max_prediction_error_idx = (outputs - labels).argmax().item()
        text = tokenizer.decode(input_ids[max_prediction_error_idx], skip_special_tokens=True)
        prediction, actual = outputs[max_prediction_error_idx].item(), labels[max_prediction_error_idx].item()
        if prediction - actual > biggest_errors["max"]["prediction"] - biggest_errors["max"]["actual"]:
            biggest_errors["max"] = dict(text=text, prediction=prediction, actual=actual)
        
        min_prediction_error_idx = (outputs - labels).argmin().item()
        text = tokenizer.decode(input_ids[min_prediction_error_idx], skip_special_tokens=True)
        prediction, actual = outputs[min_prediction_error_idx].item(), labels[min_prediction_error_idx].item()
        if prediction - actual < biggest_errors["min"]["prediction"] - biggest_errors["min"]["actual"]:
            biggest_errors["min"] = dict(text=text, prediction=prediction, actual=actual)
    
    for name, direction in zip(["min", "max"], ["underestimate", "overestimate"]):
        print("\n".join([
            "="*30, f"Worst {direction}", "="*30, "{}", "="*30, "predicted = {:.2f}", "actual = {}", ""
        ]).format(*biggest_errors[name].values()))

find_worst_mistakes(my_bert_classifier, test_data, batch_size)

# %%

import numpy as np
from einops import rearrange

arr_list = [np.random.rand(7, 8, 9) for i in range(6)]

arr1 = np.stack(arr_list)
arr1 = rearrange(arr1, "(b1 b2) h w c -> (b2 h) (b1 w) c", b2=3)

arr = np.stack(arr_list)
b1b2, h_, w_, c_ = arr.shape
b2 = 5
b1 = b1b2 // b2
arr = arr.reshape((b1, b2, h_, w_, c_))
arr = np.moveaxis(arr, [0, 1, 2, 3, 4], [2, 0, 1, 3, 4])
arr = arr.reshape((b2*h_, b1*w_, c_))

# %%