import mlflow
import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from NLI.architectures import Word2Vec
from NLI.datasets import W2Vdataset


def sample_sentences(df: pd.DataFrame, sample_size: int = 1000):
    return np.random.choice(pd.concat((df['sentence1'], df['sentence2'])).str.lower().values, sample_size)


if __name__ == '__main__':
    tokenizer = Tokenizer.from_file('../data/tokenizer.json')

    train = pd.read_parquet('../data/train.parquet')
    val = pd.read_parquet('../data/dev.parquet')
    token_frequencies = np.load('../data/token_frequencies.npy')
    unigram_probs = np.load('../data/unigram_alpha.npy')

    num_train_sentences = 100_000
    num_val_sentences = 30_000

    train_sentences = sample_sentences(train, num_train_sentences)
    val_sentences = sample_sentences(val, num_val_sentences)

    train_ds = W2Vdataset(data=train_sentences, tokenizer=tokenizer, token_frequencies=token_frequencies)
    val_ds = W2Vdataset(data=val_sentences, tokenizer=tokenizer, token_frequencies=token_frequencies)

    neg_k = 20
    embedding_size = 100
    hidden_size = 256
    lr = 1e-3
    batch_size = 128
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = torch.nn.functional.binary_cross_entropy

    model = Word2Vec(tokenizer.get_vocab_size(), embedding_size=100).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    val_dl = DataLoader(val_ds, shuffle=True, batch_size=batch_size)

    mlflow.set_tracking_uri('../mlruns')
    mlflow.set_experiment('fit_w2v')
    mlflow.log_params(
        {
            "emb_size": embedding_size, "hidden_size": hidden_size, "lr": lr,
            "batch_size": batch_size, "epochs": epochs,
            "train_size": len(train_ds), "val_size": len(val_ds)
         }
    )
    unigram_dist = torch.distributions.Categorical(probs=torch.as_tensor(unigram_probs, device=device))
    train_epoch_loss = 0
    val_epoch_loss = 0
    for epoch in range(epochs):
        losses = []
        print(f'-------------------- {epoch=} --------------------')
        for ix, batch in enumerate(train_dl):
            w, c = batch
            negs = unigram_dist.sample(sample_shape=torch.Size((len(w), neg_k)))
            pos_sim, neg_sim = model(w.to(device), c.to(device), negs)
            pos_loss = loss_fn(pos_sim, torch.ones_like(pos_sim), reduction='none')
            neg_loss = loss_fn(neg_sim, torch.ones_like(neg_sim), reduction='none').sum(dim=1)
            loss = (pos_loss + neg_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metric('train_step', loss.item())
            losses.append(loss.item())
        train_epoch_loss = np.mean(losses)

        with torch.no_grad():
            losses = []
            for batch in val_dl:
                w, c = batch
                negs = unigram_dist.sample(sample_shape=torch.Size((len(w), neg_k)))
                pos_sim, neg_sim = model(w.to(device), c.to(device), negs)
                pos_loss = loss_fn(pos_sim, torch.ones_like(pos_sim), reduction='none')
                neg_loss = loss_fn(neg_sim, torch.ones_like(neg_sim), reduction='none').sum(dim=1)
                loss = (pos_loss + neg_loss).mean()

                mlflow.log_metric('val_step', loss.item())
                losses.append(loss.item())

            val_epoch_loss = np.mean(losses)

        mlflow.log_metrics({"train_epoch_loss": train_epoch_loss, "val_epoch_loss": val_epoch_loss}, step=epoch)
