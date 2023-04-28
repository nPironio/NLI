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

    train_sentences = 10000
    val_sentences = 3000

    train_sentences = sample_sentences(train, train_sentences)
    val_sentences = sample_sentences(val, val_sentences)

    train_ds = W2Vdataset(data=train_sentences, tokenizer=tokenizer)
    val_ds = W2Vdataset(data=val_sentences, tokenizer=tokenizer)

    embedding_size = 100
    hidden_size = 256
    lr = 1e-3
    batch_size = 128
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = torch.nn.BCELoss()

    model = Word2Vec(tokenizer.get_vocab_size(), embedding_size=100, hidden_size=100).to(device)
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
    train_epoch_loss = 0
    val_epoch_loss = 0
    try:
        for epoch in range(epochs):
            losses = []
            print(f'-------------------- {epoch=} --------------------')
            for ix, batch in enumerate(train_dl):
                data, target = batch
                preds = model(data.to(device))
                loss = loss_fn(preds, target.to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mlflow.log_metric('train_step', loss.item())
                losses.append(loss.item())
            train_epoch_loss = np.mean(losses)

            with torch.no_grad():
                losses = []
                for batch in val_dl:
                    data, target = batch
                    preds = model(data.to(device))
                    loss = loss_fn(preds, target.to(device))

                    mlflow.log_metric('val_step', loss.item())
                    losses.append(loss.item())

            val_epoch_loss = np.mean(losses)

            mlflow.log_metrics({"train_epoch_loss": train_epoch_loss, "val_epoch_loss": val_epoch_loss})

    except KeyboardInterrupt:
        pass
