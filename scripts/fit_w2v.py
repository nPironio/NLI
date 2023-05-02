import mlflow
import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from NLI.architectures import Word2Vec
from NLI.datasets import W2Vdataset
from NLI.models.embeddings import EmbeddingsModel


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
    lr = 1e-3
    batch_size = 128
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    val_dl = DataLoader(val_ds, shuffle=True, batch_size=batch_size)

    mlflow.set_tracking_uri('../mlruns')
    mlflow.set_experiment('fit_w2v')
    mlflow.log_params(
        {
            "emb_size": embedding_size, "lr": lr,
            "batch_size": batch_size, "epochs": epochs,
            "train_size": len(train_ds), "val_size": len(val_ds)
         }
    )
    model = EmbeddingsModel(vocab_size=tokenizer.get_vocab_size(), embedding_size=embedding_size, device=device)
    model.fit(train_dl, val_dl, unigram_probs, neg_k, epochs, lr)
    model.save()
