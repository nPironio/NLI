import mlflow
import numpy as np
from tqdm import tqdm

from NLI.architectures import DecomposableAttentionNetwork
from NLI.models import EmbeddingsModel
from NLI.datasets import NLIDataset, collate_sequences

import torch
import pandas as pd
from tokenizers import Tokenizer

from NLI.models.inference import InferenceModel

if __name__ == '__main__':
    train = pd.read_parquet('../data/train.parquet')[:100_000]
    val = pd.read_parquet('../data/dev.parquet')[:30_000]
    tokenizer = Tokenizer.from_file('../data/tokenizer.json')
    embeddings = torch.nn.Embedding.from_pretrained(torch.load('../data/w2v_embeddings.pt'))

    emb_size = 100
    attn_hidden = 200
    compare_hidden = 200
    batch_size = 512
    lr = 0.001
    epochs = 10
    patience = 5

    model = InferenceModel(
        embeddings=embeddings, tokenizer=tokenizer,
        emb_size=emb_size, attn_hidden=attn_hidden, compare_hidden=compare_hidden
    )

    train_ds = NLIDataset(train, tokenizer)
    val_ds = NLIDataset(val, tokenizer)

    train_dl = torch.utils.data.DataLoader(
        train_ds, collate_fn=collate_sequences(pad_id=tokenizer.token_to_id('[PAD]')),
        batch_size=batch_size, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, collate_fn=collate_sequences(pad_id=tokenizer.token_to_id('[PAD]')),
        batch_size=batch_size, shuffle=True
    )

    mlflow.set_tracking_uri('../mlruns')
    mlflow.set_experiment('fit_NLI')

    mlflow.log_params({
        "emb_size": emb_size,
        "attn_hidden": attn_hidden,
        "compare_hidden": compare_hidden,
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
    })

    model.fit(train_dl=train_dl, val_dl=val_dl, epochs=epochs, lr=lr, patience=patience)