from typing import Dict

import mlflow
import numpy as np
import torch.nn
from torch.nn.functional import binary_cross_entropy

from NLI.architectures import Word2Vec


class EmbeddingsModel:
    def __init__(self, device='cuda', **model_kws):
        self.device = device
        self.model = Word2Vec(**model_kws).to(device)

        self.unigram_dist: None | torch.distributions.Categorical = None
        self.embeddings: None | torch.nn.Module = None

    def forward_pass(self, w, c, neg_k):
        negs = self.unigram_dist.sample(sample_shape=torch.Size((len(w), neg_k)))
        pos_sim, neg_sim = self.model(w.to(self.device), c.to(self.device), negs)
        return pos_sim, neg_sim

    def loss(self, pos_sim, neg_sim):
        pos_loss = binary_cross_entropy(pos_sim, torch.ones_like(pos_sim), reduction='none')
        neg_loss = binary_cross_entropy(neg_sim, torch.ones_like(neg_sim), reduction='none').sum(dim=1)
        loss = (pos_loss + neg_loss).mean()
        return loss

    def step(self, w, c, neg_k):
        return self.loss(*self.forward_pass(w, c, neg_k))

    def fit(self, train_dl, val_dl, unigram_probs, neg_k=20, epochs=50, lr=0.001):
        self.unigram_dist = torch.distributions.Categorical(probs=torch.as_tensor(unigram_probs, device=self.device))
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f'-------------------- {epoch=} --------------------')
            losses = []
            for ix, batch in enumerate(train_dl):
                w, c = batch
                loss = self.step(w, c, neg_k)

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
                    loss = self.step(w, c, neg_k)

                    mlflow.log_metric('val_step', loss.item())
                    losses.append(loss.item())

                val_epoch_loss = np.mean(losses)

            mlflow.log_metrics({"train_epoch_loss": train_epoch_loss, "val_epoch_loss": val_epoch_loss}, step=epoch)

        combined_embeddings = self.model.in_embedding_layer.weight + self.model.out_embedding_layer.weight
        self.embeddings = torch.nn.Embedding.from_pretrained(combined_embeddings)

    def save(self):
        torch.save(self.embeddings.weight, '../data/w2v_embeddings.pt')

    @classmethod
    def from_file(cls, path, **kwargs):
        emb_model = cls(**kwargs)
        embedding_weights = torch.load(path)
        emb_model.embeddings = torch.nn.Embedding.from_pretrained(embedding_weights).to(emb_model.device)

        return emb_model
