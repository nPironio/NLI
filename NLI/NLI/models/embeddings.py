from typing import Dict, List

import mlflow
import numpy as np
import torch.nn
from tokenizers import Tokenizer
from torch.nn.functional import binary_cross_entropy, normalize
from tqdm import tqdm

from NLI.architectures import Word2Vec


class EmbeddingsModel:
    def __init__(self, device='cuda', tokenizer=None, **model_kws):
        self.device = device
        self.model = Word2Vec(**model_kws).to(device)

        self.unigram_dist: None | torch.distributions.Categorical = None
        self.embeddings: None | torch.nn.Module = None
        self.tokenizer: None | Tokenizer = None

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

    def obtain_embeddings(self):
        norm_in = normalize(self.model.in_embedding_layer.weight)
        norm_out = normalize(self.model.out_embedding_layer.weight)

        return normalize((norm_in + norm_out) / 2)

    def fit(self, train_dl, val_dl, unigram_probs, neg_k=20, epochs=50, lr=0.001, patience=5):
        self.unigram_dist = torch.distributions.Categorical(probs=torch.as_tensor(unigram_probs, device=self.device))
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

        best_val, best_epoch = np.inf, 0
        for epoch in range(epochs):
            print(f'-------------------- {epoch=} --------------------')
            losses = []
            for ix, batch in enumerate(tqdm(train_dl, desc="Training batch")):
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
                for batch in tqdm(val_dl, desc="Validation batch"):
                    w, c = batch
                    loss = self.step(w, c, neg_k)

                    mlflow.log_metric('val_step', loss.item())
                    losses.append(loss.item())

                val_epoch_loss = np.mean(losses)

            mlflow.log_metrics({"train_epoch_loss": train_epoch_loss, "val_epoch_loss": val_epoch_loss}, step=epoch)

            if val_epoch_loss < best_val:
                best_val = val_epoch_loss
                best_epoch = epoch
                self.embeddings = torch.nn.Embedding.from_pretrained(self.obtain_embeddings())
            elif epoch - best_epoch > patience:
                break

    def save(self):
        torch.save(self.embeddings.weight, '../data/w2v_embeddings.pt')

    @classmethod
    def from_file(cls, weights_path, tokenizer_path=None, **kwargs):
        emb_model = cls(**kwargs)
        embedding_weights = torch.load(weights_path)
        emb_model.embeddings = torch.nn.Embedding.from_pretrained(embedding_weights).to(emb_model.device)
        emb_model.tokenizer = Tokenizer.from_file(tokenizer_path) if tokenizer_path is not None else None

        return emb_model

    def encode(self, x, **encoding_kws):
        enc = self.tokenizer.encode(x, **encoding_kws).ids
        return torch.as_tensor(enc, dtype=torch.int64, device=self.device)

    def most_similar(self, words: List[str], top_k=10):
        v = self.encode(words, is_pretokenized=True)[1:-1]
        embs = self.embeddings(v)
        sims = torch.matmul(embs, self.embeddings.weight.T)

        return self.tokenizer.decode_batch(torch.topk(sims, k=top_k + 1, largest=True)[1][:, 1:].cpu().tolist())

    def document_similarity(self, x: str, y: str):
        enc_x = self.encode(x)
        enc_y = self.encode(y)

        x_embedding = self.embeddings(enc_x).sum(dim=0)
        y_embedding = self.embeddings(enc_y).sum(dim=0)

        return torch.dot(x_embedding, y_embedding).item()

    def analogy(self, m, w, k, eps=1e-5):
        encodings = torch.cat((self.encode(m)[1:-1], self.encode(w)[1:-1], self.encode(k)[1:-1]))

        embs = self.embeddings(encodings)
        emb_sims = torch.matmul(embs, self.embeddings.weight.T)

        total_sim = emb_sims[2] - emb_sims[1] + emb_sims[0]
        total_sim[encodings.flatten()] = -torch.inf

        return self.tokenizer.decode(torch.topk(total_sim, k=5, largest=True)[1].cpu().tolist())

