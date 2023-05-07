import mlflow
import numpy as np
import torch.nn
from tokenizers import Tokenizer
from tqdm import tqdm

from NLI.architectures import DecomposableAttentionNetwork


class InferenceModel:
    def __init__(self, embeddings: None | torch.nn.Module = None, device='cuda', tokenizer=None, **model_kws):
        self.device = device
        self.model = DecomposableAttentionNetwork(**model_kws).to(device)

        self.embeddings: None | torch.nn.Module = embeddings
        self.tokenizer: None | Tokenizer = tokenizer

        self.loss = torch.nn.CrossEntropyLoss()

    def forward_pass(self, v, w, v_mask, w_mask):
        v_emb = self.embeddings(v)
        w_emb = self.embeddings(w)
        pred = self.model(v_emb, w_emb, v_mask, w_mask)

        return pred

    def checkpoint(self):
        torch.save(self.embeddings.weight, '../data/inference_embeddings.pt')
        torch.save(self.model.state_dict(), '../data/inference_model.pt')

    def fit(self, train_dl, val_dl, epochs=10, lr=0.001, patience=5):
        for param in self.embeddings.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

        best_val, best_epoch = np.inf, 0
        for epoch in range(epochs):
            losses = []
            for ix, batch in enumerate(tqdm(train_dl, desc=f"Training epoch {epoch}")):
                v, w, v_mask, w_mask, target = batch
                pred = self.forward_pass(v, w, v_mask, w_mask)
                loss = self.loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mlflow.log_metric('train_step', loss.item())
                losses.append(loss.item())

            train_epoch_loss = np.mean(losses)

            with torch.no_grad():
                losses = []
                for batch in tqdm(val_dl, desc=f"Validation epoch {epoch}"):
                    v, w, v_mask, w_mask, target = batch
                    pred = self.forward_pass(v, w, v_mask, w_mask)
                    loss = self.loss(pred, target)

                    mlflow.log_metric('val_step', loss.item())
                    losses.append(loss.item())

                val_epoch_loss = np.mean(losses)

            mlflow.log_metrics({
                "train_epoch_loss": float(train_epoch_loss),
                "val_epoch_loss": float(val_epoch_loss)},
                step=epoch
            )

            if val_epoch_loss < best_val:
                best_val = val_epoch_loss
                best_epoch = epoch
                self.checkpoint()
            elif epoch - best_epoch > patience:
                break

        self.model.load_state_dict(torch.load('../data/inference_model.pt'))

    @classmethod
    def from_files(cls, embeddings_path, model_path, tokenizer_path, **kwargs):
        instance = cls(**kwargs)
        instance.embeddings = torch.nn.Embedding.from_pretrained(torch.load(embeddings_path))
        instance.model.load_state_dict(torch.load(model_path))
        instance.tokenizer = Tokenizer.from_file(tokenizer_path)

        return instance
