import torch
from torch import sigmoid


def batch_dot(v, w):
    return torch.bmm(v.unsqueeze(1), w.unsqueeze(-1))


class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Word2Vec, self).__init__()
        self.in_embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)
        self.out_embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)

    def forward(self, w, c, negs):
        w_i = self.in_embedding_layer(w)
        w_o = self.out_embedding_layer(c)
        w_neg = self.out_embedding_layer(negs)

        pos_sim = sigmoid(torch.bmm(w_o.unsqueeze(1), w_i.unsqueeze(-1)))
        neg_sim = sigmoid(torch.bmm(-w_neg, w_i.unsqueeze(-1)))

        return pos_sim.squeeze(-1), neg_sim.squeeze(1)
