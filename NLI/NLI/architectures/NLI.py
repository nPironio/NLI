import torch
from torch import softmax


class CrossAttention(torch.nn.Module):
    def __init__(self, emb_size: int = 100, hidden_size: int = 200, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = torch.nn.Sequential(
            torch.nn.Linear(emb_size, hidden_size),
            torch.nn.ReLU()
        )

    def forward(self, v, w):
        t_v = self.transform(v)
        t_w = self.transform(w)

        sim = torch.bmm(t_v, torch.transpose(t_w, 1, 2))

        alpha_v = softmax(sim, dim=2)
        alpha_w = torch.transpose(softmax(sim, dim=1), 1, 2)

        w_alignment = torch.bmm(alpha_v, w)
        v_alignment = torch.bmm(alpha_w, v)

        return w_alignment, v_alignment


class CompareLayer(torch.nn.Module):
    def __init__(self, emb_size, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = torch.nn.Linear(2 * emb_size, hidden_size)
        self.activation = torch.nn.ReLU()

    def forward(self, s, alignment):
        x = torch.cat((s, alignment), dim=-1)
        return self.activation(self.layer(x))


class OutLayer(torch.nn.Module):
    def __init__(self, input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = torch.nn.Linear(2 * input_size, 3)

    def forward(self, v, w):
        x = torch.cat((v.sum(dim=1), w.sum(dim=1)), dim=-1)
        y = self.layer(x)
        return softmax(y, dim=-1)


class DecomposableAttentionNetwork(torch.nn.Module):
    def __init__(self, emb_size: int = 100, attn_hidden: int = 200, compare_hidden: int = 200, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attention = CrossAttention(emb_size, attn_hidden)
        self.compare = CompareLayer(emb_size, compare_hidden)
        self.out = OutLayer(compare_hidden)

    def forward(self, v, w, v_mask, w_mask):
        masked_v = v * v_mask.unsqueeze(-1)
        masked_w = w * w_mask.unsqueeze(-1)

        w_alignment, v_alignment = self.attention(masked_v, masked_w)

        z_v = self.compare(masked_v, w_alignment)
        z_w = self.compare(masked_w, v_alignment)

        return self.out(z_v, z_w)
