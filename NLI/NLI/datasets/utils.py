import torch
from torch.nn.utils.rnn import pad_sequence


def create_mask(lengths):
    max_len = max(lengths)
    mask = torch.zeros(len(lengths), max_len)
    for i in range(len(mask)):
        mask[i, :lengths[i]] = 1
    return mask


def collate_sequences(pad_token):
    def f(batch):
        s1 = []
        len_s1 = []
        s2 = []
        len_s2 = []
        target = []
        for v, w, t in batch:
            s1.append(torch.as_tensor(v, dtype=torch.int64))
            len_s1.append(len(v))
            s2.append(torch.as_tensor(w, dtype=torch.int64))
            len_s2.append(len(w))
            target.append(t)

        s1 = pad_sequence(s1, batch_first=True, padding_value=pad_token)
        s2 = pad_sequence(s2, batch_first=True, padding_value=pad_token)
        s1_mask = create_mask(len_s1)
        s2_mask = create_mask(len_s2)
        target = torch.as_tensor(target, dtype=torch.float32)

        return s1, s1_mask, s2, s2_mask, target

    return f
