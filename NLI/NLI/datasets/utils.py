import torch
from torch.nn.utils.rnn import pad_sequence


def create_mask(lengths, device):
    max_len = max(lengths)
    mask = torch.zeros(len(lengths), max_len, device=device)
    for i in range(len(mask)):
        mask[i, :lengths[i]] = 1
    return mask


def collate_sequences(pad_id):
    def f(batch):
        s1 = []
        len_s1 = []
        s2 = []
        len_s2 = []
        target = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for v, w, t in batch:
            s1.append(torch.as_tensor(v, dtype=torch.int64, device=device))
            len_s1.append(len(v))
            s2.append(torch.as_tensor(w, dtype=torch.int64, device=device))
            len_s2.append(len(w))
            target.append(t)

        s1 = pad_sequence(s1, batch_first=True, padding_value=pad_id)
        s2 = pad_sequence(s2, batch_first=True, padding_value=pad_id)
        s1_mask = create_mask(len_s1, device)
        s2_mask = create_mask(len_s2, device)
        target = torch.as_tensor(target, dtype=torch.int64, device=device)

        return s1, s1_mask, s2, s2_mask, target

    return f
