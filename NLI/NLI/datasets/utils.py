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
        vs = []
        len_vs = []
        ws = []
        len_ws = []
        target = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for v, w, t in batch:
            vs.append(torch.as_tensor(v, dtype=torch.int64, device=device))
            len_vs.append(len(v))
            ws.append(torch.as_tensor(w, dtype=torch.int64, device=device))
            len_ws.append(len(w))
            target.append(t)

        vs = pad_sequence(vs, batch_first=True, padding_value=pad_id)
        ws = pad_sequence(ws, batch_first=True, padding_value=pad_id)
        vs_mask = create_mask(len_vs, device)
        ws_mask = create_mask(len_ws, device)
        target = torch.as_tensor(target, dtype=torch.int64, device=device)

        return vs, ws, vs_mask, ws_mask, target

    return f
