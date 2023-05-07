import numpy as np
import pandas as pd
import torch.utils.data
from tokenizers import Tokenizer
from tqdm import tqdm


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.s1 = self.encode(df['sentence1'].values)
        self.s2 = self.encode(df['sentence2'].values)

        self.target = df['target'].values

    def encode(self, sentences: np.ndarray, batch_size: int = 512):
        encoded = []
        for ss in tqdm(np.array_split(sentences, len(sentences)//batch_size), desc='Tokenizing data'):
            ss_enc = self.tokenizer.encode_batch(ss)
            encoded += [enc.ids for enc in ss_enc]

        return encoded

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, ix):
        return self.s1[ix], self.s2[ix], self.target[ix]
