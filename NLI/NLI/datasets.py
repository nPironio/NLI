import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm


class W2Vdataset(Dataset):
    def __init__(self, data: np.ndarray, tokenizer: Tokenizer, sentence_sample: int = 10):
        """
        Implements a skip-gram dataset for a Word2Vec algorithm, by sampling a number of (word, context) pairs from each
        sentence and sampling $$k$$ negative examples from other sentences
        :param data: sequence of sentences
        :param tokenizer: used to obtain tokenize sentences
        :param sentence_sample: maximum number of pair samples to select from each sentence.
        """
        pairs = []
        for sentence in tqdm(data):
            pairs.append(np.random.choice(tokenizer.encode(sentence).ids, size=(sentence_sample, 2), replace=True))

        self.pairs = np.concatenate(pairs).astype('i8')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx, 0], self.pairs[idx, 1]

