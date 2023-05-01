import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm


class W2Vdataset(Dataset):
    def __init__(self, data: np.ndarray, tokenizer: Tokenizer, token_frequencies: np.ndarray,
                 sentence_sample: int = 10, threshold: float = 1e-5):
        """
        Implements a skip-gram dataset for a Word2Vec algorithm, by sampling a number of (word, context) pairs from each
        sentence and sampling $$k$$ negative examples from other sentences
        :param data: sequence of sentences
        :param tokenizer: used to obtain tokenize sentences
        :param token_frequencies: used to subsample tokens in the dataset based on token frequency
        :param sentence_sample: maximum number of pair samples to select from each sentence.
        :param threshold: value used for token subsampling
        """
        self.token_frequencies = token_frequencies
        self.threshold = threshold
        pairs = []
        for sentence in tqdm(data):
            sample = np.random.choice(tokenizer.encode(sentence).ids, size=(sentence_sample, 2), replace=True)
            ixs = [i for i, (w, _) in enumerate(sample) if self.keep(w)]
            pairs.append(sample[ixs])

        self.pairs = np.concatenate(pairs).astype('i8')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx, 0], self.pairs[idx, 1]

    def keep(self, w):
        return np.random.rand() < (1-np.sqrt(self.threshold/self.token_frequencies[w]))

