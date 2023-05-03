import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm


class W2Vdataset(Dataset):
    def __init__(self, data: np.ndarray, tokenizer: Tokenizer, token_frequencies: np.ndarray,
                 threshold: float = 1e-5, batch_size: int = 512, max_k: int = 5):
        """
        Implements a skip-gram dataset for a Word2Vec algorithm, by sampling a number of (word, context) pairs from each
        sentence and sampling $$k$$ negative examples from other sentences
        :type max_k: Maximum context size to each side of the token
        :type batch_size: used for batch encoding of sentences
        :param data: sequence of sentences
        :param tokenizer: used to obtain tokenize sentences
        :param token_frequencies: used to subsample tokens in the dataset based on token frequency
        :param threshold: value used for token subsampling
        """
        self.token_frequencies = token_frequencies
        self.threshold = threshold
        pairs = []
        for sentences in tqdm(np.array_split(data, len(data)//batch_size), desc="Processing sentences for dataset"):
            encoded = tokenizer.encode_batch(sentences)
            for encoding in encoded:
                sentence = [token_id for token_id in encoding.ids if self.keep(token_id)]
                k_sample = np.random.randint(1, max_k)
                for ix, w in enumerate(sentence):
                    for c_ix in range(max(0, ix-k_sample), ix):
                        pairs.append((w, sentence[c_ix]))
                    for c_ix in range(ix+1, min(len(sentence), ix+k_sample+1)):
                        pairs.append((w, sentence[c_ix]))

        self.pairs = np.array(pairs).astype('i8')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx, 0], self.pairs[idx, 1]

    def keep(self, w):
        return np.random.rand() < (1-np.sqrt(self.threshold/self.token_frequencies[w]))

