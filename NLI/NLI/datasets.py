import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm


class W2Vdataset(Dataset):
    def __init__(self, data: np.ndarray, tokenizer: Tokenizer, neg_k: int = 5, sentence_sample: int = 10):
        """
        Implements a skip-gram dataset for a Word2Vec algorithm, by sampling a number of (word, context) pairs from each
        sentence and sampling $$k$$ negative examples from other sentences
        :param data: sequence of sentences
        :param tokenizer: used to obtain tokenize sentences
        :param neg_k: number of negative examples to sample per each (word, context) pair
        :param sentence_sample: maximum number of pair samples to select from each sentence.
        """

        pairs = []
        targets = []
        for sentence in tqdm(data):
            sentence_pairs = np.random.choice(tokenizer.encode(sentence).ids, size=(sentence_sample, 2), replace=True)

            negatives = []
            for w, _ in sentence_pairs:
                other_sentences = np.random.choice(data, neg_k)
                for other in other_sentences:
                    negatives.append([w, np.random.choice(tokenizer.encode(other).ids, 1)[0]])

            pairs.append(np.stack((*sentence_pairs, *negatives)))
            targets.append(np.concatenate((np.ones(sentence_sample), np.zeros(neg_k * len(sentence_pairs)))))

        self.pairs = np.concatenate(pairs).astype('i8')
        self.targets = np.concatenate(targets).reshape(-1, 1).astype('f4')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.targets[idx]

