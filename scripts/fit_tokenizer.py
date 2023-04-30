from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import pandas as pd
import numpy as np

if __name__ == "__main__":
    train = pd.read_parquet('../data/train.parquet')

    alpha = 3/4

    all_sentences = pd.concat((train['sentence1'], train['sentence2'])).str.lower().values

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(all_sentences, trainer, len(all_sentences))
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
    )

    ocurrences = np.zeros(tokenizer.get_vocab_size())
    for sentences in np.array_split(all_sentences, len(all_sentences)//128):
        encoded = tokenizer.encode_batch(sentences)
        for sentence in encoded:
            for id in sentence.ids:
                ocurrences[id] += 1

    unigram_alpha_unnormalized = ocurrences/ocurrences.sum() ** alpha
    unigram_alpha_norm = unigram_alpha_unnormalized / unigram_alpha_unnormalized.sum()

    np.save('../data/unigram_alpha.npy', unigram_alpha_norm)
    tokenizer.save('../data/tokenizer.json')
