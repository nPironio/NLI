from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import pandas as pd

if __name__ == "__main__":
    train = pd.read_parquet('../data/train.parquet')

    all_sentences = pd.concat((train['sentence1'], train['sentence2'])).str.lower()

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(all_sentences.values, trainer, len(all_sentences))
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
    )

    tokenizer.save('./data/tokenizer.json')

