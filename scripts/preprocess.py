import json
import pickle
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_json_list(path: Path | str):
    return [json.loads(line) for line in open(path, 'r', encoding='utf-8')]


def extract_relevant_fields(records: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records([(r['sentence1'], r['sentence2'], r['gold_label']) for r in records],
                                     columns=['sentence1', 'sentence2', 'target'])


if __name__ == "__main__":
    DATA_PATH = '../data'
    data = {}
    for split in ['train', 'dev', 'test']:
        records = load_json_list(Path(f'{DATA_PATH}/snli_1.0/snli_1.0_train.jsonl'))
        df = extract_relevant_fields(records)
        df = df[df.target != '-']
        data[split] = df

    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(data['train'].target)

    with open(f'{DATA_PATH}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    for split in ['train', 'dev', 'test']:
        df = data[split]
        df['target'] = label_encoder.transform(df['target'])
        df.to_parquet(f'{DATA_PATH}/{split}.parquet')
