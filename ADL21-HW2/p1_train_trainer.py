from bdb import effective
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch, random
from tqdm import trange
import numpy as np

# from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union

import pyarrow as pa
import pandas as pd
from datasets import Dataset

from accelerate import Accelerator

# https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
# https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df

TRAIN = "train"
DEV = "valid"
SPLITS = [TRAIN, DEV]

# model_name = ''
# logger = SummaryWriter(log_dir=f'log/intent/{model_name}')
seed = 567
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

model_checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def prepare_dataset(data, context):
    dataset = {'sent1':[], 'option0': [], 'option1': [], 'option2': [], 'option3': [],'label': []}

    for i, o in enumerate(data):
        context_list = [context[p] for p in o['paragraphs']]
        relevant = o['paragraphs'].index(o['relevant'])

        dataset['sent1'].append(o['question'])
        dataset['option0'].append(context_list[0])
        dataset['option1'].append(context_list[1])
        dataset['option2'].append(context_list[2])
        dataset['option3'].append(context_list[3])
        dataset['label'].append(relevant)

        # if len(dataset['sent1']) == 2:
        #     break

    dataset = Dataset(pa.Table.from_pandas(pd.DataFrame(dataset)))
    return dataset.map(preprocess_function, batched=True)


def preprocess_function(dataset):
    ending_names = ["option0", "option1", "option2", "option3"]
    first_sentence = [[s] * 4 for s in dataset['sent1']]
    second_sentence = [[s for s in dataset[op]] for op in ending_names]

    # Flatten everything
    first_sentences = sum(first_sentence, [])
    second_sentences = sum(second_sentence, [])
    
    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=512, padding="max_length")

    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

def main(args):
    
    context_path = 'data/context.json'
    context = json.load(open(context_path))

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, Dataset] = {
        split: prepare_dataset(split_data, context)
        for split, split_data in data.items()
    }

    model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
    
    gradient_accumulation_steps = 2
    batch_size = 1
    effective_batch_size = batch_size * gradient_accumulation_steps

    model_name = model_checkpoint.split("/")[-1]
    device = TrainingArguments(
        output_dir=f"{model_name}-finetuned-swag",
        evaluation_strategy = "epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        weight_decay=0.01,
    )


    trainer = Trainer(
        model,
        args,
        train_dataset=datasets['train'],
        eval_dataset=datasets["valid"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer, max_length=512),
        compute_metrics=compute_metrics,
    )

    result = trainer.train()
    trainer.save_model()
    metrics = result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(result)
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="",
    )

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
