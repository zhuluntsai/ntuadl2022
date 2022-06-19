from lib2to3.pgen2.tokenize import tokenize
from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab
from torch.nn.utils.rnn import pad_sequence
import spacy
from collections import Counter
import torch
import numpy as np

spacy_en = spacy.load('en_core_web_sm')

# https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch/notebook
class Vocabulary:
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_en.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ] 

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.tokenizer = Vocabulary
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        token = [self.tokenizer.tokenize(instance['text']) for instance in samples]
        vector = self.vocab.encode_batch(token)
        vector = [torch.tensor(instance) for instance in vector]
        targets = pad_sequence(vector, batch_first=True, padding_value=0)
        
        label = [self.label2idx(instance['intent']) for instance in samples]
        label = torch.tensor(label)
        label = torch.nn.functional.one_hot(label, num_classes=self.num_classes)

        length = [len(instance) for instance in targets]
        return targets, label, length

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SeqClsDataset_test(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.tokenizer = Vocabulary
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        token = [self.tokenizer.tokenize(instance['text']) for instance in samples]
        vector = self.vocab.encode_batch(token)
        vector = [torch.tensor(instance) for instance in vector]
        targets = pad_sequence(vector, batch_first=True, padding_value=0)
        
        length = [len(instance) for instance in targets]
        return targets, length

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class TaggingDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int = None,
    ):
        self.data = data
        self.tokenizer = Vocabulary
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

        if self.max_len == None:
            length = []
            for instance in self.data:
                length.append(len(instance['tokens']))
            self.max_len = np.max(length) + 1
        else:
            self.max_len = max_len


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        token = [instance['tokens'] for instance in samples]
        vector = self.vocab.encode_batch(token)
        vector = [torch.tensor(instance) for instance in vector]
        vector.append(torch.zeros(self.max_len))
        targets = pad_sequence(vector, batch_first=True, padding_value=9)[:-1]

        label = [[self.label2idx(tag) for tag in instance['tags']] for instance in samples]
        label = [torch.tensor(instance) for instance in label]
        label.append(torch.zeros(self.max_len))
        label = pad_sequence(label, batch_first=True, padding_value=9)[:-1]
        # label = torch.nn.functional.one_hot(label, num_classes=self.num_classes+1)
        
        length = [len(instance) for instance in token]
        return targets, label, length

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class TaggingDataset_test(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int = None,
    ):
        self.data = data
        self.tokenizer = Vocabulary
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

        if self.max_len == None:
            length = []
            for instance in self.data:
                length.append(len(instance['tokens']))
            self.max_len = np.max(length) + 1
        else:
            self.max_len = max_len


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        token = [instance['tokens'] for instance in samples]
        vector = self.vocab.encode_batch(token)
        vector = [torch.tensor(instance) for instance in vector]
        vector.append(torch.zeros(self.max_len))
        targets = pad_sequence(vector, batch_first=True, padding_value=9)[:-1]

        length = [len(instance) for instance in token]
        return targets, length

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]