import csv
import random

from torch.utils.data.dataset import Dataset
from dataclasses import dataclass
from transformers import BertTokenizer
import torch
import os
import logging
from transformers import  RobertaTokenizer, RobertaModel

@dataclass(frozen= True)
class InputExampleTrain:
    "A single training/test/val for Modle"
    Base_event: str
    Leaf_event: str
    Relation: str

@dataclass(frozen= True)
class InputExampleTest:
    "A single training/test/val for Modle"
    Base_event: str
    Leaf_event: str
    Pairs: list
    Descriptions: str
    Class_type: str


class RelCSKGCDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 max_seq_length: int,
                 mode: str,
                 PLM_type: str,
                 data_path = None,
                 relation_data_path = None,
                 ):
        super(RelCSKGCDataset, self).__init__()
        processor = DataProcessor()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.PLM_type= PLM_type
        self.realtion2num, self.num2relation = load_vocabulary(relation_data_path)
        self.mode = mode
        if mode == 'train':
            self.examples = processor.get_train_examples(f'{data_path}train_for_relkgc.csv', mode)
        elif mode == 'test':
            self.examples = processor.get_test_examples('./Data/little_examples_test/Human_label_Atomic_test_result_checked.csv', mode)
        elif mode == 'val':
            self.examples = processor.get_test_examples(f'{data_path}test_for_relkgc.csv', mode)

        logging.info("Current examples: %s", len(self.examples))

    def __getitem__(self, index):
        example = self.examples[index]
        Base_event = example.Base_event
        Leaf_event = example.Leaf_event

        if self.PLM_type== 'RoBerta':
            base_event_input = self.tokenizer(
                Base_event,
                add_special_tokens=True,
                max_length=50,
                padding='max_length',
                truncation=True
            )
            base_event_length = sum(base_event_input['attention_mask']) - 2

            tail_event_input = self.tokenizer(
                Leaf_event,
                add_special_tokens=True,
                max_length=50,
                padding='max_length',
                truncation=True
            )

            tail_event_input['input_ids'][0] = 2
            tail_event_length = sum(tail_event_input['attention_mask']) - 2

            inputs = {}
            inputs['input_ids'] = base_event_input['input_ids'] + tail_event_input['input_ids']
            inputs['attention_mask'] = base_event_input['attention_mask'] + tail_event_input['attention_mask']
            inputs['base_event_length'] = base_event_length
            inputs['tail_event_length'] = tail_event_length

            if self.mode == 'train':
                Relation = example.Relation

                if self.realtion2num[Relation] != self.realtion2num['NoLink']:
                    Binary_label = 1
                else:
                    Binary_label = 0

                return (
                    torch.tensor(inputs['input_ids'], dtype=torch.long),
                    torch.tensor(inputs['attention_mask'], dtype=torch.long),
                    torch.tensor(inputs['base_event_length'], dtype=torch.long),
                    torch.tensor(inputs['tail_event_length'], dtype=torch.long),
                    torch.tensor(self.realtion2num[Relation], dtype=torch.long),
                    torch.tensor([Binary_label], dtype=torch.float),
                )
            elif self.mode in ['test', 'val']:
                Pairs = example.Pairs
                Descriptions = example.Descriptions
                Class_type = example.Class_type

                return (
                    torch.tensor(inputs['input_ids'], dtype=torch.long),
                    torch.tensor(inputs['attention_mask'], dtype=torch.long),
                    torch.tensor(inputs['base_event_length'], dtype=torch.long),
                    torch.tensor(inputs['tail_event_length'], dtype=torch.long),
                    Pairs,
                    Descriptions,
                    Class_type
                )

    def __len__(self):
        return len(self.examples)

def load_vocabulary(path):
    f = open(path, 'r', encoding= 'utf-8')
    data = f.read()
    data.split('\n')
    data = data.split('\n')
    w2n = {}
    n2w = {}
    for word in data:
        if word not in w2n:
            w2n[word] = len(w2n)
            n2w[w2n[word]] = word

    return  w2n, n2w

class DataProcessor:
    def get_train_examples(self, path, mode):
        logging.info("Getting training examples at {}.".format(path))
        return self._creat_examples(path, mode)


    def get_test_examples(self, path, mode):
        logging.info("Getting testing examples at {}.".format(path))
        return self._creat_examples(path, mode)

    def get_val_examples(self, path, mode):
        logging.info("Getting validing examples at {}.".format(path))
        return self._creat_examples(path, mode)

    def _creat_examples(self, path, mode):
        examples = []
        
        f = open(path, 'r', encoding= 'utf-8')

        if mode == 'train':
            data = csv.reader(f)
            next(data)
            for triple in data:
                examples.append(
                    InputExampleTrain(
                        Base_event= triple[1],
                        Leaf_event= triple[3],
                        Relation= triple[2]
                    )
                )

            return examples

        elif mode in ['test', 'val']:
            data = csv.reader(f)

            for triple in data:
                if len(triple[4].split(']')) == 2:
                    examples.append(
                        InputExampleTest(
                            Base_event=triple[0],
                            Leaf_event=triple[1],
                            Pairs=[triple[0], triple[1]],
                            Descriptions=triple[2],
                            Class_type='intra'
                        )
                    )
                else:
                    examples.append(
                        InputExampleTest(
                            Base_event=triple[0],
                            Leaf_event=triple[1],
                            Pairs=[triple[0], triple[1]],
                            Descriptions=triple[2],
                            Class_type='inter'
                        )
                    )

            return examples