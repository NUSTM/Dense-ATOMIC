#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: Siwei Wu
@file: data_load_for_Completion.py 
@time: 2022/04/16
@contact: wusiwei@njust.edu.cn
"""
import csv

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import BertTokenizer
import torch
import os
import logging

@dataclass(frozen= True)
class InputExample:
    "A single training/test/val for Modle"
    Base_event: str
    Leaf_event: str
    Pairs: list
    Head_event_raw: str
    Leaf_event_raw: str
    Pairs_raw: list

class CompletionDataset(Dataset):
    def __init__(self,
                 test_data:list,
                 data_dir:str,
                 tokenizer,
                 max_seq_length: int,
                 relation_type,
                 model_type= 'Bert'
                 ):
        super(CompletionDataset, self).__init__()
        processor = DataProcessor()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data_dir = data_dir
        self.test_data= test_data
        if relation_type== 'norm':
            self.realtion2num, self.num2relation = load_vocabulary(os.path.join(data_dir, 'Relation/Relation_label.txt'))
        elif relation_type== 'reverse':
            self.realtion2num, self.num2relation = load_vocabulary(
                os.path.join(data_dir, 'Relation/Relation_add_reverse_label.txt'))
        elif relation_type== 'fuse':
            self.realtion2num, self.num2relation = load_vocabulary(
                os.path.join(data_dir, 'Relation/relation_fuse.txt'))
        self.examples = processor.get_test_examples(test_data)
        self.model_type= model_type

        logging.info("Current examples: %s", len(self.examples))

    def __getitem__(self, index):
        example = self.examples[index]
        Base_event = example.Base_event
        Leaf_event = example.Leaf_event
        Pairs = example.Pairs
        Pairs_raw= example.Pairs_raw

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


        if self.model_type== 'Bert':
            mask = torch.tensor(inputs['token_type_ids'], dtype=torch.long) + torch.tensor(inputs['attention_mask'],
                                                                                           dtype=torch.long)
            len_of_seq1 = (mask == 1).sum()
            mask[len_of_seq1 - 1] = 0
            mask[0] = 0

            return (
                torch.tensor(inputs['input_ids'], dtype=torch.long),
                torch.tensor(inputs['token_type_ids'], dtype=torch.long),
                torch.tensor(inputs['attention_mask'], dtype=torch.long),
                mask,
                Pairs,

                Pairs_raw
            )
        elif self.model_type== 'RoBerta':
            return (
                torch.tensor(inputs['input_ids'], dtype=torch.long),
                torch.tensor(inputs['attention_mask'], dtype=torch.long),
                torch.tensor(inputs['base_event_length'], dtype=torch.long),
                torch.tensor(inputs['tail_event_length'], dtype=torch.long),
                Pairs,
                Pairs_raw
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
    def get_train_examples(self, data):
        return self._creat_examples(data)


    def get_test_examples(self, data):
        return self._creat_examples(data)

    def get_val_examples(self, data):
        return self._creat_examples(data)

    def _creat_examples(self, data):
        examples = []

        for line in data:
            examples.append(
                InputExample(
                    Base_event= line[0],
                    Leaf_event= line[1],
                    Pairs = [line[0], line[1]],
                    Head_event_raw = line[3],
                    Leaf_event_raw= line[4],
                    Pairs_raw=[line[3],line[4]]
                )
            )

        return examples
