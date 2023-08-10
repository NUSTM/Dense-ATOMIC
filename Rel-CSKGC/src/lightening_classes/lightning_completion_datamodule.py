#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: Siwei Wu
@file: lightening_Completion_datamodule.py 
@time: 2022/04/16
@contact: wusiwei@njust.edu.cn
"""
import logging
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pytorch_lightning as pl
from src.data_load import CompletionDataset

class CompletionDataMoudle(pl.LightningDataModule):
    def __init__(self, max_length, train_batch_size, test_batch_size, val_batch_size, tokenizer, test_data, data_dir, relation_type, model_type= None):
        super(CompletionDataMoudle, self).__init__()
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_batch_size = val_batch_size
        self.model_type= model_type
        self.relation_type= relation_type
        self.test_data= test_data

    def setup(self, stage = None):
        self.test_dataset = CompletionDataset(
            test_data= self.test_data,
            data_dir= self.data_dir,
            tokenizer= self.tokenizer,
            max_seq_length= self.max_length,
            relation_type= self.relation_type,
            model_type= self.model_type
        )

    def prepare_data(self):
        logging.info(f"Please place data in {self.data_dir}")

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False
        )