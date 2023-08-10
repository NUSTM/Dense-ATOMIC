import logging
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pytorch_lightning as pl
from src.data_load import RelCSKGCDataset

class RelCSKGCDataMoudle(pl.LightningDataModule):
    def __init__(self, max_length, train_batch_size, test_batch_size, val_batch_size, tokenizer, relation_data_path, data_path = None, inference= None, PLM_type= 'Bert'):
        super(RelCSKGCDataMoudle, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_batch_size = train_batch_size#
        self.test_batch_size = test_batch_size
        self.val_batch_size = val_batch_size
        self.PLM_type= PLM_type
        self.inference = inference
        self.data_path = data_path
        self.relation_data_path = relation_data_path

    def setup(self, stage = None):
        if stage == 'train' or stage == None:
            self.train_dataset = RelCSKGCDataset(
                self.tokenizer,
                self.max_length,
                'train',
                self.PLM_type,
                self.data_path,
                self.relation_data_path,
            )

            print(self.train_dataset.__len__())

            self.val_dataset = RelCSKGCDataset(
                self.tokenizer,
                self.max_length,
                'val',
                self.PLM_type,
                self.data_path,
                self.relation_data_path,
            )

            self.test_dataset = RelCSKGCDataset(
                self.tokenizer,
                self.max_length,
                'test',
                self.PLM_type,
                self.data_path,
                self.relation_data_path,            
            )
        elif stage == 'test':
            self.test_dataset = RelCSKGCDataset(
                self.tokenizer,
                self.max_length,
                'test',
                self.PLM_type,
                self.data_path,
                self.relation_data_path,            
            )

    def prepare_data(self):
        logging.info(f"nothing")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size= self.train_batch_size,
            num_workers= 8,
            pin_memory= True,
            shuffle= True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=8,
            pin_memory=True,
        )


