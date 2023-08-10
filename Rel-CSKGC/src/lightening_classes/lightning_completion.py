#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: Siwei Wu
@file: lightning_completion_Bertbaseline4.py 
@time: 2022/04/16
@contact: wusiwei@njust.edu.cn
"""
import csv

import torch
import copy

from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import pytorch_lightning as pl

def load_vocabulary(path):
    f = open(path, 'r', encoding= 'utf-8')
    data = f.read()
    data.split('\n')
    data = data.split('\n')
    w2n = {}
    n2w = []
    for word in data:
        if word not in w2n:
            w2n[word] = len(w2n)
            n2w.append(word)

    return  w2n, n2w

class RelCSKGCCompletion(pl.LightningModule):
    def __init__(self, dropout_prob, text_dim, out_dim, lr_bert, lr_Linner, epsilon, weight_decay, model, vocabulary_path, save_path, feature_method, model2= None, task_type= 'completion', model_type= 'Bert'):
        super(RelCSKGCCompletion, self).__init__()
        self.val_predict_right_num = 0
        self.val_count = 0
        self.test_predict_right_num = 0
        self.samples_count = 0
        self.lr_bert = lr_bert,
        self.lr_Linner = lr_Linner,
        self.epsilon = epsilon,
        self.weight_decay = weight_decay
        self.model_type = model_type
        self.task_type = task_type
        self.feature_method = feature_method
        if model_type == 'RoBerta':
            self.Bert = model

        self.Linner_CLS = nn.Linear(text_dim, 1)

        self.Linner_maxpool = nn.Linear(text_dim * 2, out_dim)
        self.Sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax()
        self.CrossEntropy = nn.CrossEntropyLoss()
        self.BCE = nn.BCELoss()
        self.predict_outs = []
        self.save_path = save_path
        self.realtion2num, self.num2relation = load_vocabulary(vocabulary_path)

    def forward(self, inputs):
        features = self.Bert(input_ids=inputs[0],
                             attention_mask=inputs[1])['last_hidden_state']
        base_event_length = inputs[2]
        tail_event_length = inputs[3]
        Pairs = inputs[4]

        Pairs_raw= inputs[5]

        feture_CLS = features[:, 0, :]

        base_event_feature = features.new_zeros((features.size(0), 48, 768)).to(dtype=next(self.parameters()).dtype)
        tail_event_feature = features.new_zeros((features.size(0), 48, 768)).to(dtype=next(self.parameters()).dtype)
        
        for i in range(features.size(0)):
            base_event_feature[i][1:base_event_length[i] + 1] = features[i][1:base_event_length[i] + 1]
            tail_event_feature[i][1:tail_event_length[i] + 1] = features[i][51:tail_event_length[i] + 51]
        feature_seq1, index = torch.max(base_event_feature, dim=1)
        feature_seq2, index = torch.max(tail_event_feature, dim=1)
        feature_seq1_seq2 = torch.cat(
            (feature_seq1, feature_seq2),
            dim=1)

        output_Logits = self.Linner_CLS(feture_CLS)
        output_maxpool = self.Linner_maxpool(feature_seq1_seq2)
        return output_Logits, output_maxpool,Pairs,Pairs_raw



    def test_step(self, batch, batch_index):
        output_Logits, output_maxpool,Pairs,Pairs_raw = self(batch)
        output_CLS = self.Sigmoid(output_Logits)
        output_maxpool = self.Softmax(output_maxpool)

        predict_relation = torch.argmax(output_maxpool, dim=1).tolist()
        Heads = Pairs_raw[0]
        Tails = Pairs_raw[1]
        for i in range(len(Heads)):
            if self.num2relation[predict_relation[i]] == 'xIntent':
                if output_maxpool[i, predict_relation[i]] > 0.7:
                          self.predict_outs.append(
                              [Heads[i], Tails[i], self.num2relation[predict_relation[i]]])
            elif self.num2relation[predict_relation[i]] == 'oPersona':
                if output_maxpool[i, predict_relation[i]] > 0.9:
                          self.predict_outs.append(
                              [Heads[i], Tails[i], self.num2relation[predict_relation[i]]])
            elif output_CLS[i] > 0.99:
                if self.num2relation[predict_relation[i]] == 'xNeed':
                  if output_maxpool[i, predict_relation[i]] > 0.9:
                      self.predict_outs.append(
                          [Heads[i], Tails[i], self.num2relation[predict_relation[i]]])
                elif self.num2relation[predict_relation[i]] in ['xPersona']:
                    if self.num2relation[predict_relation[i]] == 'xPersona':
                        if output_maxpool[i, predict_relation[i]] > 0.9:
                            self.predict_outs.append(
                                [Heads[i], Tails[i], self.num2relation[predict_relation[i]]])
                elif self.num2relation[predict_relation[i]] in ['xEvent', 'oEvent']:
                    if self.num2relation[predict_relation[i]] == 'xEvent':
                        if output_maxpool[i, predict_relation[i]] > 0.9:
                            self.predict_outs.append(
                                [Heads[i], Tails[i], self.num2relation[predict_relation[i]]])
                    else:
                        if output_maxpool[i, predict_relation[i]] > 0.9:
                            self.predict_outs.append(
                                [Heads[i], Tails[i], self.num2relation[predict_relation[i]]])

    def test_epoch_end(self, outputs):
        f = open(self.save_path, 'w', encoding='utf-8')
        csv_writor = csv.writer(f)
        csv_writor.writerows(self.predict_outs)
        f.close()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.Bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay[0],
                "lr": self.lr_bert[0]
            },
            {
                "params": [p for n, p in self.Bert.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0,
                "lr": self.lr_bert[0]
            },
            {
                "params": self.Linner_CLS.parameters(),
                "weight_decay": self.weight_decay[0],
                "lr": self.lr_Linner[0]
            },
            {
                "params": self.Linner_maxpool.parameters(),
                "weight_decay": self.weight_decay[0],
                "lr": self.lr_Linner[0]
            }
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=self.epsilon[0],
            weight_decay=self.weight_decay[0]
        )


        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1295,
            num_training_steps=12947
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
