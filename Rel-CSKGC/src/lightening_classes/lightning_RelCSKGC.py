# #!/usr/bin/env python
# # -*- coding:utf-8 -*-
# """
# @author: Siwei Wu
# @file: lightning_Bertbaseline4.py
# @time: 2022/04/07
# @contact: wusiwei@njust.edu.cn
# """
import torch
import copy
import csv

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

def evaluate_predicated(result_path):
    f = open(result_path, 'r', encoding='utf-8')
    data_csv = csv.reader(f)

    predict_num = 0
    predict_intra_num = 0
    predict_inter_num = 0

    glod_num = 0
    glod_intra_num = 0
    glod_inter_num = 0

    all_num = 0
    all_intra_num = 0
    all_inter_num = 0

    for line in data_csv:
        ground_truth = line[2]
        predict_result = line[3]
        class_type = line[4]

        if ground_truth == predict_result and ground_truth != 'NoLink':
            glod_num += 1
            if class_type == 'intra':
                glod_intra_num += 1
            else:
                glod_inter_num += 1

        if predict_result != 'NoLink':
            predict_num += 1
            if class_type == 'intra':
                predict_intra_num += 1
            else:
                predict_inter_num += 1

        if ground_truth != 'NoLink':
            all_num += 1
            if class_type == 'intra':
                all_intra_num += 1
            else:
                all_inter_num += 1

    f.close()

    p = glod_num / predict_num
    r = glod_num / all_num

    p_intra = glod_intra_num / predict_intra_num
    r_intra = glod_intra_num / all_intra_num

    p_inter = glod_inter_num / predict_inter_num
    r_inter = glod_inter_num / all_inter_num

    return p, r , p_intra, r_intra, p_inter, r_inter

class RelCSKGC(pl.LightningModule):
    def __init__(self, text_dim, out_dim, lr_bert, lr_Linner, epsilon, weight_decay, feature_method, model, relation_path = None, save_path = None, inference = None, PLM_type= 'RoBert'):
        super(RelCSKGC, self).__init__()
        self.val_predict_right_num = 0
        self.val_count = 0
        self.test_predict_right_num = 0
        self.samples_count = 0
        self.lr_bert = lr_bert,
        self.lr_Linner = lr_Linner,
        self.epsilon= epsilon,
        self.weight_decay = weight_decay,
        self.PLM_type= PLM_type
        self.feature_method= feature_method
        self.inference = inference
        self.relation_path = relation_path
        if PLM_type== 'RoBerta':
            self.Bert = model

        self.Linner_CLS = nn.Linear(text_dim, 1)

        self.Linner_maxpool = nn.Linear(text_dim * 2, out_dim)
        self.Sigmoid= nn.Sigmoid()
        self.Softmax = nn.Softmax()
        self.CrossEntropy = nn.CrossEntropyLoss()
        self.BCE= nn.BCEWithLogitsLoss()
        self.predict_outs = []
        if save_path == None:
            self.save_path = f'./Data/result/{self.PLM_type}_{self.feature_method}_test_result.csv'
        else:
            self.save_path = save_path
        
        self.realtion2num, self.num2relation = load_vocabulary(self.relation_path)

    def forward(self, inputs, mode):

        if self.PLM_type== 'RoBerta':
          features = self.Bert(input_ids=inputs[0],
                               attention_mask=inputs[1])['last_hidden_state']
          base_event_length= inputs[2]
          tail_event_length = inputs[3]

          feture_CLS= features[:,0,:]

          base_event_feature = features.new_zeros((features.size(0), 48, 768)).to(dtype=next(self.parameters()).dtype)
          tail_event_feature = features.new_zeros((features.size(0), 48, 768)).to(dtype=next(self.parameters()).dtype)
          
          for i in range(features.size(0)):
              base_event_feature[i][1:base_event_length[i] + 1] = features[i][1:base_event_length[i] + 1]
              tail_event_feature[i][1:tail_event_length[i] + 1] = features[i][51:tail_event_length[i] + 51]
          feature_seq1, index = torch.max(base_event_feature, dim=1)
          feature_seq2, index = torch.max(tail_event_feature, dim=1)
          features_seq1_seq2 = torch.cat(
              (feature_seq1, feature_seq2),
              dim=1)
          
          
          output_Logits = self.Linner_CLS(feture_CLS)
          output_maxpool= self.Linner_maxpool(features_seq1_seq2)

          if mode == 'train':
              relation_id = inputs[4]
              Binary_label = inputs[5]

              return output_Logits, output_maxpool, relation_id, Binary_label

          elif mode == 'test':
              Pairs = inputs[4]
              Descriptions = inputs[5]
              Class_type = inputs[6]

              return output_Logits, output_maxpool, Pairs, Descriptions, Class_type

    def training_step(self,batch, batch_index):
        output_Logits, output_maxpool, relation_id, Binary_label = self(batch, 'train')
        loss_maxpool = self.CrossEntropy(output_maxpool, relation_id)
        loss_CLS= self.BCE(output_Logits, Binary_label)
        loss= loss_CLS+ loss_maxpool
        self.log("Training/loss", loss, on_step= True, on_epoch= True, prog_bar= True)
        return loss

    def validation_step(self, batch, batch_index):
        pass

    def validation_epoch_end(self, outputs):
        self.log('Validation/f1', 0.1, prog_bar= True)

    def test_step(self, batch, batch_index):
        if self.inference == None:
            output_Logits, output_maxpool, Pairs, Descriptions, Class_type = self(batch, 'test')
            Heads = Pairs[0]
            Tails = Pairs[1]
        else:
            output_Logits, output_maxpool, Pairs, Pairs_raw = self(batch, 'inference')
            Heads = Pairs_raw[0]
            Tails = Pairs_raw[1]
        output_CLS = self.Sigmoid(output_Logits)
        output_maxpool = self.Softmax(output_maxpool)

        predict_relation = torch.argmax(output_maxpool, dim=1).tolist()

        for i in range(len(Heads)):
            if self.num2relation[predict_relation[i]] == 'xIntent':
                if output_maxpool[i, predict_relation[i]] > 0.7:
                    self.predict_outs.append(
                        [Heads[i], Tails[i], Descriptions[i], self.num2relation[predict_relation[i]],
                         Class_type[i]])
                else:
                    self.predict_outs.append(
                        [Heads[i], Tails[i], Descriptions[i], 'NoLink', Class_type[i]])
            elif self.num2relation[predict_relation[i]] == 'oPersona':
                if output_maxpool[i, predict_relation[i]] > 0.9:
                    self.predict_outs.append(
                        [Heads[i], Tails[i], Descriptions[i], self.num2relation[predict_relation[i]],
                         Class_type[i]])
                else:
                    self.predict_outs.append(
                        [Heads[i], Tails[i], Descriptions[i], 'NoLink', Class_type[i]])
            elif output_CLS[i] > 0.99:
                if self.num2relation[predict_relation[i]] == 'xNeed':
                    if output_maxpool[i, predict_relation[i]] > 0.9:
                        self.predict_outs.append(
                            [Heads[i], Tails[i], Descriptions[i], self.num2relation[predict_relation[i]],
                             Class_type[i]])
                    else:
                        self.predict_outs.append(
                            [Heads[i], Tails[i], Descriptions[i], 'NoLink', Class_type[i]])
                elif self.num2relation[predict_relation[i]] in ['xPersona']:
                    if self.num2relation[predict_relation[i]] == 'xPersona':
                        if output_maxpool[i, predict_relation[i]] > 0.9:
                            self.predict_outs.append(
                                [Heads[i], Tails[i], Descriptions[i], self.num2relation[predict_relation[i]],
                                 Class_type[i]])
                        else:
                            self.predict_outs.append(
                                [Heads[i], Tails[i], Descriptions[i], 'NoLink', Class_type[i]])
                elif self.num2relation[predict_relation[i]] in ['xEvent', 'oEvent']:
                    if self.num2relation[predict_relation[i]] == 'xEvent':
                        if output_maxpool[i, predict_relation[i]] > 0.9:
                            self.predict_outs.append(
                                [Heads[i], Tails[i], Descriptions[i], self.num2relation[predict_relation[i]],
                                 Class_type[i]])
                        else:
                            self.predict_outs.append(
                                [Heads[i], Tails[i], Descriptions[i], 'NoLink', Class_type[i]])
                    else:
                        if output_maxpool[i, predict_relation[i]] > 0.9:
                            self.predict_outs.append(
                                [Heads[i], Tails[i], Descriptions[i], self.num2relation[predict_relation[i]],
                                 Class_type[i]])
                        else:
                            self.predict_outs.append(
                                [Heads[i], Tails[i], Descriptions[i], 'NoLink', Class_type[i]])
                else:
                    self.predict_outs.append(
                        [Heads[i], Tails[i], Descriptions[i], 'NoLink', Class_type[i]])
            else:
                self.predict_outs.append(
                    [Heads[i], Tails[i], Descriptions[i], 'NoLink', Class_type[i]])

    def test_epoch_end(self, outputs):
        # save predict result
        f = open(self.save_path, 'w', encoding='utf-8')
        csv_writor = csv.writer(f)
        csv_writor.writerows(self.predict_outs)
        f.close()

        p, r , p_intra, r_intra, p_inter, r_inter = evaluate_predicated(self.save_path)
        self.log('P:',p)
        self.log('P_intra:', p_intra)
        self.log('P_inter:', p_inter)
        self.predict_outs = []

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        if 'attention' in self.feature_method:
            optimizer_grouped_parameters= [
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
                },
                {
                    "params": self.BaseToTailAttention.parameters(),
                    "weight_decay": self.weight_decay[0],
                    "lr": self.lr_Linner[0]
                },
                {
                    "params": self.TailToBaseAttention.parameters(),
                    "weight_decay": self.weight_decay[0],
                    "lr": self.lr_Linner[0]
                },
            ]
        else:
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
                },
            ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=self.epsilon[0],
            weight_decay=self.weight_decay[0]
        )


        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1298,
            num_training_steps=12978
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
