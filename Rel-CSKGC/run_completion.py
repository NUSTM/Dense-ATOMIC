#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: Siwei Wu
@file: run_completion.py 
@time: 2022/04/16
@contact: wusiwei@njust.edu.cn
"""
from tqdm import tqdm

from src.lightening_classes import RelCSKGCCompletion
from src.lightening_classes import CompletionDataMoudle
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import csv
import random
import pandas as pd

import json

def load_atomic(path):
    f= open(path, 'r', encoding= 'utf-8')
    
    data_raw= csv.reader(f)
    next(data_raw)
    base= {}
    tail_according_base= {}
    for line in data_raw:
        relation = line[2]
        head = line[1]
        tail = line[3]
        if head not in base:
            base[head]= head
            tail_according_base[head]= []
        tail_according_base[head].append([tail, relation])
    f.close()
    return list(base), tail_according_base

def change_X_and_Y(H, T):
    head_split = H.split()
    if head_split[0] == 'PersonY':
        tail_split = T.split()
        for t, word in enumerate(head_split):
            if word == 'PersonX':
                head_split[t] = 'PersonY'
            elif word == 'PersonY':
                head_split[t] = 'PersonX'
            elif word == 'PersonX.':
                head_split[t] = 'PersonY.'
            elif word == 'PersonY.':
                head_split[t] = 'PersonX.'
            elif word == "PersonX's":
                head_split[t] = "PersonY's"
            elif word == "PersonY's":
                head_split[t] = "PersonX's"
        for t, word in enumerate(tail_split):
            if word == 'PersonX':
                tail_split[t] = 'PersonY'
            elif word == 'PersonY':
                tail_split[t] = 'PersonX'
            elif word == 'PersonX.':
                tail_split[t] = 'PersonY.'
            elif word == 'PersonY.':
                tail_split[t] = 'PersonX.'
            elif word == "PersonX's":
                tail_split[t] = "PersonY's"
            elif word == "PersonY's":
                tail_split[t] = "PersonX's"
        head = ' '.join(head_split)
        tail = ' '.join(tail_split)
        return head, tail
    else:
        return  H, T

def sample_inner(base, tail_according_base):
    samples= []
    tail_events= tail_according_base[base]
    events_add_base= tail_events.copy()
    if '_' not in base:
        events_add_base.append([base, base])
    for head in tail_events:
        for tail in events_add_base:
            if head != tail:
                H, T= change_X_and_Y(head[0], tail[0])
                samples.append([H, T, 'inner', head[0], tail[0], head[1], tail[1]])

    return samples

def sample_external(base1, base2, tail_according_base):
    samples = []
    tail1_events = tail_according_base[base1]
    tail2_events = tail_according_base[base2]
    if '_' not in base1:
        tail1_events.append([base1, base1])
    if '_' not in base2:
        tail2_events.append([base2, base2])

    for event1 in tail1_events:
        for event2 in tail2_events:
            H, T = change_X_and_Y(event1[0], event2[0])
            samples.append([H, T, 'external', event1[0], event2[0], event1[1], event2[1]])

    return samples

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    #tokenizer = RobertaTokenizer.from_pretrained('./Data/package/')
    #model = RobertaModel.from_pretrained('./Data/package/')
    task_type= 'completion'
    file_nums= 1
    trainer = pl.Trainer(
        enable_progress_bar=True,
        gpus=1,
        max_epochs=3,
        fast_dev_run=False,
        gradient_clip_val=5.0,
    )
    
    
    random.seed(123)
    num= 800
    atomic_path= './Data/decomposed_data/raw_train_for_relkgc.csv'
    bases, tail_according_base= load_atomic(atomic_path)

    df = pd.read_csv(atomic_path, index_col= None)



    base_pairs= []
    for base in bases:
        for i in range(num):
            base_pairs.append([base, random.choice(bases)])
    
    f = open('./Data/decomposed_data/base_pairs.json', 'w', encoding = 'utf-8')
    json.dump(base_pairs, f)
    f.close()



    if task_type== 'completion':
        Model = RelCSKGCCompletion(
                dropout_prob=0.2,
                text_dim=768,
                out_dim=8,
                lr_bert=2e-5,
                lr_Linner=1e-4,
                epsilon=1e-8,
                weight_decay=0.01,
                model=model,
                vocabulary_path='./Data/Relation/relation_fuse_Ioevent.txt',
                save_path='./Data/completed_data/completed_data.csv',
                feature_method='maxpooling',
                task_type='completion',
                model_type='RoBerta'
            )
        
        for i in tqdm(list(range(1223))):
            base_pairs_epoch = base_pairs[10000 * i : 10000 * (i + 1)]
            
            samples= []
            count = 0
            for pair in base_pairs_epoch:
                count += 1
                samples_external= sample_external(pair[0], pair[1], tail_according_base)
                samples.extend(samples_external)
                if count % 2000 == 0:
                    samples_tem = []
                    for sample in samples:
                        samples_tem.append('<repeat>'.join(sample))
                    samples_tem = list(set(samples_tem))
                    
                    samples = []
                    for sample in samples_tem:
                        samples.append(sample.split('<repeat>'))
            
            samples_tem = []
            for sample in samples:
                samples_tem.append('<repeat>'.join(sample))
            samples_tem = list(set(samples_tem))
            
            samples = []
            for sample in samples_tem:
                samples.append(sample.split('<repeat>'))
            
            DM = CompletionDataMoudle(
                max_length=100,
                train_batch_size=2048,
                test_batch_size=2048,
                val_batch_size=2048,
                tokenizer=tokenizer,
                test_data = samples,
                data_dir='./Data/',
                relation_type='fuse_Ioevent',
                model_type='RoBerta'
            )
    
            DM.setup('test')
            
            trainer.test(model=Model, datamodule=DM,
                         ckpt_path='./Data/logs/RoBerta_maxpooling/RelKGC-epoch=02-Validation/f1=0.10.ckpt')
            
