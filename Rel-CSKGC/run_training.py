import pytorch_lightning as pl
from src.lightening_classes import RelCSKGCDataMoudle
from src.lightening_classes import RelCSKGC
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer, BertModel
from transformers import  RobertaTokenizer, RobertaModel
import os
import argparse

def run(args):
    pl.seed_everything(1234)

    logs_path = os.path.join('./Data/logs/', args.PLM_type+'_'+ args.feature_method)
    print(logs_path)
    if os.path.exists(logs_path) == False:
        os.makedirs(logs_path)

    if args.PLM_type == 'RoBerta':
        #tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        #model = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('./Data/package/')
        model = RobertaModel.from_pretrained('./Data/package/')

    # lightningmodule checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="Validation/f1",
        dirpath=logs_path,
        filename="RelKGC-{epoch:02d}-{Validation/f1:.2f}",
        save_top_k= args.save_top_k,
        mode="max",
    )

    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        callbacks= callbacks,
        gpus=1,
        max_epochs= args.max_epochs,
        fast_dev_run=False,
        gradient_clip_val= args.gradient_clip_val,
    )

    '''max_length, train_batch_size, test_batch_size, val_batch_size'''
    DM = RelCSKGCDataMoudle(
        max_length= args.max_length,
        train_batch_size= args.train_batch_size,
        test_batch_size= args.test_batch_size,
        val_batch_size= args.val_batch_size,
        tokenizer= tokenizer,
        relation_data_path = './Data/Relation/relation_fuse_Ioevent.txt',
        data_path = './Data/decomposed_data/',
        PLM_type= args.PLM_type
    )

    '''dropout_prob, text_dim, out_dim, lr, epsion, weight_decay'''
    Model = RelCSKGC(
        text_dim= args.text_dim,
        out_dim= args.out_dim,
        lr_bert= args.lr_bert,
        lr_Linner= args.lr_Linner,
        epsilon= args.epsilon,
        weight_decay= args.weight_decay,
        feature_method= args.feature_method,
        model= model,
        relation_path = './Data/Relation/relation_fuse_Ioevent.txt',
        save_path = './Data/little_examples_test/TestResult.csv',
        PLM_type= args.PLM_type
    )

    DM.setup()

    #Training Model
    trainer.fit(model= Model, datamodule= DM)

    # test
    DM.setup('test')
    trainer.test(model= Model, datamodule= DM)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine RelKGC')
    parser.add_argument("--lr_bert", type= float, default= 2e-5,help="learning rate of bert/roberta")
    parser.add_argument("--lr_Linner", type=float, default= 1e-4, help="learning rate of linnear classifier")
    parser.add_argument("--epsilon", type= float, default= 1e-8)
    parser.add_argument("--text_dim", type= int, default= 768, help="dimension of bert/robert embedding")
    parser.add_argument("--out_dim", type= int, default= 8, help = "num of relation categories")
    parser.add_argument("--weight_decay", type= float, default= 0.01, help= "weight_decay")
    parser.add_argument("--PLM_type", type= str, default= 'RoBerta', help = "the pretrained launage model we used")
    parser.add_argument("--max_length", type= int ,default= 100, help= "the max length of tokenizer")
    parser.add_argument("--train_batch_size", type= int, default= 128, help= 'the batch size during training')
    parser.add_argument("--test_batch_size", type=int, default=1024, help='the batch size during testing')
    parser.add_argument("--val_batch_size", type=int, default=128, help='the batch size during deving')
    parser.add_argument("--feature_method", type= str, default= 'maxpooling', help= "the methond type")
    parser.add_argument("--save_top_k", type= int, default= 5, help= "set top k models are saved")
    parser.add_argument("--max_epochs", type= int, default= 5, help= "set the max iterations during training")
    parser.add_argument("--gradient_clip_val", type = float , default= 5.0, help= "set the gradient clip val")

    args = parser.parse_args()
    print(args)

    run(args)