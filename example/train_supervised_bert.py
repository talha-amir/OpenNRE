# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
import sys
import os
import argparse
import logging
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased', 
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'], 
        help='Sentence representation pooler')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true', 
        help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['none', 'semeval', 'wiki80', 'tacred'], 
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

# Hyper-parameters
parser.add_argument('--batch_size', default=64, type=int,
        help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
        help='Learning rate')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=3, type=int,
        help='Max number of training epochs')

# Seed
parser.add_argument('--seed', default=42, type=int,
        help='Seed')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if len(args.ckpt) == 0:
    args.ckpt = f'{args.dataset}_{args.pretrain_path}_{args.pooler}'
ckpt = f'ckpt/{args.ckpt}.pth.tar'

if args.dataset != 'none':
    opennre.download(args.dataset, root_path=root_path)
    args.train_file = os.path.join(
        root_path, 'benchmark', args.dataset, f'{args.dataset}_train.txt'
    )

    args.val_file = os.path.join(
        root_path, 'benchmark', args.dataset, f'{args.dataset}_val.txt'
    )

    args.test_file = os.path.join(
        root_path, 'benchmark', args.dataset, f'{args.dataset}_test.txt'
    )

    if not os.path.exists(args.test_file):
        logging.warn(
            f"Test file {args.test_file} does not exist! Use val file instead"
        )

        args.test_file = args.val_file
    args.rel2id_file = os.path.join(
        root_path, 'benchmark', args.dataset, f'{args.dataset}_rel2id.json'
    )

    args.metric = 'acc' if args.dataset == 'wiki80' else 'micro_f1'
elif (
    not os.path.exists(args.train_file)
    or not os.path.exists(args.val_file)
    or not os.path.exists(args.test_file)
    or not os.path.exists(args.rel2id_file)
):
    raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info(f'    {arg}: {getattr(args, arg)}')

rel2id = json.load(open(args.rel2id_file))

# Define the sentence encoder
if args.pooler == 'entity':
    sentence_encoder = opennre.encoder.BERTEntityEncoder(
        max_length=args.max_length, 
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
elif args.pooler == 'cls':
    sentence_encoder = opennre.encoder.BERTEncoder(
        max_length=args.max_length, 
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
else:
    raise NotImplementedError

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt='adamw'
)

# Train the model
if not args.only_test:
    framework.train_model('micro_f1')

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
logging.info('Test set results:')
logging.info(f"Accuracy: {result['acc']}")
logging.info(f"Micro precision: {result['micro_p']}")
logging.info(f"Micro recall: {result['micro_r']}")
logging.info(f"Micro F1: {result['micro_f1']}")
