import numpy
import torch
import transformers
from transformers import BertModel, BertConfig
import sys, os, shutil, re, argparse, json

config = BertConfig(vocab_size=5449)
model = BertModel(config)
config.is_decoder = True
config.decoder_start_token_id = 1
config.pad_token_id = 1

model.save_pretrained('decoder-bert')