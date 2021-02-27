import numpy
import torch
import transformers
from tokenizers import Tokenizer
from transformers import BertModel, BertConfig
import sys, os, shutil, re, argparse, json

code_tokenizer = Tokenizer.from_file("code_tokenizer.json")
code_vocab_size = code_tokenizer.get_vocab_size()
print(code_vocab_size)

config = BertConfig(vocab_size=code_vocab_size)
model = BertModel(config)
config.is_decoder = True
config.decoder_start_token_id = code_tokenizer.token_to_id('[CLS]')
config.pad_token_id = code_tokenizer.token_to_id('[PAD]')

model.save_pretrained('decoder-bert')