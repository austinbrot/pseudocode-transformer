import numpy
import torch
import transformers
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import sys, os, shutil, re, argparse, json

# config = transformers.GPT2Config.from_pretrained('gpt2-medium')
# gpt2_decoder = transformers.GPT2Model(config)

model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased','bert-base-uncased')

text_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
code_tokenizer = Tokenizer.from_file("code_tokenizer.json")

inp = text_tokenizer.encode('add 2 to i')

input_ids = torch.tensor(inp).unsqueeze(0)  # Batch size 1
print(input_ids)s
# outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, return_dict=True)
# print(outputs)
