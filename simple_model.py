import numpy
import torch
import transformers
from transformers import AutoTokenizer, pipeline
from tokenizers import Tokenizer
import sys, os, shutil, re, argparse, json

# config = transformers.GPT2Config.from_pretrained('gpt2-medium')
# gpt2_decoder = transformers.GPT2Model(config)

model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased','./decoder-bert')

text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
code_tokenizer = Tokenizer.from_file("code_tokenizer.json")

inp = text_tokenizer.encode('add 2 to i')
input_ids = torch.tensor(inp).unsqueeze(0)  # Batch size 1
decoder_input_ids = torch.tensor(code_tokenizer.encode('[PAD]').ids).unsqueeze(0)

outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)

encoded_sequence = (outputs.encoder_last_hidden_state,)
lm_logits = outputs.logits

# sample last token with highest prob
next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
print(code_tokenizer.decode(next_decoder_input_ids.numpy()[0]))

# concat
decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)


# STEP 2
outputs = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True)

encoded_sequence = (outputs.encoder_last_hidden_state,)
lm_logits = outputs.logits

# sample last token with highest prob
next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
print(code_tokenizer.decode(next_decoder_input_ids.numpy()[0]))

# concat
decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
