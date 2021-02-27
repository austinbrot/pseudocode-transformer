import numpy
import torch
import transformers
from transformers import BertModel, BertConfig
import sys, os, shutil, re, argparse, json

config = BertConfig(vocab_size=5449)
model = BertModel(config)
config.is_decoder = True

model.save_pretrained('decoder-bert')