import transformers
from transformers import EncoderDecoderModel, AutoTokenizer
from tokenizers import Tokenizer
import torch
import sys

if len(sys.argv)<2:
    raise('Please provide checkpoint num')

model = EncoderDecoderModel.from_pretrained('./checkpoints/checkpoint-{}/'.format(sys.argv[1]))
code_tokenizer = Tokenizer.from_file('code_tokenizer.json')
text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

f = open('tok-eval.tsv','r')
i = 0
for line in f:
    if i==0:
        i += 1 
        continue
    if i==50: 
        break
    src, tgt = line.strip('\n').split('\t')
    input_ids = torch.tensor(text_tokenizer.encode(src)).unsqueeze(0)
    generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id, num_beams=100)
    translation = code_tokenizer.decode(generated.numpy()[0])
    print('PSEUDO: {}, GOLD: {}, MODEL: {}'.format(src,tgt,translation))
    i += 1

f.close()	
