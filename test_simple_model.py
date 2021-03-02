import transformers
from transformers import EncoderDecoderModel, AutoTokenizer
from tokenizers import Tokenizer
import torch
import sys
from datasets import load_metric

metric = load_metric('sacrebleu')

if len(sys.argv)<3:
    print('Please provide checkpoint num and num_beams')

model = EncoderDecoderModel.from_pretrained('./checkpoints/checkpoint-{}/'.format(sys.argv[1]))
code_tokenizer = Tokenizer.from_file('code_tokenizer.json')
text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
num_beams = int(sys.argv[2])

f = open('tok-eval.tsv','r')
for i, line in enumerate(f):
    if i==0:
        continue
    if i>1000:
        break
    if i%100==0: 
        print(i)
    src, tgt = line.strip('\n').split('\t')
    input_ids = torch.tensor(text_tokenizer.encode(src)).unsqueeze(0)
    generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.bos_token_id, num_beams=num_beams)
    translation = code_tokenizer.decode(generated.numpy()[0])
    # print(tgt, translation)
    metric.add_batch(predictions=[translation],references=[[tgt]])

print(metric.compute())
f.close()	
