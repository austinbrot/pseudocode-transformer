import transformers
from transformers import EncoderDecoderModel, AutoTokenizer
from tokenizers import Tokenizer
import torch
import sys
from datasets import load_metric

metric = load_metric('sacrebleu')

chk_dir = sys.argv[1] 
chk_num = sys.argv[2] 
num_beams = int(sys.argv[3])
code_tok = False if sys.argv[4]=='false' else True
print_bool = False if sys.argv[5]=='false' else True

model = EncoderDecoderModel.from_pretrained('./{}/checkpoint-{}/'.format(chk_dir,chk_num))
code_tokenizer = Tokenizer.from_file('code_tokenizer.json') if code_tok else AutoTokenizer.from_pretrained('bert-base-uncased')
text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#print(text_tokenizer.convert_tokens_to_ids())
pad_token_id = 1 if code_tok else 0
bos_token_id = 2 if code_tok else 101
eos_token_id = 3 if code_tok else 102

f = open('tok-eval.tsv','r')
for i, line in enumerate(f):
    if i==0:
        continue
    if i>1000:
        break
    if i%50==0: 
        print(i)
    src, tgt = line.strip('\n').split('\t')
    input_ids = torch.tensor(text_tokenizer.encode(src)).unsqueeze(0)
    # max_len = max(max_len, len(tgt.split()))
    generated = model.generate(input_ids, min_length=1, max_length=100, eos_token_id=eos_token_id, pad_token_id=pad_token_id, no_repeat_ngram_size=3, decoder_start_token_id=bos_token_id, num_beams=num_beams)
    
    translation = code_tokenizer.decode(generated.numpy()[0],skip_special_tokens=True)
    # print(len(translation.split()))
    if (print_bool):
        print("GOLD: {} -------- MODEL: {}".format(tgt, translation))
    metric.add_batch(predictions=[translation],references=[[tgt]])
print(metric.compute())
f.close()	
