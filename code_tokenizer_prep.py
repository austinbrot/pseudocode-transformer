
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


t = Tokenizer(WordLevel(unk_token="[UNK]"))
t.pre_tokenizer = Whitespace()

trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]"])

# fin = open('data/tok-train-shuf.tsv','r')
# fout = open('data/tok-train-shuf-tgt.tsv','w+')

# for line in fin:
# 	src, tgt = line.strip('\n').split('\t')
# 	fout.write(tgt+'\n')
# fin.close()
# fout.close()

files = ['data/tok-train-shuf-tgt.tsv']
t.train(files, trainer)

t.save("code_tokenizer.json")
