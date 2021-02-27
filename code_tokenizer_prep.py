
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing


t = Tokenizer(WordLevel(unk_token="[UNK]"))
t.pre_tokenizer = Whitespace()

trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[CLS]"])
t.post_processor = TemplateProcessing(
    single="[CLS] $A",
    # ,
    # pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 2),
        # ("[SEP]", 3),
    ]
)

files = ['tok-train-shuf-tgt.tsv']
t.train(files, trainer)

t.save("code_tokenizer.json")
