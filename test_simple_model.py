import transformers
from transformers import EncoderDecoderModel, AutoTokenizer
from tokenizers import Tokenizer
import torch

model = EncoderDecoderModel.from_pretrained('./checkpoint-1000/')
code_tokenizer = Tokenizer.from_file('code_tokenizer.json')
text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

inps = ['add 2 to i','read T','increment j']

for inp in inps:
    print('Pseudocode: {}'.format(inp))
    input_ids = torch.tensor(text_tokenizer.encode(inp)).unsqueeze(0)
    generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)

    print('Translation: {}'.format(code_tokenizer.decode(generated.numpy()[0])))
