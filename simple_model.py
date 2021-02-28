import numpy
import torch
import transformers
from transformers import AutoTokenizer, pipeline, Seq2SeqTrainer, Seq2SeqTrainingArguments
from tokenizers import Tokenizer
import sys, os, shutil, re, argparse, json
from datasets import load_dataset

text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
code_tokenizer = Tokenizer.from_file("code_tokenizer.json")

dataset = load_dataset('csv', delimiter="\t", data_files={'train':'tok-train-shuf.tsv','eval':'tok-eval.tsv','test':'tok-test.tsv'}, download_mode="force_redownload")

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = text_tokenizer(batch["pseudo"],padding="max_length", truncation=True, max_length=128)['input_ids']
    outputs = [enc.ids for enc in code_tokenizer.encode_batch(batch["code"])]
    for i in range(len(outputs)):
        pad_token_id = code_tokenizer.token_to_id('[PAD]')
        outputs[i] += [pad_token_id]*(128 - len(outputs[i]))
    batch["input_ids"] = inputs
    batch["decoder_input_ids"] = outputs
    batch["labels"] = outputs.copy()
    batch["labels"] = [[-100 if token == pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

train_data = dataset['train'].map(
    process_data_to_model_inputs, 
    batched=True,
    load_from_cache_file=False 
)

eval_data = dataset['eval'].map(
    process_data_to_model_inputs, 
    batched=True, 
    load_from_cache_file=False
)

model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased','./decoder-bert')
print(model.num_parameters())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# inp = text_tokenizer.encode('add 2 to i')
# print(text_tokenizer.convert_ids_to_tokens(inp))
# input_ids = torch.tensor(inp).unsqueeze(0)  # Batch size 1

# outp = code_tokenizer.encode('i += 2 ;')
# print(outp.tokens)
# decoder_input_ids = torch.tensor(outp.ids).unsqueeze(0)
# print(input_ids, input_ids.shape)
# print(decoder_input_ids, decoder_input_ids.shape)

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    fp16=True, 
    output_dir="./checkpoints/",
    logging_steps=5000,
    save_steps=5000,
    eval_steps=5000,
    warmup_steps=2000,
    save_total_limit=5,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    # compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=eval_data,
)
trainer.train()





# nlp = pipeline('text2text-generation',model=model,tokenizer=text_tokenizer)
# output = nlp('add 2 to i',return_tensors=True)
# print(output[0]['generated_token_ids'].numpy())
# print(code_tokenizer.decode(output[0]['generated_token_ids'].numpy()))

# inp = text_tokenizer.encode('add 2 to i')
# input_ids = torch.tensor(inp).unsqueeze(0)  # Batch size 1
# decoder_input_ids = torch.tensor(code_tokenizer.encode('[PAD]').ids).unsqueeze(0)

# outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)

# encoded_sequence = (outputs.encoder_last_hidden_state,)
# lm_logits = outputs.logits

# # sample last token with highest prob
# next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
# print(next_decoder_input_ids)
# print(code_tokenizer.decode(next_decoder_input_ids.numpy()[0]))

# # concat
# decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)


# # STEP 2
# outputs = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True)

# encoded_sequence = (outputs.encoder_last_hidden_state,)
# lm_logits = outputs.logits

# # sample last token with highest prob
# next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
# print(code_tokenizer.decode(next_decoder_input_ids.numpy()[0]))

# # concat
# decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
