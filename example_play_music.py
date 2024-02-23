import os
import logging

from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from TALNT import add_token

# Model setup
MODEL_NAME = ""
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
logging.info("Model and tokenizer loaded")

# Add tokens to the model
token = "<play_music>"
description = "Play music command to be used to complete user requests for music"
model, tokenizer = add_token(model, tokenizer, token, description)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=512, return_tensors=None)


# Create and tokenize dataset
dataset = Dataset.from_dict(
    {
        "text": [
            "Play me Taylor Swift<play_music>Taylor Swift",
            "I'd like to hear some jazz<play_music>Jazz",
            "Can you play some rock music?<play_music>Rock",
            "Play 'Congratulations' by Post Malone<play_music>Congratulations by Post Malone",
            "I want to hear some classical music<play_music>Classical",
            "Play music by Eminem<play_music>Eminem"
        ]
    }
)
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.add_column("labels", dataset["input_ids"].copy())
print(dataset[0]["input_ids"])
print(tokenizer("<play_music>"))
data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    learning_rate=0.001  # Relatively high learning rate
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)
trainer.train()

# Evaluate
model_inputs = tokenizer(["Play some rap for me"],
                         return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs)
decoded_output = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True)[0]
print(decoded_output)
