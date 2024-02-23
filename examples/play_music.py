import os
import logging

from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import TALNT

# Model setup
MODEL_NAME = ""
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
logging.info("Model and tokenizer loaded")

# Add tokens to the model
token = "<play_music>"
description = "Play music command to be used to complete user requests for music"
model, tokenizer = TALNT.add_token(model, tokenizer, token, description)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Create and tokenize dataset
dataset = Dataset.from_dict(
    {
        "text": [
            "Play me Taylor Swift<play_music>Taylor Swift",
            "I'd like to hear some jazz<play_music>Jazz",
            "Can you play some rock music?<play_music>Rock",
            "Play 'Congratulations' by Post Malone<play_music>Congratulations by Post Malone",
            "I want to hear some classical music<play_music>Classical",
        ]
    }
)
dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    learning_rate=0.001  # Relatively high learning rate
)
trainer.train()

# Evaluate
model_inputs = tokenizer(["Play some rap for me"],
                         return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs)
decoded_output = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True)[0]
print(decoded_output)
