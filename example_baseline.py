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

# Add tokens to the model with random initializations
tokenizer.add_tokens(["<play_music>", "</play_music>"])
model.resize_token_embeddings(len(tokenizer))


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=512, return_tensors=None)


# Create and tokenize dataset
dataset = Dataset.from_dict(
    {
        "text": [
            "Can you play 'Shape of You' by Ed Sheeran? <play_music> 'Shape of You' by Ed Sheeran </play_music>",
            "I'm in the mood for some classic rock. <play_music> classic rock </play_music>",
            "Play me 'Congratulations' by Post Malone. <play_music> 'Congratulations' by Post Malone </play_music>",
            "I'd like to listen to some jazz tonight. <play_music> jazz </play_music>",
            "How about some Beethoven symphonies? <play_music> Beethoven symphonies </play_music>",
            "Put on the latest Billie Eilish album, please. <play_music> latest Billie Eilish album </play_music>",
            "Let's hear some country classics. <play_music> country classics </play_music>",
            "Play 'Hotline Bling' by Drake for me. <play_music> 'Hotline Bling' by Drake </play_music>",
            "I want to listen to 'Bohemian Rhapsody' by Queen. <play_music> 'Bohemian Rhapsody' by Queen </play_music>",
            "How about playing some smooth jazz? <play_music> smooth jazz </play_music>",
            "I feel like listening to 'Lose Yourself' by Eminem. <play_music> 'Lose Yourself' by Eminem </play_music>",
            "Could you put on some meditation music? <play_music> meditation music </play_music>",
            "I'd love to hear 'Jolene' by Dolly Parton. <play_music> 'Jolene' by Dolly Parton </play_music>",
            "Let's get some techno beats going. <play_music> techno beats </play_music>",
            "Can we listen to 'Stairway to Heaven' by Led Zeppelin? <play_music> 'Stairway to Heaven' by Led Zeppelin </play_music>",
            "I'm in the mood for some salsa music. <play_music> salsa music </play_music>",
            "Play 'Thriller' by Michael Jackson, please. <play_music> 'Thriller' by Michael Jackson </play_music>",
            "How about some indie folk to chill? <play_music> indie folk </play_music>",
            "Let's listen to 'Purple Rain' by Prince. <play_music> 'Purple Rain' by Prince </play_music>",
            "I want some upbeat workout music. <play_music> upbeat workout music </play_music>",
            "Can you play some calming nature sounds? <play_music> calming nature sounds </play_music>",
            "I'd like to hear 'Sweet Child o' Mine' by Guns N' Roses. <play_music> 'Sweet Child o' Mine' by Guns N' Roses </play_music>",
            "Let's have some reggae music playing. <play_music> reggae music </play_music>",
            "Play 'Wonderwall' by Oasis for me. <play_music> 'Wonderwall' by Oasis </play_music>",
            "I'd enjoy some live jazz recordings. <play_music> live jazz recordings </play_music>"
        ]
    }
)
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.add_column("labels", dataset["input_ids"].copy())
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
