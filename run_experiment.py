import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from TALNT import add_tokens, add_tokens_norm_weighted


def run_experiment(
    model,
    learning_rate,
    finetune_type,
    tokens_and_descriptions,
    add_token_type,
    train_dataset,
    test_dataset
):
    # Model setup
    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Add tokens to the model based on the add_token_type
    if add_token_type == "baseline":
        tokenizer.add_tokens(tokens_and_descriptions)
        model.resize_token_embeddings(len(tokenizer))
    elif add_token_type == "TALNT":
        model, tokenizer = add_tokens(
            model, tokenizer, tokens_and_descriptions["tokens"], tokens_and_descriptions["descriptions"])
    elif add_token_type == "TALNT_norm_weighted":
        model, tokenizer = add_tokens_norm_weighted(
            model, tokenizer, tokens_and_descriptions["tokens"], tokens_and_descriptions["descriptions"])
    else:
        raise ValueError("Invalid add_token_type")

    # Set up LoRA if relevant
    if finetune_type == "LORA":
        lora_config = LoraConfig(
            lora_alpha=16,
            r=8,
            lora_target_modules=["q_proj", "v_proj"],
            bias=None,
            task_type="CausalLM"
        )
        model = get_peft_model(model, lora_config)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None,
            add_special_tokens=True
        )

    train_dataset = Dataset.from_dict(train_dataset).map(
        tokenize_function, batched=True)
    train_dataset = train_dataset.add_column(
        "labels", train_dataset["input_ids"].copy())
    test_dataset = Dataset.from_dict(test_dataset).map(
        tokenize_function, batched=True)
    test_dataset = test_dataset.add_column(
        "labels", test_dataset["input_ids"].copy())
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    # Training
    losses = []
    for i in range(30):
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            learning_rate=learning_rate
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
        )
        trainer.train()
        trainer.evaluate(test_dataset)
        losses.append(trainer.state.log_history[-1]["eval_loss"])

    losses = torch.tensor(losses)
    print("Average loss:", losses.mean())
    print("Standard deviation:", losses.std())

    # Print an example output
    generated_ids = model.generate(**test_dataset[0])
    decoded_output = tokenizer.decode(
        generated_ids, skip_special_tokens=True)[0]
    print(decoded_output)
