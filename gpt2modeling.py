from transformers import GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, GPT2LMHeadModel, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="dataset/txt/NEODATASET.txt",  # Ganti dengan path dataset Anda
    block_size=50
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

model = GPT2LMHeadModel.from_pretrained("gpt2")
training_args = TrainingArguments(
    output_dir="model/",  # Ganti dengan direktori output yang diinginkan
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=5_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("model/")  # Ganti dengan direktori penyimpanan model yang diinginkan

model = GPT2LMHeadModel.from_pretrained("model/")
tokenizer = GPT2Tokenizer.from_pretrained("model/")
