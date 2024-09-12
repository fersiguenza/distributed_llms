from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from datasets import load_dataset

# Load a small pre-trained model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare dataset (using IMDB dataset as an example)
dataset = load_dataset("imdb")

# Reduce dataset size
train_dataset = dataset["train"].shuffle(seed=42).select(range(5000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = {}
tokenized_datasets["train"] = train_dataset.map(tokenize_function, batched=True)
tokenized_datasets["test"] = test_dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./sentiment_analyzer/results",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=0,
    weight_decay=0.01,
    logging_dir="./sentiment_analyzer/logs",
    logging_steps=100,
    eval_steps=500,
    save_steps=1000,
    evaluation_strategy="steps",
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

# Save the model
model.save_pretrained("./sentiment_analyzer/model")
tokenizer.save_pretrained("./sentiment_analyzer/model")