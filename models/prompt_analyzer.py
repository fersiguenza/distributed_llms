from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dataset
data = [
    {"text": "I love this product!", "label": 0},
    {"text": "What is the capital of France?", "label": 1},
    {"text": "Summarize this article for me.", "label": 2},
    # Add more examples...
]

dataset = Dataset.from_list(data)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split dataset
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./prompt_analyzer/results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./prompt_analyzer/logs",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./prompt_analyzer/model")
tokenizer.save_pretrained("./prompt_analyzer/model")