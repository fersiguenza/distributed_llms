from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, concatenate_datasets

# Load a smaller pre-trained model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Prepare dataset (using a small subset of SQuAD)
datasets = load_dataset("squad")
train_dataset = datasets["train"].shuffle(seed=42).select(range(1000))
val_dataset = datasets["validation"].shuffle(seed=42).select(range(100))

# Create custom dataset
custom_data = [
    {
        "context": "Unfortunately, there has been a problem in another station that is causing these delays. Trains will be here soon. Railway staff are working to resolve the issue as quickly as possible.",
        "question": "How is it possible that the train is delayed once again, it's been 3 times in less than 1 hour, is the train coming at all?",
        "answers": {
            "text": ["Unfortunately there has been a problem in another station that is causing these delays. Trains will be here soon."],
            "answer_start": [0]
        },
        "id": "custom_0",
        "title": "Train Delays"
    },
    {
        "context": "There have been multiple issues affecting train services today. A signal failure at a major junction is causing widespread delays. Engineers are on site working to fix the problem as quickly as possible.",
        "question": "Why are there so many train delays today?",
        "answers": {
            "text": ["A signal failure at a major junction is causing widespread delays."],
            "answer_start": [66]
        },
        "id": "custom_1",
        "title": "Train Delays"
    },
    # Add more similar examples here
]

custom_dataset = Dataset.from_dict({k: [dic[k] for dic in custom_data] for k in custom_data[0]})

# Combine SQuAD and custom dataset
combined_train_dataset = concatenate_datasets([train_dataset, custom_dataset])

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers["text"][i][0]
        start_char = answers["answer_start"][i][0]
        end_char = start_char + len(answer)
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_datasets = {}
tokenized_datasets["train"] = combined_train_dataset.map(preprocess_function, batched=True, remove_columns=combined_train_dataset.column_names)
tokenized_datasets["validation"] = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./question_answerer/results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./question_answerer/logs",
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    evaluation_strategy="steps",
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

# Save the model
model.save_pretrained("./question_answerer/model")
tokenizer.save_pretrained("./question_answerer/model")

# Test the model
test_question = "How is it possible that the train is delayed once again, it's been 3 times in less than 1 hour, is the train coming at all?"
test_context = "Train delays can occur due to various reasons such as signal failures, track maintenance, or incidents at other stations. Unfortunately, there has been a problem in another station that is causing these delays. Trains will be here soon. Railway staff are working to resolve the issue as quickly as possible."

inputs = tokenizer(test_question, test_context, return_tensors="pt")
outputs = model(**inputs)

answer_start = outputs.start_logits.argmax()
answer_end = outputs.end_logits.argmax()

answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
print(f"Test Question: {test_question}")
print(f"Model Answer: {answer}")