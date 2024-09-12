import json
import os
import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def create_kafka_client(client_type):
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            if client_type == 'producer':
                return KafkaProducer(
                    bootstrap_servers=[os.environ.get('KAFKA_BROKER')],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
            elif client_type == 'consumer':
                return KafkaConsumer(
                    'prompts_to_analyze',
                    bootstrap_servers=[os.environ.get('KAFKA_BROKER')],
                    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                    auto_offset_reset='earliest',
                    enable_auto_commit=True,
                    group_id='prompt-analyzer-group'
                )
        except NoBrokersAvailable:
            retries += 1
            print(f"Unable to connect to Kafka. Retrying in 5 seconds... (Attempt {retries}/{max_retries})")
            time.sleep(5)
    raise Exception(f"Failed to create Kafka {client_type} after multiple attempts")

# Initialize Kafka producer and consumer
try:
    producer = create_kafka_client('producer')
    consumer = create_kafka_client('consumer')
except Exception as e:
    print(f"Failed to create Kafka client: {e}")
    producer, consumer = None, None

# Load pre-trained model and tokenizer
model_name = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a text classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def analyze_prompt(task_id, prompt):
    # Classify the prompt
    result = classifier(prompt)[0]
    label = result['label']
    
    # Map the label to a task
    task_mapping = {
        'LABEL_0': 'sentiment_analysis',
        'LABEL_1': 'question_answering',
        'LABEL_2': 'other'  # You might want to handle other types of tasks
    }
    
    task = task_mapping.get(label, 'other')
    
    message = {
        'task_id': task_id,
        'task': task,
        'prompt': prompt,
        'confidence': result['score']
    }
    
    if producer:
        try:
            producer.send('nlp_tasks', message)
            print(f"Analyzed prompt: Task = {task}, Confidence = {result['score']:.2f}")
        except Exception as e:
            print(f"Failed to send message to Kafka: {e}")
    else:
        print("Kafka producer is not available. Message not sent.")

# Main loop
print("Prompt analyzer is running. Waiting for messages...")
for message in consumer:
    prompt = message.value.get('prompt')
    task_id = message.value.get('task_id')
    if prompt and task_id:
        analyze_prompt(task_id, prompt)
    else:
        print("Received message without a prompt or task_id. Skipping.")