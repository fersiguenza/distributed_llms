import json
import os
from kafka import KafkaConsumer, KafkaProducer
from transformers import pipeline
import time

# Initialize Kafka consumer and producer
consumer = KafkaConsumer(
    'nlp_tasks',
    bootstrap_servers=[os.environ.get('KAFKA_BROKER')],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=[os.environ.get('KAFKA_BROKER')],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model='./model')

print("Sentiment analyzer is running. Waiting for messages...")

def interpret_sentiment(label, score):
    if label == 'LABEL_1':
        sentiment = 'positive' if score > 0.6 else 'somewhat positive'
    else:
        sentiment = 'negative' if score > 0.6 else 'somewhat negative'
    
    intensity = 'strong' if score > 0.8 else 'moderate'
    return f"{intensity} {sentiment}"

for message in consumer:
    start_time = time.time()
    prompt = message.value['prompt']
    task_id = message.value.get('task_id', 'unknown')

    # Perform sentiment analysis
    sentiment_result = sentiment_pipeline(prompt)[0]
    sentiment = interpret_sentiment(sentiment_result['label'], sentiment_result['score'])
    processing_time = time.time() - start_time
    # Prepare the response
    response = {
        'task_id': task_id,
        'task': 'sentiment_analysis',
        'prompt': prompt,
        'sentiment_analysis': {
            'sentiment': sentiment,
            'confidence': sentiment_result['score'],
            "processing_time": processing_time
        }
    }

    # Send the result back to the 'nlp_results' topic
    producer.send('nlp_results', response)
    print(f"Processed and sent sentiment analysis result for task_id: {task_id}")