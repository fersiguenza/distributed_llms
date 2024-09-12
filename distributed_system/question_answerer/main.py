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

# Load the question-answering pipeline
qa_pipeline = pipeline('question-answering', model='./model')

print("Question answerer is running. Waiting for messages...")

# Simple context retrieval function
def get_relevant_context(question):
    contexts = [
        "Train delays can occur due to various reasons such as signal failures, track maintenance, or incidents at other stations.",
        "Railway companies often provide real-time updates on train statuses and reasons for delays.",
        "In cases of repeated delays, there might be a significant issue affecting multiple trains on the line.",
        "Railway staff typically work to resolve issues as quickly as possible to minimize disruptions.",
        "Passengers are encouraged to check for updates through official channels or station announcements.",
        "Unfortunately, there has been a problem in another station that is causing these delays. Trains will be here soon.",
        #Add more
    ]
    relevant_contexts = [ctx for ctx in contexts if any(word in ctx.lower() for word in question.lower().split())]
    return ' '.join(relevant_contexts) if relevant_contexts else ' '.join(contexts)  # Return all contexts if no match

for message in consumer:
    start_time = time.time()
    prompt = message.value['prompt']
    task_id = message.value.get('task_id', 'unknown')

    # Get relevant context for the question
    context = get_relevant_context(prompt)

    # Perform question answering
    qa_result = qa_pipeline(question=prompt, context=context)
    processing_time = time.time() - start_time
    # Prepare the response
    response = {
        'task_id': task_id,
        'task': 'question_answering',
        'prompt': prompt,
        'question_answering': {
            'answer': qa_result['answer'],
            'confidence': qa_result['score'],
            "processing_time": processing_time
        },
        'context': context
    }

    # Send the result back to the 'nlp_results' topic
    producer.send('nlp_results', response)
    print(f"Processed and sent question answering result for task_id: {task_id}")