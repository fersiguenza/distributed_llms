from flask import Flask, request, jsonify
from flask_cors import CORS
from kafka import KafkaProducer, KafkaConsumer
import json
import os
import uuid
import threading
import time

app = Flask(__name__)
CORS(app)

producer = KafkaProducer(
    bootstrap_servers=[os.environ.get('KAFKA_BROKER')],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

consumer = KafkaConsumer(
    'nlp_results',
    bootstrap_servers=[os.environ.get('KAFKA_BROKER')],
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='api-gateway-group'
)

results = {}

def consume_results():
    for message in consumer:
        task_id = message.value.get('task_id')
        if task_id:
            if task_id not in results:
                results[task_id] = {}
            results[task_id][message.value['task']] = message.value

threading.Thread(target=consume_results, daemon=True).start()

@app.route('/api/process', methods=['POST'])
def process_prompt():
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    task_id = str(uuid.uuid4())
    
    # Send task to prompt analyzer
    producer.send('prompts_to_analyze', {'task_id': task_id, 'prompt': prompt})

    return jsonify({'task_id': task_id}), 202

@app.route('/api/result/<task_id>', methods=['GET'])
def get_result(task_id):
    result = results.get(task_id, {})
    
    if 'sentiment_analysis' in result or 'question_answering' in result:
        response = {
            'task_id': task_id,
            'prompt': result.get('sentiment_analysis', {}).get('prompt') or result.get('question_answering', {}).get('prompt'),
            'mode': result.get('sentiment_analysis', {}).get('task') or result.get('question_answering', {}).get('task'),
        }
        
        if 'sentiment_analysis' in result:
            sentiment = result['sentiment_analysis']['sentiment_analysis']
            response['sentiment_analysis'] = {
                'sentiment': sentiment['sentiment'],
                'confidence': sentiment['confidence'],
                'processing_time': sentiment.get('processing_time', 'N/A')
            }
        
        if 'question_answering' in result:
            qa = result['question_answering']['question_answering']
            response['question_answering'] = {
                'answer': qa['answer'],
                'confidence': qa['confidence'],
                'processing_time': qa.get('processing_time', 'N/A')
            }
        
        response['total_processing_time'] = result.get('sentiment_analysis', {}).get('total_processing_time') or \
                                            result.get('question_answering', {}).get('total_processing_time', 'N/A')
        
        if 'sentiment_analysis' in result and 'question_answering' in result:
            del results[task_id]  # Remove the result after sending both
        
        return jsonify(response)
    else:
        return jsonify({'status': 'processing'}), 202

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)