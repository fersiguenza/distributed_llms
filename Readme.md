
# Small Language Model Project

This project demonstrates different approaches to working with small language models for specific NLP tasks. It includes implementations using Hugging Face Transformers, a distributed system with Kafka, and scripts for pre-training models.

## Project Structure

The project is organized into three main directories:

```
.
├── hugging_faces_transformer/
├── web_interface/
├── api_gateway/
├── distributed_system/
├── models/
├── docker-compose
└── README.md
```

### 1. hugging_faces_transformer/

This directory contains implementations using the Hugging Face Transformers library. It demonstrates how to use pre-trained models for sentiment analysis and question answering tasks.

Key features:
- Direct use of Hugging Face models
- Examples of inference using pipelines
- Scripts for fine-tuning models on specific tasks

To run:
1. Navigate to the directory: `cd hugging_faces_transformer`
2. Install requirements: `pip install -r requirements.txt`
3. Run the example scripts: `python main.py` 

### 2. distributed_system/

This directory contains a distributed system implementation using Kafka for message passing. It demonstrates how to break down NLP tasks into microservices for parallel processing.

Key components:
- Docker Compose file for setting up Kafka and Python services
- Prompt analyzer service
- Sentiment analysis service
- Question answering service

To run:
1. Navigate to the directory: `cd distributed_system`
2. Ensure Docker and Docker Compose are installed
3. Run `docker-compose up --build`
4. Follow the instructions in the directory's README for interacting with the system

### 3. models/

This directory contains scripts and notebooks for pre-training and fine-tuning small language models. It provides examples of how to create custom models for specific tasks.

Key features:
- Scripts for data preprocessing
- Model training notebooks
- Examples of model compression techniques

To use:
1. Navigate to the directory: `cd models`
2. Install requirements: `pip install -r requirements.txt`
3. Follow the instructions in individual notebooks or scripts for training models

### 4. Use the bash script to setup models

```bash
chmod +x setup_and_train.sh
```

3. Run the script:

```bash
./setup_and_train.sh
```

This script will:
- Create a virtual environment
- Install the required dependencies
- Create directories for each model
- Train each model and save them in their respective directories

## Requirements

- Python 3.9+
- PyTorch
- Transformers library
- Kafka (for distributed system)
- Docker and Docker Compose (for distributed system)

See individual `requirements.txt` files in each directory for specific dependencies.


## Running the entire System with the UI interface to test both approaches

1. Build and start the services:
   ```
   docker-compose up --build
   ```

2. The system will start with three main services:
   - prompt_analyzer
   - sentiment_analyzer
   - question_answerer

3. To interact with the system, you can either use the web interface to input prompts opening a web browser and navigating to `http://localhost:8080` or attach to the prompt_analyzer service. You can do this by attaching to the prompt_analyzer container:
   ```
   docker attach <repository-directory>_prompt_analyzer_1
   ```

4. Once attached, you can enter prompts. The system will automatically determine whether to perform sentiment analysis or question answering based on the presence of a question mark.

5. To view the results, you can check the logs of the sentiment_analyzer and question_answerer services:
   ```
   docker-compose logs -f sentiment_analyzer
   docker-compose logs -f question_answerer
   ```

## Stopping the System

To stop the system, use:
```
docker-compose down
```
