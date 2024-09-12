# Distributed Small Language Model System

This project demonstrates a distributed system for processing natural language tasks using small language models. It uses Kafka for message passing and Docker for containerization.

## Prerequisites

- Docker
- Docker Compose

## Project Structure

```
.
├── prompt_analyzer/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── sentiment_analyzer/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── question_answerer/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
└── README.md
```

## Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Ensure you have the trained models in the correct directories:
   - `prompt_analyzer/prompt_model/` 
   - `sentiment_analyzer/sentiment_model/`
   - `question_answerer/qa_model/`

   If you don't have these models, you'll need to train them or download pre-trained models and place them in these directories.
   You can train the models from the ./models directory

