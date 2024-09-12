async function submitPrompt() {
    const promptInput = document.getElementById('promptInput');
    const statusElement = document.getElementById('status');
    const taskType = document.getElementById('taskType');
    const response = document.getElementById('response');
    const modeToggle = document.getElementById('mode-toggle');
    const submitButton = document.getElementById('submitButton');

    const prompt = promptInput.value;
    const mode = modeToggle.checked ? 'huggingface' : 'distributed';

    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    // Disable submit button and clear previous results
    submitButton.disabled = true;
    statusElement.textContent = 'Submitting...';
    taskType.textContent = '';
    response.textContent = '';

    try {
        const res = await fetch('http://localhost:3000/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt, mode }),
        });

        const data = await res.json();
        
        if (res.status === 202) {
            // Start polling for results
            statusElement.textContent = 'Processing...';
            await pollForResult(data.task_id);
        } else {
            displayResult(data);
        }
    } catch (error) {
        console.error('Error:', error);
        statusElement.textContent = 'An error occurred while processing your request.';
    } finally {
        // Re-enable submit button
        submitButton.disabled = false;
    }
}

async function pollForResult(taskId) {
    const statusElement = document.getElementById('status');
    const resultElement = document.getElementById('result');

    while (true) {
        try {
            const res = await fetch(`http://localhost:3000/api/result/${taskId}`);
            const data = await res.json();

            if (res.status === 200) {
                // Both results are ready
                displayResult(data);
                break;
            } else if (res.status === 202) {
                // Still processing or partial results ready
                statusElement.textContent = data.status === 'partial_results_ready' ? 
                    'Partial results ready, waiting for complete analysis...' : 'Processing...';
                await new Promise(resolve => setTimeout(resolve, 1000));  // Wait 1 second
            } else {
                // Unexpected status
                throw new Error(`Unexpected status: ${res.status}`);
            }
        } catch (error) {
            console.error('Error polling for result:', error);
            statusElement.textContent = 'An error occurred while retrieving the result.';
            break;
        }
    }
}

function displayResult(data) {
    const resultElement = document.getElementById('result');
    const statusElement = document.getElementById('status');

    let resultHtml = `
        <p><strong>Prompt:</strong> ${data.prompt}</p>
        <p><strong>Mode:</strong> ${data.mode}</p>
      `;

    if (data.sentiment_analysis) {
        resultHtml += `
            <h3>Sentiment Analysis:</h3>
            <p>Sentiment: ${data.sentiment_analysis.sentiment}</p>
            <p>Confidence: ${(data.sentiment_analysis.confidence * 100).toFixed(2)}%</p>
            <p>Processing Time: ${data.sentiment_analysis.processing_time ? data.sentiment_analysis.processing_time.toFixed(4) + ' seconds' : 'N/A'}</p>
        `;
    }

    if (data.question_answering) {
        resultHtml += `
            <h3>Question Answering:</h3>
            <p>Answer: ${data.question_answering.answer}</p>
            <p>Confidence: ${(data.question_answering.confidence * 100).toFixed(2)}%</p>
            <p>Processing Time: ${data.question_answering.processing_time ? data.question_answering.processing_time.toFixed(4) + ' seconds' : 'N/A'}</p>
        `;
    }

    resultElement.innerHTML = resultHtml;
    statusElement.textContent = 'Processing complete.';
}