from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")
model = AutoModelForCausalLM.from_pretrained("Qwen/QwQ-32B")

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    message = data.get('message', '')
    app.logger.debug(f'Received message: {message}')

    try:
        # Tokenize input and generate response
        inputs = tokenizer(message, return_tensors="pt")
        app.logger.debug(f'Tokenized inputs: {inputs}')
        outputs = model.generate(**inputs)
        app.logger.debug(f'Model outputs: {outputs}')
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        app.logger.debug(f'Generated reply: {reply}')
        return jsonify({'reply': reply})
    except Exception as e:
        app.logger.error(f'Error generating reply: {e}', exc_info=True)
        return jsonify({'reply': 'Sorry, I am having trouble responding right now.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
