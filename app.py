from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

class HumanlikeChatbot:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.history = []

    def generate_response(self, user_input):
        # Add user input to history
        self.history.append(f"You: {user_input}")

        # Combine history for context
        input_text = " ".join(self.history)
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        # Set attention_mask explicitly
        attention_mask = torch.ones(input_ids.shape, dtype=input_ids.dtype, device=input_ids.device)

        # Generate response
        response_ids = self.model.generate(
            input_ids,
            max_length=2000,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=90,
            top_p=0.95,
            do_sample=True,
            temperature=0.8,  # Experiment with temperature for desired randomness
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode and refine response
        response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        response_text = self.post_process_response(response_text)

        # Add chat history
        self.history.append(f"Chatbot: {response_text}")

        # Trim history to keep it manageable
        self.history = self.history[-5:]

        return response_text

    def post_process_response(self, response):
        # Add custom post-processing logic here
        # For example, you can refine grammar, add casual language, etc.
        return response

chatbot = HumanlikeChatbot()

@app.route('/')
def index():
    return render_template('hack7.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.form['user_message']
    response = chatbot.generate_response(user_input)
    return jsonify({'message': response})

if __name__ == '__main__':
    app.run(debug=True)
