from flask import Flask, request, jsonify

app = Flask(__name__)

def predict(*args):
    return 'Hello World!'

@app.route('/')
def index():
    print("Hello There!")

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        data = request.json
        prompt = data['prompt']
        token_count = data['token_count']
        temperature = data['temperature']
        n_gen = data['n_gen']
        response = {
            "status": "success",
            "ai_results": [],
            "text_length": 0
        }
        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

# request = {
# "prompt": "your prompt",
# "token_count": 128,
# "temperature": 0.6,
# "n_gen": 4, // Defines how many alternatives will be returned.
# }
# response = {"status": "success",
# "ai_results": [{"generated_text": "generated text",
# "text_length": 753}]
# }