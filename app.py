import torch
import random
import config as cfg
from utils import seed_everything, get_tokenizer, get_model

from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = ""
seed_everything(cfg.SEED)
tokenizer = get_tokenier(special_tokens=cfg.SPECIAL_TOKENS)
model = get_model(tokenizer, 
                special_tokens=cfg.SPECIAL_TOKENS,
                load_model_path=model_path)

def join_keywords(keywords, randomize=True):
    N = len(keywords)
    if randomize: 
        M = random.choice(range(N+1))
        keywords = keywords[:M]
        random.shuffle(keywords)

    return ','.join(keywords)

def predict(prompt: str, token_count: int, temperature: float, n_gen: int, keywords=None) -> dict:
    output = {}
    try:
        if keywords:
            kw = join_keywords(keywords, randomize=False)
            prompt = cfg.SPECIAL_TOKENS['bos_token'] + input_text + \
                cfg.SPECIAL_TOKENS['sep_token'] + kw + cfg.SPECIAL_TOKENS['sep_token']
        else:
            prompt = cfg.SPECIAL_TOKENS['bos_token'] + input_text + \
                cfg.SPECIAL_TOKENS['sep_token'] + cfg.SPECIAL_TOKENS['sep_token']

        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        device = torch.device("cuda")
        generated = generated.to(device)

        model.eval()
        sample_outputs = model.generate(generated, 
                                        do_sample=True,   
                                        min_length=50, 
                                        max_length=cfg.MAXLEN,
                                        top_k=30,                                 
                                        top_p=0.7,        
                                        temperature=temperature,
                                        repetition_penalty=2.0,
                                        num_return_sequences=n_gen
                                        )
        if sample_outputs:                            
            output['status'] = 'success'
            output['ai_results'] = []
            for i, sample_output in enumerate(sample_outputs):
                text = tokenizer.decode(sample_output, skip_special_tokens=True)
                output['ai_results'].append({'generated_text': text, "text_length": len(text)})

    except Exception as e:
        output['status'] = 'error'

    return output

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
        response = predict(prompt, token_count, temperature, n_gen)
        # response = {
        #     "status": "success",
        #     "ai_results": [],
        #     "text_length": 0
        # }
        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
