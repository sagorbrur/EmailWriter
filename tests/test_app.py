import requests

data = {
    "prompt": "your prompt",
    "token_count": 128,
    "temperature": 0.6,
    "n_gen": 4
}

response = requests.post('http://localhost:5000/generate', json=data)
print(response.json())
