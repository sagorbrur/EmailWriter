import requests

def test():
    data = {
        "prompt": "leave application",
        "token_count": 128,
        "temperature": 0.6,
        "n_gen": 4,
        "keywords": ["sick", "days"]
    }
    
    response = requests.post('http://localhost:5000/generate', json=data)
    print(response.json())

if __name__ == '__main__':
    test()
