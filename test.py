import requests, os
from dotenv import load_dotenv
load_dotenv()
headers = {'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}', 'Content-Type': 'application/json'}
r = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json={
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Xin chào"}]
})
print(r.status_code)
print(r.json())
