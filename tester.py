import requests
import os

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:5000",   # required by OpenRouter
    "X-Title": "Pregnancy Nutrition Chatbot"   # any name you like
}

payload = {
    "model": "upstage/solar-pro-3:free",
    "messages": [
        {"role": "user", "content": "Give a healthy Indian breakfast for a pregnant woman."}
    ],
    "max_tokens": 200
}

res = requests.post(url, headers=headers, json=payload)

print(res.status_code)
print(res.json())
