import os
import requests
from flask import Flask, request, Response
from dotenv import load_dotenv
from llm import LLM

load_dotenv()
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

app = Flask(__name__)
chatbot = LLM()

@app.route("/telegram", methods=['POST'])
def reply_telegram():
    msg = request.get_json()

    if 'message' in msg:
        m = msg['message']['text']
        chat_id = msg['message']['chat']['id']
    elif 'edited_message' in msg:
        m = msg['edited_message']['text']
        chat_id = msg['edited_message']['chat']['id']
    else:
        return Response('ok', status=200)

    if m == '/start':
        bot_response = "¡Hola! Soy tu alumno de pruebas. ¿Qué me vas a enseñar hoy?"
    else:
        bot_response = chatbot.chat(chat_id, m)

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": bot_response
    }
    
    requests.post(url, json=payload)

    return Response('ok', status=200)

if __name__ == '__main__':
    app.run(port=5000)