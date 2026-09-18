import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

class LLM:
    def __init__(self):
        self.active_chats = {}
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        with open("prompt.txt", "r", encoding="utf-8") as file:
            self.prompt = file.read()

    def chat(self, id, text):
        if id not in self.active_chats:
            self.active_chats[id] = self.client.chats.create(
                model='gemini-3.5-flash-lite',
                config=types.GenerateContentConfig(
                    system_instruction=self.prompt,
                    temperature=0.7,
                    max_output_tokens=1500,
                )
            )

        response = self.active_chats[id].send_message(text)

        return response.text