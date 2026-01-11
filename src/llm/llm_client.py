import os
from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    _client = None

    @classmethod
    def initialize(cls):
        if cls._client is None:
            base_dir = os.path.dirname(__file__)
            env_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'vars.env'))
            load_dotenv(env_path)

            cls._client = OpenAI(
                base_url=os.getenv("LLM_BASE_URL"),
                api_key=os.getenv("LLM_API_KEY", "none")
            )

    @classmethod
    def chat(cls, messages, model="llama-3.3-70b-instruct", **kwargs):
        cls.initialize()

        response = cls._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

        return response.choices[0].message.content
