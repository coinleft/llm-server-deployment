import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.responses import StreamingResponse

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://tvxzgqb7gpbph2-8000.proxy.runpod.net/v1"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

resp = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": "hi"}],
        stream=False
    )

content = resp.choices[0].message.content
print(content)