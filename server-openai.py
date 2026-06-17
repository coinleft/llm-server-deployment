import os
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://tvxzgqb7gpbph2-8000.proxy.runpod.net/v1"

# 同步、异步双客户端
sync_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

async_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL
)

app = FastAPI(title="API Service")

# 请求体模型


class ChatRequest(BaseModel):
    prompt: str
    model: str = "Qwen/Qwen3-8B"


# 1. 同步接口：一次性返回完整回答，非流式
@app.post("/chat")
def chat_sync(req: ChatRequest):
    print(req)
    """同步对话接口，等待全部生成完成后一次性返回文本"""
    resp = sync_client.chat.completions.create(
        # openai compatible parameters
        # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        model=req.model,
        messages=[{"role": "user", "content": req.prompt}],
        stream=True,
        # extra_body={"top_k": req.top_k}

        # vLLM supports some parameters that are not supported by OpenAI, top_k for example.
        # You can pass these parameters to vLLM using the OpenAI client in the extra_body parameter of your requests, i.e. extra_body={"top_k": 50} for top_k.

        # https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/#internal-data-structures
        # use_beam_search: bool = False
        # top_k: int | None = None
        # min_p: float | None = None
        # repetition_penalty: float | None = None
        # length_penalty: float = 1.0
        # stop_token_ids: list[int] | None = []
        # include_stop_str_in_output: bool = False
        # ignore_eos: bool = False
        # min_tokens: int = 0
        # skip_special_tokens: bool = True
        # spaces_between_special_tokens: bool = True
        # truncate_prompt_tokens: Annotated[int, Field(ge=-1, le=_INT64_MAX)] | None = None
        # truncation_side: Literal["left", "right"] | None = Field(
        #     default=None,
        #     description=(
        #         "Which side to truncate from when truncate_prompt_tokens is active. "
        #         "'right' keeps the first N tokens. "
        #         "'left' keeps the last N tokens."
        #     ),
        # )
        # allowed_token_ids: list[int] | None = None
        # prompt_logprobs: int | None = None

    )
    content = resp.choices[0].message.content
    return {
        "success": True,
        "model": req.model,
        "content": content
    }


# 2. 异步流式接口：逐块实时输出
async def stream_generator(prompt: str, model: str):
    stream = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    # 正确取值：先判断 choices 是否存在且不为空
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """异步流式对话接口，SSE 实时返回分段内容"""
    generator = stream_generator(req.prompt, req.model)
    return StreamingResponse(generator, media_type="text/event-stream")


# 测试用根路由
@app.get("/")
def root():
    return {"msg": "FastAPI service running",
            "endpoints": ["/chat", "/chat/stream"]}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
