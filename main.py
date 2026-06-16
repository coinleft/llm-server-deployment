from fastapi import FastAPI
from pydantic import BaseModel
import httpx
from fastapi.responses import StreamingResponse

app = FastAPI(title="LLM中转服务")
# vLLM地址
# VLLM_URL = "http://127.0.0.1:8000"
VLLM_URL = "https://tvxzgqb7gpbph2-8000.proxy.runpod.net/v1/chat/completions"
LLM_API_KEY = "sk-tvxzgqb7gpbph2"


class ChatReq(BaseModel):
    id: str
    # openai compatible parameters
    # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    model: str = 'Qwen/Qwen3-8B'
    messages: list
    temperature: float = 0.7
    stream: bool = False

    # vllm extra parameters for sampling
    # https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/#internal-data-structures
    use_beam_search: bool = False
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    length_penalty: float = 1.0
    stop_token_ids: list[int] | None = []
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    # truncate_prompt_tokens: Annotated[int, Field(ge=-1, le=_INT64_MAX)] | None = None
    # truncation_side: Literal["left", "right"] | None = Field(
    #     default=None,
    #     description=(
    #         "Which side to truncate from when truncate_prompt_tokens is active. "
    #         "'right' keeps the first N tokens. "
    #         "'left' keeps the last N tokens."
    #     ),
    # )
    allowed_token_ids: list[int] | None = None
    prompt_logprobs: int | None = None


@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    # 1. 前端参数校验（限制范围，防止非法值）
    req.temperature = max(0.0, min(2.0, req.temperature))

    # 2. 组装转发给vLLM的payload
    payload = {
        "id": req.id,
        "model": req.model,
        "messages": req.messages,
        "temperature": req.temperature,
        "stream": req.stream
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    print(payload)
    print(headers)

    # 3. 异步转发流式
    async def stream_generator():
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", VLLM_URL, json=payload, headers=headers) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

    if req.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(VLLM_URL, json=payload, headers=headers)
            return r.json()

@app.get("/")
def root():
    return {"msg": "FastAPI service running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
