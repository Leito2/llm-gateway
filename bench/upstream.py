"""A trivial OpenAI-compatible upstream with zero latency, the 'provider' every benchmark talks to.

python -m uvicorn bench.upstream:app --port 9001
"""

import time

from fastapi import FastAPI, Request

app = FastAPI()
_REPLY = "Benchmark reply."


@app.post("/v1/chat/completions")
async def chat(request: Request) -> dict:
    body = await request.json()
    return {
        "id": "chatcmpl-bench",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "m"),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": _REPLY}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
