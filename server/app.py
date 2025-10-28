import asyncio
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from server.autogen_session import Session

app = FastAPI()
SESS: Dict[str, Session] = {}  # 簡易メモリ

@app.post("/session/create")
async def create_session(session_id: str):
    if session_id in SESS:
        raise HTTPException(409, "already exists")
    SESS[session_id] = Session()
    return {"ok": True}

@app.get("/session/stream")
async def stream(session_id: str):
    if session_id not in SESS:
        raise HTTPException(404, "no session")
    session = SESS[session_id]

    async def sse():
        async for who, content in session.stream_run():
            yield f"data: {who}|:|{content}\n\n"
        yield "data: __END__\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")

@app.post("/session/input")
async def post_input(session_id: str, who: str, text: str):
    if session_id not in SESS:
        raise HTTPException(404, "no session")
    if who not in ("traveler_B", "traveler_D"):
        raise HTTPException(400, "who must be traveler_B or traveler_D")
    SESS[session_id].hio.feed(who, text)
    return {"ok": True}

@app.get("/session/log")
async def get_log(session_id: str):
    if session_id not in SESS:
        raise HTTPException(404, "no session")
    md = SESS[session_id].get_log_markdown()
    return PlainTextResponse(md)
