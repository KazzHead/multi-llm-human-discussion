# server/app.py
from typing import Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio

from server.autogen_session import Session

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://multi-llm-human-discussion.vercel.app",
]

SESS: Dict[str, Session] = {}

app = FastAPI()

# --- CORS: Next.js(3000) からのプリフライト(OPTIONS)を許可 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,          # 実運用は next のホストに限定推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 入力モデル ---
class CreateSessionIn(BaseModel):
    session_id: str

class PostInputIn(BaseModel):
    session_id: str
    who: str           # "traveler_B" or "traveler_D"
    text: str

# --- メモリ上にセッションを保持（簡易実装） ---
SESS: Dict[str, Session] = {}

@app.post("/session/create")
async def create_session(payload: CreateSessionIn, bg: BackgroundTasks):
    session_id = payload.session_id
    if not session_id:
        raise HTTPException(400, "session_id required")
    if session_id in SESS:
        return {"ok": True, "started": bool(SESS[session_id]._run_task)}

    session = Session()
    SESS[session_id] = session
    bg.add_task(session.stream_run) 
    # session._run_task = asyncio.create_task(session.stream_run())
    return {"ok": True, "started": True}

@app.get("/session/stream")
async def stream(session_id: str):
    if session_id not in SESS:
        raise HTTPException(404, "no session")
    session = SESS[session_id]
    listener = session.add_listener()

    async def sse():
        for who, content in session.messages:
            yield f"data: {json.dumps({'type':'message','who':who,'content':content}, ensure_ascii=False)}\n\n"
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(listener.get(), timeout=15)
                    if ev.get("type") == "__END__":
                        yield "data: __END__\n\n"
                        break
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # 心拍（コメント）
        finally:
            session.remove_listener(listener)

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

class TypingIn(BaseModel):
    session_id: str
    who: str
    active: bool

@app.post("/session/typing")
async def post_typing(payload: TypingIn):
    sid, who, active = payload.session_id, payload.who, payload.active
    if sid not in SESS:
        raise HTTPException(404, "no session")
    if who not in ("traveler_B", "traveler_D", "moderator", "traveler_A", "traveler_C"):
        raise HTTPException(400, "bad who")
    SESS[sid].set_typing(who, active)
    return {"ok": True}

@app.post("/session/input")
async def post_input(payload: PostInputIn):
    session_id = payload.session_id
    who = payload.who
    text = payload.text
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

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
