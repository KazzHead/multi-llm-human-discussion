# server/app.py
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio

from server.autogen_session import Session, TRAVELER_ROLES

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
    ai_travelers: Optional[List[str]] = None
    wishes_md: Optional[str] = None

class PostInputIn(BaseModel):
    session_id: str
    who: str           # "traveler_B" or "traveler_D"
    text: str

class StopSessionIn(BaseModel):
    session_id: str

# --- メモリ上にセッションを保持（簡易実装） ---
SESS: Dict[str, Session] = {}

@app.post("/session/create")
async def create_session(payload: CreateSessionIn, bg: BackgroundTasks):
    session_id = payload.session_id
    if not session_id:
        raise HTTPException(400, "session_id required")
    if session_id in SESS:
        session = SESS[session_id]
        return {
            "ok": True,
            "started": bool(session._run_task),
            "config": session.get_config(),
        }

    ai_travelers = payload.ai_travelers or []
    invalid = [r for r in ai_travelers if r not in TRAVELER_ROLES]
    if invalid:
        raise HTTPException(400, f"invalid traveler ids: {invalid}")

    session = Session(ai_travelers=ai_travelers, wishes_md=payload.wishes_md)
    SESS[session_id] = session
    # Prefer Session.start if available; fallback to BackgroundTasks
    try:
        if hasattr(session, "start"):
            session.start()  # type: ignore[attr-defined]
        else:
            bg.add_task(session.stream_run)
    except Exception:
        bg.add_task(session.stream_run)
    # session._run_task = asyncio.create_task(session.stream_run())
    return {"ok": True, "started": True, "config": session.get_config()}

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
    if who not in TRAVELER_ROLES:
        raise HTTPException(400, "invalid traveler")
    session = SESS[session_id]
    if session.is_ai_traveler(who):
        raise HTTPException(400, f"{who} is AI-controlled")
    try:
        session.hio.feed(who, text)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}

@app.get("/session/log")
async def get_log(session_id: str):
    if session_id not in SESS:
        raise HTTPException(404, "no session")
    md = SESS[session_id].get_log_markdown()
    return PlainTextResponse(md)


@app.get("/session/config")
async def get_session_config(session_id: str):
    if session_id not in SESS:
        raise HTTPException(404, "no session")
    return SESS[session_id].get_config()

@app.get("/session/list")
async def list_sessions():
    items = []
    for sid, sess in SESS.items():
        try:
            started = bool(getattr(sess, "started", True)) or True
        except Exception:
            started = True
        items.append({
            "id": sid,
            "started": started,
            "config": sess.get_config(),
        })
    return {"sessions": items}

@app.post("/session/stop")
async def stop_session(payload: StopSessionIn):
    sid = payload.session_id
    if sid not in SESS:
        raise HTTPException(404, "no session")
    sess = SESS.pop(sid)
    try:
        if hasattr(sess, "stop"):
            await sess.stop()  # type: ignore[attr-defined]
        else:
            sess.broadcast({"type": "__END__"})
    except Exception:
        pass
    return {"ok": True}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
