"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import type { Role, ChatMessage, SSEEvent } from "@/app/types";

const TURN_ORDER = [
  "moderator",
  "traveler_A",
  "traveler_B",
  "traveler_C",
  "traveler_D",
] as const;

// const API_BASE = `http://${window.location.hostname}:8000`;
const API_BASE = "https://multi-llm-human-discussion.onrender.com";

type Speaker = (typeof TURN_ORDER)[number];

function nextSpeaker(after: Speaker): Speaker {
  const i = TURN_ORDER.indexOf(after);
  const j = (i + 1) % TURN_ORDER.length;
  return TURN_ORDER[j];
}

function guessNextSpeaker(messages: ChatMessage[]): Speaker {
  // まだ何も来てなければ最初は moderator から開始する前提
  if (messages.length === 0) return "moderator";
  const last = messages[messages.length - 1].who as Speaker;
  return nextSpeaker(last);
}

export default function Room() {
  const params = useParams<{ id: string }>();
  const rawId = params?.id;
  const sessionId = Array.isArray(rawId) ? rawId[0] : rawId;

  const [role, setRole] = useState<Role>("traveler_B");
  const [msgs, setMsgs] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const sseRef = useRef<EventSource | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  const [typingMap, setTypingMap] = useState<Record<string, boolean>>({});
  const [finished, setFinished] = useState(false);

  const [isTyping, setIsTyping] = useState(false);

  // SSE接続
  useEffect(() => {
    if (!sessionId || sseRef.current) return;
    const es = new EventSource(
      `${API_BASE}/session/stream?session_id=${encodeURIComponent(sessionId)}`
    );
    sseRef.current = es;

    es.onmessage = (e) => {
      if (!e?.data) return;
      if (e.data === "__END__") {
        setFinished(true);
        es.close();
        sseRef.current = null;
        return;
      }
      try {
        const ev = JSON.parse(e.data) as SSEEvent;
        if (ev.type === "message") {
          setMsgs((prev) => [
            ...prev,
            { who: ev.who, content: ev.content, ts: Date.now() },
          ]);
          if (ev.content.trim() === "【合意確定】") setFinished(true);
          // 話した人は typing を下げておく（保険）
          setTypingMap((m) => ({ ...m, [ev.who]: false }));
        } else if (ev.type === "typing") {
          setTypingMap((m) => ({ ...m, [ev.who]: ev.active }));
        }
      } catch {}
    };

    es.onerror = () => {
      es.close();
      sseRef.current = null;
    };

    return () => {
      es.close();
      sseRef.current = null;
    };
  }, [sessionId]);

  // 自動スクロール
  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [msgs.length]);

  const typingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sendTyping = (active: boolean) => {
    if (!sessionId) return;
    fetch(`${API_BASE}/session/typing`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, who: role, active }),
    }).catch(() => {});
  };

  const myAgentName: Speaker = role; // B か D
  const currentNext: Speaker = useMemo(() => guessNextSpeaker(msgs), [msgs]);
  const isMyTurn = currentNext === myAgentName;

  const onInputChange: React.ChangeEventHandler<HTMLTextAreaElement> = (e) => {
    const el = e.currentTarget;
    setInput(el.value);

    // 入力中フラグ（300msデバウンス）
    if (typingTimer.current) clearTimeout(typingTimer.current);
    sendTyping(true);
    typingTimer.current = setTimeout(() => sendTyping(false), 800);

    // 自動リサイズ
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 240) + "px"; // 上限240px
  };

  const send = async () => {
    if (!sessionId || !input.trim() || !isMyTurn || finished) return;
    await fetch(`${API_BASE}/session/input`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        who: role,
        text: input.trim(),
      }),
    });
    setInput("");
    // 高さリセット
    const ta = document.getElementById(
      "chat-input"
    ) as HTMLTextAreaElement | null;
    if (ta) {
      ta.style.height = "auto";
    }
  };

  if (!sessionId)
    return <main style={{ padding: 24 }}>セッションID取得中…</main>;

  return (
    <main
      style={{
        padding: 0,
        height: "100vh",
        display: "grid",
        gridTemplateRows: "56px 1fr 112px",
      }}
    >
      {/* ヘッダ */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          padding: "0 16px",
          borderBottom: "1px solid #eee",
        }}
      >
        <div style={{ fontWeight: 700, flex: 1 }}>Room: {sessionId}</div>
        <div style={{ display: "flex", gap: 8 }}>
          <label>役割</label>
          <select
            value={role}
            onChange={(e) => setRole(e.target.value as Role)}
            style={{ padding: "6px 8px" }}
          >
            <option value="traveler_B">旅行者B（自分）</option>
            <option value="traveler_D">旅行者D（自分）</option>
          </select>
        </div>
      </header>

      {/* メッセージリスト */}
      <div
        ref={listRef}
        style={{
          overflowY: "auto",
          background: "#f5f7fb",
          padding: "12px 12px 0 12px",
        }}
      >
        {msgs.map((m, i) => {
          const mine = m.who === myAgentName;
          const nameMap: Record<string, string> = {
            moderator: "司会",
            traveler_A: "旅行者A",
            traveler_B: "旅行者B",
            traveler_C: "旅行者C",
            traveler_D: "旅行者D",
          };
          return (
            <div
              key={i}
              style={{
                display: "flex",
                justifyContent: mine ? "flex-end" : "flex-start",
                margin: "6px 0",
              }}
            >
              {!mine && (
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: 18,
                    background: "#e0e7ff",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 12,
                    marginRight: 8,
                    flex: "0 0 auto",
                  }}
                >
                  {nameMap[m.who]?.replace("旅行者", "旅")}
                </div>
              )}
              <div
                style={{
                  maxWidth: "76%",
                  background: mine ? "#a7f3d0" : "#fff",
                  border: "1px solid rgba(0,0,0,0.08)",
                  borderRadius: 12,
                  padding: "10px 12px",
                  boxShadow: "0 1px 1px rgba(0,0,0,0.04)",
                }}
              >
                {!mine && (
                  <div
                    style={{ fontSize: 12, color: "#6b7280", marginBottom: 4 }}
                  >
                    {nameMap[m.who] ?? m.who}
                  </div>
                )}
                <div
                  style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}
                >
                  {m.content}
                </div>
              </div>
              {mine && <div style={{ width: 36, height: 36, marginLeft: 8 }} />}
            </div>
          );
        })}
      </div>

      {/* 入力中インジケータ
      <div style={{ padding: "4px 12px", color: "#6b7280", fontSize: 12 }}>
        {Object.entries(typingMap)
          .filter(([who, active]) => active && who !== myAgentName)
          .map(([who]) => {
            const nameMap: Record<string, string> = {
              moderator: "司会",
              traveler_A: "旅行者A",
              traveler_B: "旅行者B",
              traveler_C: "旅行者C",
              traveler_D: "旅行者D",
            };
            return (
              <span key={who} style={{ marginRight: 12 }}>
                {nameMap[who] ?? who} が入力中…
              </span>
            );
          })}
      </div> */}

      {/* 入力欄 */}
      <div
        style={{
          borderTop: "1px solid #eee",
          padding: 12,
          display: "grid",
          gridTemplateColumns: "1fr 92px",
          gap: 8,
          background: "#fff",
        }}
      >
        <div style={{ display: "grid", alignItems: "center" }}>
          <textarea
            id="chat-input"
            value={input}
            onChange={onInputChange}
            onBlur={() => sendTyping(false)}
            placeholder={"ここにメッセージを入力"}
            rows={1}
            style={{
              width: "100%",
              resize: "none", // ← ユーザによるドラッグリサイズ禁止
              padding: 12,
              borderRadius: 8,
              border: "1px solid #ddd",
              outline: "none",
              overflowY: "auto", // ← スクロール可
              maxHeight: "240px",
            }}
          />
        </div>
        <button
          onClick={send}
          disabled={!isMyTurn || !input.trim()}
          style={{
            borderRadius: 8,
            border: "1px solid #10b981",
            background: isMyTurn && input.trim() ? "#10b981" : "#a7f3d0",
            color: "#fff",
            fontWeight: 700,
          }}
        >
          送信
        </button>

        <div
          style={{
            gridColumn: "1 / span 2",
            fontSize: 12,
            color: "#6b7280",
            display: "flex",
            gap: 8,
          }}
        >
          <span>
            次の番：
            <b>
              {currentNext === "moderator"
                ? "司会"
                : currentNext === "traveler_A"
                ? "旅行者A"
                : currentNext === "traveler_B"
                ? "旅行者B"
                : currentNext === "traveler_C"
                ? "旅行者C"
                : "旅行者D"}
            </b>
          </span>
          <a
            href={`${API_BASE}/session/log?session_id=${encodeURIComponent(
              sessionId
            )}`}
            target="_blank"
            rel="noreferrer"
          >
            Markdownログ
          </a>
        </div>
      </div>
    </main>
  );
}
