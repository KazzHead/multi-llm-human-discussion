"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import {
  type Role,
  type ChatMessage,
  type SSEEvent,
  type SessionConfig,
  TRAVELER_ROLES,
} from "@/app/types";

const TRAVELER_LABELS: Record<Role, string> = {
  traveler_A: "旅行者A",
  traveler_B: "旅行者B",
  traveler_C: "旅行者C",
  traveler_D: "旅行者D",
};

type Speaker = Role | "moderator";

const DEFAULT_TURN_ORDER: Speaker[] = ["moderator", ...TRAVELER_ROLES];

// const API_BASE = `http://${window.location.hostname}:8000`;
const API_BASE = "https://multi-llm-human-discussion.onrender.com";

function nextSpeaker(order: Speaker[], after: Speaker): Speaker {
  if (order.length === 0) return after;
  const i = order.indexOf(after);
  const nextIndex = i >= 0 ? (i + 1) % order.length : 0;
  return order[nextIndex];
}

function guessNextSpeaker(order: Speaker[], messages: ChatMessage[]): Speaker {
  if (order.length === 0) return "moderator";
  if (messages.length === 0) return order[0];
  const last = messages[messages.length - 1].who as Speaker;
  return nextSpeaker(order, last);
}

export default function Room() {
  const params = useParams<{ id: string }>();
  const rawId = params?.id;
  const sessionId = Array.isArray(rawId) ? rawId[0] : rawId;

  const [turnOrder, setTurnOrder] = useState<Speaker[]>(DEFAULT_TURN_ORDER);
  const [role, setRole] = useState<Role | null>(null);
  const [sessionConfig, setSessionConfig] = useState<SessionConfig | null>(
    null
  );
  const [msgs, setMsgs] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const sseRef = useRef<EventSource | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  const [typingMap, setTypingMap] = useState<Record<string, boolean>>({});
  const [finished, setFinished] = useState(false);

  const humanRoles = useMemo<Role[]>(() => {
    if (sessionConfig) return sessionConfig.human_travelers;
    return ["traveler_B", "traveler_D"];
  }, [sessionConfig]);

  useEffect(() => {
    if (!sessionId) return;
    let aborted = false;
    const loadConfig = async () => {
      try {
        const res = await fetch(
          `${API_BASE}/session/config?session_id=${encodeURIComponent(
            sessionId
          )}`
        );
        if (!res.ok) throw new Error("failed to load config");
        const data = (await res.json()) as SessionConfig;
        if (!aborted) {
          setSessionConfig(data);
          setTurnOrder(DEFAULT_TURN_ORDER);
        }
      } catch {
        if (!aborted) {
          setSessionConfig({
            ai_travelers: TRAVELER_ROLES.filter(
              (t) => !["traveler_B", "traveler_D"].includes(t)
            ),
            human_travelers: ["traveler_B", "traveler_D"],
          });
          setTurnOrder(DEFAULT_TURN_ORDER);
        }
      }
    };
    loadConfig();
    return () => {
      aborted = true;
    };
  }, [sessionId]);

  useEffect(() => {
    setRole((prev) => {
      if (prev && TRAVELER_ROLES.includes(prev)) return prev;
      return humanRoles[0] ?? TRAVELER_ROLES[0] ?? null;
    });
  }, [humanRoles]);

  useEffect(() => {
    if (!role) {
      setInput("");
    }
  }, [role]);

  // SSE 接続
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
    if (!sessionId || !role) return;
    fetch(`${API_BASE}/session/typing`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, who: role, active }),
    }).catch(() => {});
  };

  const myAgentName = role;
  const currentNext: Speaker = useMemo(
    () => guessNextSpeaker(turnOrder, msgs),
    [turnOrder, msgs]
  );
  const isMyTurn = role ? currentNext === role : false;
  const isSelectedAi = useMemo(() => {
    if (!role) return false;
    if (sessionConfig) return sessionConfig.ai_travelers.includes(role);
    return ["traveler_A", "traveler_C"].includes(role);
  }, [role, sessionConfig]);

  const onInputChange: React.ChangeEventHandler<HTMLTextAreaElement> = (e) => {
    const el = e.currentTarget;
    setInput(el.value);

    // 入力中フラグ（800ms デバウンス）
    if (typingTimer.current) clearTimeout(typingTimer.current);
    if (role && !isSelectedAi) {
      sendTyping(true);
      typingTimer.current = setTimeout(() => sendTyping(false), 800);
    }

    // 自動リサイズ
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 240) + "px";
  };

  const send = async () => {
    if (
      !sessionId ||
      !role ||
      !input.trim() ||
      !isMyTurn ||
      finished ||
      isSelectedAi
    )
      return;
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
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <label>役割</label>
          <select
            value={role ?? ""}
            onChange={(e) => {
              const value = e.target.value as Role | "";
              if (value) setRole(value as Role);
            }}
            style={{ padding: "6px 8px" }}
          >
            {TRAVELER_ROLES.map((r) => (
              <option key={r} value={r}>
                {TRAVELER_LABELS[r]}
              </option>
            ))}
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
          const mine = role ? m.who === myAgentName : false;
          const nameMap: Record<string, string> = {
            moderator: "司会",
            ...TRAVELER_ROLES.reduce(
              (acc, key) => ({ ...acc, [key]: TRAVELER_LABELS[key] }),
              {} as Record<string, string>
            ),
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
                  {nameMap[m.who] ?? m.who}
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
            style={{
              width: "100%",
              resize: "none",
              padding: 12,
              borderRadius: 8,
              border:
                role && isMyTurn && !finished
                  ? "1px solid #10b981"
                  : "1px solid #ddd",
              outline: "none",
              overflowY: "auto",
              maxHeight: "240px",
              backgroundColor:
                !role || finished || !isMyTurn || isSelectedAi
                  ? "#f5f5f5"
                  : "#fff",
            }}
          />
        </div>
        <button
          onClick={send}
          disabled={
            !role || finished || !isMyTurn || !input.trim() || isSelectedAi
          }
          style={{
            borderRadius: 8,
            border: "1px solid #10b981",
            background:
              role && isMyTurn && !finished && input.trim() && !isSelectedAi
                ? "#10b981"
                : "#a7f3d0",
            color: "#fff",
            fontWeight: 700,
            cursor:
              role && isMyTurn && !finished && input.trim() && !isSelectedAi
                ? "pointer"
                : "not-allowed",
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
            次の番:{" "}
            <b>
              {currentNext === "moderator"
                ? "司会"
                : TRAVELER_LABELS[currentNext as Role] ?? currentNext}
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
