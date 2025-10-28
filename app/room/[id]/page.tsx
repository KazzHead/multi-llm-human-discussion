"use client";
import { useEffect, useRef, useState } from "react";
import type { Role, StreamEvent } from "@/app/types";

export default function Room({ params }: { params: { id: string } }) {
  const sessionId = params.id;
  const [role, setRole] = useState<Role>("traveler_B");
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [input, setInput] = useState("");
  const sseRef = useRef<EventSource | null>(null);

  useEffect(() => {
    // 接続（片方のPCが接続すればサーバ側で開始）
    const es = new EventSource(
      `http://localhost:8000/session/stream?session_id=${sessionId}`
    );
    sseRef.current = es;
    es.onmessage = (e) => {
      if (!e?.data) return;
      if (e.data === "__END__") {
        es.close();
        return;
      }
      const [who, content] = String(e.data).split("|:|");
      setEvents((prev) => [...prev, { who, content }]);
    };
    es.onerror = () => {
      // 接続が切れたらクローズ
      es.close();
    };
    return () => {
      es.close();
    };
  }, [sessionId]);

  const send = async () => {
    if (!input.trim()) return;
    await fetch("http://localhost:8000/session/input", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, who: role, text: input }),
    });
    setInput("");
  };

  return (
    <main style={{ padding: 24, display: "grid", gap: 12 }}>
      <h2>Room: {sessionId}</h2>

      <div>
        <label>あなたの役割: </label>
        <select value={role} onChange={(e) => setRole(e.target.value as Role)}>
          <option value="traveler_B">旅行者B（人間）</option>
          <option value="traveler_D">旅行者D（人間）</option>
        </select>
      </div>

      <div
        style={{
          border: "1px solid #ccc",
          padding: 12,
          height: 380,
          overflow: "auto",
        }}
      >
        {events.map((ev, i) => (
          <div key={i} style={{ marginBottom: 8 }}>
            <strong>[{ev.who}]</strong> {ev.content}
          </div>
        ))}
      </div>

      <textarea
        placeholder="複数行可。貼り付け→送信"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        rows={6}
      />
      <button onClick={send}>送信</button>

      <a
        href={`http://localhost:8000/session/log?session_id=${sessionId}`}
        target="_blank"
        rel="noreferrer"
      >
        Markdownログを開く
      </a>
    </main>
  );
}
