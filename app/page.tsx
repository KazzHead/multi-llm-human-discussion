"use client";
import React, { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

const TRAVELERS = [
  { id: "traveler_A", label: "旅行者A" },
  { id: "traveler_B", label: "旅行者B" },
  { id: "traveler_C", label: "旅行者C" },
  { id: "traveler_D", label: "旅行者D" },
] as const;

// const API_BASE = `http://${window.location.hostname}:8000`;
const API_BASE = "https://multi-llm-human-discussion.onrender.com";

type RoomItem = {
  id: string;
  started: boolean;
  config: { ai_travelers: string[]; human_travelers: string[] };
};

export default function Home() {
  const [sessionId, setSessionId] = useState<string>("");
  const [aiMap, setAiMap] = useState<Record<string, boolean>>({
    traveler_A: true,
    traveler_B: false,
    traveler_C: true,
    traveler_D: false,
  });
  const [rooms, setRooms] = useState<RoomItem[]>([]);
  const router = useRouter();

  const selectedAi = useMemo(
    () =>
      TRAVELERS.filter((t) => aiMap[t.id])
        .map((t) => t.id)
        .sort(),
    [aiMap]
  );

  const toggleTraveler = (id: (typeof TRAVELERS)[number]["id"]) => {
    setAiMap((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const create = async () => {
    if (!sessionId) return;
    await fetch(`${API_BASE}/session/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, ai_travelers: selectedAi }),
    });
    router.push(`/room/${sessionId}`);
  };

  const loadRooms = async () => {
    try {
      const res = await fetch(`${API_BASE}/session/list`);
      if (!res.ok) return;
      const data = (await res.json()) as { sessions: RoomItem[] };
      setRooms(data.sessions || []);
    } catch {}
  };

  const stopRoom = async (id: string) => {
    try {
      await fetch(`${API_BASE}/session/stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: id }),
      });
    } catch {}
    loadRooms();
  };

  useEffect(() => {
    loadRooms();
  }, []);

  return (
    <main style={{ padding: 24 }}>
      <h1>AgentChat 実験ロビー</h1>
      <input
        placeholder="セッションID"
        value={sessionId}
        onChange={(e) => setSessionId(e.target.value)}
      />
      <button onClick={create}>作成</button>
      <section style={{ marginTop: 24 }}>
        <h2 style={{ fontSize: 18 }}>参加者の役割設定</h2>
        <p style={{ marginBottom: 8 }}>
          チェックされた旅行者はAI、未チェックは人間として参加します。
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: 8,
            maxWidth: 480,
          }}
        >
          {TRAVELERS.map((t) => (
            <label
              key={t.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                border: "1px solid #ccc",
                borderRadius: 8,
                padding: "8px 12px",
              }}
            >
              <input
                type="checkbox"
                checked={!!aiMap[t.id]}
                onChange={() => toggleTraveler(t.id)}
              />
              <span>
                {t.label}
                <span style={{ fontSize: 12, color: "#666", marginLeft: 4 }}>
                  {aiMap[t.id] ? "AI" : "人間"}
                </span>
              </span>
            </label>
          ))}
        </div>
      </section>

      <section style={{ marginTop: 24 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <h2 style={{ fontSize: 18, margin: 0 }}>アクティブな部屋</h2>
          <button onClick={loadRooms}>更新</button>
        </div>
        {rooms.length === 0 ? (
          <p style={{ color: "#666" }}>現在アクティブな部屋はありません</p>
        ) : (
          <ul style={{ paddingLeft: 16 }}>
            {rooms.map((r) => (
              <li
                key={r.id}
                style={{
                  margin: "6px 0",
                  display: "flex",
                  gap: 8,
                  alignItems: "center",
                }}
              >
                <a
                  href={`/room/${encodeURIComponent(r.id)}`}
                  style={{ textDecoration: "underline" }}
                >
                  {r.id}
                </a>
                <span
                  style={{
                    fontSize: 12,
                    color: r.started ? "#059669" : "#6b7280",
                  }}
                >
                  {r.started ? "稼働中" : "停止"}
                </span>
                <button onClick={() => stopRoom(r.id)}>停止</button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <p>既存のIDがある場合は直接URL入力でもOK: /room/[id]</p>
    </main>
  );
}
