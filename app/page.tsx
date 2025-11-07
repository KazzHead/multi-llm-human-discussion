"use client";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

const TRAVELERS = [
  { id: "traveler_A", label: "旅行者A" },
  { id: "traveler_B", label: "旅行者B" },
  { id: "traveler_C", label: "旅行者C" },
  { id: "traveler_D", label: "旅行者D" },
] as const;

const API_BASE = `http://${window.location.hostname}:8000`;
// const API_BASE = "https://multi-llm-human-discussion.onrender.com";

export default function Home() {
  const [sessionId, setSessionId] = useState<string>("");
  const [aiMap, setAiMap] = useState<Record<string, boolean>>({
    traveler_A: true,
    traveler_B: false,
    traveler_C: true,
    traveler_D: false,
  });
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
      body: JSON.stringify({
        session_id: sessionId,
        ai_travelers: selectedAi,
      }),
    });
    router.push(`/room/${sessionId}`);
  };

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
          チェックされた旅行者はAIエージェント、未チェックは人間として参加します。
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
      <p>既存のIDがある場合は直接URL入力でもOK: /room/[id]</p>
    </main>
  );
}
