"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function Home() {
  const [sessionId, setSessionId] = useState<string>("");
  const router = useRouter();

  const create = async () => {
    if (!sessionId) return;
    await fetch("http://localhost:8000/session/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
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
      <p>既存のIDがある場合は直接URL入力でもOK: /room/[id]</p>
    </main>
  );
}
