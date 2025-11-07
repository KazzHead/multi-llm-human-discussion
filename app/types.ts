export type Role = "traveler_B" | "traveler_D";

export type ChatMessage = {
  who: string; // "moderator" | "traveler_A" | "traveler_B" | "traveler_C" | "traveler_D"
  content: string;
  ts?: number; // 任意: クライアント受信時刻
};

export type SSEMessageEvent = { type: "message"; who: string; content: string };
export type SSETypingEvent = { type: "typing"; who: string; active: boolean };
export type SSEEndEvent = { type: "__END__" };
export type SSEEvent = SSEMessageEvent | SSETypingEvent | SSEEndEvent;
