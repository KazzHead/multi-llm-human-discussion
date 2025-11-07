import asyncio
from typing import Dict, List, Tuple, Optional, AsyncIterator
from datetime import datetime, timezone, timedelta
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# === 型 ===
Message = Tuple[str, str]  # (source, content)

# === ユーティリティ ===
def jst_now_iso() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S %Z")

# ==== 人間入力キュー（B/D別） ====
class HumanIO:
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue[str]] = {
            "traveler_B": asyncio.Queue(),
            "traveler_D": asyncio.Queue(),
        }

    async def wait_input(self, who: str, prompt: str = "") -> str:
        q = self.queues[who]
        text = await q.get()
        return text

    def feed(self, who: str, text: str) -> None:
        self.queues[who].put_nowait(text)

# ==== セッション ====
class Session:
    def __init__(self, wishes_md: Optional[str] = None, model_mod="gpt-5-mini", model_agent="gpt-5-mini"):
        self.wishes_md = wishes_md or ""
        self.model_mod_name = model_mod
        self.model_agent_name = model_agent
        self.hio = HumanIO()
        self.messages: list[tuple[str, str]] = []
        self.listeners: set[asyncio.Queue] = set()
        self._team = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self.typing_q: asyncio.Queue[dict] = asyncio.Queue()
        self._run_task: asyncio.Task | None = None
        self.typing_q: asyncio.Queue[dict] = asyncio.Queue()

    def add_listener(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.listeners.add(q)
        return q

    def remove_listener(self, q: asyncio.Queue) -> None:
        self.listeners.discard(q)

    def broadcast(self, ev: dict) -> None:
        for q in list(self.listeners):
            try:
                q.put_nowait(ev)
            except Exception:
                pass

    def _build_sys(self, name_ja: str, public: List[str], private: List[str]) -> str:
        # 必要ならここをあなたの build_agent_system 相当で置換
        pub = "".join(f"・{p}" for p in public)
        prv = "".join(f"・{p}" for p in private)
        return (
        f"あなたは交渉参加者の{name_ja}です。"
        +"以下はあなたの『公開してもよい希望』です。"
        +f"{pub}"
        +"あなたの役割は、自分の希望を最大限に反映した合意に到達するため、交渉で一貫して自己主張することです。"
        +"以下はあなたの『非公開の希望』です（内容を直接言及してはいけません。満たす/守る方向で交渉してください）。"
        +f"{prv}"
        +"【非譲歩ポリシー（強化版）】"
        +"1) 譲歩は一切禁止。自分の希望が満たされない案には同意しない。"
        +"2) 非公開希望は絶対条件として扱い、これに反する提案は必ず拒否する。"
        +"3) 代替案があっても満たされない限り同意しない。"
        +"4) 『まあ良い』『仕方ない』など曖昧な表現は禁止。理由を明確にして断固とした主張を行う。"
        +"5) 司会者に促されても、希望を変えたり緩和してはならない。"
        +"6) 合意文を受け入れる前に、すべての希望が満たされているか厳密に自己確認する。"
        )

    async def _input_func_B(self, *args, **kwargs) -> str:
        return await self.hio.wait_input("traveler_B")

    async def _input_func_D(self, *args, **kwargs) -> str:
        return await self.hio.wait_input("traveler_D")

    def set_typing(self, who: str, active: bool) -> None:
        self.typing_q.put_nowait({"type": "typing", "who": who, "active": bool(active)})

    async def stream_run(self):  # -> AsyncIterator[tuple[str, str]]
        # モデルクライアント
        moderator_mc = OpenAIChatCompletionClient(model=self.model_mod_name)
        agent_mc = OpenAIChatCompletionClient(model=self.model_agent_name)

        # 司会
        moderator = AssistantAgent(
            name="moderator",
            system_message=(
                "あなたは旅行計画会議の司会者です。"
                "旅行者A-Dの議論を円滑に進め、全員の意見を公平に引き出してください。"
                "あなた自身は意見を述べず、発言の順番を管理し、要点を簡潔に整理することに集中します。"
                "他の参加者の代弁や代理発言をしてはいけません。"
                "全員が明確に『賛成』『同意』『了承』などと表明した場合のみ、"
                "その直後のターンで次の語を単独で出力してください：『【合意確定】』。"
                "それ以外のタイミングではこの語を絶対に出力してはいけません。"
            ),
            model_client=moderator_mc,
        )

        # 旅行者A/C: AI
        agentA = AssistantAgent(name="traveler_A", system_message="あなたは旅行者A。", model_client=agent_mc)
        agentC = AssistantAgent(name="traveler_C", system_message="あなたは旅行者C。", model_client=agent_mc)

        # 旅行者B/D: 人間
        agentB = UserProxyAgent(name="traveler_B", input_func=self._input_func_B)
        agentD = UserProxyAgent(name="traveler_D", input_func=self._input_func_D)

        # 終了条件・順序（司会→A→B→C→D→…）
        termination = TextMentionTermination("【合意確定】") | MaxMessageTermination(50)
        team = RoundRobinGroupChat([moderator, agentA, agentB, agentC, agentD],termination_condition=termination)
        self._team = team

        task = (
            "あなたたちは4人の旅行者と司会。2泊3日の国内旅行計画を合意。"
            "できるだけ詳細に予算と各日程をまとめてください。"
            "話し合いを開始。"
        )

        # イベントから値を安全に取り出すヘルパ
        def pick(obj, *keys, default=None):
            # dict形式
            if isinstance(obj, dict):
                for k in keys:
                    if k in obj and obj[k]:
                        return obj[k]
                return default
            # 属性形式
            for k in keys:
                try:
                    v = getattr(obj, k)
                    if v:
                        return v
                except Exception:
                    pass
            # dataサブ構造
            try:
                data = getattr(obj, "data", None)
                if isinstance(data, dict):
                    for k in keys:
                        if k in data and data[k]:
                            return data[k]
            except Exception:
                pass
            return default

        # ストリーム実行（型に依存せず content があるものだけ流す）
        async for ev in team.run_stream(task=task):
            who = pick(ev, "source", "name", default="system")
            content = pick(ev, "content", default=None)
            if content:
                self.messages.append((who, content))
                self.broadcast({"type": "message", "who": who, "content": content})
        self.broadcast({"type": "__END__"})

        await agent_mc.close()


    def get_log_markdown(self) -> str:
        lines = [
            "# 交渉ログ",
            f"- 実行: {jst_now_iso()}",
            "## メッセージ",
        ]
        for who, c in self.messages:
            lines.append(f"- **[{who}]** {c}")
        return "\n".join(lines)
