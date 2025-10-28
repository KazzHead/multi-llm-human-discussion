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
    def __init__(self, wishes_md: Optional[str] = None, model_mod="gpt-5", model_agent="gpt-4o-2024-08-06"):
        self.wishes_md = wishes_md or ""
        self.model_mod_name = model_mod
        self.model_agent_name = model_agent
        self.hio = HumanIO()
        self.messages: List[Message] = []
        self._team: Optional[RoundRobinGroupChat] = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def _build_sys(self, name_ja: str, public: List[str], private: List[str]) -> str:
        # 必要ならここをあなたの build_agent_system 相当で置換
        pub = "".join(f"・{p}" for p in public)
        prv = "".join(f"・{p}" for p in private)
        return (
            f"あなたは{name_ja}です。以下は公開希望:{pub}"
            "以下は非公開希望（口外禁止・直接言及禁止）:" + prv
        )

    async def _input_func_B(self, *args, **kwargs) -> str:
        return await self.hio.wait_input("traveler_B")

    async def _input_func_D(self, *args, **kwargs) -> str:
        return await self.hio.wait_input("traveler_D")

    async def stream_run(self):  # -> AsyncIterator[tuple[str, str]]
        # モデルクライアント
        moderator_mc = OpenAIChatCompletionClient(model=self.model_mod_name)
        agent_mc = OpenAIChatCompletionClient(model=self.model_agent_name)

        # 司会
        moderator = AssistantAgent(
            name="moderator",
            system_message=(
                "あなたは旅行計画の司会。順番を管理し要点整理のみ。"
                "全員が明確に賛成等を表明した直後のみ『【合意確定】』を単独出力。"
            ),
            model_client=moderator_mc,
        )

        # 旅行者A/C: AI
        agentA = AssistantAgent(name="traveler_A", system_message="あなたは旅行者A。", model_client=agent_mc)
        agentC = AssistantAgent(name="traveler_C", system_message="あなたは旅行者C。", model_client=agent_mc)

        # 旅行者B/D: 人間
        agentB = UserProxyAgent(name="traveler_B", input_func=self._input_func_B,
                                system_message="あなたは旅行者B（人間）。順番が来たら返信。")
        agentD = UserProxyAgent(name="traveler_D", input_func=self._input_func_D,
                                system_message="あなたは旅行者D（人間）。順番が来たら返信。")

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
                yield (who, content)  # ← (who, content) で返す

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
