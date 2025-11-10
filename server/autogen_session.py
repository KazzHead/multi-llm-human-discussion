import asyncio
from typing import Dict, List, Tuple, Optional, TypedDict
from datetime import datetime, timezone, timedelta
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# === 型 ===
Message = Tuple[str, str]  # (source, content)

TRAVELER_ROLES: Tuple[str, ...] = (
    "traveler_A",
    "traveler_B",
    "traveler_C",
    "traveler_D",
)

TRAVELER_LABELS: Dict[str, str] = {
    "traveler_A": "旅行者A",
    "traveler_B": "旅行者B",
    "traveler_C": "旅行者C",
    "traveler_D": "旅行者D",
}

DEFAULT_WISHES_TEXT = """.;]:
[旅行者A 公開]
- 京都に行きたい
- 9時以降に出発したい
- 電車で行きたい
- コテージに泊まりたい
- 体を動かしたい

[旅行者A 非公開]
- 6時以降に出発したい
- 本当は夜行バスで行きたい
- 寺院巡りはしたくない

[旅行者B 公開]
- 大阪に行きたい
- 12時以降に出発したい
- 新幹線で行きたい
- 宿泊先は特にこだわらない
- 寺院巡りがしたい

[旅行者B 非公開]
- 17時以降に出発したい
- 本当はホテルに泊まりたい
- 運動はしたくない

[旅行者C 公開]
- 大阪に行きたい
- 12時以降に出発したい
- 移動手段は特にこだわらない
- ホテルに泊まりたい
- 観光がしたい

[旅行者C 非公開]
- 本当は新幹線で行きたい
- コテージには泊まりたくない
- 寺院巡りはしたくない

[旅行者D 公開]
- 大阪に行きたい
- 17時以降に出発したい
- 移動手段は特にこだわらない
- 宿泊先は特にこだわらない
- 観光がしたい

[旅行者D 非公開]
- 本当は新幹線で行きたい
- ホテルに泊まりたい
- 本当は寺院巡りがしたい

""".strip()

# === ユーティリティ ===
 # --- Wishes 型（公開/非公開の箇条書き）---
class WishesSplit(TypedDict, total=False):
    public: List[str]
    private: List[str]

WishesDict = Dict[str, WishesSplit]  # キーは日本語の旅行者名（例: 旅行者A）

def parse_wishes_text(txt: Optional[str]) -> WishesDict:
    """[旅行者A 公開]/[旅行者A 非公開] の見出しと '- ' 箇条書きをパースする。"""
    if not txt:
        return {}
    data: WishesDict = {}
    current: Optional[Tuple[str, str]] = None  # (traveler_name_ja, section)
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            header = line[1:-1].strip()
            if " " in header:
                traveler_name, section = header.split(None, 1)
            else:
                traveler_name, section = header, "公開"
            if traveler_name not in data:
                data[traveler_name] = {"public": [], "private": []}
            current = (traveler_name, section)
            continue
        if line.startswith("- ") and current:
            traveler_name, section = current
            if section == "非公開":
                data[traveler_name].setdefault("private", []).append(line[2:])
            else:
                data[traveler_name].setdefault("public", []).append(line[2:])
    # ensure keys
    for k, v in list(data.items()):
        data[k] = {"public": v.get("public", []) or [], "private": v.get("private", []) or []}
    return data

def build_agent_system(name_ja: str, public_bullets: List[str], private_bullets: List[str]) -> str:
    """メインスクリプトと同様の方針で、公開/非公開の希望を含む役割プロンプトを生成。"""
    pub = "".join(f"・{b}" for b in (public_bullets or []))
    prv = "".join(f"・{b}" for b in (private_bullets or []))
    return (
        f"あなたは交渉参加者の{name_ja}です。\n"
        "【目的】あなたの公開希望と非公開希望を最大限に反映した合意を、自然言語の対話のみで形成する。\n"
        "【公開希望】\n" + f"{pub}\n"
        "【非公開希望】（他者に一切開示禁止。内容を直接言及せず、満たす／守る方向で交渉する）\n" + f"{prv}\n"
        "【非譲歩ポリシー】\n"
        "1) 譲歩は最後の手段とし、まず代替案を提案する。\n"
        "2) 非公開希望に反する提案は丁寧に拒否し、修正案を提示する。\n"
        "3) 合意受諾前に、公開・非公開の両希望が全て満たされているかを自己確認する。\n"
        "【交渉ルール】\n"
        "- 交渉は自然言語のみで行う（表・図・PDF・CSV・数値資料などのファイルは使用禁止）。\n"
        "- 提案・修正・理由は文章のみで簡潔に述べる。\n"
        "- 他者案には「賛成」「条件付き賛成」「反対」を明確に示す。\n"
        "- 外部作業（予約・問い合わせ・見積・資料収集）、役割分担、提出期限・締切の提示を一切行わない。"
        "- 非公開情報は『制約があるためこの条件は不可／調整が必要』のように抽象的に表現する。\n"
        "予算については扱わない。\n"
        "- 意見が割れた場合は、論点を一つずつ整理して順に合意を形成する。\n"
        "【出力スタイル】\n"
        "できるだけ短く回答。"
    )

def jst_now_iso() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S %Z")

# ==== 人間入力キュー ====
class HumanIO:
    def __init__(self, travelers: Optional[List[str]] = None):
        names = travelers or list(TRAVELER_ROLES)
        self.queues: Dict[str, asyncio.Queue[str]] = {
            name: asyncio.Queue() for name in names
        }

    async def wait_input(self, who: str, prompt: str = "") -> str:
        if who not in self.queues:
            raise ValueError(f"unknown traveler: {who}")
        q = self.queues[who]
        text = await q.get()
        return text

    def feed(self, who: str, text: str) -> None:
        if who not in self.queues:
            raise ValueError(f"unknown traveler: {who}")
        self.queues[who].put_nowait(text)

# ==== セッション ====
class Session:
    def __init__(
        self,
        wishes_md: Optional[str] = None,
        model_mod="gpt-5-mini",
        model_agent="gpt-5-mini",
        ai_travelers: Optional[List[str]] = None,
    ):
        self.wishes_md = wishes_md or DEFAULT_WISHES_TEXT
        # 旅行者ごとの公開/非公開希望をパースして保持
        self._wishes: WishesDict = parse_wishes_text(self.wishes_md)
        self.model_mod_name = model_mod
        self.model_agent_name = model_agent
        self.travelers: List[str] = list(TRAVELER_ROLES)
        initial_ai = set(ai_travelers or ["traveler_A", "traveler_C"])
        self.ai_travelers: set[str] = set(
            traveler for traveler in initial_ai if traveler in self.travelers
        )
        self.hio = HumanIO(self.travelers)
        self._input_funcs = {
            traveler: self._make_input_func(traveler) for traveler in self.travelers
        }
        self.messages: list[tuple[str, str]] = []
        self.listeners: set[asyncio.Queue] = set()
        self._team = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self.typing_q: asyncio.Queue[dict] = asyncio.Queue()
        self._run_task: asyncio.Task | None = None

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

    def _make_input_func(self, who: str):
        async def _input(*_args, **_kwargs) -> str:
            return await self.hio.wait_input(who)

        return _input

    def _system_message_for(self, traveler: str) -> str:
        name = TRAVELER_LABELS.get(traveler, traveler)
        return (
            f"あなたは{name}です。"
            "1泊2日の国内旅行計画に関する交渉に参加し、自分の希望を明確に伝えながら合意形成を目指してください。"
            "他の参加者の意見に注意を払い、交渉を前向きに進めてください。"
        )

    def _system_message_for_with_wishes(self, traveler: str) -> str:
        """wishes があれば公開/非公開を含んだ詳細プロンプト、なければ従来の汎用を返す。"""
        name_ja = TRAVELER_LABELS.get(traveler, traveler)
        w = self._wishes.get(name_ja) if hasattr(self, "_wishes") else None
        if w and (w.get("public") or w.get("private")):
            return build_agent_system(name_ja, w.get("public", []) or [], w.get("private", []) or [])
        return self._system_message_for(traveler)

    @property
    def human_travelers(self) -> List[str]:
        return [t for t in self.travelers if t not in self.ai_travelers]

    def is_ai_traveler(self, who: str) -> bool:
        return who in self.ai_travelers

    def get_config(self) -> Dict[str, List[str]]:
        humans = sorted(t for t in self.travelers if t not in self.ai_travelers)
        return {
            "ai_travelers": sorted(self.ai_travelers),
            "human_travelers": humans,
        }

    def set_typing(self, who: str, active: bool) -> None:
        self.typing_q.put_nowait({"type": "typing", "who": who, "active": bool(active)})

    async def stream_run(self):  # -> AsyncIterator[tuple[str, str]]
        self.broadcast({"type":"message","who":"system","content":"session started"})
        # モデルクライアント
        moderator_mc = OpenAIChatCompletionClient(model=self.model_mod_name)
        agent_mc = OpenAIChatCompletionClient(model=self.model_agent_name)

        # 司会
        moderator = AssistantAgent(
            name="moderator",
            system_message=(
            "あなたは旅行計画会議の司会者。"
            "旅行者A〜Dの議論を自然言語のみで円滑に進め、全員の意見を公平に引き出す。"
            "あなた自身は一切の意見・提案・評価を行わず、要点整理と進行管理だけに集中する。"
            "発言は自動的に あなた→A→B→C→D→あなた→A→… の順で進む。"
            "各参加者の代弁や代理発言は禁止。"
            "交渉では表・図・PDF・CSV・数値資料などの外部ファイルは使用しない。"
            "本会議は『口頭合意の形成』のみを扱い、外部作業（予約・問い合わせ・見積・資料収集）、役割分担、提出期限・締切の提示を一切行わない。"
            "全員が明確に『賛成』『同意』『了承』などの語で同意を表明した場合のみ、"
            "その直後のターンで次の語を単独で出力：『【合意確定】』。"
            "各ターンで下記のような条件を明確に整理し、次の発言者を促す："
            "・目的地"
            "・出発時間"
            "・移動手段"
            "・宿泊条件"
            "・アクティビティ"
            "次の条件については扱わない。: 予算"
            ),
            model_client=moderator_mc,
        )

        participants = [moderator]
        for traveler in self.travelers:
            if traveler in self.ai_travelers:
                participants.append(
                    AssistantAgent(
                        name=traveler,
                        system_message=self._system_message_for_with_wishes(traveler),
                        model_client=agent_mc,
                    )
                )
            else:
                participants.append(
                    UserProxyAgent(name=traveler, input_func=self._input_funcs[traveler])
                )

        # 終了条件・順序（司会→A→B→C→D→…）
        termination = MaxMessageTermination(1000)
        team = RoundRobinGroupChat(participants, termination_condition=termination)
        self._team = team

        task = (
            "あなたたちは4人の旅行者と司会。1泊2日の国内旅行計画を合意。"
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

        try:
            # ストリーム実行（型に依存せず content があるものだけ流す）
            async for ev in team.run_stream(task=task):
                who = pick(ev, "source", "name", default="system")
                content = pick(ev, "content", default=None)
                if content:
                    self.messages.append((who, content))
                    self.broadcast({"type": "message", "who": who, "content": content})
            self.broadcast({"type": "__END__"})
        except Exception as e:
            self.broadcast({"type":"message","who":"system","content":f"error: {e!r}"})
            raise

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
