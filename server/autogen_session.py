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

- 自然の多い場所（海や山）に行きたい
- カヌーやハイキングなど体を動かす体験をしたい
- ご当地の郷土料理を食べたい
- 温泉に入りたい
- 移動は新幹線か特急
- 出発は午前中
- 夜は全員で一緒に食事

[旅行者A 非公開]

- 片道4時間以内
- 宿泊費は1泊6,000円以内
- 乗換えは少なくしたい
- 混雑する観光地は避けたい
- 和室で眠りたい

[旅行者B 公開]

- ご当地グルメやお酒を楽しみたい
- 温泉付きの宿
- 有名な観光名所を少し回りたい
- 景観の良い場所で写真を撮りたい
- 宿はできれば個室

[旅行者B 非公開]

- 移動は片道3時間以内
- 高速バスは不可
- 徒歩観光は1時間以内
- 夜は早めに切り上げたい
- 宿は簡素すぎる場所は避けたい

[旅行者C 公開]

- 観光は少しだけできれば満足
- 高価なアクティビティは不要
- 全員と一緒に過ごすことを最優先
- お土産は小さなもので良い
- 宿は質素でOK
- 食事は安い定食屋で十分

[旅行者C 非公開]

- 移動は片道2時間以内
- 高級料理は避けたい
- 混雑する観光地は避けたい
- 夜は静かに過ごしたい
- 宿泊は全員同じ宿にしたい
- 旅行費用は2万円以内

[旅行者D 公開]

- 歴史的な街並みや文化遺産を巡りたい
- 地元の伝統工芸や体験教室に参加したい
- 美術館や博物館にも行きたい
- 落ち着いた雰囲気の宿に泊まりたい
- 食事は地元の老舗店で味わいたい
- 朝はゆっくり出発したい

[旅行者D 非公開]

- 予算は交通費込みで3万円以内
- 宿泊は静かで清潔な場所を希望
- 団体行動より少人数行動を好む
- 食事に待ち時間が長い店は避けたい
- 移動中に読書できる時間がほしい

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
        f"あなたは交渉参加者の{name_ja}です。"
        "あなたの役割は、あなたの公開希望と非公開希望を最大限に反映した合意に到達するため、交渉で一貫して自己主張することです。"
        "回答は出来るだけ簡潔に、短く"
        "以下はあなたの『公開してもよい希望』です。"
        f"{pub}"
        "以下はあなたの『非公開の希望』です（内容を直接言及してはいけません。満たす/守る方向で交渉してください）。"
        f"{prv}"
        "【非譲歩ポリシー】"
        "1) 譲歩幅は最小限に、代替案の提示を優先し、希望の放棄を避ける。"
        "2) あなたの非公開希望に反する提案は丁寧に拒否し別案を要求する。"
        "3) 合意案を受け入れる前に、すべての希望が満たされているか厳密に自己確認する。"
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
            "2泊3日の国内旅行計画に関する交渉に参加し、自分の希望を明確に伝えながら合意形成を目指してください。"
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
                "あなたは旅行計画会議の司会者です。"
                "旅行者A-Dの議論を円滑に進め、全員の意見を公平に引き出してください。"
                "あなた自身は意見を述べず、発言の順番を管理し、要点を簡潔に整理することに集中します。"
                "他の参加者の代弁や代理発言をしてはいけません。"
                "交渉では表・図・PDF・CSV・数値資料などの外部ファイルは使用せず、自然言語のやりとりのみを扱います。"
                "ルールや方針の説明は不要です。"
                "全員が明確に『賛成』『同意』『了承』などと表明した場合のみ、"
                "その直後のターンで次の語を単独で出力してください：『【合意確定】』。"
                "それ以外のタイミングではこの語を絶対に出力してはいけません。"
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
