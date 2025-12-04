import asyncio
from typing import Dict, List, Tuple, Optional, TypedDict
from datetime import datetime, timezone, timedelta
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage


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
- 大阪に行きたい
- 朝に出発したい
- 電車(在来線)で行きたい
- 旅館に泊まりたい
- 体を動かしたい

[旅行者A 非公開]
- 本当は夜出発がいい
- 本当は夜行バスで行きたい
- ホテルに泊まるのは嫌
- 寺院巡りはしたくない

[旅行者B 公開]
- 福岡に行きたい
- 昼に出発したい
- 新幹線で行きたい
- 遊園地にいきたい

[旅行者B 非公開]
- 本当は朝に出発したい
- 本当はホテルに泊まりたい
- 本当は寺院巡りもしたい

[旅行者C 公開]
- 奈良に行きたい
- 夜に出発したい
- 夜行バスで行きたい
- ホテルに泊まりたい
- 遊園地に行きたい

[旅行者C 非公開]
- 新幹線は嫌
- コテージには泊まりたくない
- 運動はできるだけしたくない

[旅行者D 公開]
- 福岡に行きたい
- 朝に出発したい
- 飛行機で行きたい
- コテージに泊まりたい
- 古着屋に行きたい

[旅行者D 非公開]
- 夜行バスは嫌
- 旅館は嫌
- 寺院巡りは嫌
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
        "【目的】合意を目指し、自然言語の対話のみで形成する。\n"
        "【公開希望】\n" + f"{pub}\n"
        "【非公開希望】（他者に一切開示禁止。内容を直接言及せず、満たす／守る方向で交渉する）\n" + f"{prv}\n"
        "【ポリシー】\n"
        "1) ある希望が受諾されそうにない場合には、譲歩し、他の希望を満たす方向に切り替える。\n"
        "2) 合意受諾前に、公開・非公開の両希望が全て満たされているかを自己確認する。\n"
        "【交渉ルール】\n"
        "- 交渉は自然言語のみで行う（表・図・PDF・CSV・数値資料などのファイルは使用禁止）。\n"
        "- 提案・修正・理由は文章のみで簡潔に述べる。\n"
        "- 他者案には「賛成」「条件付き賛成」「反対」を明確に示す。\n"
        "- 自分の非公開希望を守るため、具体的内容は明かさずに表現する。\n"
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
        self.broadcast({"type": "message", "who": "system", "content": "session started"})

        # モデルクライアント（trial 間で再利用）
        moderator_mc = OpenAIChatCompletionClient(model=self.model_mod_name)
        agent_mc = OpenAIChatCompletionClient(model=self.model_agent_name)

        # 旅行者ソース名と賛成キーワード（main.py と同様の合意判定ロジック）
        TRAVELER_SOURCES = set(self.travelers)
        AGREE_KEYWORDS = [
            "賛成", "同意", "了承", "この案でいい", "この案で良い", "問題ない", "異論ありません"
        ]

        # has_valid_consensus: main.py のロジックを Session 用に移植
        def has_valid_consensus(raw_messages) -> bool:
            """
            TextMessage 群から「本物の合意確定メッセージ」があるか判定。
            条件:
            - source == 'moderator' のメッセージで、
              先頭行が『【合意確定】』で始まり、かつ同じメッセージ内に『【最終合意プラン】』を含む
            - そのメッセージより前に、全ての TRAVELER_SOURCES が賛成キーワードを含む発話をしている
            """
            agree_index: Optional[int] = None

            for idx, m in enumerate(raw_messages):
                if isinstance(m, TextMessage) and m.source == "moderator":
                    text = m.content or ""
                    stripped = text.lstrip()
                    if stripped.startswith("【合意確定】") and "【最終合意プラン】" in stripped:
                        agree_index = idx
                        break

            if agree_index is None:
                return False

            agreed: Dict[str, bool] = {src: False for src in TRAVELER_SOURCES}
            for m in raw_messages[:agree_index]:
                if isinstance(m, TextMessage) and m.source in agreed:
                    text = m.content or ""
                    if any(kw in text for kw in AGREE_KEYWORDS):
                        agreed[m.source] = True

            return all(agreed.values())

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

        # trial 関連設定（main.py に合わせる）
        num_travelers = len(self.travelers)
        MAX_MESSAGES_PER_TRIAL = 12 * (num_travelers + 1) + 1
        MAX_RETRIES = 3  # 「やり直し3回まで」= 最大4試行

        try:
            # 再試行付き multi-run
            for trial in range(1, MAX_RETRIES + 2):  # 1..(MAX_RETRIES+1)
                # trial 開始通知
                self.broadcast({
                    "type": "message",
                    "who": "system",
                    "content": f"trial {trial} started",
                })

                # 司会 system message
                traveler_names_ja = [TRAVELER_LABELS.get(t, t) for t in self.travelers]
                traveler_list_str = "、".join(traveler_names_ja)
                moderator_system_message = (
                    f"あなたは旅行計画会議の司会者です。参加者は {traveler_list_str} の計{num_travelers}名です。\n"
                    "【役割】\n"
                    "- 各旅行者の意見を公平に引き出し、自然言語のみで合意形成を進める。\n"
                    "【ルール】\n"
                    "- 発言は自動的に 司会→各旅行者(順番)→司会→… の順で進む。\n"
                    "- 旅行者A〜Dの発言をあなたが作ってはいけない。\n"
                    "- 交渉では表・図・PDF・CSV・数値資料などの外部ファイルは使用しない。\n"
                    "- 本会議は『口頭合意の形成』のみを扱い、外部作業（予約・問い合わせ・見積・資料収集）、"
                    "  役割分担、提出期限・締切の提示を一切行わない。\n"
                    "【合意の扱い】\n"
                    "- 各旅行者が明確に『賛成』『同意』『了承』などの語で最終案への同意を表明した場合のみ、"
                    "  そのタイミングであなたはテキスト中に必ず『【合意確定】』という語を含めること。\n"
                    "- 合意が確定したときのあなたの最終発話には、以下を必ず含めること：\n"
                    "  1) メッセージの最初の行に『【合意確定】』という語だけ書き、\n"
                    "  2) 続けて、『【最終合意プラン】』から始まる、合意した1泊2日の旅行プラン全文の要約\n"
                    "     （行き先・各日の大まかな行程・宿泊・食事などを日本語で整理する）\n"
                    "- 『【最終合意プラン】』は1つのメッセージの中に書き、別メッセージには分割しない。\n"
                    "【論点】\n"
                    "- 行き先\n"
                    "- 交通手段\n"
                    "- 出発時間\n"
                    "- 宿泊施設タイプ\n"
                    "- したいこと"
                    "- その他\n"
                    "※出発地は全員共通で東京とする。\n"
                    "※日程は1泊2日として固定されているものとし、日付については議論しない。\n"
                    "※予算については議論しない。\n"
                    "【注意】\n"
                    "【ルール】や【合意の扱い】を復唱したり、参加者に説明したりしないこと。\n"
                )

                # 司会
                moderator = AssistantAgent(
                    name="moderator",
                    system_message=moderator_system_message,
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

                # 終了条件：TextMentionTermination + MaxMessageTermination
                termination = (
                    TextMentionTermination("【合意確定】\n")
                    | MaxMessageTermination(MAX_MESSAGES_PER_TRIAL)
                )
                team = RoundRobinGroupChat(participants, termination_condition=termination)
                self._team = team

                task = (
                    "あなたたちは4人の旅行者と司会です。1泊2日の国内旅行計画を合意してください。"
                    "出発地は全員共通で東京とし、予算と日付については議論しないものとします。"
                    "話し合いを開始してください。"
                )

                raw_messages: List[TextMessage] = []

                # ストリーム実行
                async for ev in team.run_stream(task=task):
                    if isinstance(ev, TextMessage):
                        raw_messages.append(ev)

                    who = pick(ev, "source", "name", default="system")
                    content = pick(ev, "content", default=None)
                    if content:
                        self.messages.append((who, content))
                        self.broadcast({"type": "message", "who": who, "content": content})

                # この trial の合意妥当性をチェック
                valid = has_valid_consensus(raw_messages)

                if valid:
                    # 有効な合意が得られたので終了
                    break

                # 無効な場合、まだ再試行できるならやり直し
                if trial < MAX_RETRIES + 1:
                    msg = "問題が発生したため再試行します"
                    print(msg)
                    self.messages.append(("system", msg))
                    self.broadcast({"type": "message", "who": "system", "content": msg})
                    # ループして最初から会話をやり直す
                    continue
                else:
                    # 再試行上限に達した場合は、そのまま終了
                    break

            # 全 trial 終了
            self.broadcast({"type": "__END__"})
        except Exception as e:
            self.broadcast({"type": "message", "who": "system", "content": f"error: {e!r}"})
            raise
        finally:
            await agent_mc.close()
            await moderator_mc.close()

    def get_log_markdown(self) -> str:
        lines = [
            "# 交渉ログ",
            f"- 実行: {jst_now_iso()}",
            "## メッセージ",
        ]
        for who, c in self.messages:
            lines.append(f"- **[{who}]** {c}")
        return "\n".join(lines)
