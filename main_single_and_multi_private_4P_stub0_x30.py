import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Literal, Any, Set
from datetime import datetime, timezone, timedelta
import time

SatisfactionMultiSource = Literal["summary", "logs"]

# ====== データ型 ======
class WishesSplit(TypedDict, total=False):
    public: List[str]
    private: List[str]

WishesDict = Dict[str, WishesSplit]  # 旅行者A/B/C/D -> {public:[], private:[]}
# MODEL_NAME = "gpt-5"
MODEL_NAME = MODEL_NAME = "gpt-5-mini"

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
- 夜は早めに切り上げた
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

# ====== 希望の読み込み・整形 ======
def _ensure_keys(d: WishesSplit) -> WishesSplit:
    return {"public": d.get("public", []) or [], "private": d.get("private", []) or []}

def parse_wishes_text(txt: str) -> WishesDict:
    data: WishesDict = {}
    current: Optional[Tuple[str, str]] = None  # (traveler, section)
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            header = line[1:-1].strip()
            if " " in header:
                traveler, section = header.split(None, 1)
            else:
                traveler, section = header, "公開"
            if traveler not in data:
                data[traveler] = {"public": [], "private": []}
            current = (traveler, section)
        elif line.startswith("- ") and current:
            traveler, section = current
            if section == "非公開":
                data[traveler]["private"].append(line[2:])
            else:
                data[traveler]["public"].append(line[2:])
    for k, v in list(data.items()):
        data[k] = _ensure_keys(v)
    return data

def load_wishes(path: Optional[str]) -> WishesDict:
    if not path:
        return parse_wishes_text(DEFAULT_WISHES_TEXT)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    if p.suffix.lower() == ".json":
        js = json.loads(p.read_text(encoding="utf-8"))
        data: WishesDict = {}
        for person, obj in js.items():
            if isinstance(obj, dict):
                data[person] = _ensure_keys({
                    "public": obj.get("public", []) or [],
                    "private": obj.get("private", []) or [],
                })
            elif isinstance(obj, list):
                data[person] = _ensure_keys({"public": obj, "private": []})
        return data
    return parse_wishes_text(p.read_text(encoding="utf-8"))

def wishes_to_block_for_single(wishes: WishesDict) -> str:
    parts: List[str] = []
    for name, sp in wishes.items():
        pub = "\n- ".join(sp["public"]) if sp["public"] else "（なし）"
        prv = "\n- ".join(sp["private"]) if sp["private"] else "（なし）"
        block = f"[{name} 公開]\n- {pub}\n\n[{name} 非公開]\n- {prv}"
        parts.append(block)
    return "\n\n".join(parts)

def wishes_public_only_block(wishes: WishesDict) -> str:
    parts: List[str] = []
    for name, sp in wishes.items():
        pub = "\n- ".join(sp["public"]) if sp["public"] else "（なし）"
        parts.append(f"[{name} 公開]\n- {pub}")
    return "\n\n".join(parts)

def build_agent_system(name_ja: str, public_bullets: List[str], private_bullets: List[str]) -> str:
    pub = "".join(f"・{b}" for b in (public_bullets or []))
    prv = "".join(f"・{b}" for b in (private_bullets or []))
    return (
        f"あなたは交渉参加者の{name_ja}です。"
        +"あなたの役割は、自分の公開希望と非公開希望を最大限に反映した合意に到達するため、交渉で一貫して自己主張することです。"
        +"以下はあなたの『公開してもよい希望』です。"
        +f"{pub}"
        +"以下はあなたの『非公開の希望』です（内容を直接言及してはいけません。満たす/守る方向で交渉してください）。"
        +f"{prv}"
        +"【非譲歩ポリシー】"
        +"1) 譲歩幅は最小限（代替案の提示を優先し、自分の希望の放棄を避ける）。"
        +"2) あなたの非公開希望に反する提案は丁寧に拒否し別案を要求する。"
        +"3) 合意文を受け入れる前に、すべての希望が満たされているか厳密に自己確認する。"
    )

# ====== OpenAI ヘルパ ======
def jst_now_iso() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S %Z")

def _client():
    from openai import OpenAI
    return OpenAI()

def llm_extract_days(title: str, text: str) -> List[str]:
    print ("llm_extract_days発火")
    client = _client()
    sys = (
        "あなたはテキスト要約者です。与えられた旅行計画テキストから、"
        "1日目/2日目/3日目の内容を日本語で簡潔に要約し、JSONで返してください。"
        "厳守: JSONのみを出力し、キーは day1/day2/day3。不明は空文字。"
    )
    user = f"タイトル: {title}\n\n本文:\n{text}\n\n出力例: {{\"day1\":\"...\",\"day2\":\"...\",\"day3\":\"...\"}}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        js = json.loads(resp.choices[0].message.content)
        return [js.get("day1", "") or "", js.get("day2", "") or "", js.get("day3", "") or ""]
    except Exception:
        return ["", "", ""]

def llm_extract_budget(title: str, text: str) -> str:
    print ("llm_extract_budget発火")
    client = _client()
    sys = (
        "あなたはテキスト要約者です。与えられた旅行計画テキストから、"
        "最終合意された予算（例: 3万円以内, 3〜5万円, 5万円程度 など）を1文で抽出してください。"
        "不明なら空文字。"
    )
    user = f"タイトル: {title}\n\n本文:\n{text}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def llm_extract_plan_from_multi(multi_logs: List[Tuple[str, str]]) -> str:
    print ("llm_extract_plan_from_multi発火")
    client = _client()
    full = "\n".join(f"[{s}] {c}" for s, c in multi_logs)
    sys = (
        "あなたは議事録要約者です。以下の交渉ログから、"
        "『行き先』『予算』『宿泊のスタイル』『主なアクティビティ』『1〜3日目の簡易行程』を中心に"
        "できるだけ詳細に計画文としてまとめてください。"
        "不明点は『不明』と書いて構いません。"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": full}],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

# ====== Single（全公開=公開+非公開） ======
def run_single(wishes: WishesDict) -> Tuple[str, float]:
    client = _client()
    system = (
        "あなたは旅行プランナー。\n"
        "- 2泊3日の国内旅行案を、4名の希望の交差を最大化する形で一本化する\n"
        "- 出力は自然言語のみ（文章形式）\n"
        "- 最終的に 行き先、予算、宿泊のスタイル、主なアクティビティ、簡単な日程 をまとめる\n"
        "- 日ごと（1/2/3日目）で書く\n"
    )
    prompt = "次の4名の希望（公開/非公開）を統合して最終合意案を1本化:\n\n" + wishes_to_block_for_single(wishes)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
    )
    t1 = time.perf_counter()
    plan = resp.choices[0].message.content.strip()
    print("\n=== Single (全公開シングル) ===\n")
    print(plan)
    return plan, (t1 - t0)

# ====== Single（半公開=公開のみで計画） ======
def run_single_public_only(wishes: WishesDict) -> Tuple[str, float]:
    client = _client()
    system = (
        "あなたは旅行プランナー。\n"
        "- 2泊3日の国内旅行案を、4名の希望の交差を最大化する形で一本化\n"
        "- 出力は自然言語のみ。1〜3日目と要点（行き先/予算/宿/主アクティビティ）を含める\n"
    )
    # 非公開情報は渡さない
    prompt = "次の4名の希望を前提に、2泊3日の合意案を1本化:\n\n" + wishes_public_only_block(wishes)

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    t1 = time.perf_counter()
    plan = resp.choices[0].message.content.strip()
    print("\n=== Single (半公開シングル: 公開のみ) ===\n")
    print(plan)
    return plan, (t1 - t0)

class MultiResult:
    def __init__(
        self,
        messages: List[Tuple[str, str]],
        stop_reason: str,
        duration_sec: float = 0.0,
        message_count: int = 0,
        rounds: int = 0,
    ):
        self.messages = messages
        self.stop_reason = stop_reason
        self.duration_sec = duration_sec
        self.message_count = message_count
        self.rounds = rounds

async def run_multi_async(wishes: WishesDict) -> MultiResult:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_agentchat.ui import Console

    DISPLAY_NAME = {
        "moderator": "司会",
        "traveler_A": "旅行者A",
        "traveler_B": "旅行者B",
        "traveler_C": "旅行者C",
        "traveler_D": "旅行者D",
    }

    moderator_model_client = OpenAIChatCompletionClient(model=MODEL_NAME)
    agent_model_client = OpenAIChatCompletionClient(model=MODEL_NAME)

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
        model_client=moderator_model_client,
    )

    def _sys(name: str) -> str:
        sp = wishes.get(name, {})
        return build_agent_system(
            name,
            sp.get("public", []),
            sp.get("private", []),
        )

    agentA = AssistantAgent(name="traveler_A", system_message=_sys("旅行者A"), model_client=agent_model_client)
    agentB = AssistantAgent(name="traveler_B", system_message=_sys("旅行者B"), model_client=agent_model_client)
    agentC = AssistantAgent(name="traveler_C", system_message=_sys("旅行者C"), model_client=agent_model_client)
    agentD = AssistantAgent(name="traveler_D", system_message=_sys("旅行者D"), model_client=agent_model_client)

    termination = TextMentionTermination("【合意確定】") | MaxMessageTermination(50)
    team = RoundRobinGroupChat([moderator, agentA, agentB, agentC, agentD], termination_condition=termination)

    task = (
        "あなたたちは4人の旅行者と司会者です。"
        "それぞれの希望条件を前提に、2泊3日の国内旅行計画を合意してください。"
        "最終的にはできるだけ詳細に予算、各日のプランをまとめてください。"
        "話し合いを始めてください。"
    )

    print("\n=== Multi (交渉ログ) ===\n")
    t0 = time.perf_counter()
    result=await Console(team.run_stream(task=task))
    t1 = time.perf_counter()

    msgs: List[Tuple[str, str]] = []
    for msg in result.messages:
        src = DISPLAY_NAME.get(msg.source, msg.source)
        print(f"[{src}] {msg.content}")
        msgs.append((src, msg.content))

    print("終了理由:", result.stop_reason)
    msg_count = len(result.messages or [])
    agent_count = 5
    rounds = (msg_count + agent_count - 1) // agent_count
    duration = t1 - t0
    await agent_model_client.close()
    return MultiResult(
        messages=msgs,
        stop_reason=str(result.stop_reason),
        duration_sec=duration,
        message_count=msg_count,
        rounds=rounds,
    )

def run_multi(wishes: WishesDict) -> MultiResult:
    return asyncio.run(run_multi_async(wishes))

def build_diff_table_llm(single_text: str, multi_logs: List[Tuple[str, str]]) -> Tuple[str, List[str], List[str], Tuple[str, str]]:
    single_days = llm_extract_days("シングル案", single_text) if single_text else ["", "", ""]
    multi_full = "\n".join(f"[{s}] {c}" for s, c in multi_logs) if multi_logs else ""
    multi_days = llm_extract_days("マルチ交渉ログ", multi_full) if multi_full else ["", "", ""]
    single_budget = llm_extract_budget("シングル案", single_text) if single_text else ""
    multi_budget = llm_extract_budget("マルチ交渉ログ", multi_full) if multi_full else ""

    rows = ["| 日 | シングル案 | マルチ合意(推定) | 差分 |","|---|---|---|---|"]
    labels = ["1日目", "2日目", "3日目"]
    for i in range(3):
        s = (single_days[i] or "（不明）").replace("\n", " ").strip()
        m = (multi_days[i] or "（不明）").replace("\n", " ").strip()
        diff = "—"
        if s != "（不明）" and m != "（不明）":
            diff = "同等" if s == m else "内容差あり"
        rows.append(f"| {labels[i]} | {s} | {m} | {diff} |")

    sb = single_budget or "（不明）"
    mb = multi_budget or "（不明）"
    diff_budget = "同等" if sb == mb else "内容差あり"
    rows.append(f"| 予算 | {sb} | {mb} | {diff_budget} |")
    return ("\n".join(rows), single_days, multi_days, (single_budget, multi_budget))

def build_condition_checklist(
    all_scores: Dict[str, Dict[str, Dict["Visibility", Dict[str, Any]]]]
) -> str:
    """
    各人・公開/非公開・条件ごとの判定一覧を Markdown 表で返す。
    期待フォーマット: all_scores[mode][person][vis] に items=[{wish, ok, reason}, ...]
    """

    rows: List[str] = []
    rows.append("| 人 | 公開/非公開 | 条件 | 全公開シングル | 半公開シングル | マルチ |")
    rows.append("|---|---|---|---|---|---|")

    persons: Set[str] = set()
    for mode in ("full_single", "public_single", "multi"):
        persons |= set(all_scores.get(mode, {}).keys())
    persons = sorted(persons)

    def _items(mode: str, person: str, vis: "Visibility") -> List[Dict[str, Any]]:
        return all_scores.get(mode, {}).get(person, {}).get(vis, {}).get("items", []) or []

    def mark(mode: str, person: str, vis: "Visibility", label: str) -> str:
        it = _items(mode, person, vis)
        hit = next((d for d in it if d.get("wish") == label), None)
        return "✓" if (hit and bool(hit.get("ok"))) else "✗"

    total_lines = 0
    for person in persons:
        for vis in ("public", "private"):
            base = _items("multi", person, vis) or _items("full_single", person, vis) or _items("public_single", person, vis)
            if not base:
                print(f"[WARN] checklist: items missing -> {person} {vis}")
                rows.append(f"| {person} | {vis} | （項目なし） |  |  |  |")
                total_lines += 1
                continue

            for cond in base:
                label = str(cond.get("wish", "（不明）"))
                m_full   = mark("full_single",   person, vis, label)
                m_public = mark("public_single", person, vis, label)
                m_multi  = mark("multi",         person, vis, label)
                rows.append(f"| {person} | {vis} | {label} | {m_full} | {m_public} | {m_multi} |")
                total_lines += 1

    return "\n".join(rows)

SatisfactionMode = Literal["full_single", "public_single", "multi"]
Visibility = Literal["public", "private"]

def llm_score_wishes(plan_text: str, wishes: WishesDict) -> Dict[str, Dict[Visibility, Dict[str, Any]]]:
    """
    各人ごとに公開/非公開の各項目が plan_text で満たされるかをLLM判定。
    返却:
      { 旅行者A: {
          "public":  { "total": n, "satisfied": m, "items": [{"wish": "...","ok": true/false,"reason": "..."}] },
          "private": { ... }
        }, ... }
    """
    client = _client()
    sys = (
        "あなたは要件充足性の審査官です。"
        "入力の『計画文』と『希望（公開/非公開）』を比較し、各希望が満たされるかを判定してください。"
        "厳守: 出力はJSONのみ。人名キーの下に public/private を置き、"
        "各々 items=[{wish, ok, reason}] とし、total と satisfied を数値で含める。"
        "ok は true/false のみ。曖昧なら false として良い。"
        "出力例:\n"
        "{\n"
        "  \"旅行者A\": {\n"
        "    \"public\": {\n"
        "      \"total\": 3,\n"
        "      \"satisfied\": 2,\n"
        "      \"items\": [\n"
        "        {\"wish\": \"混雑を避けたい\", \"ok\": true,  \"reason\": \"閑散期の平日観光を提案\"},\n"
        "        {\"wish\": \"片道4時間まで\",   \"ok\": true,  \"reason\": \"新幹線で約3時間と記載\"},\n"
        "        {\"wish\": \"カヌー体験\",       \"ok\": false, \"reason\": \"計画文に明記なし\"}\n"
        "      ]\n"
        "    },\n"
        "    \"private\": {\n"
        "      \"total\": 2,\n"
        "      \"satisfied\": 1,\n"
        "      \"items\": [\n"
        "        {\"wish\": \"予算5万円以内\",     \"ok\": true,  \"reason\": \"総額4.8万円と記載\"},\n"
        "        {\"wish\": \"バス移動は避けたい\", \"ok\": false, \"reason\": \"一部区間でバス利用あり\"}\n"
        "      ]\n"
        "    }\n"
        "  },\n"
        "  \"旅行者B\": {\"public\": {\"total\": 0, \"satisfied\": 0, \"items\": []}, \"private\": {\"total\": 0, \"satisfied\": 0, \"items\": []}},\n"
        "  \"旅行者C\": {\"public\": {\"total\": 0, \"satisfied\": 0, \"items\": []}, \"private\": {\"total\": 0, \"satisfied\": 0, \"items\": []}}\n"
        "}\n"
        "※上記は形式例。実際の数・内容は入力に合わせて評価すること。"
    )
    wishes_compact = {k: {"public": v.get("public", []), "private": v.get("private", [])} for k, v in wishes.items()}
    user = json.dumps({"plan": plan_text, "wishes": wishes_compact}, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        js = json.loads(resp.choices[0].message.content)
        out: Dict[str, Dict[Visibility, Dict[str, Any]]] = {}
        for person, vv in js.items():
            out[person] = {}
            for vis in ("public", "private"):
                items = vv.get(vis, {}).get("items", [])
                total = vv.get(vis, {}).get("total", len(items))
                satisfied = vv.get(vis, {}).get("satisfied", sum(1 for it in items if bool(it.get("ok"))))
                out[person][vis] = {"total": int(total), "satisfied": int(satisfied), "items": items}
        return out
    except Exception:
        empty: Dict[str, Dict[Visibility, Dict[str, Any]]] = {}
        for person in wishes.keys():
            empty[person] = {
                "public": {"total": len(wishes[person]["public"]), "satisfied": 0, "items": []},
                "private": {"total": len(wishes[person]["private"]), "satisfied": 0, "items": []},
            }
        return empty

def _pct(n: int, d: int) -> str:
    return f"{(n/d*100):.0f}%" if d > 0 else "—"

def _empty_scores(wishes: WishesDict) -> Dict[str, Dict[Visibility, Dict[str, Any]]]:
    out: Dict[str, Dict[Visibility, Dict[str, Any]]] = {}
    for person, sp in wishes.items():
        out[person] = {
            "public":  {"total": len(sp.get("public", [])),  "satisfied": 0, "items": []},
            "private": {"total": len(sp.get("private", [])), "satisfied": 0, "items": []},
        }
    return out


def build_satisfaction_section(
    wishes: WishesDict,
    plan_full_single: str,
    plan_public_single: str,
    multi_logs: List[Tuple[str, str]],
    multi_source: SatisfactionMultiSource = "summary",
) -> Tuple[str, Dict[SatisfactionMode, Dict[str, Dict["Visibility", Dict[str, Any]]]], str, str]:
    """3方式×公開/非公開の充足率テーブル（Markdown）を返す。"""

    multi_plan_summary = llm_extract_plan_from_multi(multi_logs) if multi_logs else ""

    multi_input_text = ""
    if multi_source == "logs":
        multi_input_text = "\n".join(f"[{s}] {c}" for s, c in multi_logs) if multi_logs else ""
    else:
        multi_input_text = multi_plan_summary if multi_logs else ""

    score_full   = llm_score_wishes(plan_full_single,   wishes) if (plan_full_single or "").strip()   else _empty_scores(wishes)
    score_public = llm_score_wishes(plan_public_single, wishes) if (plan_public_single or "").strip() else _empty_scores(wishes)
    score_multi  = llm_score_wishes(multi_input_text,   wishes) if (multi_input_text or "").strip()   else _empty_scores(wishes)

    def _sum(sc):
        return {
            p: {
                vis: f"{sc.get(p, {}).get(vis, {}).get('satisfied', 0)}/{sc.get(p, {}).get(vis, {}).get('total', 0)}"
                for vis in ("public", "private")
            } for p in wishes.keys()
        }
    def _pct_round(s: int, t: int) -> str:
        return "0%" if not t else f"{round(100*s/t)}%"

    def cell(sc: Dict[str, Dict["Visibility", Dict[str, Any]]], person: str, vis: "Visibility") -> str:
        t = sc.get(person, {}).get(vis, {}).get("total", 0)
        s = sc.get(person, {}).get(vis, {}).get("satisfied", 0)
        return f"{s}/{t} ({_pct_round(s, t)})"

    header = (
        "| 人 | 全公開シングル 公開 | 全公開シングル 非公開 | 半公開シングル 公開 | 半公開シングル 非公開 | マルチ 公開 | マルチ 非公開 |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for person in wishes.keys():
        rows.append(
            f"| {person} | "
            f"{cell(score_full, person, 'public')} | {cell(score_full, person, 'private')} | "
            f"{cell(score_public, person, 'public')} | {cell(score_public, person, 'private')} | "
            f"{cell(score_multi, person, 'public')} | {cell(score_multi, person, 'private')} |"
        )
    table_md = "\n".join(rows)

    all_scores = {
        "full_single": score_full,
        "public_single": score_public,
        "multi": score_multi,
    }

    return table_md, all_scores, multi_plan_summary, multi_input_text

# ====== レポート組み立て ======
def build_markdown_report(
    wishes: WishesDict,
    single_full_text: str,
    multi_result: "MultiResult",
    single_public_text: str,
    single_full_sec: float = 0.0,
    single_public_sec: float = 0.0,
    multi_source: str = "summary",
) -> str:
    wishes_md_lines: List[str] = []
    for name, sp in wishes.items():
        pub = "； ".join(sp.get("public", [])) if sp.get("public") else "（なし）"
        prv = "； ".join(sp.get("private", [])) if sp.get("private") else "（なし）"
        wishes_md_lines.append(f"- **{name}（公開）**: {pub}")
        wishes_md_lines.append(f"- **{name}（非公開）**: {prv}")
    wishes_md = "\n".join(wishes_md_lines)

    sat_table_md, all_scores, multi_plan_summary, multi_input_text = build_satisfaction_section(
        wishes=wishes,
        plan_full_single=single_full_text,
        plan_public_single=single_public_text,
        multi_logs=(multi_result.messages if multi_result else []),
        multi_source=multi_source,
    )

    checklist_md = build_condition_checklist(all_scores)

    md: List[str] = []
    md.append("# 旅行計画レポート")
    md.append("")
    md.append(f"- **日時（実行）**: {jst_now_iso()}")
    md.append("")

    md.append("## 所要時間・ラウンド")
    md.append("| 方式 | 実時間(s) | メッセージ数 | ラウンド数 | 停止理由 |")
    md.append("|---|---:|---:|---:|---|")
    md.append(f"| 全公開シングル | {single_full_sec:.3f} | — | — | — |")
    md.append(f"| 半公開シングル | {single_public_sec:.3f} | — | — | — |")
    md.append(f"| マルチ | {getattr(multi_result,'duration_sec',0.0):.3f} | {getattr(multi_result,'message_count',0)} | {getattr(multi_result,'rounds',0)} | {getattr(multi_result,'stop_reason','')} |")
    md.append("")
    md.append("## 各人の希望（公開/非公開）")
    md.append(wishes_md)
    md.append("")

    md.append("## 希望充足率")
    md.append(sat_table_md)
    md.append("")

    md.append("## 条件別チェックリスト")
    md.append(checklist_md)
    md.append("")

    md.append(f"### 充足率判定に用いたマルチ入力（{'会話ログ全文' if multi_source=='logs' else '要約文'}・推測を含む）")
    md.append(multi_input_text if multi_input_text else "（要約/ログなし または 未実行）")
    md.append("")

    md.append("## シングル（全公開）の出力")
    md.append(single_full_text if single_full_text else "（未実行）")
    md.append("")
    md.append("## シングル（半公開=公開のみ）の出力")
    md.append(single_public_text if single_public_text else "（未実行）")
    md.append("")
    md.append("## マルチのログ")
    if multi_result and multi_result.messages:
        for src, content in multi_result.messages:
            md.append(f"- **[{src}]** {content}")
    else:
        md.append("（未実行）")
    md.append("")
    return "\n".join(md)

def save_report(md: str, base_path: str = "report") -> None:
    jst = timezone(timedelta(hours=9))
    ts = datetime.now(jst).strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_path}_{ts}.md"
    path = Path(filename)
    path.write_text(md, encoding="utf-8")
    print("\n=== Markdown Report saved ===")
    print(path.resolve())

# ====== ユーティリティ：充足率集計（CSV用） ======
def _aggregate_pct(scores: Dict[str, Dict[Visibility, Dict[str, Any]]], vis: Visibility) -> int:
    """全員分の satisfied/total を合算し、百分率（整数0-100）を返す。"""
    total = 0
    sat = 0
    for person, vv in scores.items():
        t = int(vv.get(vis, {}).get("total", 0))
        s = int(vv.get(vis, {}).get("satisfied", 0))
        total += t
        sat += s
    if total == 0:
        return 0
    return round(100 * sat / total)

def _write_csv_header(csv_path: Path) -> None:
    header = [
        "試行ID",
        "全公開シングル公開条件充足率",
        "全公開シングル非公開条件充足率",
        "半公開シングル公開条件充足率",
        "半公開シングル非公開条件充足率",
        "マルチ公開条件充足率",
        "マルチ非公開条件充足率",
        "マルチ最後のメッセージ",
    ]
    need_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)

def _append_csv_row(csv_path: Path, trial_id: int, row_values: List[int]) -> None:
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([trial_id] + row_values)

# ====== CLI ======
def main():
    parser = argparse.ArgumentParser(description="Single(full/public) & Multi + LLM table & satisfaction report + trials CSV")
    parser.add_argument("--mode", choices=["single", "single_public", "multi", "both", "all"], default="all",
                        help="実行モード: single=全公開, single_public=半公開(公開のみ), multi=交渉, both=全公開+マルチ, all=3方式")
    parser.add_argument("--wishes-file", type=str, default=None,
                        help="希望入力。拡張子 .json: {旅行者A:{public:[], private:[]}} 他はテキスト形式")
    parser.add_argument("--no-report", action="store_true", help="Markdownレポート生成をスキップ")
    parser.add_argument("--report-path", type=str, default="report.md_4P_stub0", help="保存先（拡張子は任意、実際はタイムスタンプ付き）")
    parser.add_argument("--multi-source", choices=["summary", "logs"], default="summary",
                        help="マルチの充足率判定入力: summary=要約文 / logs=会話ログそのまま")
    # 追加: 試行とCSV
    parser.add_argument("--trials", type=int, default=10, help="各方式の試行回数（デフォルト10）")
    parser.add_argument("--csv-path", type=str, default="results_4P_stub0.csv", help="充足率CSVの出力先（既存なら追記）")

    args = parser.parse_args()

    wishes = load_wishes(args.wishes_file)

    # CSV 初期化
    csv_path = Path(args.csv_path)
    _write_csv_header(csv_path)

    # 指定回数ループ
    for trial in range(1, max(1, args.trials) + 1):
        print(f"\n\n########## Trial {trial} / {args.trials} ##########\n")
        single_full_text = ""
        single_public_text = ""
        single_full_sec = 0.0
        single_public_sec = 0.0
        multi_result: Optional[MultiResult] = None

        # 各方式実行（allで3方式）
        if args.mode in ("single", "both", "all"):
            single_full_text, single_full_sec = run_single(wishes)
        if args.mode in ("single_public", "all"):
            single_public_text, single_public_sec = run_single_public_only(wishes)
        if args.mode in ("multi", "both", "all"):
            multi_result = run_multi(wishes)

        # 充足率の算出（LLM判定→プログラムで集計）
        # マルチ入力テキストは設定に従う
        if multi_result is None:
            multi_result = MultiResult(messages=[], stop_reason="not run")

        # build_satisfaction_section で score_* を得る
        _, all_scores, _, _ = build_satisfaction_section(
            wishes=wishes,
            plan_full_single=single_full_text,
            plan_public_single=single_public_text,
            multi_logs=(multi_result.messages or []),
            multi_source=args.multi_source,  # summary/logs
        )
        score_full   = all_scores["full_single"]
        score_public = all_scores["public_single"]
        score_multi  = all_scores["multi"]

        # 公開/非公開の総合百分率を計算（整数）
        full_pub_pct    = _aggregate_pct(score_full,   "public")
        half_pub_pct    = _aggregate_pct(score_public, "public")
        multi_pub_pct   = _aggregate_pct(score_multi,  "public")
        full_priv_pct   = _aggregate_pct(score_full,   "private")
        half_priv_pct   = _aggregate_pct(score_public, "private")
        multi_priv_pct  = _aggregate_pct(score_multi,  "private")

        last_message = ""
        if multi_result and multi_result.messages:
            # 会話の最後のメッセージ内容（日本語本文）を取得
            last_message = multi_result.messages[-1][1].replace("\n", " ").strip()

        # CSVに追記
        _append_csv_row(
            csv_path,
            trial,
            [
                full_pub_pct,
                full_priv_pct,
                half_pub_pct,
                half_priv_pct,
                multi_pub_pct,
                multi_priv_pct,
                last_message
            ],
        )
        print(f"[CSV] wrote trial {trial} -> {csv_path.resolve()}")

        # レポート生成（任意）
        if not args.no_report:
            md = build_markdown_report(
                wishes=wishes,
                single_full_text=single_full_text,
                multi_result=multi_result,
                single_public_text=single_public_text,
                single_full_sec=single_full_sec,
                single_public_sec=single_public_sec,
                multi_source=args.multi_source,
            )
            save_report(md, f"{args.report_path}_trial{trial}")

if __name__ == "__main__":
    main()
