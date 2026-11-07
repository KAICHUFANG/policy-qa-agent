"""FastAPI 服务：三个接口。

M1（第 1 阶段）只求「最短链路能跑通」，所以这里是：
    /health      服务活着吗（部署后第一件事就是 curl 它）
     /corpus      加载了多少文档、多少条款（答非所问时先看这里）
     /ask         问一个问题，返回答案 + 出处 + 用量

M1 的检索是「全量投喂」：把所有条款拼进提示词。制度文档小的时候这招最准，
但它有天花板——条款一多，token 成本和延迟就顶不住。M2 再换成向量检索，
并且用同一套评测集对比两条路的 通过率 / 幻觉率 / 延迟。
"""

import re
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from app import corpus as corpus_mod
from app import llm, prompts
from app.retrieval import Retriever

app = FastAPI(title="制度问答 Agent", version="0.1.0")

# 启动时加载一次：制度文档是「读多写少」的数据，每次请求都去读盘是浪费。
# 以后要支持上传新制度，就在这里留一个 reload 的入口（M3 做）。
_CLAUSES = corpus_mod.load_clauses()
_RETRIEVER = Retriever(_CLAUSES)
# 出处校验表：(文档名, 条款号) → 条款对象
_INDEX = {(c.doc, c.number): c for c in _CLAUSES}

# 答案里的引用有两种常见写法，必须都认：
#   ① 分开写：《员工手册》第13条；《员工手册》第15条
#   ② 合并写：《差旅与费用报销制度》第4条、第5条   ← 模型很爱这么写
# 旧版只认 ①，结果把合法的第二、三个条号当成「没引用」丢掉，
# 评测里表现为「明明答对了却判没命中条款」。
# 教训：测试报错时先怀疑自己的工具，不要一上来就改业务逻辑。
# 条款号可能出现的字符：阿拉伯数字，或中文数字。
# 条款号可能出现的字符：阿拉伯数字，或中文数字（含「十百千万亿」位权）。
# 上百条条款是真实存在的（M3 要把语料扩到 500 条），所以位权必须全支持。
_CN_NUM = r"\d零〇一二三四五六七八九十百千万亿两"

_CITE_RE = re.compile(
    rf"《([^》]+)》\s*((?:第\s*[{_CN_NUM}]+\s*条[、,，和与及\s]*)+)"
)

_NUM_RE = re.compile(rf"第\s*([{_CN_NUM}]+)\s*条")

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}          # 位权：乘上去
_CN_SECTIONS = {"万": 10_000, "亿": 100_000_000}        # 分级：前面一整段先算完再乘


def _normalize_number(raw: str) -> int | None:
    """把条款号统一成阿拉伯数字。

    为什么必须有这一步（而不是只改正则）：
        正则只能把「十三」这两个字**找出来**，没法把它**算成 13**。
        _INDEX 的 key 是从文档标题原样切的（"第13条"），拿「第十三条」去查必然查不到——
        字典只认字符串完全相等，不认「意思相同」。所以要先归一化，再比对。

    返回 None 表示这个写法不认：调用方**不许猜**一个号码，否则会把认不出的写法
    记成一个假条款，反过来冤枉模型「编造」。

    规则（用户拍板）：
        "13"       → 13      阿拉伯数字直接转（含 "05" 这种前导零）
        "十三"     → 13      「十」前面没有数字 = 隐含一个 1
        "一十三"   → 13      开头的「一」不需要特殊处理，通用规则自然算对
        "二十三"   → 23      二×10 + 三
        "十"       → 10
        "两"       → 2       容忍「两」的写法
        "五百"     → 500     M3 扩到 500 条时会用到
        "一百零三" → 103     「零」只表示这一位空着，不参与累加
        "十二万三千" → 123000 「万」是分级：前面一整段先算完，再乘 10000
    """
    s = raw.strip()
    if not s:
        return None

    # 全是阿拉伯数字：int 会顺手吃掉前导零
    if s.isdigit():
        return int(s)

    total = 0      # 已经结算完的部分（遇到「万/亿」时把整段上提）
    section = 0    # 当前这一节累加的值（十百千算在这里）
    number = 0     # 刚读到的数字，等着乘下一个位权

    for ch in s:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]                    # 先记住，等位权
        elif ch in _CN_UNITS:
            # 「十三」：位权前面没有数字时，隐含一个 1
            section += (1 if number == 0 else number) * _CN_UNITS[ch]
            number = 0
        elif ch in _CN_SECTIONS:
            section += number                          # 把这一节收尾
            total += section * _CN_SECTIONS[ch]
            section, number = 0, 0
        else:
            return None                                # 不认识的字符：宁可判无效，也不猜

    return total + section + number


class AskIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=500,
                          description="员工的问题，例：出差去上海住宿能报多少")
    # 不传就用 config.RETRIEVAL_MODE。显式传是为了能「同一个进程里跑三种检索对比」。
    mode: str | None = Field(None, description="full / bm25 / vector，不传用默认")
    top_k: int | None = Field(None, ge=1, le=58, description="检索取几条条款")


class Citation(BaseModel):
    doc: str
    number: str
    heading: str
    valid: bool          # 这条引用是否真实存在于制度文档里


def verify_citations(answer: str) -> List[Citation]:
    """把答案里出现的引用逐个对照制度文档校验。

    为什么这段一定要用代码做，不能靠模型自觉：
        模型很会「顺便」编一个《员工手册》第99条，读起来毫无破绽。
        评审（和面试官）第一眼要看的就是「出处能不能对上」。
        这里是确定性的字符串比对，不依赖模型的自觉性。
    """
    seen, out = set(), []
    for doc, chain in _CITE_RE.findall(answer):
        for num in _NUM_RE.findall(chain):
            value = _normalize_number(num)
            # 认不出的写法：不许猜号码，原样拼回去 → 查不到 → 落进 invalid_citations。
            # 让失败「看得见」；如果直接丢掉，就分不清「模型没写引用」和「工具瞎了」。
            number = f"第{num}条" if value is None else f"第{value}条"
            key = (doc.strip(), number)
            if key in seen:
                continue
            seen.add(key)
            clause = _INDEX.get(key)
            out.append(Citation(doc=key[0], number=key[1],
                                heading=clause.heading if clause else "",
                                valid=clause is not None))
    return out


def _clauses_text(clauses: List[corpus_mod.Clause]) -> str:
    return "\n\n".join(c.render() for c in clauses)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model": config.MODEL,
        "retrieval_mode": config.RETRIEVAL_MODE,
        "clause_count": len(_CLAUSES),
    }


@app.get("/corpus")
def corpus_info() -> Dict[str, Any]:
    return corpus_mod.stats()


@app.post("/ask")
def ask(body: AskIn) -> Dict[str, Any]:
    question = body.question.strip()

    # ---- 按模式取条款（full / bm25 / vector）----
    used = _RETRIEVER.search(question, mode=body.mode, k=body.top_k)
    clauses_text = _clauses_text(used) #转成上下文

    if len(clauses_text)<=config.MAX_CLAUSES_CHARS:
        result = llm.chat(system=prompts.SYSTEM,
                      user=prompts.build_user_prompt(question, clauses_text))
    else:
        raise HTTPException(status_code=400,detail="超过条款字数上限")
    answer = result["text"]
    citations = verify_citations(answer)
    refused = ("没有" in answer and "规定" in answer) or "没有关于" in answer

    return {
        "question": question,
        "answer": answer,
        "citations": [c.dict() for c in citations],
        "invalid_citations": [c.dict() for c in citations if not c.valid],
        "refused": refused,          # 模型有没有老实说「制度里没写」
        "retrieval": {
            "mode": (body.mode or config.RETRIEVAL_MODE).lower(),
            "clauses_used": len(used),
            "chars_used": len(clauses_text),
            "cites": [c.cite for c in used],       # 检索到了哪些条款（排障用）
        },
        "usage": result["usage"],
        "latency_ms": result["latency_ms"],
        "attempts": result["attempts"],
    }

@app.get("/clause")
def clause(doc:str,number:str)->dict:
    c=_INDEX.get((doc,number))
    if c is None:
        raise HTTPException(status_code=404, detail="没有这份文档或这一条")
    return {"doc":c.doc,"number":c.number,"chapter":c.chapter,"heading":c.heading,"text":c.text}
