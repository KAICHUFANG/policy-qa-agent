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

from fastapi import FastAPI
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
_CITE_RE = re.compile(r"《([^》]+)》\s*((?:第\d+条[、,，和与及\s]*)+)")
_NUM_RE = re.compile(r"第(\d+)条")


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
            number = f"第{num}条"
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
    clauses_text = _clauses_text(used)

    result = llm.chat(system=prompts.SYSTEM,
                      user=prompts.build_user_prompt(question, clauses_text))

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
