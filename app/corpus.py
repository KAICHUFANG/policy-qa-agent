"""制度文档 → 「一条一条条款」。

为什么要切到「条款」这个粒度，而不是按 500 字切块：
    RAG（Retrieval-Augmented Generation，检索增强生成）里最要命的问题是
    「答得对但出处对不上」。按字数硬切，一条「第5条 住宿标准」很可能被切成两半，
    检索回来半条，答案就缺一半条件；而且没法说清「这条规定在第几条」。

    制度文档天生有条款结构（### 第N条），顺着它切，出处就是天然的、不用猜。
    这就是「文档自带的标题结构，比任何切块算法都可靠」。
"""

from dataclasses import dataclass
from typing import List

import config

# 条款标题形如：### 第5条 住宿标准（每人每晚，含税）
_CLAUSE_PREFIX = "### 第"


@dataclass
class Clause:
    """一条制度条款，带出处信息。"""
    doc: str        # 来自哪份文档（文件名，不含扩展名）
    heading: str    # 条款标题，如「第5条 住宿标准（每人每晚，含税）」
    chapter: str    # 所属章，如「第二章 差旅标准」
    text: str       # 条款正文

    @property
    def number(self) -> str:
        """条款号，如「第5条」。用于校验答案里的引用是否真实存在。"""
        return self.heading.split(" ")[0]

    @property
    def cite(self) -> str:
        """出处标签，答案里必须带这个格式。"""
        return f"《{self.doc}》{self.number}"

    def render(self) -> str:
        return f"[{self.cite}]（{self.chapter} · {self.heading}）\n{self.text}"


def _parse_doc(path) -> List[Clause]:
    """把一份 markdown 制度切成条款列表。

    规则（简单但够用）：
      · 「## 」开头  → 进入新的一章
      · 「### 第」开头的行 → 新条款开始
      · 其余行 → 追加到当前条款正文
    """
    doc = path.stem
    lines = path.read_text(encoding="utf-8").splitlines()

    clauses: List[Clause] = []
    chapter = ""
    heading = ""
    buf: List[str] = []

    def flush() -> None:
        if heading:
            clauses.append(Clause(doc=doc, heading=heading, chapter=chapter,
                                  text="\n".join(buf).strip()))

    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            flush()                    # 上一章结束，把最后一条收掉
            chapter, heading, buf = line[3:].strip(), "", []
            continue
        if line.startswith(_CLAUSE_PREFIX):
            flush()
            heading, buf = line[4:].strip(), []
            continue
        if heading:
            buf.append(line)

    flush()
    return clauses


def load_clauses() -> List[Clause]:
    """读 corpus/ 下所有 .md，返回全部条款（按文档名、条款顺序）。"""
    out: List[Clause] = []
    for path in sorted(config.CORPUS_DIR.glob("*.md")):
        out.extend(_parse_doc(path))
    return out


def stats() -> dict:
    """给 /corpus 接口用：一眼看出加载了多少文档、多少条、多少字。"""
    clauses = load_clauses()
    docs: dict = {}
    for c in clauses:
        docs[c.doc] = docs.get(c.doc, 0) + 1
    return {
        "documents": docs,
        "clause_count": len(clauses),
        "total_chars": sum(len(c.text) for c in clauses),
    }
