"""检索层：三种取条款的方式，同一个出口。

为什么要做成「三档可切换」而不是直接上向量检索：
    没有基线的优化没法证明。全量投喂（full）是准确率的天花板但成本最高；
    关键词检索（bm25）几乎零成本；向量检索（vector）能处理「同义不同词」。
    三种方式用同一套题目跑，才知道每种方式到底换来了什么、又付出了什么。

    M2 的产出就是这张对比表，不是「我把关键词换成了向量」这句话。

三档说明：
    full    把所有条款塞进提示词。条款少时最准，条款一多就顶到 token 上限。
    bm25    纯 python 实现的 BM25（Best Matching 25，经典关键词相关性算法）。
            中文不装分词器怎么办：用「字符二元组」当词（见 _tokens）。
            对制度文档这种用词重复度高的文本，它其实很强。
    vector  Chroma 向量库 + bge-small-zh-v1.5 中文语义向量。
            优势在同义表达（「多久内提交」↔「报销时限」）。
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional

import config
from app import netfix
from app.corpus import Clause

# ---------------------------------------------------------------- 关键词检索

_CJK = re.compile(r"[\u4e00-\u9fff]")
_SPLIT = re.compile(r"[^\w\u4e00-\u9fff]+")


def _tokens(text: str) -> List[str]:
    """中文不装分词器也能检索：切成「字符二元组」+ 单字。

    为什么二元组够用：
        「报销时限」→ 报销/销时/时限 + 报/销/时/限。
        提问里出现「多久内提交报销」时，会命中「报销」这个二元组；
        而「时限」这类词也保留了。中文文档的检索，二元组召回率通常已经很好，
        真正的语义鸿沟（「能报多少钱」↔「报销标准」）才需要向量。
    """
    out: List[str] = []
    for chunk in _SPLIT.split(text or ""):
        if not chunk:
            continue
        if _CJK.search(chunk):
            out.extend(chunk)                                        # 单字
            out.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))  # 二元组
        else:
            out.append(chunk.lower())
    return out


class BM25:
    """标准 BM25。公式不用背，知道三个量的含义就够：

        tf  —— 该词在这条条款里出现几次（越多越相关，但有上限）
        idf —— 该词在整个文档集里有多罕见（越罕见越有区分度）
        dl  —— 这条条款有多长（长条款天然容易命中，要做归一化）
    """

    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tokens(d) for d in docs]
        self.doc_len = [len(t) for t in self.docs]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.docs else 0.0

        df: Counter = Counter()
        for toks in self.docs:
            df.update(set(toks))
        n = len(self.docs)
        self.idf: Dict[str, float] = {
            w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }
        self.tf = [Counter(t) for t in self.docs]

    def scores(self, query: str) -> List[float]:
        q = _tokens(query)
        out: List[float] = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = self.doc_len[i] or 1
            for w in q:
                if w not in tf:
                    continue
                f = tf[w]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


# ---------------------------------------------------------------- 向量检索

class VectorIndex:
    """Chroma 向量库。嵌入模型用 bge-small-zh-v1.5（512 维，中文语义）。

    为什么不用 Chroma 自带的默认嵌入函数：
        默认是 all-MiniLM-L6-v2，英文模型，中文检索质量差；
        而且它要在首次使用时从 S3 下模型，这台机器上直连超时。
        所以这里**自己算向量、把向量交给 Chroma 存**，
        Chroma 只负责「向量 + 元数据 + 近邻查询」，嵌入模型自己选。
    """

    def __init__(self) -> None:
        netfix.init()                     # 必须在 import fastembed 之前
        import chromadb
        self._client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self._col = self._client.get_or_create_collection(
            "clauses", metadata={"hnsw:space": "cosine"})
        self._embedder = None

    def _model(self):
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(config.EMBED_MODEL,
                                           cache_dir=str(config.EMBED_CACHE))
        return self._embedder

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._model().embed(texts)]

    def build(self, clauses: List[Clause], force: bool = False) -> int:
        """建索引。条数对得上就不重建（Chroma 是持久化的）。"""
        if not force and self._col.count() == len(clauses):
            return self._col.count()
        if force and self._col.count():
            self._client.delete_collection("clauses")
            self._col = self._client.get_or_create_collection(
                "clauses", metadata={"hnsw:space": "cosine"})
        # 嵌入文本带上章名与条款标题：「第5条 住宿标准」本身就是最强的检索信号
        texts = [f"{c.chapter} {c.heading}\n{c.text}" for c in clauses]
        self._col.add(
            ids=[f"{c.doc}#{c.number}" for c in clauses],
            documents=[c.text for c in clauses],
            metadatas=[{"doc": c.doc, "number": c.number,
                        "heading": c.heading, "chapter": c.chapter} for c in clauses],
            embeddings=self._embed(texts),
        )
        return self._col.count()

    def search(self, query: str, k: int) -> List[Clause]:
        res = self._col.query(query_embeddings=self._embed([query]), n_results=k)
        out: List[Clause] = []
        for meta, doc in zip(res["metadatas"][0], res["documents"][0]):
            # heading 形如「第5条 住宿标准」，Clause.number 从它里面取条款号
            out.append(Clause(doc=meta["doc"], heading=meta["heading"],
                              chapter=meta["chapter"], text=doc))
        return out


# ---------------------------------------------------------------- 统一出口

class Retriever:
    """按 mode 取条款。第一次用到时才加载索引（启动快，也方便只跑 full 模式）。"""

    def __init__(self, clauses: List[Clause]):
        self.clauses = clauses
        self._bm25: Optional[BM25] = None
        self._vec: Optional[VectorIndex] = None

    def search(self, query: str, mode: Optional[str] = None,
               k: Optional[int] = None) -> List[Clause]:
        mode = (mode or config.RETRIEVAL_MODE).lower()
        k = k or config.TOP_K

        if mode == "full":
            return self.clauses

        if mode == "bm25":
            if self._bm25 is None:
                self._bm25 = BM25([f"{c.chapter} {c.heading} {c.text}"
                                   for c in self.clauses])
            scores = self._bm25.scores(query)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [self.clauses[i] for i in order[:k] if scores[i] > 0]

        if mode == "vector":
            if self._vec is None:
                self._vec = VectorIndex()
                self._vec.build(self.clauses)
            return self._vec.search(query, k)

        if mode == "hybrid":
            return self._hybrid(query, k)

        raise ValueError(f"未知的检索模式：{mode}（可选 full / bm25 / vector / hybrid）")

    # ---- 混合检索用 RRF 融合 ----
    # RRF（Reciprocal Rank Fusion，基于排名的倒数融合）：
    #     得分 = Σ 1 / (60 + 名次)
    # 它只看名次、不看原始分数——这是关键优势：
    #     BM25 的分数是无界的，余弦相似度在 0-1，两者根本不能直接相加；
    #     用名次就自动归一了，不需要调权重，也不需要归一化分数。
    # 60 是原论文里的经验常数，作用是压平头部名次之间的差距。
    def _hybrid(self, query: str, k: int) -> List[Clause]:
        pool = max(k * 2, 10) #候选池
        lists = [self.search(query, mode="bm25", k=pool),
                 self.search(query, mode="vector", k=pool)]
        fused: Dict[str, float] = {} #分值表
        by_cite: Dict[str, Clause] = {} #本体表
        for items in lists:
            for rank, c in enumerate(items, start=1):
                fused[c.cite] = fused.get(c.cite, 0.0) + 1.0 / (60 + rank)
                by_cite[c.cite] = c
        order = sorted(fused, key=lambda key: fused[key], reverse=True)
        return [by_cite[key] for key in order[:k]]
