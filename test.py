import sys
sys.path.insert(0, r"C:\Users\xiong\Desktop\制度问答Agent")

from app.main import _RETRIEVER
from typing import Any, Dict, List, Literal

used = _RETRIEVER.search("年假", mode="bm25", k=5)

clause_text: List[Any] = []

for rank, c in enumerate(used, start=0):
    if len(c.text) >= 50:
        clause_text.append(
            {"rank": rank, "doc": c.doc, "heading": c.heading, "chapter": c.chapter, "text": c.text[0:49] + "..."})
    else:
        clause_text.append(
            {"rank": rank, "doc": c.doc, "heading": c.heading, "chapter": c.chapter, "text": c.text[0:49]})

print(clause_text)