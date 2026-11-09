import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJ = EVAL_DIR.parent
sys.path.insert(0, str(PROJ))

from app.main import _normalize_number, verify_citations

print("=" * 72)
print("一、_normalize_number 单元用例")
cases = [
    ("13", 13),
    ("十三", 13),
    ("一十三", 13),
    ("二十三", 23),
    ("十", 10),
    ("二十", 20),
    ("两", 2),
    ("35", 35),
    ("99", 99),
    ("一百", 100),
    ("一百零三", 103),
    ("一百二十三", 123),
    ("五百", 500),
    ("一千", 1000),
    ("两千零五", 2005),
    ("十二万三千", 123000),
]
ok = True
for raw, want in cases:
    got = _normalize_number(raw)
    flag = "OK " if got == want else "FAIL"
    if got != want:
        ok = False
    print(f"  [{flag}] {raw!r:12} -> {got!r:8} (期望 {want})")

print("=" * 72)
print("二、_normalize_number 负面用例（必须返回 None，不许瞎猜）")
bad = ["", "abc", "第", "拾"]
for raw in bad:
    got = _normalize_number(raw)
    flag = "OK " if got is None else "FAIL"
    if got is not None:
        ok = False
    print(f"  [{flag}] {raw!r:8} -> {got!r}")

print("=" * 72)
print("三、verify_citations 端到端用例")
cite_cases = [
    ("《员工手册》第十三条", 1, True, "中文数字，真实存在"),
    ("《员工手册》第一十三条", 1, True, "一十三，真实存在"),
    ("《员工手册》第 13 条", 1, True, "带空格"),
    ("《员工手册》第13条、第 15 条", 2, True, "混合写法，两条都要抓到"),
    ("《员工手册》第99条", 1, False, "真编的，必须判无效"),
    ("《员工手册》第四条", 1, True, "简单中文数字"),
    ("《员工手册》第一百二十三条", 1, False, "百位能解析，但制度里没有这条"),
]
for text, want_n, want_valid, note in cite_cases:
    got = verify_citations(text)
    good_n = len(got) == want_n
    good_v = all(c.valid for c in got) is want_valid and len(got) > 0
    flag = "OK " if (good_n and good_v) else "FAIL"
    if not (good_n and good_v):
        ok = False
    print(f"  [{flag}] {note}")
    print(f"         {text}  ->  {[(c.number, c.valid) for c in got]}")

print("=" * 72)
print("总结果:", "全部通过" if ok else "有失败")
sys.exit(0 if ok else 1)