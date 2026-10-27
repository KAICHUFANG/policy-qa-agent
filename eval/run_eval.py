"""三档检索对比评测。

用法：
    uv run python eval/run_eval.py                 # full / bm25 / vector 全跑
    uv run python eval/run_eval.py --modes bm25    # 只跑一档
    uv run python eval/run_eval.py --ids C01 C17   # 只跑指定题

为什么走 HTTP 接口（TestClient）而不是直接调函数：
    要测的就是「线上会怎么跑」。绕过接口去调函数，很容易把「接口层漏传参数」
    这类问题测掉。TestClient 不占端口、不起服务，代价几乎为零。

评分是确定性的（见 cases.json 的 scoring 段），不调模型裁判：
    同一套题要在三种检索方式上对比，评分开销必须为零、结果必须可复现。
"""

import argparse
import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJ = EVAL_DIR.parent
sys.path.insert(0, str(PROJ))

from fastapi.testclient import TestClient          # noqa: E402

import config                                     # noqa: E402
from app.main import app                          # noqa: E402

MODES = ["full", "bm25", "vector"]


def load_cases() -> list:
    with open(EVAL_DIR / "cases.json", encoding="utf-8") as f:
        return json.load(f)["cases"]


def judge(case: dict, resp: dict) -> tuple:
    """返回 (是否通过, 失败原因列表)。全部是确定性断言。"""
    reasons = []

    cites = {f"《{c['doc']}》{c['number']}" for c in resp["citations"] if c["valid"]}
    for want in case.get("expect_cites", []):
        if want not in cites:
            reasons.append(f"没命中条款 {want}（实得 {sorted(cites) or '无'}）")

    for text in case.get("must_contain", []):
        if text not in resp["answer"]:
            reasons.append(f"答案里没有「{text}」")

    for text in case.get("must_not_contain", []):
        if text in resp["answer"]:
            reasons.append(f"出现了不该有的「{text}」")

    if resp["invalid_citations"]:
        bad = [f"《{c['doc']}》{c['number']}" for c in resp["invalid_citations"]]
        reasons.append(f"引用了不存在的条款 {bad}")

    if "expect_refused" in case and resp["refused"] != case["expect_refused"]:
        want = "应该拒答" if case["expect_refused"] else "不该拒答"
        reasons.append(f"{want}，实际 refused={resp['refused']}")

    return (not reasons), reasons


def run_mode(client, mode: str, cases: list, top_k: int | None = None) -> dict:
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        payload = {"question": case["q"], "mode": mode}
        if top_k:
            payload["top_k"] = top_k
        r = client.post("/ask", json=payload)
        wall = int((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            rows.append({"id": case["id"], "ok": False,
                         "reasons": [f"HTTP {r.status_code}: {r.text[:120]}"]})
            continue
        resp = r.json()
        ok, reasons = judge(case, resp)
        rows.append({
            "id": case["id"], "category": case["category"], "q": case["q"],
            "ok": ok, "reasons": reasons,
            "answer": resp["answer"],
            "cites": [f"《{c['doc']}》{c['number']}" for c in resp["citations"]],
            "retrieved": resp["retrieval"]["clauses_used"],
            "chars": resp["retrieval"]["chars_used"],
            "tokens": resp["usage"]["total_tokens"],
            "latency_ms": resp["latency_ms"],
            "wall_ms": wall,
        })

    n = len(rows)
    passed = sum(1 for r in rows if r["ok"])
    return {
        "mode": mode,
        "passed": passed,
        "total": n,
        "pass_rate": round(100 * passed / n, 1) if n else 0.0,
        "avg_tokens": round(sum(r.get("tokens", 0) for r in rows) / n, 0) if n else 0,
        "avg_retrieved": round(sum(r.get("retrieved", 0) for r in rows) / n, 1) if n else 0,
        "avg_chars": round(sum(r.get("chars", 0) for r in rows) / n, 0) if n else 0,
        "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in rows) / n, 0) if n else 0,
        "hallucinated_cites": sum(1 for r in rows if not r["ok"] and
                                  any("不存在的条款" in x for x in r["reasons"])),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--top-k", type=int, default=None,
                    help="覆盖检索条数（模拟条款变多、只能给 3 条的场景）")
    args = ap.parse_args()

    config.check()
    cases = load_cases()
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    client = TestClient(app)

    print("=" * 96)
    print(f"制度问答 · 检索方式对比　题目 {len(cases)} 条 × 方式 {len(modes)} 种")
    print(config.describe())
    print("=" * 96)

    results = []
    for mode in modes:
        print(f"\n▶ 跑 {mode} ……", flush=True)
        res = run_mode(client, mode, cases, args.top_k)
        results.append(res)
        print(f"  通过 {res['passed']}/{res['total']}"
              f"　平均 {res['avg_tokens']:.0f} tokens"
              f"　检索 {res['avg_retrieved']} 条/{res['avg_chars']:.0f} 字"
              f"　模型延迟 {res['avg_latency_ms']:.0f}ms")

    # ---- 汇总表 ----
    print("\n" + "=" * 96)
    print("汇总")
    print("=" * 96)
    head = f"{'方式':<8}{'通过':>8}{'通过率':>9}{'平均tokens':>12}{'平均检索条款':>14}{'平均延迟':>10}"
    print(head)
    print("-" * 96)
    for r in results:
        print(f"{r['mode']:<8}{r['passed']}/{r['total']:<5}{r['pass_rate']:>8.1f}%"
              f"{r['avg_tokens']:>12.0f}{r['avg_retrieved']:>14.1f}"
              f"{r['avg_latency_ms']:>9.0f}ms")

    # 相对 full 的节省
    base = next((r for r in results if r["mode"] == "full"), None)
    if base and base["avg_tokens"]:
        print("\n相对 full（全量投喂）的 token 消耗：")
        for r in results:
            if r["mode"] == "full":
                continue
            print(f"  {r['mode']:<7} {r['avg_tokens'] / base['avg_tokens'] * 100:>5.1f}%"
                  f"　（省 {100 - r['avg_tokens'] / base['avg_tokens'] * 100:.1f}%）")

    # ---- 逐题差异（只看失败，方便定位是检索问题还是提示词问题）----
    for r in results:
        bad = [x for x in r["rows"] if not x["ok"]]
        if not bad:
            continue
        print(f"\n[{r['mode']}] 未通过 {len(bad)} 条：")
        for x in bad:
            print(f"  {x['id']} {x['category']}｜{x['q']}")
            for reason in x["reasons"]:
                print(f"      · {reason}")

    out = EVAL_DIR / "report.json"
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(cases),
        "results": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "rows": {r["mode"]: r["rows"] for r in results},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {out}")

    # 退出码：只要有一种方式没全过就返回 1，方便以后接 CI
    return 0 if all(r["passed"] == r["total"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
