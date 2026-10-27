"""环境修补：两个实测踩到的坑，集中在这里处理一次。

这个文件看着「不像业务代码」，但它解决的是「换台机器就跑不起来」的那类问题。

坑 1：NO_PROXY 里的方括号 IPv6
    某些工具（比如 Cherry Studio）会把 NO_PROXY 设成
        localhost,127.*,...,[::1]
    注意那个 **[::1]** —— 带方括号。httpx / urllib3 解析代理列表时会按
    「最后一个冒号后面是端口」来切，于是把 ":1]" 当端口，直接抛
        InvalidURL: Invalid port: ':1]'
    表现是「模型调用正常、但一下载模型就报 InvalidURL」，极难联想。
    修法：去掉方括号形式的 IPv6 条目（保留 127.0.0.1 / localhost 就够了）。

坑 2：huggingface.co 在国内直连不通
    bge 这类模型权重放在 HuggingFace 上。设 HF_ENDPOINT 指向 hf-mirror.com 即可。
    这跟「配全局代理」不是一回事：只影响这一个下载源。

两个坑都是在同一台机器上真实撞到的，写在这里，换环境时不用再猜一遍。
"""

import os

# 方括号 IPv6：会被代理解析器当成端口，必须去掉
_BAD_NOPROXY = {"[::1]", "::1", "[::]"}


def sanitize_proxy_env() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(key)
        if not raw:
            continue
        kept = [x.strip() for x in raw.split(",")
                if x.strip() and x.strip() not in _BAD_NOPROXY]
        cleaned = ",".join(kept)
        os.environ[key] = cleaned


def use_hf_mirror(url: str = "https://hf-mirror.com") -> None:
    """模型权重走镜像。已经有值就不覆盖（允许用环境变量自定义）。"""
    os.environ.setdefault("HF_ENDPOINT", url)
    # Windows 不支持软链接缓存，会有噪音警告；关掉
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def init() -> None:
    sanitize_proxy_env()
    use_hf_mirror()
