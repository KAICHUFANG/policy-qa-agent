"""让 `uvicorn app.main:app` 也能 import 到根目录下的 config.py。

直跑 `python -m uvicorn` 时当前目录会自动进 sys.path，但用命令行脚本
`uvicorn app.main:app` 时不一定——表现是「python 里能跑，uvicorn 起来就 ModuleNotFoundError」。
把项目根目录显式放进 sys.path，这条坑就没了。
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
