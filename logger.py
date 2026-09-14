"""集中式错误日志：把异常与工具报错写入 logs/agent_error.log（轮转，防无限增长）。"""

import logging
import os
import re
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_ERR_WORDS = (
    "出错", "错误", "失败", "超时", "异常", "无法", "未找到", "不存在", "解析失败",
    "不可用", "被拒绝", "离线", "未运行", "未就绪", "未收到", "未能", "拒绝连接",
    # 常见英文错误
    "error", "exception", "traceback", "failed", "timeout", "refused",
    "not found", "does not exist", "not running", "unavailable",
)
_ERROR_RE = re.compile("|".join(re.escape(w) for w in _ERR_WORDS), re.IGNORECASE)


def _build() -> logging.Logger:
    logger = logging.getLogger("agent")
    if logger.handlers:
        return logger
    os.makedirs(_LOG_DIR, exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        os.path.join(_LOG_DIR, "agent_error.log"),
        maxBytes=2_000_000, backupCount=3, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


log = _build()


def error(msg: str, exc: BaseException = None) -> None:
    """记录一条错误（可带异常对象）。"""
    log.error(msg + (f"  |  {exc!r}" if exc is not None else ""))


def warning(msg: str) -> None:
    log.warning(msg)


def is_error(text: str) -> bool:
    """判断工具返回结果是否为错误状态（忽略大小写）。"""
    return _ERROR_RE.search(text) is not None
