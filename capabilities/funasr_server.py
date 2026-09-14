#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SenseVoice ASR 常驻服务(CPU)。

stdin 每行读 {"audio_path": "<wav>"}, stdout 每行回 RESULT:{json}.
模型只加载一次, 避免每次调用重载。日志输出到 stderr。
"""
import json
import os
import re
import sys

_MODEL = "iic/SenseVoiceSmall"
_DEVICE = "cpu"
_LANG = "zh"


def _clean_tags(text):
    """去掉 SenseVoice 输出的 <|zh|> <|NEUTRAL|> <|HAPPY|> 等特殊标记。"""
    return re.sub(r"<\|[^|>]*\|>", "", text).strip()


def main():
    from funasr import AutoModel

    sys.stderr.write("加载 SenseVoice 模型(首次会下载 ~2GB)...\n")
    sys.stderr.flush()
    try:
        model = AutoModel(model=_MODEL, device=_DEVICE, disable_update=True)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"模型加载失败: {e}\n")
        sys.stderr.flush()
        sys.exit(1)
    sys.stderr.write("模型就绪, 等待请求...\n")
    sys.stderr.flush()
    # 通知上层(voice.py)模型已就绪
    print("VOICE_READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print('RESULT:' + json.dumps({"text": "", "error": "bad request"}), flush=True)
            continue
        audio = req.get("audio_path", "")
        if not audio or not os.path.exists(audio):
            print('RESULT:' + json.dumps({"text": "", "error": "audio not found"}), flush=True)
            continue
        try:
            res = model.generate(input=audio, language=_LANG, use_itn=True)
            raw = res[0].get("text", "") if res else ""
            text = _clean_tags(str(raw))
            print('RESULT:' + json.dumps({"text": text}), flush=True)
        except Exception as e:  # noqa: BLE001
            print('RESULT:' + json.dumps({"text": "", "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
