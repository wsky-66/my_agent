"""Optional SenseVoice model worker, loaded once."""
import re
from core.results import ToolResult
from workers.protocol import options, serve


class AsrBackend:
    def __init__(self, opts):
        self.model = None

    def __call__(self, action, params):
        if self.model is None:
            from funasr import AutoModel
            self.model = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
        if action == "warmup":
            return ToolResult.success("语音识别就绪")
        if action != "transcribe":
            raise ValueError("未知语音动作")
        result = self.model.generate(input=params["audio_path"], language="zh", use_itn=True)
        text = re.sub(r"<\|[^|>]*\|>", "", str(result[0].get("text", "") if result else "")).strip()
        return ToolResult.success("语音识别完成", {"text": text})


if __name__ == "__main__":
    serve(AsrBackend(options()))
