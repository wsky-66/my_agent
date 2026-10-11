"""Optional multimodal description service."""
import base64
import mimetypes
from pathlib import Path
from config import config
from core.results import ToolResult
from logger import measure

_client = None


def describe(image_path, prompt):
    global _client
    from openai import OpenAI
    if _client is None:
        config.validate()
        _client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL, timeout=config.LLM_TIMEOUT)
    path = Path(image_path)
    url = "data:" + (mimetypes.guess_type(str(path))[0] or "image/png") + ";base64," + base64.b64encode(path.read_bytes()).decode()
    with measure("vision.description"):
        response = _client.chat.completions.create(model=config.MODEL, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}], max_tokens=2000)
    text = (response.choices[0].message.content or "").strip()
    return ToolResult.success(text) if text else ToolResult.failure("EMPTY_VISION_REPLY", "模型未返回画面描述")
