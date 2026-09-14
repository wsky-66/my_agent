import os
from dotenv import load_dotenv

load_dotenv()

for _key in ("ALL_PROXY", "all_proxy"):
    _val = os.environ.get(_key, "")
    if _val.startswith("socks"):
        os.environ.pop(_key, None)


class Config:
    API_KEY: str = os.getenv("LLM_API_KEY", "")
    BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    MAX_TOOL_ROUNDS: int = int(os.getenv("LLM_MAX_TOOL_ROUNDS", "25"))
    SHELL_TIMEOUT: int = int(os.getenv("SHELL_TIMEOUT", "30"))

    # DDS 算法平台（Grounding DINO 开放集检测）
    DDS_API_TOKEN: str = os.getenv("DDS_API_TOKEN", "")
    DDS_BASE_URL: str = os.getenv("DDS_BASE_URL", "https://api.deepdataspace.com")

    @classmethod
    def validate(cls):
        if not cls.API_KEY:
            raise ValueError(
                "LLM_API_KEY 未设置。请在 .env 文件中配置或设置环境变量。\n"
                "  cp .env.example .env  # 然后编辑 .env 填入你的 API Key"
            )


config = Config()
