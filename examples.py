"""
扩展工具示例 - 演示如何注册自定义工具。

只需在代码中导入 registry 并使用 @registry.register 装饰器即可注册新工具。
LLM 会根据函数的类型注解和描述自动选择并调用合适的工具。

用法：
  1. 参考下面的示例编写你的工具函数
  2. 在 main.py 中 import 你的工具模块
  3. 启动 Agent，用自然语言描述任务即可
"""

from tools import registry


@registry.register(description="查询指定城市的天气信息（模拟）")
def get_weather(city: str) -> str:
    """
    查询指定城市的天气信息。

    :param city: 城市名称，如 "北京"、"上海"
    """
    weather_data = {
        "北京": "晴，25°C，湿度 40%",
        "上海": "多云，28°C，湿度 70%",
        "深圳": "阵雨，30°C，湿度 85%",
        "成都": "阴，22°C，湿度 60%",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


@registry.register(description="搜索本地文件，返回匹配的文件路径列表")
def find_files(pattern: str) -> str:
    """
    在项目目录下搜索文件。

    :param pattern: glob 模式，如 "*.py"、"**/*.txt"
    """
    import glob as g
    import os

    results = g.glob(os.path.expanduser(pattern), recursive=True)
    if not results:
        return f"未找到匹配 '{pattern}' 的文件"
    return "\n".join(results[:50])
