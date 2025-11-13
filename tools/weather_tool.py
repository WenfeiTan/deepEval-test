# tools/weather_tool.py
from deepeval.tracing import observe

@observe()
def weather_tool(city: str, date: str = "today") -> str:
    """
    示例天气工具。现在是假的：直接返回一个固定文案。
    将来你可以在这里换成调用真实天气 API 或 MCP weather server。
    """
    # TODO: 换成真实 API 调用
    return f"{city} {date} 的天气：晴，25°C，微风。"
