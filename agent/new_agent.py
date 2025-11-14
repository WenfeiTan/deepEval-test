"""
from colleague_sdk import AgentClient

client = AgentClient(...)

answer = client.chat("帮我查一下今天上海的天气")
"""

# agent/colleague_agent.py
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

from colleague_sdk import AgentClient  # 假设是这样

task_completion_metric = TaskCompletionMetric()
client = AgentClient(...)


@observe(metrics=[task_completion_metric])
def colleague_agent(query: str) -> str:
    """
    这是一个“薄薄的 wrapper”，把同事的 agent 包成 eval 框架能用的接口。
    """
    answer = client.chat(query)

    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=answer,
        )
    )
    return answer


"""
from mcp_client import MCPClient

client = MCPClient(server_url="...")

result = client.call_tool("get_weather", {"city": "上海", "date": "today"})

"""

# agent/colleague_mcp_agent.py
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

from mcp_client import MCPClient  # 伪代码，按真实的来

task_completion_metric = TaskCompletionMetric()
client = MCPClient(...)


@observe(metrics=[task_completion_metric])
def colleague_mcp_agent(query: str) -> str:
    """
    这里你可以：
    - 要么把 query 直接发给同事的“总 agent”（如果他已经写好）
    - 要么自己写一点逻辑：调用 MCP server 提供的工具，再组织答案
    """
    # 假设同事已经在 MCP 里实现了一个完整 agent，你只需发 query：
    answer = client.chat(query)

    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=answer,
        )
    )
    return answer
