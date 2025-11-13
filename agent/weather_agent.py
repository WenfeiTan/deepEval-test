# agent/weather_agent.py
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.metrics import TaskCompletionMetric, ArgumentCorrectnessMetric

from tools.weather_tool import weather_tool

# 指标先在这里 new 出来（你可以挪到 metrics/ 里，和之前框架统一）
task_completion_metric = TaskCompletionMetric()
arg_correctness_metric = ArgumentCorrectnessMetric()

# ===== 1. 你已经 setup 好的 Vertex 模型（示意） =====
from vertexai.generative_models import GenerativeModel
vertexai_chat_model = GenerativeModel("gemini-1.5-pro")  # 模型名按你实际来

def call_vertex_chat(prompt: str) -> str:
    chat = vertexai_chat_model.start_chat()
    resp = chat.send_message(prompt)
    return resp.text
# =====================================================


@observe(metrics=[arg_correctness_metric])
def llm_component(query: str):
    """
    这里当成 component-level：
    让 LLM 帮我们“理解”用户是不是在问天气 & 抽取城市、日期。

    为了简单，我就用 prompt 方式，让模型输出一个 JSON 字符串作为 tool 参数。
    （这样也能测 ArgumentCorrectness：看它的 JSON 合不合理）
    """
    prompt = f"""
你是一个参数抽取助手。用户的问题如下：

{query}

如果用户是在问天气，请输出一个 JSON，格式如下（不要多余文字）：
{{"is_weather": true, "city": "城市名", "date": "今天/明天/..."}}

如果不是问天气，输出：
{{"is_weather": false}}
"""

    raw = call_vertex_chat(prompt)
    # 这里简单一点，假设模型能老实输出 JSON（如果不行你再加点 post-processing）
    return raw


@observe(metrics=[task_completion_metric])
def weather_agent(query: str) -> str:
    """
    最终 Agent：
    1. 用 llm_component 抽取 tool 参数（component-level，测 arg correctness）
    2. 如果 is_weather = true，则调用 weather_tool
    3. 否则就让 Vertex 直接回答
    4. 把这次调用包装成一个 LLMTestCase，给 TaskCompletionMetric 用
    """
    import json

    raw_args = llm_component(query)
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        # 参数抽取全崩了，就直接当普通聊天
        answer = call_vertex_chat(query)
        update_current_span(
            test_case=LLMTestCase(
                input=query,
                actual_output=answer,
            )
        )
        return answer

    # 这里就可以测 ArgumentCorrectness：city/date 抽得好不好
    if args.get("is_weather"):
        city = args.get("city", "上海")
        date = args.get("date", "today")
        weather_info = weather_tool(city=city, date=date)
        answer = f"根据查询，{weather_info}"
    else:
        answer = call_vertex_chat(query)

    # 挂一个 test case，给 TaskCompletion 用
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=answer,
        )
    )
    return answer
