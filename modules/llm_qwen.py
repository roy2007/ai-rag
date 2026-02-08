# modules/llm_qwen.py
import dashscope
from dashscope import Generation

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def call_qwen_for_answer(question: str, context: str) -> str:
    prompt = f"""你是一个企业知识助手，请基于以下参考资料回答问题。
要求：
1. 仅使用参考资料内容，不要编造；
2. 若资料不足，回答“未找到相关信息”；
3. 回答简洁准确。

参考资料：
{context}

问题：{question}

请回答："""
    resp = Generation.call(
        model="qwen-max",
        prompt=prompt,
        max_tokens=300,
        temperature=0.3
    )
    return resp.output.text if resp.status_code == 200 else "模型调用失败"


def qwen_intent(query: str) -> str:
    prompt = f"""判断用户意图，只输出一个类别：fact_query / operation_request / chitchat
问题：{query}"""
    resp = Generation.call(model="qwen-max", prompt=prompt, max_tokens=10)
    return resp.output.text.strip()

def qwen_rewrite(query: str) -> list[str]:
    prompt = f"""生成3个与以下问题语义相同但表述不同的问题，每行一个：
{query}"""
    resp = Generation.call(model="qwen-max", prompt=prompt, max_tokens=100)
    return [line.strip() for line in resp.output.text.split('\n') if line.strip()]