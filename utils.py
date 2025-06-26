"""
utils - 数据分析智能体使用的工具函数

Author: 骆昊
Version: 0.1
Date: 2025/6/25
"""
import json
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

PROMPT_TEMPLATE = """数据分析助手登场！🚀数据分析就像一场冒险，而我就是你的向导。✨下面是我的魔法指南，让我们一起探索数据的奥秘：

1. **思考阶段 (Thought)** ：首先，我会像侦探一样分析你的请求类型（文字回答/表格/图表），并验证数据是否合适。🔍

2. **行动阶段 (Action)** ：根据分析结果，我会用以下魔法格式来回应你：✨
   - **纯文字回答**：（简洁明了，绝不啰嗦）
     {"answer": "不超过50个字符的明确答案"}👏

   - **表格数据**：（整齐排列，一目了然）
     {"table":{"columns":["列名1", "列名2", ...], "data":[["第一行值1", "值2", ...], ["第二行值1", "值2", ...]]}}📊

   - **柱状图**：（直观展示，高低立现）
     {"bar":{"columns": ["A", "B", "C", ...], "data":[35, 42, 29, ...]}}📈

   - **折线图**：（趋势尽显，一目了然）
     {"line":{"columns": ["A", "B", "C", ...], "data": [35, 42, 29, ...]}}📈

3. **格式校验要求**：（别担心，我会小心谨慎，确保万无一失）🛡️
   - 字符串值必须使用英文双引号（别问我为什么，这是规矩）🤔
   - 数值类型不得添加引号（数字就是数字，不需要伪装）🚫
   - 确保数组闭合无遗漏（我可不想漏掉任何重要信息）✅

   错误案例：（反面教材，千万别学）💩
   {'columns':['Product', 'Sales'], data:[[A001, 200]]}

   正确案例：（这才是正确的打开方式）👍
   {"columns":["product", "sales"], "data":[["A001", 200]]}

注意：响应数据的"output"中不要有换行符、制表符以及其他格式符号。（保持整洁，是我的原则）🧹

当前用户请求如下：\n"""


def dataframe_agent(df, query):
    model = ChatOpenAI(
        base_url='https://twapi.openai-hk.com/v1',
        api_key=st.secrets['API_KEY'],
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=8192
    )
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=32,
        allow_dangerous_code=True,
        verbose=True
    )

    prompt = PROMPT_TEMPLATE + query

    try:
        response = agent.invoke({"input": prompt})
        return json.loads(response["output"])
    except Exception as err:
        print(err)
        return {"answer": "暂时无法提供分析结果，请稍后重试！"}