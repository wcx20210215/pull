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


@st.cache_data(ttl=1800)  # 缓存30分钟
def cached_dataframe_analysis(df_hash, query_hash, query):
    """缓存数据分析结果"""
    # 实际的分析逻辑会在dataframe_agent中执行
    return None

def get_enhanced_model(model_choice="gpt-4o"):
    """获取增强的AI模型"""
    model_configs = {
        "gpt-4o": {
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 8192
        },
        "gpt-4o-mini": {
            "model": "gpt-4o-mini", 
            "temperature": 0,
            "max_tokens": 4096
        },
        "gpt-4-turbo": {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.2,
            "max_tokens": 8192
        }
    }
    
    config = model_configs.get(model_choice, model_configs["gpt-4o-mini"])
    
    return ChatOpenAI(
        base_url='https://twapi.openai-hk.com/v1',
        api_key=st.secrets['API_KEY'],
        **config
    )

def dataframe_agent(df, query, model_choice="gpt-4o", use_cache=True):
    """增强版数据分析智能体"""
    
    # 生成缓存键
    df_hash = hash(str(df.values.tobytes()) + str(df.columns.tolist()))
    query_hash = hash(query)
    
    # 尝试从缓存获取结果
    if use_cache:
        try:
            cached_result = cached_dataframe_analysis(df_hash, query_hash, query)
            if cached_result:
                return cached_result
        except:
            pass
    
    # 选择模型
    model = get_enhanced_model(model_choice)
    
    # 创建智能体
    try:
        agent = create_pandas_dataframe_agent(
            llm=model,
            df=df,
            verbose=True,
            allow_dangerous_code=True
        )
    except Exception as e:
        st.error(f"创建智能体时出错: {str(e)}")
        return {"answer": "抱歉，AI智能体初始化失败，请稍后重试。"}

    # 增强的提示词
    enhanced_prompt = PROMPT_TEMPLATE + f"""
    
    数据集信息：
    - 行数：{len(df)}
    - 列数：{len(df.columns)}
    - 列名：{', '.join(df.columns.tolist())}
    - 数值列：{', '.join(df.select_dtypes(include=['number']).columns.tolist())}
    - 文本列：{', '.join(df.select_dtypes(include=['object']).columns.tolist())}
    
    用户问题：{query}
    
    请根据以上信息提供准确、有用的分析结果。
    """

    try:
        # 执行分析
        response = agent.invoke({"input": enhanced_prompt})
        result = json.loads(response["output"])
        
        # 验证结果格式
        if not isinstance(result, dict):
            raise ValueError("返回结果格式不正确")
            
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return {"answer": "分析结果格式错误，请重新尝试"}
    except Exception as err:
        print(f"分析错误: {err}")
        error_messages = [
            "暂时无法提供分析结果，请稍后重试！",
            "数据分析遇到问题，请检查数据格式或简化问题",
            "AI分析超时，请尝试更简单的问题",
            "分析过程中出现错误，请重新描述您的需求"
        ]
        import random
        return {"answer": random.choice(error_messages)}

def multi_model_analysis(df, query, models=["gpt-4o", "gpt-4o-mini"]):
    """多模型集成分析"""
    results = []
    
    for model in models:
        try:
            result = dataframe_agent(df, query, model_choice=model, use_cache=False)
            results.append({"model": model, "result": result})
        except Exception as e:
            print(f"模型 {model} 分析失败: {e}")
            continue
    
    if not results:
        return {"answer": "所有模型分析都失败了，请检查数据或问题"}
    
    # 简单的结果合并策略（可以根据需要改进）
    if len(results) == 1:
        return results[0]["result"]
    
    # 如果有多个结果，返回第一个成功的结果，并添加备注
    primary_result = results[0]["result"]
    if "answer" in primary_result:
        primary_result["answer"] += f" (基于{len(results)}个模型的分析结果)"
    
    return primary_result