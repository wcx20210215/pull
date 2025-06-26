"""
utils - 数据分析智能体使用的工具函数

Author: 骆昊
Version: 0.1
Date: 2025/6/25
"""
import json
import streamlit as st
import time
from typing import List, Dict, Any, Generator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.schema import HumanMessage, AIMessage

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

def get_enhanced_model(model_choice="gpt-4o-mini"):
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

def dataframe_agent(df, query, model_choice="gpt-4o-mini", use_cache=True):
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


# ==================== 对话历史管理和流式响应功能 ====================

def initialize_conversation_memory():
    """初始化对话记忆"""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = int(time.time())

def add_message_to_memory(message: str, role: str = "user"):
    """将消息添加到对话记忆中
    
    Args:
        message: 消息内容
        role: 角色，"user" 或 "assistant"
    """
    initialize_conversation_memory()
    
    message_data = {
        "role": role,
        "content": message,
        "timestamp": time.time(),
        "conversation_id": st.session_state.current_conversation_id
    }
    
    st.session_state.conversation_history.append(message_data)
    
    # 限制历史记录长度，保留最近20条消息
    if len(st.session_state.conversation_history) > 20:
        st.session_state.conversation_history = st.session_state.conversation_history[-20:]

def get_conversation_history() -> List[Dict[str, Any]]:
    """获取完整的对话历史
    
    Returns:
        对话历史列表
    """
    initialize_conversation_memory()
    return st.session_state.conversation_history

def clear_conversation_history():
    """清空对话历史"""
    st.session_state.conversation_history = []
    st.session_state.current_conversation_id = int(time.time())

def format_conversation_for_ai(df) -> str:
    """将对话历史格式化为AI可理解的上下文
    
    Args:
        df: 当前数据框
        
    Returns:
        格式化的对话上下文
    """
    history = get_conversation_history()
    
    if not history:
        return ""
    
    context = "\n\n=== 对话历史上下文 ===\n"
    
    # 只包含最近5轮对话
    recent_history = history[-10:] if len(history) > 10 else history
    
    for msg in recent_history:
        role_name = "用户" if msg["role"] == "user" else "AI助手"
        context += f"{role_name}: {msg['content']}\n"
    
    context += "\n=== 当前数据集信息 ===\n"
    context += f"行数: {len(df)}, 列数: {len(df.columns)}\n"
    context += f"列名: {', '.join(df.columns.tolist())}\n"
    context += "\n请基于以上对话历史和当前问题提供连贯的回答。\n\n"
    
    return context

def stream_dataframe_agent(df, query: str, model_choice: str = "gpt-4o-mini") -> Generator[str, None, None]:
    """流式数据分析智能体
    
    Args:
        df: 数据框
        query: 用户查询
        model_choice: 模型选择
        
    Yields:
        流式响应的文本块
    """
    try:
        # 添加用户消息到记忆
        add_message_to_memory(query, "user")
        
        # 获取对话历史上下文
        conversation_context = format_conversation_for_ai(df)
        
        # 构建增强的查询
        enhanced_query = conversation_context + PROMPT_TEMPLATE + f"""
        
        数据集信息：
        - 行数：{len(df)}
        - 列数：{len(df.columns)}
        - 列名：{', '.join(df.columns.tolist())}
        - 数值列：{', '.join(df.select_dtypes(include=['number']).columns.tolist())}
        - 文本列：{', '.join(df.select_dtypes(include=['object']).columns.tolist())}
        
        用户问题：{query}
        
        请根据以上信息和对话历史提供准确、有用的分析结果。
        """
        
        # 获取模型
        model = get_enhanced_model(model_choice)
        
        # 创建智能体
        agent = create_pandas_dataframe_agent(
            llm=model,
            df=df,
            verbose=True,
            allow_dangerous_code=True
        )
        
        # 流式执行分析
        full_response = ""
        
        # 由于langchain的agent不直接支持流式，我们模拟流式输出
        yield "🤔 正在分析数据...\n"
        time.sleep(0.5)
        
        yield "📊 正在处理查询...\n"
        time.sleep(0.5)
        
        # 执行分析
        response = agent.invoke({"input": enhanced_query})
        result_text = response["output"]
        
        yield "✨ 分析完成，正在生成回答...\n\n"
        time.sleep(0.3)
        
        # 模拟逐字输出
        words = result_text.split()
        for i, word in enumerate(words):
            yield word + " "
            if i % 3 == 0:  # 每3个词暂停一下
                time.sleep(0.1)
        
        full_response = result_text
        
        # 将AI响应添加到记忆
        add_message_to_memory(full_response, "assistant")
        
    except Exception as e:
        error_msg = f"❌ 分析过程中出现错误: {str(e)}"
        yield error_msg
        add_message_to_memory(error_msg, "assistant")

def display_conversation_history():
    """显示对话历史"""
    history = get_conversation_history()
    
    if not history:
        st.info("暂无对话历史")
        return
    
    st.markdown("### 💬 对话历史")
    
    for msg in history:
        role_icon = "🧑‍💻" if msg["role"] == "user" else "🤖"
        role_name = "用户" if msg["role"] == "user" else "AI助手"
        
        with st.chat_message(msg["role"]):
             st.markdown(f"**{role_icon} {role_name}**")
             st.markdown(msg["content"])
             st.caption(f"时间: {time.strftime('%H:%M:%S', time.localtime(msg['timestamp']))}")