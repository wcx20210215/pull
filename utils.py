"""
utils - 数据分析智能体使用的工具函数

Author: 骆昊
Version: 0.1
Date: 2025/6/25
"""
import json
import re
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

PROMPT_TEMPLATE = """你是一个专业的数据分析助手。请严格按照以下格式要求回答用户问题：

**重要：你的回答必须是有效的JSON格式，不能包含任何其他文本！**

根据用户问题类型，选择以下其中一种JSON格式回答：

1. **纯文字回答**：
{"answer": "你的分析结果"}

2. **表格数据**：
{"table": {"columns": ["列名1", "列名2"], "data": [["值1", "值2"], ["值3", "值4"]]}}

3. **柱状图数据**：
{"bar": {"columns": ["类别1", "类别2"], "data": [数值1, 数值2]}}

4. **折线图数据**：
{"line": {"columns": ["时间1", "时间2"], "data": [数值1, 数值2]}}

5. **饼图数据**：
{"pie": {"columns": ["类别1", "类别2"], "data": [数值1, 数值2]}}

**格式要求**：
- 必须使用英文双引号
- 字符串值用双引号包围
- 数值不用引号
- 确保JSON格式完整有效
- 不要添加任何解释文字

**示例**：
正确：{"answer": "销售总额为1000万元"}
错误：分析结果：{"answer": "销售总额为1000万元"}

当前用户请求如下：\n"""


def clean_and_fix_json(text):
    """清理和修复JSON格式"""
    if not text or not isinstance(text, str):
        return None
    
    # 移除可能的前缀和后缀文本
    text = text.strip()
    
    # 查找JSON开始和结束位置
    json_start = text.find('{')
    if json_start == -1:
        return None
    
    # 从最后一个}开始向前查找完整的JSON
    json_end = text.rfind('}') + 1
    if json_end <= json_start:
        return None
    
    json_text = text[json_start:json_end]
    
    # 尝试修复常见的JSON格式问题
    try:
        # 移除可能的控制字符
        json_text = re.sub(r'[\n\r\t]', '', json_text)
        
        # 修复单引号为双引号
        json_text = re.sub(r"(?<!\\)'", '"', json_text)
        
        # 修复没有引号的键名
        json_text = re.sub(r'(\w+):', r'"\1":', json_text)
        
        # 尝试解析
        result = json.loads(json_text)
        return result
    except:
        # 如果修复失败，尝试原始文本
        try:
            return json.loads(json_text)
        except:
            return None


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
        output = response["output"]
        
        # 打印原始输出用于调试
        print(f"原始AI输出: {output}")
        
        # 使用新的JSON清理函数
        if output.strip():
            # 尝试清理和解析JSON
            cleaned_result = clean_and_fix_json(output)
            
            if cleaned_result and isinstance(cleaned_result, dict):
                return cleaned_result
            
            # 如果JSON解析失败，返回文本答案
            return {"answer": output.strip()}
        else:
            return {"answer": "AI返回了空的响应，请重新尝试"}
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"原始输出: {response.get('output', 'No output')}")
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