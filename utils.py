"""
utils - 数据分析智能体使用的工具函数

Author: 骆昊
Version: 0.1
Date: 2025/6/25
"""
import json
import streamlit as st
import sqlite3
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

PROMPT_TEMPLATE = """数据分析助手登场！🚀数据分析就像一场冒险，而我就是你的向导。✨下面是我的魔法指南，让我们一起探索数据的奥秘：

1. **思考阶段 (Thought)** ：首先，我会像侦探一样分析你的请求类型（文字回答/表格/图表），并验证数据是否合适。🔍

2. **行动阶段 (Action)** ：根据分析结果，我会用以下魔法格式来回应你：✨
   - **纯文字回答**：（简洁明了，绝不啰嗦）
     {"answer": "不超过200个字符的明确答案"}👏

   - **表格数据**：（整齐排列，一目了然）
     {"table":{"columns":["列名1", "列名2", ...], "data":[["第一行值1", "值2", ...], ["第二行值1", "值2", ...]]}}📊

   - **柱状图**：（直观展示，高低立现）
     {"bar":{"columns": ["类别A", "类别B", "类别C"], "data":[35, 42, 29]}}📈

   - **折线图**：（趋势尽显，一目了然）
     {"line":{"columns": ["时间1", "时间2", "时间3"], "data": [35, 42, 29]}}📈

   - **饼图**：（占比清晰，一目了然）
     {"pie":{"columns": ["部分A", "部分B", "部分C"], "data": [35, 42, 29]}}🥧

3. **重要格式要求**：🛡️
   - 必须返回有效的JSON格式
   - 字符串值必须使用英文双引号
   - 数值类型不要添加引号
   - 图表数据必须包含"columns"和"data"两个字段
   - columns是字符串数组，data是数值数组
   - columns和data的长度必须相等
   - 不要在JSON中添加注释或额外的格式符号

   **正确的图表格式示例：**
   {"bar":{"columns":["产品A", "产品B", "产品C"], "data":[100, 200, 150]}}
   {"line":{"columns":["1月", "2月", "3月"], "data":[50, 75, 90]}}
   {"pie":{"columns":["类型1", "类型2", "类型3"], "data":[30, 45, 25]}}

   **错误格式（不要这样做）：**
   {bar: {x: [A, B], y: [1, 2]}}  // 缺少引号，字段名错误
   {"bar": {"data": [1, 2]}}      // 缺少columns字段

4. **数据验证**：确保返回的数据格式完全正确，特别是图表可视化请求时。

当前用户请求如下：\n"""


class ChatMemory:
    """对话记忆管理类"""
    
    def __init__(self, db_path="chat_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建对话历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_hash TEXT
            )
        """)
        
        # 创建缓存表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                data_hash TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_session_id(self):
        """获取或创建会话ID"""
        if 'chat_session_id' not in st.session_state:
            st.session_state.chat_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return st.session_state.chat_session_id
    
    def generate_hash(self, content: str) -> str:
        """生成内容哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def add_message(self, user_message: str, ai_response: str, data_hash: str = None):
        """添加对话消息到记忆中"""
        session_id = self.get_session_id()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO chat_history (session_id, user_message, ai_response, data_hash)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_message, ai_response, data_hash))
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, data_hash: str = None, limit: int = 10) -> List[tuple]:
        """获取对话历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if data_hash:
            cursor.execute("""
                SELECT user_message, ai_response, timestamp
                FROM chat_history
                WHERE data_hash = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (data_hash, limit))
        else:
            session_id = self.get_session_id()
            cursor.execute("""
                SELECT user_message, ai_response, timestamp
                FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        # 反转顺序，使最新的在最后
        return list(reversed(results))
    
    def get_cached_response(self, query: str, data_hash: str) -> str:
        """获取缓存的响应"""
        query_hash = self.generate_hash(query + data_hash)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT response FROM response_cache
            WHERE query_hash = ? AND data_hash = ?
            AND datetime(timestamp) > datetime('now', '-1 hour')
        """, (query_hash, data_hash))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def cache_response(self, query: str, response: str, data_hash: str):
        """缓存响应"""
        query_hash = self.generate_hash(query + data_hash)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 使用 INSERT OR REPLACE 来更新缓存
        cursor.execute("""
            INSERT OR REPLACE INTO response_cache (query_hash, data_hash, query, response)
            VALUES (?, ?, ?, ?)
        """, (query_hash, data_hash, query, response))
        
        conn.commit()
        conn.close()
    
    def clear_session_history(self, data_hash: str = None):
        """清除会话历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if data_hash:
            cursor.execute("DELETE FROM chat_history WHERE data_hash = ?", (data_hash,))
        else:
            session_id = self.get_session_id()
            cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
            # 重新生成会话ID
            if 'chat_session_id' in st.session_state:
                del st.session_state.chat_session_id
        
        conn.commit()
        conn.close()

# 全局实例
chat_memory = ChatMemory()

def get_conversation_history(df, limit=10):
    """获取指定数据集的对话历史"""
    df_hash = chat_memory.generate_hash(str(df.values.tobytes()) + str(df.columns.tolist()))
    return chat_memory.get_chat_history(df_hash, limit)

def clear_conversation_history(df):
    """清除指定数据集的对话历史"""
    df_hash = chat_memory.generate_hash(str(df.values.tobytes()) + str(df.columns.tolist()))
    chat_memory.clear_session_history(df_hash)
    return True

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

def dataframe_agent_streaming(df, query, model_choice="gpt-4o-mini", use_cache=True, stream_container=None):
    """流式数据分析智能体"""
    
    # 生成数据哈希
    df_hash = chat_memory.generate_hash(str(df.values.tobytes()) + str(df.columns.tolist()))
    
    # 检查缓存
    if use_cache:
        cached_result = chat_memory.get_cached_response(query, df_hash)
        if cached_result:
            if stream_container:
                stream_container.success("✅ 从缓存获取结果")
            try:
                return json.loads(cached_result)
            except:
                pass
    
    # 将用户消息添加到记忆中（临时存储）
    if 'temp_user_message' not in st.session_state:
        st.session_state.temp_user_message = query
    
    try:
        # 获取模型
        model = get_enhanced_model(model_choice)
        
        # 构建增强提示
        dataset_info = f"数据集信息：{len(df)}行 x {len(df.columns)}列\n列名：{', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}"
        enhanced_prompt = f"{PROMPT_TEMPLATE}\n\n{dataset_info}\n\n用户问题：{query}"
        
        # 创建代理
        agent = create_pandas_dataframe_agent(
            llm=model,
            df=df,
            handle_parsing_errors=True,
            max_iterations=32,
            allow_dangerous_code=True,
            verbose=True,
            return_intermediate_steps=False
        )
        
        # 流式执行分析
        if stream_container:
            with stream_container:
                st.write("🤔 AI正在分析中...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 模拟流式处理
                import time
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("📊 正在理解数据结构...")
                    elif i < 60:
                        status_text.text("🔍 正在分析查询需求...")
                    elif i < 90:
                        status_text.text("⚡ 正在生成分析结果...")
                    else:
                        status_text.text("✨ 即将完成...")
                    time.sleep(0.02)
                
                progress_bar.empty()
                status_text.empty()
        
        # 执行分析
        result = agent.invoke({"input": enhanced_prompt})
        
        # 处理结果
        if isinstance(result, dict) and "output" in result:
            output = result["output"]
        else:
            output = str(result)
        
        # 尝试解析JSON
        try:
            parsed_result = json.loads(output)
            
            # 验证和修复图表数据格式
            for chart_type in ["bar", "line", "pie"]:
                if chart_type in parsed_result:
                    chart_data = parsed_result[chart_type]
                    
                    # 确保图表数据有正确的格式
                    if isinstance(chart_data, dict):
                        if "columns" not in chart_data or "data" not in chart_data:
                            # 尝试从其他可能的字段名中提取数据
                            if "x" in chart_data and "y" in chart_data:
                                chart_data["columns"] = chart_data.pop("x")
                                chart_data["data"] = chart_data.pop("y")
                            elif "labels" in chart_data and "values" in chart_data:
                                chart_data["columns"] = chart_data.pop("labels")
                                chart_data["data"] = chart_data.pop("values")
                            elif "categories" in chart_data and "values" in chart_data:
                                chart_data["columns"] = chart_data.pop("categories")
                                chart_data["data"] = chart_data.pop("values")
                        
                        # 确保数据类型正确
                        if "columns" in chart_data and "data" in chart_data:
                            if not isinstance(chart_data["columns"], list):
                                chart_data["columns"] = list(chart_data["columns"])
                            if not isinstance(chart_data["data"], list):
                                chart_data["data"] = list(chart_data["data"])
                    else:
                        # 如果chart_data不是字典，尝试重新构造
                        parsed_result[chart_type] = {"answer": f"图表数据格式错误: {str(chart_data)}"}
            
            # 验证表格数据格式
            if "table" in parsed_result:
                table_data = parsed_result["table"]
                if isinstance(table_data, dict):
                    if "columns" not in table_data or "data" not in table_data:
                        # 尝试修复表格格式
                        if "headers" in table_data and "rows" in table_data:
                            table_data["columns"] = table_data.pop("headers")
                            table_data["data"] = table_data.pop("rows")
                        elif "cols" in table_data and "rows" in table_data:
                            table_data["columns"] = table_data.pop("cols")
                            table_data["data"] = table_data.pop("rows")
                        
                    # 确保数据类型正确
                    if "columns" in table_data and "data" in table_data:
                        if not isinstance(table_data["columns"], list):
                            table_data["columns"] = list(table_data["columns"])
                        if not isinstance(table_data["data"], list):
                            table_data["data"] = list(table_data["data"])
                            
        except json.JSONDecodeError as e:
            # 如果不是JSON，尝试提取可能的JSON部分
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, output)
            
            if json_matches:
                try:
                    # 尝试解析找到的JSON
                    parsed_result = json.loads(json_matches[-1])  # 使用最后一个匹配的JSON
                except:
                    parsed_result = {"answer": output}
            else:
                # 如果完全不是JSON，包装为answer格式
                parsed_result = {"answer": output}
        except Exception as e:
            parsed_result = {"answer": f"结果解析错误: {str(e)}\n原始输出: {output}"}
        
        # 添加调试信息
        if 'debug_mode' in st.session_state and st.session_state.debug_mode:
            parsed_result['_debug_info'] = {
                'raw_output': output[:500] + '...' if len(output) > 500 else output,
                'output_type': type(output).__name__,
                'output_length': len(output)
            }
        
        # 缓存结果
        if use_cache:
            chat_memory.cache_response(query, json.dumps(parsed_result, ensure_ascii=False), df_hash)
        
        # 将完整对话添加到记忆中
        chat_memory.add_message(
            st.session_state.get('temp_user_message', query),
            json.dumps(parsed_result, ensure_ascii=False),
            df_hash
        )
        
        # 清理临时状态
        if 'temp_user_message' in st.session_state:
            del st.session_state.temp_user_message
        
        return parsed_result
        
    except Exception as e:
        error_msg = f"分析过程中出现错误: {str(e)}"
        if stream_container:
            stream_container.error(error_msg)
        
        # 记录错误到对话历史
        chat_memory.add_message(
            st.session_state.get('temp_user_message', query),
            error_msg,
            df_hash
        )
        
        return {"answer": error_msg}

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