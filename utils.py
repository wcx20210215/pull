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
from datetime import datetime
from typing import List, Dict, Any
import time
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# 记忆管理类
class ConversationMemory:
    def __init__(self, db_path="conversation_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建对话记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                question_hash TEXT,
                data_hash TEXT,
                timestamp DATETIME,
                response_time REAL
            )
        """)
        
        # 创建快速回答缓存表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quick_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_hash TEXT UNIQUE,
                data_hash TEXT,
                question TEXT,
                answer TEXT,
                hit_count INTEGER DEFAULT 1,
                last_used DATETIME,
                created_at DATETIME
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_question_hash(self, question: str) -> str:
        """生成问题的哈希值"""
        return hashlib.md5(question.lower().strip().encode('utf-8')).hexdigest()
    
    def get_data_hash(self, df) -> str:
        """生成数据的哈希值"""
        try:
            data_str = str(df.shape) + str(df.columns.tolist()) + str(df.dtypes.tolist())
            return hashlib.md5(data_str.encode('utf-8')).hexdigest()
        except:
            return "unknown"
    
    def add_conversation(self, session_id: str, question: str, answer: str, 
                        data_hash: str, response_time: float):
        """添加对话记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        question_hash = self.get_question_hash(question)
        
        cursor.execute("""
            INSERT INTO conversations 
            (session_id, question, answer, question_hash, data_hash, timestamp, response_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, question, answer, question_hash, data_hash, 
               datetime.now(), response_time))
        
        conn.commit()
        conn.close()
    
    def get_quick_answer(self, question: str, data_hash: str) -> Dict[str, Any]:
        """获取快速回答"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        question_hash = self.get_question_hash(question)
        
        cursor.execute("""
            SELECT answer, hit_count FROM quick_answers 
            WHERE question_hash = ? AND data_hash = ?
        """, (question_hash, data_hash))
        
        result = cursor.fetchone()
        
        if result:
            # 更新使用次数和最后使用时间
            cursor.execute("""
                UPDATE quick_answers 
                SET hit_count = hit_count + 1, last_used = ?
                WHERE question_hash = ? AND data_hash = ?
            """, (datetime.now(), question_hash, data_hash))
            
            conn.commit()
            conn.close()
            
            return {
                "found": True,
                "answer": result[0],
                "hit_count": result[1] + 1
            }
        
        conn.close()
        return {"found": False}
    
    def save_quick_answer(self, question: str, answer: str, data_hash: str):
        """保存快速回答"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        question_hash = self.get_question_hash(question)
        
        cursor.execute("""
            INSERT OR REPLACE INTO quick_answers 
            (question_hash, data_hash, question, answer, hit_count, last_used, created_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
        """, (question_hash, data_hash, question, answer, datetime.now(), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """获取对话历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT question, answer, timestamp, response_time 
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (session_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "question": row[0],
            "answer": row[1],
            "timestamp": row[2],
            "response_time": row[3]
        } for row in reversed(results)]
    
    def get_popular_questions(self, data_hash: str, limit: int = 5) -> List[Dict]:
        """获取热门问题"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT question, hit_count, last_used 
            FROM quick_answers 
            WHERE data_hash = ? 
            ORDER BY hit_count DESC 
            LIMIT ?
        """, (data_hash, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "question": row[0],
            "hit_count": row[1],
            "last_used": row[2]
        } for row in results]

# 流式输出处理器
class StreamingResponseHandler:
    def __init__(self):
        self.response_container = None
        self.current_text = ""
    
    def setup_container(self, container):
        """设置输出容器"""
        self.response_container = container
        self.current_text = ""
    
    def stream_text(self, text: str, delay: float = 0.02):
        """流式输出文本"""
        if self.response_container:
            for char in text:
                self.current_text += char
                self.response_container.markdown(self.current_text + "▌")
                time.sleep(delay)
            # 移除光标
            self.response_container.markdown(self.current_text)
    
    def stream_json_response(self, response_data: Dict, delay: float = 0.02):
        """流式输出JSON响应"""
        if "answer" in response_data:
            self.stream_text(response_data["answer"], delay)
        return response_data

# 全局记忆管理器实例
memory_manager = ConversationMemory()
streaming_handler = StreamingResponseHandler()

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

def get_enhanced_model():
    return ChatOpenAI(
        base_url='https://twapi.openai-hk.com/v1',
        api_key=st.secrets['API_KEY'],
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=4096
    )

def dataframe_agent(df, query, use_cache=True,
                   session_id=None, enable_streaming=False, stream_container=None):
    """增强版数据分析智能体 - 支持记忆和流式输出"""
    
    start_time = time.time()
    
    # 生成数据哈希
    data_hash = memory_manager.get_data_hash(df)
    
    # 检查是否有快速回答
    if use_cache:
        quick_answer = memory_manager.get_quick_answer(query, data_hash)
        if quick_answer["found"]:
            try:
                cached_result = json.loads(quick_answer["answer"])
                
                # 如果启用流式输出
                if enable_streaming and stream_container:
                    streaming_handler.setup_container(stream_container)
                    return streaming_handler.stream_json_response(cached_result, delay=0.01)
                
                # 记录快速回答的使用
                if session_id:
                    response_time = time.time() - start_time
                    memory_manager.add_conversation(
                        session_id, query, quick_answer["answer"], 
                        data_hash, response_time
                    )
                
                return cached_result
            except json.JSONDecodeError:
                # 如果缓存的答案不是有效JSON，继续正常处理
                pass
    
    # 生成缓存键（保持原有缓存逻辑）
    df_hash = hash(str(df.values.tobytes()) + str(df.columns.tolist()))
    query_hash = hash(query)
    
    # 尝试从原有缓存获取结果
    if use_cache:
        try:
            cached_result = cached_dataframe_analysis(df_hash, query_hash, query)
            if cached_result:
                return cached_result
        except:
            pass
    
    # 获取模型
    model = get_enhanced_model()
    
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
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        # 保存到记忆中
        result_json = json.dumps(result, ensure_ascii=False)
        if session_id:
            memory_manager.add_conversation(
                session_id, query, result_json, data_hash, response_time
            )
        
        # 保存为快速回答（如果结果有效）
        if "answer" in result or "table" in result or "bar" in result or "line" in result:
            memory_manager.save_quick_answer(query, result_json, data_hash)
        
        # 如果启用流式输出
        if enable_streaming and stream_container:
            streaming_handler.setup_container(stream_container)
            return streaming_handler.stream_json_response(result)
            
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        error_result = {"answer": "分析结果格式错误，请重新尝试"}
        
        # 记录错误到对话历史
        if session_id:
            response_time = time.time() - start_time
            memory_manager.add_conversation(
                session_id, query, json.dumps(error_result, ensure_ascii=False), 
                data_hash, response_time
            )
        
        return error_result
        
    except Exception as err:
        print(f"分析错误: {err}")
        error_messages = [
            "暂时无法提供分析结果，请稍后重试！",
            "数据分析遇到问题，请检查数据格式或简化问题",
            "AI分析超时，请尝试更简单的问题",
            "分析过程中出现错误，请重新描述您的需求"
        ]
        import random
        error_result = {"answer": random.choice(error_messages)}
        
        # 记录错误到对话历史
        if session_id:
            response_time = time.time() - start_time
            memory_manager.add_conversation(
                session_id, query, json.dumps(error_result, ensure_ascii=False), 
                data_hash, response_time
            )
        
        return error_result

def multi_model_analysis(df, query):
    """数据分析 - 使用单一模型"""
    try:
        result = dataframe_agent(df, query, use_cache=False)
        return result
    except Exception as e:
        print(f"模型分析失败: {e}")
        return {"answer": "分析失败，请检查数据或问题"}

# 记忆管理辅助函数
def get_session_id():
    """获取或创建会话ID"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}_{hash(str(time.time()))}"
    return st.session_state.session_id

def display_conversation_history(df, limit=5):
    """显示对话历史"""
    session_id = get_session_id()
    data_hash = memory_manager.get_data_hash(df)
    
    history = memory_manager.get_conversation_history(session_id, limit)
    
    if history:
        for i, conv in enumerate(history):
            question_preview = conv['question'][:50] if len(conv['question']) > 50 else conv['question']
            st.markdown(f"**💬 问题 {i+1}:** {question_preview}...")
            
            with st.container():
                st.markdown(f"**完整问题:** {conv['question']}")
                try:
                    answer_data = json.loads(conv['answer'])
                    if "answer" in answer_data:
                        st.markdown(f"**回答:** {answer_data['answer']}")
                    if "table" in answer_data:
                        st.markdown("**数据表格:**")
                        result_df = pd.DataFrame(answer_data["table"]["data"],
                                               columns=answer_data["table"]["columns"])
                        st.dataframe(result_df, use_container_width=True)
                except:
                    st.markdown(f"**回答:** {conv['answer']}")
                
                st.caption(f"⏱️ 响应时间: {conv['response_time']:.2f}秒 | 🕐 时间: {conv['timestamp']}")
                st.divider()
    else:
        st.info("暂无对话历史")

def display_popular_questions(df, limit=5):
    """显示热门问题"""
    data_hash = memory_manager.get_data_hash(df)
    popular = memory_manager.get_popular_questions(data_hash, limit)
    
    if popular:
        for i, item in enumerate(popular):
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📝 {item['question']}", key=f"popular_{i}"):
                    st.session_state.selected_question = item['question']
                    st.rerun()
            with col2:
                st.caption(f"🔥 {item['hit_count']}次")
    else:
        st.info("暂无热门问题")

def get_memory_stats(df):
    """获取记忆统计信息"""
    session_id = get_session_id()
    data_hash = memory_manager.get_data_hash(df)
    
    # 获取统计信息
    conn = sqlite3.connect(memory_manager.db_path)
    cursor = conn.cursor()
    
    # 当前会话问题数
    cursor.execute(
        "SELECT COUNT(*) FROM conversations WHERE session_id = ?", 
        (session_id,)
    )
    session_count = cursor.fetchone()[0]
    
    # 当前数据集的快速回答数
    cursor.execute(
        "SELECT COUNT(*) FROM quick_answers WHERE data_hash = ?", 
        (data_hash,)
    )
    quick_answers_count = cursor.fetchone()[0]
    
    # 总对话数
    cursor.execute("SELECT COUNT(*) FROM conversations")
    total_conversations = cursor.fetchone()[0]
    
    # 平均响应时间
    cursor.execute(
        "SELECT AVG(response_time) FROM conversations WHERE data_hash = ?", 
        (data_hash,)
    )
    avg_response_time = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "session_count": session_count,
        "quick_answers_count": quick_answers_count,
        "total_conversations": total_conversations,
        "avg_response_time": avg_response_time
    }

def clear_session_memory(session_id):
    """清除指定会话的记忆"""
    conn = sqlite3.connect(memory_manager.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "DELETE FROM conversations WHERE session_id = ?", 
        (session_id,)
    )
    
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return deleted_count

def clear_all_memory():
    """清除所有记忆数据"""
    conn = sqlite3.connect(memory_manager.db_path)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM conversations")
    cursor.execute("DELETE FROM quick_answers")
    
    conn.commit()
    conn.close()
    
    return True