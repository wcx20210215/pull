import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from utils import dataframe_agent
from datetime import datetime
import io
def create_chart(input_data, chart_type):
    """生成统计图表"""
    df_data = pd.DataFrame(
        data={
            "x": input_data["columns"],
            "y": input_data["data"]
        }
    ).set_index("x")
    if chart_type == "bar":
        fig = px.bar(x=input_data["columns"], y=input_data["data"], 
                     title="柱状图分析", color=input_data["data"])
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "line":
        fig = px.line(x=input_data["columns"], y=input_data["data"], 
                      title="趋势分析", markers=True)
        st.plotly_chart(fig, use_container_width=True)

def display_data_overview(df):
    """显示数据概览"""
    st.subheader("📊 数据概览")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总行数", len(df))
    with col2:
        st.metric("总列数", len(df.columns))
    with col3:
        st.metric("数值列", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("缺失值", df.isnull().sum().sum())
    
    # 数据类型分布
    st.subheader("📋 列信息")
    col_info = pd.DataFrame({
        '列名': df.columns,
        '数据类型': df.dtypes.astype(str),
        '非空值': df.count(),
        '缺失值': df.isnull().sum(),
        '唯一值': df.nunique()
    })
    st.dataframe(col_info, use_container_width=True)

def data_cleaning_section(df):
    """数据清洗功能"""
    st.subheader("🧹 数据清洗")
    
    cleaning_option = st.selectbox(
        "选择清洗操作",
        ["查看缺失值", "删除缺失值", "填充缺失值", "删除重复值", "数据类型转换"]
    )
    
    if cleaning_option == "查看缺失值":
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            st.write("缺失值统计:")
            st.bar_chart(missing_data)
        else:
            st.success("数据中没有缺失值！")
    
    elif cleaning_option == "删除缺失值":
        if st.button("删除包含缺失值的行"):
            cleaned_df = df.dropna()
            st.success(f"删除了 {len(df) - len(cleaned_df)} 行包含缺失值的数据")
            st.session_state["df"] = cleaned_df
            st.rerun()
    
    elif cleaning_option == "填充缺失值":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col_to_fill = st.selectbox("选择要填充的列", numeric_cols)
            fill_method = st.selectbox("填充方法", ["均值", "中位数", "众数"])
            
            if st.button("执行填充"):
                if fill_method == "均值":
                    df[col_to_fill].fillna(df[col_to_fill].mean(), inplace=True)
                elif fill_method == "中位数":
                    df[col_to_fill].fillna(df[col_to_fill].median(), inplace=True)
                elif fill_method == "众数":
                    df[col_to_fill].fillna(df[col_to_fill].mode()[0], inplace=True)
                
                st.session_state["df"] = df
                st.success(f"已用{fill_method}填充 {col_to_fill} 列的缺失值")
                st.rerun()
    
    elif cleaning_option == "删除重复值":
        duplicates = df.duplicated().sum()
        st.write(f"发现 {duplicates} 行重复数据")
        if duplicates > 0 and st.button("删除重复值"):
            df_cleaned = df.drop_duplicates()
            st.session_state["df"] = df_cleaned
            st.success(f"删除了 {duplicates} 行重复数据")
            st.rerun()

def statistical_analysis(df):
    """统计分析功能"""
    st.subheader("📈 统计分析")
    
    analysis_type = st.selectbox(
        "选择分析类型",
        ["描述性统计", "相关性分析", "分布分析"]
    )
    
    if analysis_type == "描述性统计":
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            st.write("数值列统计摘要:")
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.warning("没有数值列可以进行统计分析")
    
    elif analysis_type == "相关性分析":
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          title="相关性热力图")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("需要至少2个数值列才能进行相关性分析")
    
    elif analysis_type == "分布分析":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("选择要分析的列", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                # 直方图
                fig_hist = px.histogram(df, x=selected_col, title=f"{selected_col} 分布直方图")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # 箱线图
                fig_box = px.box(df, y=selected_col, title=f"{selected_col} 箱线图")
                st.plotly_chart(fig_box, use_container_width=True)

def advanced_visualization(df):
    """高级可视化功能"""
    st.subheader("🎨 高级可视化")
    
    chart_type = st.selectbox(
        "选择图表类型",
        ["散点图", "饼图", "热力图", "小提琴图"]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if chart_type == "散点图" and len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X轴", numeric_cols)
        with col2:
            y_col = st.selectbox("Y轴", numeric_cols)
        
        color_col = None
        if len(categorical_cols) > 0:
            color_col = st.selectbox("颜色分组（可选）", ["无"] + list(categorical_cols))
            color_col = None if color_col == "无" else color_col
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                        title=f"{x_col} vs {y_col} 散点图")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "饼图" and len(categorical_cols) > 0:
        cat_col = st.selectbox("选择分类列", categorical_cols)
        value_counts = df[cat_col].value_counts()
        
        fig = px.pie(values=value_counts.values, names=value_counts.index, 
                    title=f"{cat_col} 分布饼图")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "热力图" and len(numeric_cols) > 1:
        selected_cols = st.multiselect("选择要显示的列", numeric_cols, default=list(numeric_cols[:5]))
        if selected_cols:
            corr_data = df[selected_cols].corr()
            fig = px.imshow(corr_data, text_auto=True, aspect="auto", 
                          title="选定列相关性热力图")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("当前数据不支持所选图表类型，请检查数据类型和列数量")

def export_data_section(df):
    """数据导出功能"""
    st.subheader("💾 数据导出")
    
    export_format = st.selectbox("选择导出格式", ["CSV", "Excel", "JSON"])
    
    if export_format == "CSV":
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="下载CSV文件",
            data=csv,
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    elif export_format == "Excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='数据', index=False)
        
        st.download_button(
            label="下载Excel文件",
            data=output.getvalue(),
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    elif export_format == "JSON":
        json_data = df.to_json(orient='records', force_ascii=False, indent=2)
        st.download_button(
            label="下载JSON文件",
            data=json_data,
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
# 页面配置
st.set_page_config(
    page_title="小脑瓜数据分析智能体",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主标题
st.title("🧠 小脑瓜数据分析智能体")
st.markdown("---")

# 侧边栏功能选择
with st.sidebar:
    st.header("🎯 功能选择")
    function_choice = st.selectbox(
        "选择功能模块",
        ["数据上传", "数据概览", "数据清洗", "统计分析", "高级可视化", "AI问答", "数据导出"]
    )
    
    st.markdown("---")
    st.markdown("### 📋 使用说明")
    st.markdown("""
    1. 首先上传数据文件
    2. 选择相应功能模块
    3. 根据需要进行数据分析
    4. 导出处理结果
    """)

# 数据上传区域
if function_choice == "数据上传" or "df" not in st.session_state:
    st.subheader("📁 数据上传")
    option = st.radio("请选择数据文件类型:", ("Excel", "CSV"))
    file_type = "xlsx" if option == "Excel" else "csv"
    data = st.file_uploader(f"上传你的{option}数据文件", type=file_type)
    if data:
        if file_type == "xlsx":
            wb = openpyxl.load_workbook(data)
            sheet_option = st.radio(label="请选择要加载的工作表：", options=wb.sheetnames)
            st.session_state["df"] = pd.read_excel(data, sheet_name=sheet_option)
        else:
            st.session_state["df"] = pd.read_csv(data)
        
        st.success("✅ 数据上传成功！")
        with st.expander("🔍 预览原始数据"):
            st.dataframe(st.session_state["df"], use_container_width=True)

# 功能模块展示
if "df" in st.session_state:
    df = st.session_state["df"]
    
    if function_choice == "数据概览":
        display_data_overview(df)
    
    elif function_choice == "数据清洗":
        data_cleaning_section(df)
    
    elif function_choice == "统计分析":
        statistical_analysis(df)
    
    elif function_choice == "高级可视化":
        advanced_visualization(df)
    
    elif function_choice == "数据导出":
        export_data_section(df)
    
    elif function_choice == "AI问答":
        st.subheader("🤖 AI智能问答")
        
        # 快速问题模板
        st.markdown("#### 💡 快速问题模板")
        quick_questions = [
            "显示数据的基本统计信息",
            "找出数值最大的前5行数据", 
            "生成销售额的柱状图",
            "显示各类别的分布情况",
            "计算数值列之间的相关性"
        ]
        
        selected_template = st.selectbox("选择问题模板（可选）", ["自定义问题"] + quick_questions)
        
        if selected_template != "自定义问题":
            query = st.text_area(
                "💬 请输入你关于数据集的问题或可视化需求：",
                value=selected_template,
                height=100
            )
        else:
            query = st.text_area(
                "💬 请输入你关于数据集的问题或可视化需求：",
                placeholder="例如：显示销售额最高的前5个地区的柱状图",
                height=100
            )
        
        button = st.button("🚀 生成回答", type="primary")
        
        if query and button:
            with st.spinner("🤔 AI正在思考中，请稍等..."):
                result = dataframe_agent(df, query)
                
                st.markdown("### 🎯 分析结果")
                
                if "answer" in result:
                    st.success(result["answer"])
                
                if "table" in result:
                    st.markdown("#### 📊 数据表格")
                    result_df = pd.DataFrame(result["table"]["data"],
                                           columns=result["table"]["columns"])
                    st.dataframe(result_df, use_container_width=True)
                
                if "bar" in result:
                    st.markdown("#### 📊 柱状图分析")
                    create_chart(result["bar"], "bar")
                
                if "line" in result:
                    st.markdown("#### 📈 趋势分析")
                    create_chart(result["line"], "line")

else:
    if function_choice != "数据上传":
        st.warning("⚠️ 请先上传数据文件才能使用此功能")
        st.info("👈 请在侧边栏选择'数据上传'功能")