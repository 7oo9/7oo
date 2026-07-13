import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from prophet import Prophet
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
.metric-card{
    background-color:#ffffff;
    padding:20px 24px;
    border-radius:12px;
    border:1px solid #e2e8f0;
    text-align:left;
}
.metric-card .label{
    color:#64748b;
    font-size:13px;
    font-weight:500;
    text-transform:uppercase;
    letter-spacing:0.5px;
}
.metric-card .value{
    color:#1e293b;
    font-size:28px;
    font-weight:600;
    margin-top:8px;
}
.metric-card .delta{
    font-size:12px;
    margin-top:4px;
    padding:2px 8px;
    border-radius:4px;
    display:inline-block;
}
h1{
    color:#1e293b;
    font-weight:600;
}
.stSelectbox > div > div {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="湖南景点热度分析与短期预测系统", layout="wide")
st.title("热度趋势预测")

@st.cache_data
def load_raw_data():
    data = pd.read_excel("景点热度_含指数.xlsx")
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce")
    data = data.dropna(subset=["日期", "综合热度指数"])
    return data

df = load_raw_data()
scenic_all = df["景点"].unique().tolist()

cluster_dict = {
    "类别0｜持续高热型": ["张家界"],
    "类别1｜稳定均衡热点": ["五一广场", "岳阳楼", "凤凰古城", "岳麓山", "韶山", "芙蓉镇", "橘子洲"],
    "类别2｜双峰宗教季节性": ["南岳衡山"],
    "类别3｜夏季避暑主导型": ["东江湖", "仰天湖"]
}

with st.sidebar:
    st.header("交互控制面板")
    st.divider()
    select_scenic = st.selectbox("选择查看景区", scenic_all)
    st.divider()
    st.subheader("聚类说明")
    for cluster_name, scenic_list in cluster_dict.items():
        st.markdown(f"**{cluster_name}**")
        st.write("、".join(scenic_list))
        st.write("")
    st.divider()
    st.caption("数据来源：百度指数、抖音、小红书")
    st.caption("分析模型：K-Means + Prophet")

scenic_df = df[df["景点"] == select_scenic][["日期", "综合热度指数"]].sort_values("日期").reset_index(drop=True)
scenic_df.columns = ["ds", "y"]

if len(scenic_df) < 10:
    st.error(f"⚠️ {select_scenic} 有效月度数据不足，无法完成预测！")
else:
    model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True)
    model.fit(scenic_df)
    future = model.make_future_dataframe(periods=3, freq="ME")
    forecast = model.predict(future)

    st.markdown(f"#### {select_scenic} 热度趋势与短期预测")
    
    max_date = scenic_df["ds"].max()
    pred_only = forecast[forecast["ds"] > max_date]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=scenic_df['ds'], y=scenic_df['y'],
        mode='markers', marker=dict(color='#1e293b', size=6),
        name='历史观测值'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', line=dict(color='#3b82f6', width=2),
        name='拟合曲线'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        mode='lines', line=dict(color='rgba(59, 130, 246, 0.2)', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        mode='lines', line=dict(color='rgba(59, 130, 246, 0.2)', width=0),
        fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)',
        name='置信区间'
    ))
    
    fig.add_trace(go.Scatter(
        x=pred_only['ds'], y=pred_only['yhat'],
        mode='lines', line=dict(color='#ef4444', width=3),
        name='未来3个月预测'
    ))
    
    fig.add_trace(go.Scatter(
        x=pred_only['ds'], y=pred_only['yhat_upper'],
        mode='lines', line=dict(color='rgba(239, 68, 68, 0.2)', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=pred_only['ds'], y=pred_only['yhat_lower'],
        mode='lines', line=dict(color='rgba(239, 68, 68, 0.2)', width=0),
        fill='tonexty', fillcolor='rgba(239, 68, 68, 0.1)',
        name='预测区间'
    ))
    
    fig.update_layout(
        title=f'{select_scenic} 热度时序拟合与短期预测',
        xaxis_title='月度日期',
        yaxis_title='综合热度指数',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400,
        margin=dict(l=50, r=50, t=60, b=50),
        template='plotly_white'
    )
    
    fig.update_xaxes(tickangle=40)
    fig.update_yaxes(gridcolor='#f1f5f9')
    
    st.plotly_chart(fig, use_container_width=True)

    recent_6m = scenic_df["y"].tail(6).mean()
    pred_3m_mean = pred_only["yhat"].mean()
    diff = pred_3m_mean - recent_6m
    if diff > 0.02:
        trend_tag = "热度上升"
        trend_color = "#22c55e"
        delta_color = "background-color:#dcfce7;color:#166534;"
    elif diff < -0.02:
        trend_tag = "热度下降"
        trend_color = "#ef4444"
        delta_color = "background-color:#fee2e2;color:#991b1b;"
    else:
        trend_tag = "热度平稳"
        trend_color = "#64748b"
        delta_color = "background-color:#f1f5f9;color:#475569;"

    train = scenic_df[scenic_df["ds"] <= "2025-12-31"]
    test = scenic_df[scenic_df["ds"] >= "2026-01-01"]
    mae, rmse = np.nan, np.nan

    if len(test) >= 3:
        val_model = Prophet(weekly_seasonality=False, daily_seasonality=False)
        val_model.fit(train)
        val_future = val_model.make_future_dataframe(periods=5, freq="ME")
        val_pred = val_model.predict(val_future)
        val_pred = val_pred.assign(ym=val_pred["ds"].dt.to_period("M"))
        test = test.assign(ym=test["ds"].dt.to_period("M"))
        merge_val = pd.merge(test, val_pred[["ym", "yhat"]], on="ym")
        mae = mean_absolute_error(merge_val["y"], merge_val["yhat"])
        mse_score = mean_squared_error(merge_val["y"], merge_val["yhat"])
        rmse = np.sqrt(mse_score)

    st.markdown("#### 量化指标")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">近6月均值</div>
            <div class="value">{recent_6m:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">未来3月预测</div>
            <div class="value">{pred_3m_mean:.4f}</div>
            <div class="delta" style="{delta_color}">{diff:+.4f}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">趋势判断</div>
            <div class="value" style="color:{trend_color}">{trend_tag}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">RMSE误差</div>
            <div class="value">{rmse:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("解读说明"):
        st.markdown("""
        - **蓝色曲线**：历史拟合热度，包含置信区间
        - **红色线段**：未来3个月核心预测区间
        - **综合热度**：取值0~1，差值阈值±0.02区分上升/平稳/下降
        - **RMSE**：数值越小，模型拟合精度越高
        """)

st.divider()
st.caption("湖南文旅多源网络热度时序分析可视化演示系统")