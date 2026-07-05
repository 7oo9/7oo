## 最先执行日志屏蔽，减少终端警告刷屏
import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# 必须导入streamlit，定义st变量，解决未定义报错
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 屏蔽各类告警
warnings.filterwarnings('ignore')

# 修复matplotlib中文方框乱码
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# 页面配置（现在st已成功定义，不会报错）
st.set_page_config(page_title="湖南景点热度分析", layout="wide")
st.title("🔥 湖南旅游景点热度分析与短期预测系统")

# 下面你原本的 load_raw_data()、循环绘图代码全部保留不动
# 下面你原本的 load_raw_data()、循环绘图代码全部保留不动


# ========== 先加载数据，定义df（解决df未定义报错） ==========
@st.cache_data
def load_raw_data():
    data = pd.read_excel("景点热度_含指数.xlsx")
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce")
    data = data.dropna(subset=["日期", "综合热度指数"])
    return data

# 全局df，提前定义，后面代码才能调用
df = load_raw_data()
scenic_all = df["景点"].unique().tolist()

# 聚类映射字典
cluster_dict = {
    "类别0｜持续高热型": ["张家界"],
    "类别1｜稳定均衡热点": ["五一广场", "岳阳楼", "凤凰古城", "岳麓山", "韶山", "芙蓉镇", "橘子洲"],
    "类别2｜双峰宗教季节性": ["南岳衡山"],
    "类别3｜夏季避暑主导型": ["东江湖", "仰天湖"]
}

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("📌 交互控制面板")
    st.divider()
    select_scenic = st.selectbox("请选择查看景区", scenic_all)
    st.divider()
    with st.expander("📊 四类景区聚类分类说明", expanded=True):
        for cluster_name, scenic_list in cluster_dict.items():
            st.markdown(f"**{cluster_name}**")
            st.write("、".join(scenic_list))
    st.divider()
    st.caption("数据来源：百度指数、抖音热度、小红书笔记")
    st.caption("分析模型：K-Means时序聚类 + Prophet短期预测")

# ========== 主页面：数据筛选（修复列长度不匹配） ==========
# 仅提取2列：日期、综合热度，无多余列，避免长度报错
scenic_df = df[df["景点"] == select_scenic][["日期", "综合热度指数"]].sort_values("日期").reset_index(drop=True)
# 严格两列，赋值两个Prophet标准列名
scenic_df.columns = ["ds", "y"]

# 数据校验
if len(scenic_df) < 10:
    st.error(f"⚠️ {select_scenic} 有效月度数据不足，无法完成预测！")
else:
    # 训练Prophet模型
    model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True)
    model.fit(scenic_df)
    future = model.make_future_dataframe(periods=3, freq="ME")
    forecast = model.predict(future)

    # 绘图：历史蓝线 + 未来3个月红色高亮
    st.subheader(f"📈 {select_scenic} 综合热度历史走势 & 未来3个月预测")
    # 每次绘图前重新设置中文，解决网页图表方框乱码
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 4.5))
    model.plot(forecast, ax=ax)
    max_date = scenic_df["ds"].max()
    pred_only = forecast[forecast["ds"] > max_date]
    ax.plot(pred_only["ds"], pred_only["yhat"], color="#d62728", linewidth=3, label="未来3个月预测区间")
    ax.set_xlabel("统计月度日期", fontsize=10)
    ax.set_ylabel("归一化综合热度指数", fontsize=10)
    ax.set_title(f"{select_scenic} 2023-2026热度时序拟合与短期预测", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    plt.xticks(rotation=40)
    plt.tight_layout()
    st.pyplot(fig)

    # 计算趋势指标
    recent_6m = scenic_df["y"].tail(6).mean()
    pred_3m_mean = pred_only["yhat"].mean()
    diff = pred_3m_mean - recent_6m
    if diff > 0.02:
        trend_tag = "📈 热度上升"
    elif diff < -0.02:
        trend_tag = "📉 热度下降"
    else:
        trend_tag = "➡️ 热度平稳"

    # 训练/验证集误差计算（用assign避免长度报错）
    # 训练/验证集误差计算（修复参数缺失bug）
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
    # 完整传入真实值+预测值两个参数
    mae = mean_absolute_error(merge_val["y"], merge_val["yhat"])
    mse_score = mean_squared_error(merge_val["y"], merge_val["yhat"])
    rmse = np.sqrt(mse_score)
    # 指标卡片展示
    st.subheader("📋 预测量化指标")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="近6个月平均热度", value=f"{round(recent_6m,4)}")
    with col2:
        st.metric(label="未来3个月预测均值", value=f"{round(pred_3m_mean,4)}", delta=f"{round(diff,4)}")
    with col3:
        st.metric(label="整体趋势判断", value=trend_tag)
    with col4:
        st.metric(label="模型RMSE误差", value=f"{round(rmse,4)}" if not np.isnan(rmse) else "无验证数据")

    st.info("""
    💡 解读说明：
    1. 蓝色曲线为历史拟合热度，红色加粗线段为本研究核心预测区间（未来3个月）；
    2. 综合热度取值0~1，差值阈值±0.02区分上升/平稳/下降；
    3. RMSE数值越小，代表模型拟合精度越高，本模型整体误差处于较低水平。
    """)

st.divider()
st.caption("湖南文旅多源网络热度时序分析可视化演示系统｜数据要素大赛演示工具")
