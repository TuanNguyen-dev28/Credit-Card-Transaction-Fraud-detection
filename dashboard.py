"""
Credit Card Fraud Detection - Streamlit Dashboard
==============================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluator import ModelEvaluator, find_optimal_threshold

# Page config
st.set_page_config(
    page_title="Phát Hiện Gian Lận Thẻ Tín Dụng",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .fraud-alert {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .safe-transaction {
        background-color: #ccffcc;
        border-left: 5px solid #00cc00;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# Title
st.markdown('<p class="main-header">💳 Trang Quản Lý Phát Hiện Gian Lận Thẻ Tín Dụng</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Cài Đặt")

# Load model
@st.cache_resource
def load_model():
    """Load the trained model."""
    model_path = Path(__file__).parent / "saved_models" / "model.pkl"
    fe_path = Path(__file__).parent / "saved_models" / "feature_engineer.pkl"
    preproc_path = Path(__file__).parent / "saved_models" / "preprocessor.pkl"

    model = None
    feature_engineer = None
    preprocessor = None

    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    if fe_path.exists():
        with open(fe_path, "rb") as f:
            feature_engineer = pickle.load(f)

    if preproc_path.exists():
        with open(preproc_path, "rb") as f:
            preprocessor = pickle.load(f)

    return model, feature_engineer, preprocessor


model, feature_engineer, preprocessor = load_model()

if model is None:
    st.warning("⚠️ Chưa có model đã huấn luyện. Vui lòng chạy huấn luyện trước.")
    st.code("python main.py train")
    st.stop()

# Model selection
st.sidebar.subheader("Thông Tin Model")
st.sidebar.info(f"**Model:** {getattr(model, 'name', 'Unknown')}")

# Threshold slider
threshold = st.sidebar.slider(
    "Ngưỡng Gian Lận",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Giao dịch có xác suất gian lận trên ngưỡng này sẽ bị đánh dấu"
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng Quan",
    "🔍 Kiểm Tra Giao Dịch",
    "📈 Hiệu Suất Model",
    "📉 Phân Tích Ngưỡng"
])

# Tab 1: Overview
with tab1:
    st.header("📊 Tổng Quan Dữ Liệu")

    # Load sample data
    try:
        train_df = pd.read_csv(
            Path(__file__).parent / "dataset" / "fraudTrain.csv",
            nrows=10000
        )
        test_df = pd.read_csv(
            Path(__file__).parent / "dataset" / "fraudTest.csv",
            nrows=10000
        )
        df = pd.concat([train_df, test_df], ignore_index=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Tổng Giao Dịch", f"{len(df):,}")
        with col2:
            fraud_count = df["is_fraud"].sum()
            st.metric("Số Giao Dịch Gian Lận", f"{fraud_count:,}")
        with col3:
            fraud_rate = df["is_fraud"].mean() * 100
            st.metric("Tỷ Lệ Gian Lận", f"{fraud_rate:.3f}%")
        with col4:
            st.metric("Số Loại", df["category"].nunique())

        st.divider()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Phân Bố Gian Lận")
            fig = px.pie(
                names=["Hợp Lệ", "Gian Lận"],
                values=[len(df) - fraud_count, fraud_count],
                hole=0.4,
                color=["#2ecc71", "#e74c3c"]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Giao Dịch Theo Loại")
            category_counts = df["category"].value_counts()
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                color=category_counts.values,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                height=400,
                xaxis_title="Loại Giao Dịch",
                yaxis_title="Số Lượng",
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Phân Bố Số Tiền Giao Dịch")
            fig = px.histogram(
                df,
                x="amt",
                color="is_fraud",
                barmode="overlay",
                nbins=50,
                color_discrete_map={0: "#2ecc71", 1: "#e74c3c"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Tỷ Lệ Gian Lận Theo Loại")
            category_fraud = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False)
            fig = px.bar(
                x=category_fraud.index,
                y=category_fraud.values * 100,
                color=category_fraud.values,
                color_continuous_scale="Reds"
            )
            fig.update_layout(
                height=400,
                xaxis_title="Loại Giao Dịch",
                yaxis_title="Tỷ Lệ Gian Lận (%)",
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")

# Tab 2: Transaction Scanner
with tab2:
    st.header("🔍 Kiểm Tra Giao Dịch")
    st.write("Nhập thông tin giao dịch để kiểm tra khả năng gian lận:")

    col1, col2 = st.columns(2)

    with col1:
        amt = st.number_input("Số Tiền Giao Dịch ($)", min_value=0.01, value=100.0, step=1.0)
        category = st.selectbox("Loại Giao Dịch", [
            "food_dining", "gas_transport", "grocery_pos", "grocery_net",
            "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
            "personal_care", "shopping_net", "shopping_pos", "travel", "entertainment"
        ], index=10)
        merchant = st.text_input("Tên Cửa Hàng", "Amazon")
        cc_num = st.text_input("Số Thẻ (4 số cuối)", "1234")

    with col2:
        trans_date = st.date_input("Ngày Giao Dịch", datetime.now())
        trans_time = st.time_input("Giờ Giao Dịch", datetime.now().time())
        gender = st.selectbox("Giới Tính", ["M", "F"])
        state = st.selectbox("Bang", ["NY", "CA", "TX", "FL", "PA", "IL", "OH", "GA", "NC", "MI"])

    # Advanced options
    with st.expander("Tùy Chọn Nâng Cao"):
        col1, col2 = st.columns(2)
        with col1:
            city_pop = st.number_input("Dân Số Thành Phố", value=1000000, step=10000)
            lat = st.number_input("Vĩ Độ Chủ Thẻ", value=40.7128, format="%.4f")
            long = st.number_input("Kinh Độ Chủ Thẻ", value=-74.0060, format="%.4f")
        with col2:
            merch_lat = st.number_input("Vĩ Độ Cửa Hàng", value=40.7580, format="%.4f")
            merch_long = st.number_input("Kinh Độ Cửa Hàng", value=-73.9855, format="%.4f")
            age = st.number_input("Tuổi Chủ Thẻ", min_value=18, max_value=100, value=35)

    if st.button("🔍 Kiểm Tra Giao Dịch", type="primary"):
        # Prepare transaction data
        trans_datetime = datetime.combine(trans_date, trans_time)

        transaction_data = {
            "trans_date_trans_time": trans_datetime,
            "cc_num": f"123456789012{cc_num}",
            "merchant": f"fraud_{merchant}",
            "category": category,
            "amt": float(amt),
            "first": "Test",
            "last": "User",
            "gender": gender,
            "street": "123 Test St",
            "city": "Test City",
            "state": state,
            "zip": "12345",
            "lat": float(lat),
            "long": float(long),
            "city_pop": int(city_pop),
            "job": "Test Job",
            "dob": datetime(1990, 1, 1),
            "trans_num": f"test_{int(time.time())}",
            "unix_time": int(trans_datetime.timestamp()),
            "merch_lat": float(merch_lat),
            "merch_long": float(merch_long)
        }

        # Process and predict
        try:
            df = pd.DataFrame([transaction_data])

            if feature_engineer:
                df = feature_engineer.transform(df)

            # Drop columns
            drop_cols = [
                "trans_date_trans_time", "cc_num", "merchant", "first", "last",
                "street", "city", "job", "dob", "trans_num", "unix_time", "zip",
                "lat", "long", "merch_lat", "merch_long", "is_fraud"
            ]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            if preprocessor:
                df = preprocessor.transform(df)

            df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

            # Predict
            prob = float(model.predict_proba(df)[:, 1][0])
            is_fraud = prob >= threshold

            # Display result
            st.divider()

            if is_fraud:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h3>⚠️ CẢNH BÁO GIAN LẬN</h3>
                    <p>Giao dịch này có <strong>{prob*100:.2f}%</strong> khả năng là gian lận.</p>
                    <p><strong>Khuyến nghị:</strong> Xem xét và xác minh giao dịch này trước khi phê duyệt.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-transaction">
                    <h3>✅ GIAO DỊCH AN TOÀN</h3>
                    <p>Giao dịch này có <strong>{(1-prob)*100:.2f}%</strong> khả năng là hợp lệ.</p>
                    <p><strong>Khuyến nghị:</strong> Giao dịch có vẻ bình thường. Có thể xử lý.</p>
                </div>
                """, unsafe_allow_html=True)

            # Progress bar visualization
            st.write("**Đánh Giá Rủi Ro:**")
            progress_color = "red" if prob >= threshold else "green"
            st.progress(prob, text=f"Xác Suất Gian Lận: {prob*100:.2f}%")

            # Risk meter
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": progress_color},
                    "steps": [
                        {"range": [0, 50], "color": "lightgreen"},
                        {"range": [50, 75], "color": "yellow"},
                        {"range": [75, 100], "color": "red"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "value": threshold * 100
                    }
                },
                title={"text": "Mức Rủi Ro Gian Lận %"}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, width='stretch')

        except Exception as e:
            st.error(f"Lỗi khi xử lý giao dịch: {e}")

# Tab 3: Model Performance
with tab3:
    st.header("📈 Hiệu Suất Model")

    # Load test results if available
    results_path = Path(__file__).parent / "saved_models" / "training_results.csv"

    if results_path.exists():
        results_df = pd.read_csv(results_path)

        st.subheader("So Sánh Các Model")

        # Display metrics table
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            best_model = results_df.loc[results_df["test_f1"].idxmax()]
            st.metric("Model Tốt Nhất", best_model["model"])
        with col2:
            st.metric("F1 Score Cao Nhất", f"{best_model['test_f1']:.4f}")
        with col3:
            st.metric("ROC-AUC Cao Nhất", f"{best_model['test_roc_auc']:.4f}")
        with col4:
            st.metric("Recall Cao Nhất", f"{best_model['test_recall']:.4f}")

        st.divider()

        # Chart comparing models
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Điểm F1", "ROC-AUC", "Recall", "Precision"],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        metrics = ["test_f1", "test_roc_auc", "test_recall", "test_precision"]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for metric, (row, col) in zip(metrics, positions):
            fig.add_trace(
                go.Bar(
                    x=results_df["model"],
                    y=results_df[metric],
                    marker_color=px.colors.qualitative.Set2
                ),
                row=row, col=col
            )

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    else:
        st.info("Chưa có kết quả huấn luyện. Chạy huấn luyện để xem các chỉ số hiệu suất.")

        # Show sample metrics
        st.subheader("Các Chỉ Số Mong Đợi (từ huấn luyện)")

        metrics_placeholder = st.empty()
        metrics_placeholder.info("Huấn luyện model trước bằng lệnh:")
        st.code("python main.py train", language="bash")

# Tab 4: Threshold Analysis
with tab4:
    st.header("📉 Phân Tích Ngưỡng")
    st.write("Điều chỉnh ngưỡng để cân bằng giữa phát hiện gian lận và báo động sai.")

    # Interactive threshold analysis
    threshold_test = st.slider(
        "Chọn Ngưỡng",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    st.write(f"### Ngưỡng Hiện Tại: {threshold_test:.2f}")

    # Show metrics at different thresholds
    st.subheader("Các Chỉ Số Theo Ngưỡng")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_data = []

    # Load test data for threshold analysis
    try:
        test_df = pd.read_csv(
            Path(__file__).parent / "dataset" / "fraudTest.csv",
            nrows=5000
        )

        # Quick threshold analysis
        if feature_engineer:
            test_df = feature_engineer.transform(test_df)

        drop_cols = [
            "trans_date_trans_time", "cc_num", "merchant", "first", "last",
            "street", "city", "job", "dob", "trans_num", "unix_time", "zip",
            "lat", "long", "merch_lat", "merch_long"
        ]
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")

        if preprocessor:
            test_df = preprocessor.transform(test_df)

        test_df = test_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        if "is_fraud" in test_df.columns:
            y_true = test_df["is_fraud"]
            X_test = test_df.drop(columns=["is_fraud"])

            # Get probabilities
            y_proba = model.predict_proba(X_test)[:, 1]

            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)

                tp = ((y_true == 1) & (y_pred == 1)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                tn = ((y_true == 0) & (y_pred == 0)).sum()
                fn = ((y_true == 1) & (y_pred == 0)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                threshold_data.append({
                    "Ngưỡng": thresh,
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1 Score": round(f1, 4),
                    "Phát Hiện Gian Lận": int(tp),
                    "Báo Động Sai": int(fp)
                })

            threshold_df = pd.DataFrame(threshold_data)

            # Display chart
            fig = px.line(
                threshold_df,
                x="Ngưỡng",
                y=["Precision", "Recall", "F1 Score"],
                markers=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

            # Display table
            st.dataframe(threshold_df, width='stretch')

            # Recommendations
            st.divider()
            st.subheader("Khuyến Nghị")

            best_f1_idx = threshold_df["F1 Score"].idxmax()
            best_thresh = threshold_df.loc[best_f1_idx, "Ngưỡng"]

            st.success(f"""
            **Dựa trên tối ưu F1 Score:**
            - Ngưỡng khuyến nghị: **{best_thresh:.2f}**
            - Ở ngưỡng này, phát hiện được **{threshold_df.loc[best_f1_idx, 'Phát Hiện Gian Lận']}** giao dịch gian lận
            - Với **{threshold_df.loc[best_f1_idx, 'Báo Động Sai']}** báo động sai
            """)

    except Exception as e:
        st.error(f"Lỗi khi phân tích ngưỡng: {e}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Trang Quản Lý Phát Hiện Gian Lận Thẻ Tín Dụng | Xây dựng với Streamlit</p>
</div>
""", unsafe_allow_html=True)
