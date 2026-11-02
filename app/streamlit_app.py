import os
import sys
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

# Add project root to sys.path so inference module can be imported anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_API_URL = os.getenv("API_URL", "").rstrip("/")

st.set_page_config(
    page_title="Distributed Log Anomaly Detection",
    page_icon="🖥️",
    layout="wide"
)

@st.cache_resource(show_spinner="Loading LoRA DistilBERT Model into Memory...")
def load_local_predictor():
    from inference.predictor import LogPredictor
    return LogPredictor()

# Check sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    use_remote_api = st.checkbox("Connect to External FastAPI Server", value=bool(DEFAULT_API_URL))
    
    api_url = ""
    api_connected = False
    
    if use_remote_api:
        api_url = st.text_input("FastAPI Backend URL", value=DEFAULT_API_URL or "http://127.0.0.1:8000")
        st.markdown("---")
        st.subheader("Model Status")
        try:
            health = requests.get(f"{api_url}/", timeout=4)
            if health.status_code == 200:
                st.success("API Connected (Remote Mode)")
                api_connected = True
            else:
                st.warning(f"API Returned {health.status_code}")
        except Exception:
            st.error("API Offline — Falling back to Direct Mode")
    else:
        st.success("⚡ Running in Direct Mode (100% Free, In-Memory Model)")
        st.caption("Model runs directly inside Streamlit — no external backend or cloud fees needed!")

# If not connected to a remote API, load the local in-memory predictor
local_predictor = None
if not api_connected:
    try:
        local_predictor = load_local_predictor()
    except Exception as e:
        st.error(f"Error loading local model: {e}")

def run_single_prediction(log_text: str):
    if api_connected and api_url:
        resp = requests.post(f"{api_url}/predict", json={"log": log_text}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    elif local_predictor is not None:
        return local_predictor.predict(log_text)
    else:
        raise RuntimeError("No predictor available (neither remote API nor local model could be initialized).")

def run_batch_prediction(log_list: list):
    if api_connected and api_url:
        resp = requests.post(f"{api_url}/batch_predict", json={"logs": log_list}, timeout=120)
        resp.raise_for_status()
        return resp.json()
    elif local_predictor is not None:
        return local_predictor.predict_batch(log_list)
    else:
        raise RuntimeError("No predictor available (neither remote API nor local model could be initialized).")

st.title("Distributed System Log Anomaly Detection")

tab1, tab2 = st.tabs(
    [
        "Single Log",
        "Upload Log File"
    ]
)

with tab1:
    st.markdown("##### Quick Test Samples:")
    col_a, col_b, col_c, col_d = st.columns(4)

    if "log_input" not in st.session_state:
        st.session_state["log_input"] = "Receiving block blk_-1608999687919862906 src: /10.250.19.102:50010 dest: /10.250.19.102:50010"

    with col_a:
        if st.button("🟢 Normal Event"):
            st.session_state["log_input"] = "Receiving block blk_-1608999687919862906 src: /10.250.19.102:50010 dest: /10.250.19.102:50010"
    with col_b:
        if st.button("🚨 Disk Failure"):
            st.session_state["log_input"] = "DiskErrorException: Disk failure detected on /data/hdfs/data1 during write of block blk_8338592096525859757"
    with col_c:
        if st.button("🚨 Replica Corrupt"):
            st.session_state["log_input"] = "FATAL: All replicas corrupted for block blk_7915466481811490367"
    with col_d:
        if st.button("🚨 Connection Timeout"):
            st.session_state["log_input"] = "java.net.SocketTimeoutException: 60000 millis timeout while waiting for channel on DataNode 10.40.129.173:50010"

    log = st.text_area(
        "Enter Log",
        value=st.session_state["log_input"],
        height=140,
        placeholder="e.g. Receiving block blk_-1608999687919862906 src: /10.250.19.102:50010 dest: /10.250.19.102:50010"
    )

    if st.button("Predict"):
        if log.strip():
            try:
                with st.spinner("Classifying log event..."):
                    result = run_single_prediction(log)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Prediction", result["label"])
                with c2:
                    st.metric("Confidence", f'{result["confidence"]:.2f}%')
                with c3:
                    st.metric("Latency", f'{result["latency_ms"]:.2f} ms')

                if result["prediction"] == 1:
                    st.error("🚨 Anomaly Detected")
                else:
                    st.success("✅ Normal Log")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab2:
    uploaded_file = st.file_uploader(
        "Upload .log File",
        type=["log", "txt"]
    )

    if uploaded_file is not None:
        logs = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        ).splitlines()

        # Remove empty lines
        logs = [l for l in logs if l.strip()]

        if st.button("Run Batch Prediction"):
            try:
                with st.spinner(f"Scoring {len(logs)} log lines..."):
                    predictions = run_batch_prediction(logs)

                df = pd.DataFrame(predictions)
                total_logs = len(df)
                anomalies = (df["prediction"] == 1).sum()
                normal = total_logs - anomalies
                avg_conf = df["confidence"].mean()
                avg_latency = df["latency_ms"].mean()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Logs", total_logs)
                c2.metric("Normal", normal)
                c3.metric("Anomalies", anomalies)
                c4.metric("Avg Confidence", f"{avg_conf:.2f}%")
                c5.metric("Avg Latency", f"{avg_latency:.2f} ms")

                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
