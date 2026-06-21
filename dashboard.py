"""
Project dashboard: Multilingual Health QA Challenge (Zindi x ITU x HASH)
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Multilingual Health QA — Project Dashboard",
    page_icon="🌍",
    layout="wide",
)

def metric_card(label, value):
    st.markdown(
        f"""
        <div style="background-color:#1E222A; border:1px solid #333944;
                    border-radius:10px; padding:18px 20px; text-align:left;">
            <div style="color:#9CA3AF; font-size:14px; margin-bottom:6px;">{label}</div>
            <div style="color:#FAFAFA; font-size:28px; font-weight:700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.title("🌍 Multilingual Health Question Answering")
st.markdown("##### Low-Resource African Languages Challenge — Zindi × ITU × HASH")
st.markdown(
    "A multilingual model that answers maternal, sexual, and reproductive health "
    "questions in the same language they were asked — across **English, Akan, "
    "Amharic, Luganda, and Swahili**."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Public Leaderboard Score", "0.278")
with col2:
    metric_card("Languages", "5")
with col3:
    metric_card("Training Examples", "29,815")
with col4:
    metric_card("Model", "mT5-small")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Dataset", "🎯 Performance", "🛠️ Approach"])

with tab1:
    st.header("Dataset Overview")
    dataset_info = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Rows": [29815, 6686, 2618],
        "Has Answers": ["Yes", "Yes", "No (generated)"],
    })
    st.dataframe(dataset_info, use_container_width=True, hide_index=True)

    st.subheader("Training Data by Language & Country")
    subset_counts = pd.DataFrame({
        "Subset": ["Eng_Uga", "Aka_Gha", "Eng_Gha", "Eng_Eth", "Lug_Uga", "Eng_Ken", "Swa_Ken", "Amh_Eth"],
        "Language": ["English", "Akan", "English", "English", "Luganda", "English", "Swahili", "Amharic"],
        "Country": ["Uganda", "Ghana", "Ghana", "Ethiopia", "Uganda", "Kenya", "Kenya", "Ethiopia"],
        "Training Rows": [7624, 4455, 4443, 3915, 3383, 2080, 2070, 1845],
    })
    fig_dist = px.bar(
        subset_counts.sort_values("Training Rows", ascending=True),
        x="Training Rows", y="Subset", color="Language", orientation="h",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_dist.update_layout(height=420, margin=dict(t=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"))
    st.plotly_chart(fig_dist, use_container_width=True)
    st.info("💡 **Note:** Data is unevenly distributed — English (Uganda) has 4x more "
        "examples than Amharic, which contributes to uneven performance across languages.")

with tab2:
    st.header("Model Performance by Language")
    st.caption("ROUGE scores measured on a held-out validation sample (1,000 examples).")
    perf = pd.DataFrame({
        "Subset": ["Aka_Gha", "Eng_Eth", "Eng_Gha", "Swa_Ken", "Eng_Uga", "Eng_Ken", "Lug_Uga", "Amh_Eth"],
        "ROUGE-1": [0.4006, 0.3721, 0.3552, 0.2384, 0.2598, 0.2253, 0.1546, 0.0247],
        "ROUGE-L": [0.2510, 0.3465, 0.2586, 0.1782, 0.1898, 0.1594, 0.1225, 0.0288],
    })
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(name="ROUGE-1", x=perf["Subset"], y=perf["ROUGE-1"], marker_color="#4FC3F7"))
    fig_perf.add_trace(go.Bar(name="ROUGE-L", x=perf["Subset"], y=perf["ROUGE-L"], marker_color="#F06292"))
    fig_perf.update_layout(barmode="group", yaxis_title="Score", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"))
    st.plotly_chart(fig_perf, use_container_width=True)

    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        metric_card("Overall ROUGE-1 F1", "0.2713")
    with perf_col2:
        metric_card("Overall ROUGE-L F1", "0.1979")

    st.info("💡 **Insight:** English and Akan subsets perform strongly (ROUGE-1 above 0.35). "
        "Amharic (Ge'ez script) lags significantly — likely a tokenizer coverage "
        "limitation rather than a data volume issue alone.")

with tab3:
    st.header("Approach")
    approach_col1, approach_col2 = st.columns(2)
    with approach_col1:
        st.subheader("Model & Training")
        st.markdown("""
        - **Model**: `google/mt5-small`, fine-tuned end-to-end
        - **Input format**: question + language/country tag as prefix
        - **Training**: 3 epochs, effective batch size 16
        - **Precision**: `bf16` mixed precision
        - **Hardware**: Single 8GB consumer GPU
        - **Seed**: fixed at 42 for reproducibility
        """)
    with approach_col2:
        st.subheader("Generation")
        st.markdown("""
        - **Decoding**: beam search (4 beams)
        - **Anti-repetition**: `no_repeat_ngram_size=3`, `repetition_penalty=1.3`
        - **Impact**: ~22% improvement in ROUGE-1 over plain beam search
        - **Early stopping** enabled for cleaner sentence endings
        """)
    st.divider()
    st.subheader("Key Decisions")
    st.markdown("""
    1. **Benchmarked before committing** — measured real training speed on a small
       sample before running full epochs, avoiding wasted hours on infeasible configs.
    2. **mT5-base was tested and rejected** — looked stronger on paper, but showed
       unstable gradients and would have needed 20+ hours per run on the available
       hardware. mT5-small trained stably in ~3 hours.
    3. **Fixed a generation bug, not a training bug** — the biggest single ROUGE
       improvement came from tuning decoding parameters, not retraining.
    4. **Compliance-first** — open-source tools only, no AutoML, fixed seed,
       fully reproducible from the public repo.
    """)

st.divider()
st.caption("Built for the Zindi × ITU × HASH Multilingual Health QA Challenge — June 2026")
