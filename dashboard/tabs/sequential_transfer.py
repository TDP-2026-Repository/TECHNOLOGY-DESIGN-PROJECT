"""
Tab 5: Sequential Transfer Learning - Model C
Owner: Fin (Project Lead) + Bikram (assist)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (
    section_header, load_csv, apply_plotly_theme,
    LABELS, LABEL_COLORS,
)

def render():
    section_header(
        "Sequential Transfer Learning - Model C",
        "Fin (Lead) & Bikram (assist)",
    )

    # ── Try loading results ──
    model_c_results = load_csv("model_c_results.csv")

    # Fin's CSV uses a different schema (Strategy/Accuracy/F1 Score) — normalise it
    if model_c_results is not None:
        model_c_results = model_c_results.rename(columns={
            "Strategy": "stage",
            "Accuracy": "accuracy",
            "F1 Score": "f1_macro",
            "Precision": "precision_macro",
            "Recall": "recall_macro",
        })
        if "f1_weighted" not in model_c_results.columns:
            model_c_results["f1_weighted"] = model_c_results["f1_macro"]

    st.markdown("### Sequential Fine-tuning Pipeline")

    st.markdown(
        """
    <div class="highlight-box">
        <strong>Core hypothesis:</strong> Sequential fine-tuning (general → domain) should outperform 
        both single-domain training and naive data mixing by allowing BERT to first learn broad 
        emotion semantics, then adapt to financial vocabulary without catastrophic forgetting.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    col1, col2, col3 = st.columns([1, 0.2, 1])

    with col1:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">Stage 1: General Pre-training</div>
            <ul>
                <li><code>bert-base-uncased</code></li>
                <li>Fine-tune on GoEmotions (43,404 samples)</li>
                <li>Learn general emotion patterns</li>
                <li>Save checkpoint</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            "<div style='text-align:center; padding-top:60px; font-size:2rem; color:#00d4ff'>→</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">Stage 2: Domain Adaptation</div>
            <ul>
                <li>Load Stage 1 checkpoint</li>
                <li>Fine-tune on FPB Train (3,392 samples)</li>
                <li>Lower learning rate to prevent forgetting</li>
                <li>Evaluate on FPB Test (727 samples)</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### Why Sequential, Not Mixed?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">:material/close: Naive Mixing (TF-IDF Model C)</div>
            <p>Concatenating GoEmotions + FPB at 13:1 ratio <strong>diluted</strong> 
            financial vocabulary in TF-IDF weights.</p>
            <p>Result: Model C <strong>underperformed</strong> Model B (0.543 vs 0.563 macro F1)</p>
            <p style="color:#e74c3c">Bag-of-words cannot selectively weight domain tokens.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">:material/check_circle: Sequential Fine-tuning (BERT Model C)</div>
            <p>BERT's self-attention can <strong>selectively attend</strong> to domain-relevant 
            tokens regardless of training order.</p>
            <p>Sequential training lets the model build general representations first, 
            then <strong>specialize</strong> for financial text.</p>
            <p style="color:#2ecc71">Expected: Best of both worlds.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    if model_c_results is None:
        st.markdown("### Results")
        st.info(
            ":material/folder: **Fin:** Export Model C results to `dashboard/data/model_c_results.csv` "
            "with columns: stage, accuracy, f1_macro, f1_weighted, "
            "f1_fear, f1_joy, f1_neutral, f1_optimism, f1_sadness"
        )

        st.markdown("### Targets to Exceed")
        targets = pd.DataFrame({
            "Benchmark": [
                "TF-IDF Model B (best baseline)",
                "TF-IDF Model C (naive mix)",
                "BERT Model B (if available)",
            ],
            "F1 (Macro)": [0.563, 0.543, "TBD"],
            "Accuracy": [0.762, 0.755, "TBD"],
        })
        st.dataframe(targets, width="stretch", hide_index=True)
        return

    st.markdown("### Results")

    cols = st.columns(len(model_c_results))
    for col, (_, row) in zip(cols, model_c_results.iterrows()):
        with col:
            st.metric(
                label=row["stage"],
                value=f"F1: {row['f1_macro']:.3f}",
                delta=f"Acc: {row['accuracy']:.3f}",
            )

    st.markdown("### Full Pipeline Comparison")

    all_models = [
        {"Model": "TF-IDF A (SVM)", "F1 (macro)": 0.258, "Type": "TF-IDF Baseline"},
        {"Model": "TF-IDF B (SVM)", "F1 (macro)": 0.563, "Type": "TF-IDF Baseline"},
        {"Model": "TF-IDF C (SVM)", "F1 (macro)": 0.543, "Type": "TF-IDF Baseline"},
    ]

    for _, row in model_c_results.iterrows():
        all_models.append({
            "Model": f"BERT C - {row['stage']}",
            "F1 (macro)": row["f1_macro"],
            "Type": "BERT Sequential",
        })

    fig = px.bar(
        pd.DataFrame(all_models),
        x="Model",
        y="F1 (macro)",
        color="Type",
        title="Model C Sequential vs All Baselines",
        color_discrete_map={
            "TF-IDF Baseline": "#8899aa",
            "BERT Sequential": "#00d4ff",
        },
        text="F1 (macro)",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=450, yaxis_range=[0, 1])
    st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    perclass_cols = ["f1_fear", "f1_joy", "f1_neutral", "f1_optimism", "f1_sadness"]
    if len(model_c_results) >= 2 and all(c in model_c_results.columns for c in perclass_cols):
        st.markdown("### Per-Class F1: Before vs After Domain Adaptation")

        stage1 = model_c_results.iloc[0]
        stage2 = model_c_results.iloc[-1]

        perclass_comp = []
        for lbl, col_name in zip(LABELS, perclass_cols):
            perclass_comp.append({"Class": lbl, "Stage": stage1["stage"], "F1": stage1[col_name]})
            perclass_comp.append({"Class": lbl, "Stage": stage2["stage"], "F1": stage2[col_name]})

        fig = px.bar(
            pd.DataFrame(perclass_comp),
            x="Class",
            y="F1",
            color="Stage",
            barmode="group",
            title="Per-Class F1 Before and After FPB Fine-tuning",
            color_discrete_sequence=["#4C72B0", "#00d4ff"],
        )
        fig.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(apply_plotly_theme(fig), width="stretch")