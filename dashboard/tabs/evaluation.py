"""
Tab 7: Cross-Model Evaluation & Comparison
Owner: Himanshu (Evaluation Engineer)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (
    section_header, load_csv, apply_plotly_theme,
    LABELS, LABEL_COLORS,
)


def render():
    section_header(
        "Cross-Model Evaluation & Comparison",
        "Himanshu - Evaluation Engineer",
    )

    baseline_df = load_csv("task2_baseline_results.csv")
    bert_df = load_csv("bert_results.csv")
    model_c_df = load_csv("model_c_results.csv")

    # Same schema fix as the Sequential Transfer tab
    if model_c_df is not None:
        model_c_df = model_c_df.rename(columns={
            "Strategy": "stage",
            "Accuracy": "accuracy",
            "F1 Score": "f1_macro",
            "Precision": "precision_macro",
            "Recall": "recall_macro",
        })
        if "f1_weighted" not in model_c_df.columns:
            model_c_df["f1_weighted"] = model_c_df["f1_macro"]
    llm_df = load_csv("llm_results.csv")

    st.markdown("### Data Availability")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("TF-IDF Baselines", ":material/check_circle: Ready" if baseline_df is not None else ":material/cancel: Missing")
    s2.metric("BERT Models", ":material/check_circle: Ready" if bert_df is not None else ":material/pending: Pending")
    s3.metric("Sequential Transfer", ":material/check_circle: Ready" if model_c_df is not None else ":material/pending: Pending")
    s4.metric("LLM Experiments", ":material/check_circle: Ready" if llm_df is not None else ":material/pending: Pending")

    st.markdown("---")

    all_results = []

    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            best_per_condition = baseline_df.loc[
                baseline_df.groupby("Model Condition")["F1 (macro)"].idxmax()
            ]
        for _, row in best_per_condition.iterrows():
            all_results.append({
                "Approach": row["Model Condition"],
                "Method": f"TF-IDF + {row['Classifier']}",
                "Accuracy": row["Accuracy"],
                "F1 (Macro)": row["F1 (macro)"],
                "F1 (Weighted)": row["F1 (weighted)"],
                "Type": "TF-IDF Baseline",
            })

    if bert_df is not None:
        for _, row in bert_df.iterrows():
            all_results.append({
                "Approach": f"BERT {row['model']}",
                "Method": "BERT fine-tuned",
                "Accuracy": row["accuracy"],
                "F1 (Macro)": row["f1_macro"],
                "F1 (Weighted)": row["f1_weighted"],
                "Type": "BERT",
            })

    if model_c_df is not None:
        final_stage = model_c_df.iloc[-1]
        all_results.append({
            "Approach": "Sequential Transfer (Model C)",
            "Method": "BERT GoEmotions → FPB",
            "Accuracy": final_stage["accuracy"],
            "F1 (Macro)": final_stage["f1_macro"],
            "F1 (Weighted)": final_stage["f1_weighted"],
            "Type": "Sequential Transfer",
        })

    if llm_df is not None:
        for _, row in llm_df.iterrows():
            all_results.append({
                "Approach": f"{row['model']} ({row['strategy']})",
                "Method": "LLM prompting",
                "Accuracy": row["accuracy"],
                "F1 (Macro)": row["f1_macro"],
                "F1 (Weighted)": row["f1_weighted"],
                "Type": "LLM",
            })

    if not all_results:
        st.warning("No results available yet. CSVs will populate this tab automatically.")
        return

    results_df = pd.DataFrame(all_results)

    st.markdown("### F1 (Macro) - All Approaches")

    metric_choice = st.selectbox(
        "Select metric",
        ["F1 (Macro)", "F1 (Weighted)", "Accuracy"],
        key="eval_metric",
    )

    color_map = {
        "TF-IDF Baseline": "#8899aa",
        "BERT": "#00d4ff",
        "Sequential Transfer": "#2ecc71",
        "LLM": "#e74c3c",
    }

    fig = px.bar(
        results_df.sort_values(metric_choice, ascending=True),
        x=metric_choice,
        y="Approach",
        color="Type",
        orientation="h",
        title=f"{metric_choice} - All Models Compared",
        color_discrete_map=color_map,
        text=metric_choice,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=max(400, len(all_results) * 50), xaxis_range=[0, 1])
    st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    st.markdown("---")

    st.markdown("### Full Results Table")

    st.dataframe(
        results_df.sort_values("F1 (Macro)", ascending=False).style.format({
            "Accuracy": "{:.4f}",
            "F1 (Macro)": "{:.4f}",
            "F1 (Weighted)": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )

    st.markdown("---")

    st.markdown("### Key Takeaways")

    best = results_df.loc[results_df["F1 (Macro)"].idxmax()]
    worst = results_df.loc[results_df["F1 (Macro)"].idxmin()]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
        <div class="dashboard-card">
            <div class="card-title">:material/emoji_events: Best Performing</div>
            <p><strong>{best['Approach']}</strong></p>
            <p>Method: {best['Method']}</p>
            <p>F1 (Macro): <span style="color:#2ecc71; font-size:1.3rem; font-weight:bold">{best['F1 (Macro)']:.4f}</span></p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="dashboard-card">
            <div class="card-title">:material/trending_down: Weakest Performing</div>
            <p><strong>{worst['Approach']}</strong></p>
            <p>Method: {worst['Method']}</p>
            <p>F1 (Macro): <span style="color:#e74c3c; font-size:1.3rem; font-weight:bold">{worst['F1 (Macro)']:.4f}</span></p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    if baseline_df is not None:
        st.markdown("---")
        st.markdown("### Domain Gap Analysis")

        best_a = baseline_df[baseline_df["Model Condition"] == "Model A (General Only)"]["F1 (macro)"].max()
        best_b = baseline_df[baseline_df["Model Condition"] == "Model B (Domain Only)"]["F1 (macro)"].max()
        gap = best_b - best_a

        c1, c2, c3 = st.columns(3)
        c1.metric("Model A (General)", f"{best_a:.3f}")
        c2.metric("Model B (Domain)", f"{best_b:.3f}")
        c3.metric("Domain Gap", f"+{gap:.3f}", delta=f"{gap*100:.1f} pp")

        st.markdown(
            f"""
        <div class="highlight-box">
            <strong>Core finding:</strong> A {gap*100:.1f} percentage point macro F1 gap exists 
            between identical architectures trained on different data. This confirms that 
            domain adaptation is necessary and training data composition is as important as 
            model architecture for financial sentiment classification.
        </div>
        """,
            unsafe_allow_html=True,
        )