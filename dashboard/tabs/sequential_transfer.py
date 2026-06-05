"""
Tab 5: Sequential Transfer Learning (Transformer Stage)
Owner: Fin (Project Lead) + Bikram (assist)

Reads the canonical transformer results from bert_results.csv (Strategy 1/2/3).
Note: A/B/C labelling belongs to the TF-IDF baseline stage only; the transformer
stage uses Strategy 1 (Domain-only), Strategy 2 (Sequential), Strategy 3 (Mixed).
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
        "Transformer Fine-tuning & Sequential Transfer",
        "Fin (Lead) & Bikram (assist)",
    )

    # ── Canonical transformer results live in bert_results.csv ──
    # Schema: model, accuracy, f1_macro, f1_weighted, precision_macro,
    #         recall_macro, f1_fear, f1_joy, f1_neutral, f1_optimism, f1_sadness
    results = load_csv("bert_results.csv")
    if results is not None:
        results = results.rename(columns={"model": "stage"})

    st.markdown("### Sequential Fine-tuning Pipeline")

    st.markdown(
        """
    <div class="highlight-box">
        <strong>Core hypothesis:</strong> Sequential fine-tuning (general &rarr; domain) should match or
        exceed single-domain training by allowing BERT to first learn broad emotion
        semantics, then adapt to financial vocabulary without catastrophic forgetting.
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
            "<div style='text-align:center; padding-top:60px; font-size:2rem; color:#2a9d8f'>&rarr;</div>",
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
                <li>Lower learning rate to limit forgetting</li>
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
            <div class="card-title">:material/close: Naive Mixing (TF-IDF Baseline, Model C)</div>
            <p>Concatenating GoEmotions + FPB at roughly 13:1 <strong>diluted</strong>
            financial vocabulary in the TF-IDF weights.</p>
            <p>Result: the mixed baseline did not beat the domain-only baseline
            (0.543 vs 0.563 macro F1).</p>
            <p style="color:#e76f51">Bag-of-words cannot selectively weight domain tokens.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">:material/check_circle: Transformer Stage (Strategy 1/2/3)</div>
            <p>BERT's self-attention can <strong>selectively attend</strong> to
            domain-relevant tokens regardless of training mix.</p>
            <p>Sequential transfer builds general representations first, then
            <strong>specialises</strong> on financial text.</p>
            <p style="color:#2a9d8f">All three strategies cluster in the low-to-mid 0.80s accuracy.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    if results is None:
        st.markdown("### Results")
        st.info(
            ":material/folder: Transformer results not found. Ensure `bert_results.csv` "
            "is present in the data folder (columns: model, accuracy, f1_macro, "
            "f1_weighted, f1_fear, f1_joy, f1_neutral, f1_optimism, f1_sadness)."
        )

        st.markdown("### Targets to Exceed")
        targets = pd.DataFrame({
            "Benchmark": [
                "TF-IDF Model B (best baseline)",
                "TF-IDF Model C (naive mix)",
            ],
            "F1 (Macro)": [0.563, 0.543],
            "Accuracy": [0.762, 0.755],
        })
        st.dataframe(targets, width="stretch", hide_index=True)
        return

    st.markdown("### Results")

    cols = st.columns(len(results))
    for col, (_, row) in zip(cols, results.iterrows()):
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

    for _, row in results.iterrows():
        all_models.append({
            "Model": f"BERT - {row['stage']}",
            "F1 (macro)": row["f1_macro"],
            "Type": "BERT Transformer",
        })

    fig = px.bar(
        pd.DataFrame(all_models),
        x="Model",
        y="F1 (macro)",
        color="Type",
        title="Transformer Strategies vs TF-IDF Baselines",
        color_discrete_map={
            "TF-IDF Baseline": "#8899aa",
            "BERT Transformer": "#2a9d8f",
        },
        text="F1 (macro)",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=450, yaxis_range=[0, 1])
    st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    # ── Per-class F1 across the three strategies ──
    perclass_cols = ["f1_fear", "f1_joy", "f1_neutral", "f1_optimism", "f1_sadness"]
    if all(c in results.columns for c in perclass_cols):
        st.markdown("### Per-Class F1 by Strategy")
        perclass_comp = []
        for _, r in results.iterrows():
            for lbl, col_name in zip(LABELS, perclass_cols):
                perclass_comp.append({
                    "Class": lbl,
                    "Strategy": r["stage"],
                    "F1": r[col_name],
                })
        fig = px.bar(
            pd.DataFrame(perclass_comp),
            x="Class",
            y="F1",
            color="Strategy",
            barmode="group",
            title="Per-Class F1 Across Transformer Strategies",
            color_discrete_sequence=["#264653", "#2a9d8f", "#e76f51"],
        )
        fig.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(apply_plotly_theme(fig), width="stretch")

        st.caption(
            "Note: Fear F1 is 0.00 across all strategies because the test set "
            "contains only 7 Fear samples \u2014 too few to learn reliably. This is a "
            "documented data limitation, not a model fault."
        )