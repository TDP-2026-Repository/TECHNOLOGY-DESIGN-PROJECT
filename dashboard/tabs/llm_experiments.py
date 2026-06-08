"""
Tab 6: LLM Zero-shot & Few-shot Experiments
Owner: Kevin (LLM Analyst)

STATUS: Not yet complete. Placeholder structure ready for results.
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
        "LLM Zero-shot & Few-shot Experiments",
        "Kevin - LLM Analyst",
    )

    llm_results = load_csv("llm_results.csv")

    st.markdown("### Experiment Design")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">Gemma 3 4B</div>
            <ul>
                <li>Google's lightweight open model</li>
                <li>Zero-shot: classify with label descriptions only</li>
                <li>Few-shot: 1–5 examples per class from FPB train</li>
                <li>No fine-tuning - prompt engineering only</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">Llama 3.2 3B</div>
            <ul>
                <li>Meta's compact instruction-tuned model</li>
                <li>Same zero-shot and few-shot protocols</li>
                <li>Comparison: architecture vs model size effect</li>
                <li>No fine-tuning - prompt engineering only</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    st.markdown(
        """
    <div class="highlight-box">
        <strong>Research question:</strong> Can modern small LLMs classify financial sentiment 
        without any training, purely through in-context learning? How does this compare to 
        fine-tuned BERT and TF-IDF baselines?
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    with st.expander(":material/chat: Prompt Templates"):
        st.markdown("**Zero-shot prompt:**")
        st.code(
            """
            Classify the sentiment of the following text into one of these categories: Optimism, Joy, Neutral, Fear, Sadness.\nRespond with only the category name.\n\nText: {text}\nCategory:
            """,
            language="text",
        )

        st.markdown("**Few-shot prompt:** Same as above, preceded by 5 labelled examples (one for each class).")
        #st.markdown("*Update with Kevin's actual prompts when available.*")

    st.markdown("---")

    if llm_results is None:
        st.markdown("### Results")
        st.info(
            ":material/folder: **Kevin:** Export your LLM results to `dashboard/data/llm_results.csv` "
            "with columns: model, strategy, accuracy, f1_macro, f1_weighted, "
            "f1_fear, f1_joy, f1_neutral, f1_optimism, f1_sadness"
        )

        st.markdown("### Baselines for Context")
        baselines = pd.DataFrame({
            "Model": ["TF-IDF B (SVM)", "BERT B (if available)"],
            "F1 (Macro)": [0.563, "TBD"],
            "Notes": [
                "Best TF-IDF baseline (fine-tuned on 3,392 samples)",
                "BERT fine-tuned on same data",
            ],
        })
        st.dataframe(baselines, width="stretch", hide_index=True)
        return

    st.markdown("### Results Overview")

    cols = st.columns(len(llm_results))
    for col, (_, row) in zip(cols, llm_results.iterrows()):
        with col:
            st.metric(
                label=f"{row['model']} ({row['strategy']})",
                value=f"F1: {row['f1_macro']:.3f}",
                delta=f"Acc: {row['accuracy']:.3f}",
            )

    st.markdown("### Detailed Comparison")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_models = st.multiselect(
            "Models",
            llm_results["model"].unique().tolist(),
            default=llm_results["model"].unique().tolist(),
        )
    with filter_col2:
        selected_strategies = st.multiselect(
            "Strategies",
            llm_results["strategy"].unique().tolist(),
            default=llm_results["strategy"].unique().tolist(),
        )

    filtered = llm_results[
        (llm_results["model"].isin(selected_models))
        & (llm_results["strategy"].isin(selected_strategies))
    ]

    filtered_display = filtered.copy()
    filtered_display["label"] = filtered_display["model"] + " - " + filtered_display["strategy"]

    fig = px.bar(
        filtered_display,
        x="label",
        y="f1_macro",
        color="model",
        title="Macro F1 - LLM Experiments",
        text="f1_macro",
        color_discrete_sequence=["#e74c3c", "#9b59b6"],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=400, yaxis_range=[0, 1], xaxis_title="")
    st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    st.markdown("### LLM vs Fine-tuned Models")

    all_models = [
        {"Model": "TF-IDF B (SVM)", "F1 (macro)": 0.563, "Type": "Fine-tuned Baseline"},
    ]
    for _, row in filtered.iterrows():
        all_models.append({
            "Model": f"{row['model']} ({row['strategy']})",
            "F1 (macro)": row["f1_macro"],
            "Type": "LLM (no training)",
        })

    fig = px.bar(
        pd.DataFrame(all_models),
        x="Model",
        y="F1 (macro)",
        color="Type",
        title="LLM vs Fine-tuned Approaches",
        color_discrete_map={
            "Fine-tuned Baseline": "#8899aa",
            "LLM (no training)": "#e74c3c",
        },
        text="F1 (macro)",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=400, yaxis_range=[0, 1])
    st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    with st.expander(":material/table_view: Full Results Table"):
        st.dataframe(filtered, width="stretch", hide_index=True)