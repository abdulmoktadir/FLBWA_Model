import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ============================================================
# Page Setup
# ============================================================
st.set_page_config(
    page_title="Fuzzy LBWA App",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }

    .hero-box {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 10px 28px rgba(0,0,0,0.12);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: 0.2px;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.92;
    }

    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.1rem 1.1rem 0.9rem 1.1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fbff 100%);
        border: 1px solid rgba(37, 99, 235, 0.15);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 20px rgba(30, 64, 175, 0.08);
    }

    .small-note {
        font-size: 0.92rem;
        color: #475569;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 700;
    }

    .top-factor {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #86efac;
        color: #14532d;
        font-weight: 700;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">📊 Fuzzy Level-Based Weight Assessment (LBWA)</div>
    <div class="hero-subtitle">
        Excel-aligned model: TFN → Fuzzy Influence → Reference Weight → Fuzzy Weights → Defuzzification → Normalization
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Helper Functions
# ============================================================
def scalar_divide_tfn(a, tfn):
    """
    For positive TFN B = (l, m, u):
    a / B = (a/u, a/m, a/l)
    """
    return np.column_stack([
        a / tfn[:, 2],
        a / tfn[:, 1],
        a / tfn[:, 0],
    ])

def defuzzify_weighted(tfn):
    """
    Weighted defuzzification:
    crisp = (l + 4m + u) / 6
    """
    return (tfn[:, 0] + 4 * tfn[:, 1] + tfn[:, 2]) / 6

def make_excel_file(input_df, tfn_df, influence_df, fuzzy_weight_df, result_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        input_df.to_excel(writer, sheet_name="Input", index=False)
        tfn_df.to_excel(writer, sheet_name="TFN", index=False)
        influence_df.to_excel(writer, sheet_name="Influence", index=False)
        fuzzy_weight_df.to_excel(writer, sheet_name="FuzzyWeights", index=False)
        result_df.to_excel(writer, sheet_name="Results", index=False)
    output.seek(0)
    return output.getvalue()

def highlight_top_factor(row):
    if row["Rank"] == 1:
        return ["background-color: #dcfce7; font-weight: 700; color: #14532d;"] * len(row)
    return [""] * len(row)

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("⚙️ Configuration")

num_factors = int(
    st.sidebar.number_input("Number of Factors", min_value=2, value=5, step=1)
)
num_experts = int(
    st.sidebar.number_input("Number of Experts", min_value=1, value=5, step=1)
)
theta = st.sidebar.number_input(
    "Theta (θ)",
    min_value=0.0001,
    value=2.1,
    step=0.1,
    format="%.4f"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Input Guide")
st.sidebar.info(
    "Edit the table directly.\n\n"
    "Use numeric Qi values like your Excel workbook.\n"
    "Example: 1, 1, 2, 5, 5"
)

# ============================================================
# Default Input Table
# ============================================================
factor_names = [f"Factor {i+1}" for i in range(num_factors)]
default_df = pd.DataFrame({
    "Factor": factor_names,
    "Qi": [1.0] * num_factors
})

for i in range(num_experts):
    default_df[f"E{i+1}"] = [0.0] * num_factors

# ============================================================
# Input Section
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("1) Enter Factor Information")
st.markdown(
    '<div class="small-note">Edit the table below for factor names, Qi values, and expert scores.</div>',
    unsafe_allow_html=True
)

edited_df = st.data_editor(
    default_df,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    column_config={
        "Factor": st.column_config.TextColumn("Factor", required=True),
        "Qi": st.column_config.NumberColumn("Qi", min_value=0.0, step=1.0, format="%.2f"),
        **{
            f"E{i+1}": st.column_config.NumberColumn(
                f"E{i+1}", min_value=0.0, step=0.1, format="%.4f"
            )
            for i in range(num_experts)
        }
    },
    key="lbwa_editor"
)

factor_options = list(range(num_factors))

def factor_label_func(idx):
    val = str(edited_df.iloc[idx]["Factor"]).strip()
    return f"{idx+1}. {val if val else f'Factor {idx+1}'}"

reference_index = st.selectbox(
    "Reference / Main Factor",
    options=factor_options,
    index=0,
    format_func=factor_label_func
)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Run Button
# ============================================================
run_model = st.button("▶ Run LBWA Model")

# ============================================================
# Processing
# ============================================================
if run_model:
    try:
        input_df = edited_df.copy()

        # Clean factor names
        input_df["Factor"] = input_df["Factor"].astype(str).str.strip()
        input_df["Factor"] = [
            name if name else f"Factor {i+1}"
            for i, name in enumerate(input_df["Factor"])
        ]

        # Validate required columns
        expected_cols = ["Factor", "Qi"] + [f"E{i+1}" for i in range(num_experts)]
        missing_cols = [c for c in expected_cols if c not in input_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        # Numeric conversion
        input_df["Qi"] = pd.to_numeric(input_df["Qi"], errors="coerce")
        for c in [f"E{i+1}" for i in range(num_experts)]:
            input_df[c] = pd.to_numeric(input_df[c], errors="coerce")

        if input_df["Qi"].isna().any():
            st.error("Qi contains invalid or empty values.")
            st.stop()

        if input_df[[f"E{i+1}" for i in range(num_experts)]].isna().any().any():
            st.error("One or more expert score cells contain invalid or empty values.")
            st.stop()

        data = input_df.iloc[:, 2:].astype(float).values
        qi_arr = input_df["Qi"].astype(float).values

        if np.any(qi_arr < 0):
            st.error("Qi values must be non-negative.")
            st.stop()

        if np.any(data < 0):
            st.error("Expert scores must be non-negative.")
            st.stop()

        if theta <= 0:
            st.error("Theta (θ) must be greater than zero.")
            st.stop()

        # --------------------------------------------------------
        # Step 1: TFN = (min, mean, max)
        # --------------------------------------------------------
        tfn = np.column_stack([
            np.min(data, axis=1),
            np.mean(data, axis=1),
            np.max(data, axis=1)
        ])

        tfn_df = pd.DataFrame(tfn, columns=["l", "m", "u"])
        tfn_df.insert(0, "Qi", qi_arr)
        tfn_df.insert(0, "Factor", input_df["Factor"])

        # --------------------------------------------------------
        # Step 2: Fuzzy Influence
        # denominator = (Qi*theta + l, Qi*theta + m, Qi*theta + u)
        # influence = theta / denominator with reversed TFN division
        # --------------------------------------------------------
        denominator_tfn = np.column_stack([
            qi_arr * theta + tfn[:, 0],
            qi_arr * theta + tfn[:, 1],
            qi_arr * theta + tfn[:, 2]
        ])

        influence = scalar_divide_tfn(theta, denominator_tfn)

        influence_df = pd.DataFrame(influence, columns=["l", "m", "u"])
        influence_df.insert(0, "Qi", qi_arr)
        influence_df.insert(0, "Factor", input_df["Factor"])

        # --------------------------------------------------------
        # Step 3: Reference Weight
        # --------------------------------------------------------
        mask_others = np.ones(num_factors, dtype=bool)
        mask_others[reference_index] = False

        ref_weight = np.array([
            1 / (1 + np.sum(influence[mask_others, 2])),
            1 / (1 + np.sum(influence[mask_others, 1])),
            1 / (1 + np.sum(influence[mask_others, 0]))
        ])

        # --------------------------------------------------------
        # Step 4: Fuzzy Weights
        # --------------------------------------------------------
        fuzzy_weights = np.zeros_like(influence)
        fuzzy_weights[reference_index] = ref_weight

        for i in range(num_factors):
            if i != reference_index:
                fuzzy_weights[i] = ref_weight * influence[i]

        fuzzy_weight_df = pd.DataFrame(fuzzy_weights, columns=["l", "m", "u"])
        fuzzy_weight_df.insert(0, "Factor", input_df["Factor"])

        # --------------------------------------------------------
        # Step 5: Defuzzification and normalization
        # --------------------------------------------------------
        crisp_values = defuzzify_weighted(fuzzy_weights)
        crisp_sum = np.sum(crisp_values)

        if crisp_sum == 0:
            st.error("The sum of crisp values is zero, so normalization cannot be done.")
            st.stop()

        normalized_weights = crisp_values / crisp_sum

        result_df = pd.DataFrame({
            "Factor": input_df["Factor"],
            "Qi": qi_arr,
            "Crisp Value": crisp_values,
            "Normalized Weight": normalized_weights
        })

        result_df["Rank"] = result_df["Normalized Weight"].rank(
            ascending=False,
            method="dense"
        ).astype(int)

        result_df = result_df[
            ["Rank", "Factor", "Qi", "Crisp Value", "Normalized Weight"]
        ].sort_values("Normalized Weight", ascending=False).reset_index(drop=True)

        excel_data = make_excel_file(
            input_df=input_df,
            tfn_df=tfn_df,
            influence_df=influence_df,
            fuzzy_weight_df=fuzzy_weight_df,
            result_df=result_df
        )

        top_factor = result_df.iloc[0]["Factor"]
        top_weight = result_df.iloc[0]["Normalized Weight"]

        # ========================================================
        # Summary
        # ========================================================
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("2) Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Reference Factor", input_df.iloc[reference_index]["Factor"])
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Sum of Crisp Values", f"{crisp_sum:.10f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Sum of Final Weights", f"{normalized_weights.sum():.10f}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="top-factor">🏆 Highest-ranked factor: {top_factor} '
            f'&nbsp;&nbsp;|&nbsp;&nbsp; Weight = {top_weight:.6f}</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ========================================================
        # Detailed Tables
        # ========================================================
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("3) Detailed Computation")

        tab1, tab2, tab3, tab4 = st.tabs([
            "TFN",
            "Influence",
            "Fuzzy Weights",
            "Final Results"
        ])

        with tab1:
            st.dataframe(tfn_df, use_container_width=True)

        with tab2:
            st.dataframe(influence_df, use_container_width=True)

        with tab3:
            st.dataframe(fuzzy_weight_df, use_container_width=True)

        with tab4:
            styled_result = (
                result_df.style
                .apply(highlight_top_factor, axis=1)
                .format({
                    "Qi": "{:.2f}",
                    "Crisp Value": "{:.10f}",
                    "Normalized Weight": "{:.10f}"
                })
            )
            st.dataframe(styled_result, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ========================================================
        # Visualization
        # ========================================================
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("4) Weight Distribution")

        chart_df = result_df.copy()
        chart_df["Label"] = np.where(
            chart_df["Rank"] == 1,
            "🏆 " + chart_df["Factor"],
            chart_df["Factor"]
        )
        chart_df = chart_df.set_index("Label")[["Normalized Weight"]]
        st.bar_chart(chart_df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ========================================================
        # Export
        # ========================================================
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("5) Export")
        st.download_button(
            label="📥 Download Excel Results",
            data=excel_data,
            file_name="LBWA_output_styled.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while running the model: {e}")
