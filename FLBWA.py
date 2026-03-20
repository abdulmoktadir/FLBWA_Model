# ============================================================
# Streamlit App: Fuzzy Level-Based Weight Assessment (LBWA)
# Corrected to match the Excel workbook logic
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Fuzzy LBWA App", layout="wide")
st.title("Fuzzy Level-Based Weight Assessment (LBWA)")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def scalar_divide_tfn(a, tfn):
    """
    For positive TFN B = (l, m, u):
    a / B = (a/u, a/m, a/l)
    """
    return np.column_stack([
        a / tfn[:, 2],  # l = a / u
        a / tfn[:, 1],  # m = a / m
        a / tfn[:, 0],  # u = a / l
    ])

def defuzzify_weighted(tfn):
    """
    Weighted defuzzification used in the Excel file:
    crisp = (l + 4m + u) / 6
    """
    return (tfn[:, 0] + 4 * tfn[:, 1] + tfn[:, 2]) / 6

# ------------------------------------------------------------
# Sidebar configuration
# ------------------------------------------------------------
st.sidebar.header("Configuration")

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

st.sidebar.info(
    "Use numeric Qi levels, like your Excel sheet: 1, 1, 2, 5, 5.\n\n"
    "Choose the reference/main factor below. In your workbook, the first factor "
    "('Economic') is the reference factor."
)

# ------------------------------------------------------------
# Input section
# ------------------------------------------------------------
st.subheader("Enter Factor Information")

factors = []
qi_values = []
expert_data = []

for i in range(num_factors):
    st.markdown(f"### Factor {i+1}")
    c1, c2 = st.columns([2, 1])

    with c1:
        factor_name = st.text_input(
            f"Factor Name {i+1}",
            value=f"Factor {i+1}",
            key=f"factor_{i}"
        )

    with c2:
        qi = st.number_input(
            f"Qi (Level) {i+1}",
            min_value=0.0,
            value=1.0,
            step=1.0,
            key=f"qi_{i}"
        )

    factors.append(factor_name)
    qi_values.append(qi)

    cols = st.columns(num_experts)
    row_vals = []
    for j in range(num_experts):
        val = cols[j].number_input(
            f"E{j+1}",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key=f"val_{i}_{j}"
        )
        row_vals.append(val)

    expert_data.append(row_vals)

factor_labels = [
    f"{i+1}. {factors[i] if factors[i].strip() else f'Factor {i+1}'}"
    for i in range(num_factors)
]

reference_index = st.selectbox(
    "Reference / Main Factor",
    options=list(range(num_factors)),
    format_func=lambda x: factor_labels[x],
    index=0
)

# ------------------------------------------------------------
# Build input DataFrame
# ------------------------------------------------------------
df = pd.DataFrame(expert_data, columns=[f"E{i+1}" for i in range(num_experts)])
df.insert(0, "Qi", qi_values)
df.insert(0, "Factor", factors)

st.subheader("Input Data")
st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------
# Run model
# ------------------------------------------------------------
if st.button("Run LBWA Model"):

    data = df.iloc[:, 2:].astype(float).values
    qi_arr = df["Qi"].astype(float).values

    if np.any(qi_arr < 0):
        st.error("Qi values must be non-negative.")
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
    tfn_df.insert(0, "Factor", df["Factor"])

    st.subheader("1) Triangular Fuzzy Numbers (TFN)")
    st.dataframe(tfn_df, use_container_width=True)

    # --------------------------------------------------------
    # Step 2: Fuzzy influence
    # Excel logic:
    # denominator_i = (Qi*theta + l, Qi*theta + m, Qi*theta + u)
    # influence_i = theta / denominator_i
    # For TFN division:
    # influence = (theta/(Qiθ+u), theta/(Qiθ+m), theta/(Qiθ+l))
    # --------------------------------------------------------
    denominator_tfn = np.column_stack([
        qi_arr * theta + tfn[:, 0],
        qi_arr * theta + tfn[:, 1],
        qi_arr * theta + tfn[:, 2]
    ])

    influence = scalar_divide_tfn(theta, denominator_tfn)

    influence_df = pd.DataFrame(influence, columns=["l", "m", "u"])
    influence_df.insert(0, "Qi", qi_arr)
    influence_df.insert(0, "Factor", df["Factor"])

    st.subheader("2) Fuzzy Influence Function")
    st.dataframe(influence_df, use_container_width=True)

    # --------------------------------------------------------
    # Step 3: Reference fuzzy weight
    # Workbook logic for reference factor r:
    # wr_l = 1 / (1 + sum(other u-values))
    # wr_m = 1 / (1 + sum(other m-values))
    # wr_u = 1 / (1 + sum(other l-values))
    # --------------------------------------------------------
    mask_others = np.ones(num_factors, dtype=bool)
    mask_others[reference_index] = False

    ref_weight = np.array([
        1 / (1 + np.sum(influence[mask_others, 2])),  # l uses other u
        1 / (1 + np.sum(influence[mask_others, 1])),  # m uses other m
        1 / (1 + np.sum(influence[mask_others, 0]))   # u uses other l
    ])

    # --------------------------------------------------------
    # Step 4: Fuzzy weights
    # Reference factor gets ref_weight directly
    # Others: weight_i = ref_weight * influence_i
    # --------------------------------------------------------
    fuzzy_weights = np.zeros_like(influence)
    fuzzy_weights[reference_index] = ref_weight

    for i in range(num_factors):
        if i != reference_index:
            fuzzy_weights[i] = ref_weight * influence[i]

    fuzzy_weight_df = pd.DataFrame(fuzzy_weights, columns=["l", "m", "u"])
    fuzzy_weight_df.insert(0, "Factor", df["Factor"])

    st.subheader("3) Fuzzy Weights")
    st.dataframe(fuzzy_weight_df, use_container_width=True)

    # --------------------------------------------------------
    # Step 5: Defuzzification and normalization
    # Excel uses: (l + 4m + u) / 6
    # --------------------------------------------------------
    crisp_values = defuzzify_weighted(fuzzy_weights)
    crisp_sum = np.sum(crisp_values)

    if crisp_sum == 0:
        st.error("The sum of crisp values is zero, so normalization cannot be done.")
        st.stop()

    normalized_weights = crisp_values / crisp_sum

    result_df = pd.DataFrame({
        "Factor": df["Factor"],
        "Qi": qi_arr,
        "Crisp Value": crisp_values,
        "Normalized Weight": normalized_weights
    })

    st.subheader("4) Final Results")
    st.dataframe(result_df, use_container_width=True)

    st.write(f"**Sum of Crisp Values:** {crisp_sum:.10f}")
    st.write(f"**Sum of Normalized Weights:** {normalized_weights.sum():.10f}")

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    st.subheader("Weight Distribution")
    chart_df = result_df[["Factor", "Normalized Weight"]].set_index("Factor")
    st.bar_chart(chart_df)

    # --------------------------------------------------------
    # Export to Excel
    # --------------------------------------------------------
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Input", index=False)
        tfn_df.to_excel(writer, sheet_name="TFN", index=False)
        influence_df.to_excel(writer, sheet_name="Influence", index=False)
        fuzzy_weight_df.to_excel(writer, sheet_name="FuzzyWeights", index=False)
        result_df.to_excel(writer, sheet_name="Results", index=False)

    st.download_button(
        label="Download Results",
        data=output.getvalue(),
        file_name="LBWA_output_corrected.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================================================
# Run:
# streamlit run app.py
# ============================================================
