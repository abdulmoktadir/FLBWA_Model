# ============================================================
# Streamlit App: Fuzzy Level-Based Weight Assessment (LBWA)
# Manual Input Version
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Fuzzy LBWA App", layout="wide")

st.title("Fuzzy Level-Based Weight Assessment (LBWA)")

# ------------------------------------------------------------
# Helper Function: Scalar ÷ TFN
# For positive TFN B=(l,m,u):
# a / B = (a/u, a/m, a/l)
# ------------------------------------------------------------
def scalar_divide_tfn(a, tfn):
    return np.column_stack([
        a / tfn[:, 2],  # l = a / u
        a / tfn[:, 1],  # m = a / m
        a / tfn[:, 0],  # u = a / l
    ])

# ------------------------------------------------------------
# User Inputs
# ------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

num_factors = int(st.sidebar.number_input("Number of Factors", min_value=2, value=5, step=1))
num_experts = int(st.sidebar.number_input("Number of Experts", min_value=1, value=3, step=1))
r = st.sidebar.number_input("Elasticity Coefficient (r)", min_value=0.0, value=2.1, step=0.1)

st.subheader("📥 Enter Factor Information")

# ------------------------------------------------------------
# Dynamic Input Table
# ------------------------------------------------------------
factors = []
levels = []
expert_data = []

for i in range(num_factors):
    st.markdown(f"### Factor {i+1}")
    col1, col2 = st.columns(2)

    with col1:
        factor_name = st.text_input(f"Factor Name {i+1}", value=f"Factor {i+1}", key=f"f_{i}")
    with col2:
        level = st.selectbox(f"Level {i+1}", [f"L{j}" for j in range(1, 10)], key=f"l_{i}")

    factors.append(factor_name)
    levels.append(level)

    exp_vals = []
    cols = st.columns(num_experts)
    for j in range(num_experts):
        val = cols[j].number_input(
            f"E{j+1}",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key=f"val_{i}_{j}"
        )
        exp_vals.append(val)

    expert_data.append(exp_vals)

# Convert to DataFrame
df = pd.DataFrame(expert_data, columns=[f"E{i+1}" for i in range(num_experts)])
df.insert(0, "Level", levels)
df.insert(0, "Factor", factors)

st.subheader("📊 Input Data")
st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------
# Run Model
# ------------------------------------------------------------
if st.button("▶️ Run LBWA Model"):

    data = df.iloc[:, 2:].astype(float).values

    # ------------------------------------------------------------
    # Step 1: TFN Calculation
    # TFN = (min, mean, max)
    # ------------------------------------------------------------
    tfn = np.column_stack([
        np.min(data, axis=1),
        np.mean(data, axis=1),
        np.max(data, axis=1)
    ])

    tfn_df = pd.DataFrame(tfn, columns=["l", "m", "u"])
    tfn_df.insert(0, "Factor", df["Factor"])

    st.subheader("🔺 Triangular Fuzzy Numbers (TFN)")
    st.dataframe(tfn_df, use_container_width=True)

    # ------------------------------------------------------------
    # Step 2: Fuzzy Influence Function
    # influence = 1 / (1 + TFN^r)
    #
    # For TFN division:
    # If B = (l, m, u), then 1/B = (1/u, 1/m, 1/l)
    # ------------------------------------------------------------
    tfn_power = np.column_stack([
        np.power(tfn[:, 0], r),
        np.power(tfn[:, 1], r),
        np.power(tfn[:, 2], r)
    ])

    denominator_tfn = tfn_power + 1.0  # (1+l^r, 1+m^r, 1+u^r)

    influence = scalar_divide_tfn(1.0, denominator_tfn)

    influence_df = pd.DataFrame(influence, columns=["l", "m", "u"])
    influence_df.insert(0, "Factor", df["Factor"])

    st.subheader("⚙️ Fuzzy Influence Function")
    st.dataframe(influence_df, use_container_width=True)

    # ------------------------------------------------------------
    # Step 3: Defuzzification
    # ------------------------------------------------------------
    crisp = influence.mean(axis=1)

    if np.sum(crisp) == 0:
        st.error("The sum of crisp values is zero, so weights cannot be normalized.")
        st.stop()

    weights = crisp / np.sum(crisp)

    result_df = pd.DataFrame({
        "Factor": df["Factor"],
        "Weight": weights
    })

    st.subheader("🏁 Final Weights")
    st.dataframe(result_df, use_container_width=True)

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------
    st.subheader("📈 Weight Distribution")
    st.bar_chart(result_df.set_index("Factor"))

    # ------------------------------------------------------------
    # Download Results
    # ------------------------------------------------------------
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        tfn_df.to_excel(writer, sheet_name="TFN", index=False)
        influence_df.to_excel(writer, sheet_name="Influence", index=False)
        result_df.to_excel(writer, sheet_name="Weights", index=False)

    st.download_button(
        "📥 Download Results",
        data=output.getvalue(),
        file_name="LBWA_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================================================
# Run using:
# streamlit run app.py
# ============================================================
