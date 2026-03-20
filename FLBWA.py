# ============================================================
# Streamlit App: Fuzzy Level-Based Weight Assessment (LBWA)
# Manual Input Version
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fuzzy LBWA App", layout="wide")

st.title("Fuzzy Level-Based Weight Assessment (LBWA)")

# ------------------------------------------------------------
# User Inputs
# ------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

num_factors = st.sidebar.number_input("Number of Factors", min_value=2, value=5)
num_experts = st.sidebar.number_input("Number of Experts", min_value=1, value=3)

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
        factor_name = st.text_input(f"Factor Name {i+1}", key=f"f_{i}")
    with col2:
        level = st.selectbox(f"Level {i+1}", [f"L{j}" for j in range(1, 10)], key=f"l_{i}")

    factors.append(factor_name)
    levels.append(level)

    exp_vals = []
    cols = st.columns(num_experts)
    for j in range(num_experts):
        val = cols[j].number_input(f"E{j+1}", key=f"val_{i}_{j}")
        exp_vals.append(val)

    expert_data.append(exp_vals)

# Convert to DataFrame
df = pd.DataFrame(expert_data, columns=[f"E{i+1}" for i in range(num_experts)])
df.insert(0, "Level", levels)
df.insert(0, "Factor", factors)

st.subheader("📊 Input Data")
st.dataframe(df)

# ------------------------------------------------------------
# Step 1: TFN Calculation
# ------------------------------------------------------------
if st.button("▶️ Run LBWA Model"):

    data = df.iloc[:, 2:].values

    tfn = []
    for row in data:
        tfn.append([
            np.min(row),
            np.mean(row),
            np.max(row)
        ])

    tfn = np.array(tfn)

    tfn_df = pd.DataFrame(tfn, columns=["l", "m", "u"])
    tfn_df.insert(0, "Factor", df["Factor"])

    st.subheader("🔺 Triangular Fuzzy Numbers (TFN)")
    st.dataframe(tfn_df)

    # ------------------------------------------------------------
    # Step 2: Elasticity Coefficient
    # ------------------------------------------------------------
    r = st.number_input("Elasticity Coefficient (r)", value=2.1)

    # ------------------------------------------------------------
    # Step 3: Fuzzy Influence Function
    # ------------------------------------------------------------
    influence = 1 / (1 + np.power(tfn, r))

    influence_df = pd.DataFrame(influence, columns=["l", "m", "u"])
    influence_df.insert(0, "Factor", df["Factor"])

    st.subheader("⚙️ Fuzzy Influence Function")
    st.dataframe(influence_df)

    # ------------------------------------------------------------
    # Step 4: Defuzzification
    # ------------------------------------------------------------
    crisp = influence.mean(axis=1)

    weights = crisp / np.sum(crisp)

    result_df = pd.DataFrame({
        "Factor": df["Factor"],
        "Weight": weights
    })

    st.subheader("🏁 Final Weights")
    st.dataframe(result_df)

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------
    st.subheader("📈 Weight Distribution")
    st.bar_chart(result_df.set_index("Factor"))

    # ------------------------------------------------------------
    # Download Results
    # ------------------------------------------------------------
    output = pd.ExcelWriter("LBWA_output.xlsx", engine='xlsxwriter')
    tfn_df.to_excel(output, sheet_name='TFN', index=False)
    influence_df.to_excel(output, sheet_name='Influence', index=False)
    result_df.to_excel(output, sheet_name='Weights', index=False)
    output.close()

    with open("LBWA_output.xlsx", "rb") as f:
        st.download_button("📥 Download Results", f, file_name="LBWA_output.xlsx")

# ============================================================
# Run using:
# streamlit run app.py
# ============================================================
