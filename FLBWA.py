import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fuzzy LBWA Model", layout="wide")

st.title("🔷 Fuzzy Level-Based Weight Assessment (LBWA)")

# --------------------------------------------------
# Upload Excel
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📊 Input Data")
    st.dataframe(df)

    # --------------------------------------------------
    # Step 3.1: Select Best Factor
    # --------------------------------------------------
    st.subheader("Step 3.1: Select Best Factor")
    best_factor = st.selectbox("Choose the most important factor", df["Factor"])

    # --------------------------------------------------
    # Step 3.2: Levels
    # --------------------------------------------------
    st.subheader("Step 3.2: Factor Levels")
    levels = df["Level"].unique()
    st.write("Detected Levels:", levels)

    # --------------------------------------------------
    # Step 3.3: Convert to TFN
    # --------------------------------------------------
    st.subheader("Step 3.3: TFN Conversion")

    expert_cols = [col for col in df.columns if "Ex" in col]

    def compute_tfn(row):
        values = row[expert_cols].values
        return pd.Series([
            np.min(values),
            np.mean(values),
            np.max(values)
        ], index=["l", "m", "u"])

    tfn_df = df.apply(compute_tfn, axis=1)
    df_tfn = pd.concat([df[["Factor", "Level"]], tfn_df], axis=1)

    st.write("TFN Values:")
    st.dataframe(df_tfn)

    # --------------------------------------------------
    # Step 3.4: Elasticity Coefficient
    # --------------------------------------------------
    st.subheader("Step 3.4: Elasticity Coefficient")
    sigma = st.number_input("Enter Elasticity Coefficient (σ)", value=2.1)

    # --------------------------------------------------
    # Step 3.5: Fuzzy Influence Function
    # --------------------------------------------------
    st.subheader("Step 3.5: Fuzzy Influence Function")

    max_level = df["Level"].str.extract('(\d+)').astype(int).max()[0]

    def influence(row):
        level = int(row["Level"][1:])
        l, m, u = row["l"], row["m"], row["u"]

        f_l = 1 / (1 + sigma * l / level)
        f_m = 1 / (1 + sigma * m / level)
        f_u = 1 / (1 + sigma * u / level)

        return pd.Series([f_l, f_m, f_u], index=["fl", "fm", "fu"])

    infl_df = df_tfn.apply(influence, axis=1)
    df_infl = pd.concat([df_tfn, infl_df], axis=1)

    st.write("Fuzzy Influence Values:")
    st.dataframe(df_infl)

    # --------------------------------------------------
    # Step 3.6: Compute Weights
    # --------------------------------------------------
    st.subheader("Step 3.6: Fuzzy Weights")

    # Identify best factor row
    best_row = df_infl[df_infl["Factor"] == best_factor].iloc[0]

    sum_fl = df_infl["fl"].sum()
    sum_fm = df_infl["fm"].sum()
    sum_fu = df_infl["fu"].sum()

    # Weight of best factor
    w_best = pd.Series([
        best_row["fl"] / sum_fl,
        best_row["fm"] / sum_fm,
        best_row["fu"] / sum_fu
    ], index=["wl", "wm", "wu"])

    weights = []

    for _, row in df_infl.iterrows():
        wl = row["fl"] / sum_fl
        wm = row["fm"] / sum_fm
        wu = row["fu"] / sum_fu

        weights.append([wl, wm, wu])

    weight_df = pd.DataFrame(weights, columns=["wl", "wm", "wu"])
    final_df = pd.concat([df_infl[["Factor"]], weight_df], axis=1)

    st.write("Final Fuzzy Weights:")
    st.dataframe(final_df)

    # --------------------------------------------------
    # Defuzzification
    # --------------------------------------------------
    st.subheader("🔷 Defuzzified Weights")

    final_df["Crisp Weight"] = final_df[["wl", "wm", "wu"]].mean(axis=1)

    st.dataframe(final_df)

    # --------------------------------------------------
    # Download
    # --------------------------------------------------
    st.download_button(
        "📥 Download Results",
        final_df.to_csv(index=False),
        file_name="LBWA_results.csv"
    )
