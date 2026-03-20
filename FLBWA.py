# ============================================================
# Streamlit App: Fuzzy Level-Based Weight Assessment (LBWA)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fuzzy LBWA App", layout="wide")

st.title("📊 Fuzzy Level-Based Weight Assessment (LBWA)")
st.markdown("Stepwise implementation with transparent calculations")

# ============================================================
# STEP 1: INPUT FACTORS
# ============================================================
st.header("Step 1: Define Factors")

num_factors = st.number_input("Number of Factors", min_value=2, value=5)

factors = []
for i in range(num_factors):
    f = st.text_input(f"Factor {i+1}", value=f"F{i+1}")
    factors.append(f)

# ============================================================
# STEP 2: SELECT BEST FACTOR
# ============================================================
st.header("Step 2: Select Best Factor")

best_factor = st.selectbox("Choose the most important factor", factors)

# ============================================================
# STEP 3: ASSIGN LEVELS
# ============================================================
st.header("Step 3: Assign Levels (L1 = most important)")

levels = {}
for f in factors:
    levels[f] = st.number_input(f"Level for {f}", min_value=1, step=1)

level_df = pd.DataFrame({
    "Factor": list(levels.keys()),
    "Level": list(levels.values())
})

st.subheader("Assigned Levels")
st.dataframe(level_df)

# ============================================================
# STEP 4: EXPERT INPUT
# ============================================================
st.header("Step 4: Expert Evaluation")

num_experts = st.number_input("Number of Experts", min_value=1, value=3)

ratings = {}
for f in factors:
    st.subheader(f"Ratings for {f}")
    ratings[f] = []
    cols = st.columns(num_experts)
    for e in range(num_experts):
        val = cols[e].number_input(f"E{e+1}", key=f"{f}_{e}")
        ratings[f].append(val)

rating_df = pd.DataFrame(ratings, index=[f"E{i+1}" for i in range(num_experts)])

st.subheader("Expert Ratings Matrix")
st.dataframe(rating_df)

# ============================================================
# STEP 5: TFN CONVERSION
# ============================================================
st.header("Step 5: Convert to TFN")

# Simple TFN: (min, avg, max)
TFN = {}
for f in factors:
    vals = np.array(ratings[f])
    TFN[f] = [vals.min(), vals.mean(), vals.max()]

tfn_df = pd.DataFrame(TFN, index=["l", "m", "u"]).T

st.subheader("Triangular Fuzzy Numbers (TFN)")
st.dataframe(tfn_df)

# ============================================================
# STEP 6: ELASTICITY COEFFICIENT
# ============================================================
st.header("Step 6: Elasticity Coefficient")

phi = st.number_input("Elasticity coefficient (ϕ)", value=2.1)

# ============================================================
# STEP 7: FUZZY INFLUENCE FUNCTION
# ============================================================
st.header("Step 7: Fuzzy Influence Function")

influence = {}
for f in factors:
    l, m, u = TFN[f]
    influence[f] = [
        1 / (1 + phi * l),
        1 / (1 + phi * m),
        1 / (1 + phi * u)
    ]

infl_df = pd.DataFrame(influence, index=["l", "m", "u"]).T

st.subheader("Fuzzy Influence Function")
st.dataframe(infl_df)

# ============================================================
# STEP 8: WEIGHT CALCULATION
# ============================================================
st.header("Step 8: Compute Weights")

# Normalize using best factor
best_infl = np.array(influence[best_factor])

weights = {}
for f in factors:
    weights[f] = np.array(influence[f]) / best_infl

# Normalize final weights
sum_weights = np.sum(list(weights.values()), axis=0)

final_weights = {}
for f in factors:
    final_weights[f] = weights[f] / sum_weights

weight_df = pd.DataFrame(final_weights, index=["l", "m", "u"]).T

st.subheader("Final Fuzzy Weights")
st.dataframe(weight_df)

# ============================================================
# STEP 9: DEFUZZIFICATION
# ============================================================
st.header("Step 9: Crisp Weights")

crisp_weights = {}
for f in factors:
    l, m, u = final_weights[f]
    crisp_weights[f] = (l + m + u) / 3

crisp_df = pd.DataFrame({
    "Factor": list(crisp_weights.keys()),
    "Crisp Weight": list(crisp_weights.values())
}).sort_values(by="Crisp Weight", ascending=False)

st.subheader("Final Ranking")
st.dataframe(crisp_df)

# ============================================================
# DOWNLOAD RESULTS
# ============================================================
st.header("Download Results")

output = pd.concat([
    level_df.set_index("Factor"),
    tfn_df,
    infl_df,
    weight_df
], axis=1)

st.download_button(
    "Download Results as CSV",
    data=output.to_csv().encode("utf-8"),
    file_name="LBWA_results.csv",
    mime="text/csv"
)
