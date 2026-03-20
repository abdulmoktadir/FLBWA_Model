import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# ============================================================
# Page Setup & Theme
# ============================================================
st.set_page_config(
    page_title="Fuzzy LBWA Analyzer Pro",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for a "Professional Dashboard" feel
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #fcfdfe;
    }
    
    /* Global font tweaks */
    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    /* Card Styling */
    .content-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    /* Buttons */
    .stButton > button {
        background: #2563eb;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
        border: none;
        height: 3rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Helper Logic
# ============================================================
def scalar_divide_tfn(a, tfn):
    return np.column_stack([a / tfn[:, 2], a / tfn[:, 1], a / tfn[:, 0]])

def defuzzify_weighted(tfn):
    return (tfn[:, 0] + 4 * tfn[:, 1] + tfn[:, 2]) / 6

def make_excel_file(input_df, tfn_df, influence_df, fuzzy_weight_df, result_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        input_df.to_excel(writer, sheet_name="Input Data", index=False)
        tfn_df.to_excel(writer, sheet_name="TFN Values", index=False)
        influence_df.to_excel(writer, sheet_name="Fuzzy Influence", index=False)
        fuzzy_weight_df.to_excel(writer, sheet_name="Fuzzy Weights", index=False)
        result_df.to_excel(writer, sheet_name="Final Weights", index=False)
    return output.getvalue()

# ============================================================
# Sidebar Configuration
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3222/3222672.png", width=80)
    st.title("Model Settings")
    
    with st.expander("🔢 Dimensions", expanded=True):
        num_factors = st.number_input("Number of Factors", 2, 50, 5)
        num_experts = st.number_input("Number of Experts", 1, 20, 3)
    
    with st.expander("⚙️ Parameters", expanded=True):
        theta = st.number_input("Elasticity (θ)", 0.0001, 10.0, 2.1, format="%.2f")
    
    st.divider()
    st.caption("Fuzzy LBWA Method v2.0")

# ============================================================
# Main UI
# ============================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚖️ Fuzzy LBWA Analyzer Pro</div>
    <div class="hero-subtitle">Multi-Criteria Decision Making with Level-Based Weight Assessment</div>
</div>
""", unsafe_allow_html=True)

# 1. Input Section
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.subheader("1. Data Entry")

col_a, col_b = st.columns([3, 1])

with col_a:
    factor_names = [f"Factor {i+1}" for i in range(num_factors)]
    default_data = {"Factor": factor_names, "Qi": [1.0] * num_factors}
    for i in range(num_experts):
        default_data[f"Expert {i+1}"] = [0.0] * num_factors
    
    df_template = pd.DataFrame(default_data)
    edited_df = st.data_editor(
        df_template,
        use_container_width=True,
        hide_index=True,
        column_config={"Qi": st.column_config.NumberColumn("Rank Index (Qi)", help="Preference level of factor")}
    )

with col_b:
    st.info("Select the most important factor as the Reference.")
    reference_idx = st.selectbox(
        "Reference Factor",
        options=range(num_factors),
        format_func=lambda x: edited_df.iloc[x]["Factor"]
    )
st.markdown('</div>', unsafe_allow_html=True)

# 2. Execution
if st.button("🚀 Calculate Weights", use_container_width=True):
    with st.spinner("Calculating fuzzy coefficients..."):
        try:
            # Data Extraction
            data = edited_df.iloc[:, 2:].astype(float).values
            qi_arr = edited_df["Qi"].astype(float).values
            factors = edited_df["Factor"].values

            # Step 1: TFN Construction
            tfn = np.column_stack([np.min(data, axis=1), np.mean(data, axis=1), np.max(data, axis=1)])
            
            # Step 2: Fuzzy Influence
            denom = np.column_stack([qi_arr * theta + tfn[:, 0], qi_arr * theta + tfn[:, 1], qi_arr * theta + tfn[:, 2]])
            influence = scalar_divide_tfn(theta, denom)
            
            # Step 3 & 4: Fuzzy Weights
            mask = np.ones(num_factors, dtype=bool)
            mask[reference_idx] = False
            ref_w = np.array([
                1 / (1 + np.sum(influence[mask, 2])),
                1 / (1 + np.sum(influence[mask, 1])),
                1 / (1 + np.sum(influence[mask, 0]))
            ])
            
            f_weights = np.zeros_like(influence)
            f_weights[reference_idx] = ref_w
            for i in range(num_factors):
                if i != reference_idx: f_weights[i] = ref_w * influence[i]
            
            # Step 5: Normalization
            crisp = defuzzify_weighted(f_weights)
            norm_w = crisp / np.sum(crisp)

            # Results Preparation
            res_df = pd.DataFrame({
                "Rank": pd.Series(norm_w).rank(ascending=False, method='dense').astype(int),
                "Factor": factors,
                "Normalized Weight": norm_w,
                "Crisp Value": crisp
            }).sort_values("Rank")

            # UI Display
            st.toast("Calculations complete!", icon="✅")
            
            # 3. Visualization Section
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.subheader("2. Result Overview")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Top Factor", res_df.iloc[0]["Factor"])
            m2.metric("Ref. Weight", f"{ref_w[1]:.4f}")
            m3.metric("Consistency", "High" if theta > 1 else "Normal")
            
            # Interactive Chart
            fig = px.bar(
                res_df, x="Normalized Weight", y="Factor", 
                orientation='h', text_auto='.4f',
                color="Normalized Weight",
                color_continuous_scale="Blues",
                template="simple_white"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 4. Detailed Data
            with st.expander("📄 View Full Computation Tables"):
                tab1, tab2 = st.tabs(["Final Rankings", "Fuzzy TFN Details"])
                with tab1:
                    st.dataframe(res_df.style.background_gradient(subset=["Normalized Weight"], cmap="Greens"), use_container_width=True)
                with tab2:
                    st.write("Fuzzy Weights (l, m, u)")
                    st.dataframe(pd.DataFrame(f_weights, columns=["Lower", "Medium", "Upper"], index=factors))

            # 5. Export
            excel_bytes = make_excel_file(edited_df, pd.DataFrame(tfn), pd.DataFrame(influence), pd.DataFrame(f_weights), res_df)
            st.download_button(
                label="📥 Download Professional Report (Excel)",
                data=excel_bytes,
                file_name="LBWA_Analysis_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Computation Error: {str(e)}")
