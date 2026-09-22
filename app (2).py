import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Telecom Customer Churn & Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 2. DATA LOADING ARCHITECTURE
# ==========================================
@st.cache_data
def load_datasets():
    cleaned_file = "telecom_churn_cleaned.csv"
    predictions_file = "customer_risk_predictions.csv"

    # Verify dataset existence
    if not os.path.exists(cleaned_file):
        st.error(
            f"❌ Missing dataset: '{cleaned_file}'. Please place it in the project root folder."
        )
        st.stop()

    if not os.path.exists(predictions_file):
        st.error(
            f"❌ Missing dataset: '{predictions_file}'. Please place it in the project root folder."
        )
        st.stop()

    # Load datasets safely
    df_cleaned = pd.read_csv(cleaned_file)
    df_predictions = pd.read_csv(predictions_file)

    # Clean whitespace in column headers
    df_cleaned.columns = df_cleaned.columns.str.strip()
    df_predictions.columns = df_predictions.columns.str.strip()

    # Numeric conversions
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(
                df_cleaned[col], errors="coerce"
            ).fillna(0)
        if col in df_predictions.columns:
            df_predictions[col] = pd.to_numeric(
                df_predictions[col], errors="coerce"
            ).fillna(0)

    # Standardize target columns
    if "Churn" in df_cleaned.columns:
        df_cleaned["Churn"] = df_cleaned["Churn"].astype(str).str.strip()

    if "Actual_Churn" in df_predictions.columns:
        df_predictions["Actual_Churn"] = (
            df_predictions["Actual_Churn"].astype(str).str.strip()
        )

    # Normalize Tenure Groups if needed
    tenure_bins = [-1, 12, 24, 48, 60, float("inf")]
    tenure_labels = [
        "0–12 months",
        "13–24 months",
        "25–48 months",
        "49–60 months",
        "61+ months",
    ]
    if "Tenure_Group" not in df_predictions.columns and "tenure" in df_predictions.columns:
        df_predictions["Tenure_Group"] = pd.cut(
            df_predictions["tenure"],
            bins=tenure_bins,
            labels=tenure_labels,
            right=True,
        )

    return df_cleaned, df_predictions


# Load data into reusable variables
df_cleaned, df_predictions = load_datasets()


# ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
st.sidebar.title("📌 Navigation")

sections = [
    "1. Executive Overview",
    "2. Churn Analysis",
    "3. Prediction & Risk",
    "4. Business Action Plan",
    "5. Customer Risk Explorer",
]

selected_section = st.sidebar.radio("Go to Section:", sections)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Platform Info")
st.sidebar.info(
    """
    **Dataset Scope:** 7,043 Accounts  
    **Model:** Logistic Regression  
    **Risk Classification:**  
    • High Risk ($\ge 60\%$)  
    • Medium Risk ($30\% - 59\%$)  
    • Low Risk ($< 30\%$)
    """
)


# Helper function to generate clean bar charts
def create_churn_bar_chart(
    series, title, xlabel, ylabel="Churn Rate (%)", color="#3498db"
):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        series.index,
        series.values,
        color=color,
        width=0.55,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.2,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(series.values) * 1.20 if len(series) > 0 and max(series.values) > 0 else 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


# ==========================================
# 4. SECTION 1: EXECUTIVE OVERVIEW
# ==========================================
if selected_section == "1. Executive Overview":
    st.title("📡 Telecom Customer Churn & Retention Intelligence")
    st.subheader("Executive Overview | Data-Driven Customer Retention")
    st.markdown("---")

    total_customers = len(df_cleaned)

    churn_yes_count = (df_cleaned["Churn"] == "Yes").sum()
    overall_churn_rate = (churn_yes_count / total_customers) * 100

    high_risk_count = (df_predictions["Risk_Level"] == "High Risk").sum()
    medium_risk_count = (df_predictions["Risk_Level"] == "Medium Risk").sum()

    avg_monthly_charges = df_cleaned["MonthlyCharges"].mean()

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Customers", f"{total_customers:,}")
    with kpi2:
        st.metric("Overall Churn Rate", f"{overall_churn_rate:.2f}%")
    with kpi3:
        st.metric(
            "High Risk Customers",
            f"{high_risk_count:,}",
            f"{high_risk_count/total_customers*100:.1f}% of total",
            delta_color="inverse",
        )
    with kpi4:
        st.metric(
            "Medium Risk Customers",
            f"{medium_risk_count:,}",
            f"{medium_risk_count/total_customers*100:.1f}% of total",
            delta_color="off",
        )
    with kpi5:
        st.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("### 🛡️ Customer Churn Risk Distribution")

        risk_order = ["Low Risk", "Medium Risk", "High Risk"]
        risk_counts = (
            df_predictions["Risk_Level"].value_counts().reindex(risk_order)
        )

        fig_risk, ax_risk = plt.subplots(figsize=(6, 4))
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        bars = ax_risk.bar(
            risk_counts.index,
            risk_counts.values,
            color=colors,
            width=0.55,
            edgecolor="black",
            linewidth=0.8,
        )

        for bar in bars:
            height = bar.get_height()
            percentage = (height / total_customers) * 100
            ax_risk.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (total_customers * 0.015),
                f"{int(height):,}\n({percentage:.2f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax_risk.set_ylabel("Customer Count", fontsize=10, fontweight="bold")
        ax_risk.set_ylim(0, max(risk_counts.values) * 1.18)
        ax_risk.spines["top"].set_visible(False)
        ax_risk.spines["right"].set_visible(False)
        ax_risk.grid(axis="y", linestyle="--", alpha=0.5)

        st.pyplot(fig_risk)

    with col_chart2:
        st.markdown("### 📜 Observed Churn Rate by Contract Type")

        contract_churn = (
            df_cleaned.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reindex(["Month-to-month", "One year", "Two year"])
        )

        fig_contract = create_churn_bar_chart(
            contract_churn,
            "Churn Rate by Contract",
            "Contract Type",
            color="#3498db",
        )
        st.pyplot(fig_contract)

    st.markdown("---")

    col_insight, col_alert = st.columns([1, 1])

    with col_insight:
        st.markdown("### 💡 What the Data Tells Us")
        st.markdown(
            """
            * **Month-to-Month Contract Vulnerability:** Month-to-month contracts demonstrate the highest observed churn rate across all agreement structures.
            * **Early-Tenure Turnover Concentration:** Customer churn is heavily concentrated among accounts in their first **12 months of service**, after which retention rates stabilize significantly.
            * **Fiber Optic Service Segment Risk:** Fiber optic subscribers exhibit higher observed churn compared to DSL subscribers and customers with no internet service.
            """
        )

    with col_alert:
        st.markdown("### ⚠️ High-Risk Customer Alert")
        st.error(
            """
            **1,043 customers (14.81%)** are classified as **High Risk** based on a predicted churn probability of **60% or higher**.
            
            Among this model-identified High Risk cohort, **72.10%** have a recorded actual churn status of **Yes**.
            
            *(Note: Predicted risk reflects model probability scoring, while actual churn reflects historical outcome observations).*
            """
        )

# ==========================================
# 5. SECTION 2: CHURN ANALYSIS
# ==========================================
elif selected_section == "2. Churn Analysis":
    st.title("📊 Churn Analysis")
    st.subheader("Historical Churn Patterns & Customer Segments")
    st.markdown("---")

    contract_churn = (
        df_cleaned.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reindex(["Month-to-month", "One year", "Two year"])
    )

    tenure_bins = [-1, 12, 24, 48, 60, float("inf")]
    tenure_labels = [
        "0–12 months",
        "13–24 months",
        "25–48 months",
        "49–60 months",
        "61+ months",
    ]
    df_cleaned["Tenure_Group"] = pd.cut(
        df_cleaned["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
        right=True,
    )

    tenure_churn = (
        df_cleaned.groupby("Tenure_Group", observed=False)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reindex(tenure_labels)
    )

    internet_churn = (
        df_cleaned.groupby("InternetService")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reindex(["Fiber optic", "DSL", "No"])
    )

    charge_bins = [-1, 30, 60, 90, float("inf")]
    charge_labels = ["Below 30", "30–60", "60–90", "Above 90"]
    df_cleaned["MonthlyCharges_Group"] = pd.cut(
        df_cleaned["MonthlyCharges"],
        bins=charge_bins,
        labels=charge_labels,
        right=True,
    )

    charge_churn = (
        df_cleaned.groupby("MonthlyCharges_Group", observed=False)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reindex(charge_labels)
    )

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        fig1 = create_churn_bar_chart(
            contract_churn,
            "1. Churn Rate by Contract Type",
            "Contract Structure",
            color="#2980b9",
        )
        st.pyplot(fig1)
        st.caption(
            "💡 *Insight:* Month-to-month contracts show the highest observed churn rate, while longer-term contracts show substantially lower observed churn."
        )

    with row1_col2:
        fig2 = create_churn_bar_chart(
            tenure_churn,
            "2. Churn Rate by Tenure Group",
            "Customer Tenure Range",
            color="#16a085",
        )
        st.pyplot(fig2)
        st.caption(
            "💡 *Insight:* Churn is concentrated among customers in the early stages of their relationship, with substantially lower observed churn among longer-tenured customers."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        fig3 = create_churn_bar_chart(
            internet_churn,
            "3. Churn Rate by Internet Service",
            "Internet Service Type",
            color="#8e44ad",
        )
        st.pyplot(fig3)
        st.caption(
            "💡 *Insight:* Fiber optic customers show higher observed churn than DSL customers and customers without internet service."
        )

    with row2_col2:
        fig4 = create_churn_bar_chart(
            charge_churn,
            "4. Churn Rate by Monthly Charges Group",
            "Monthly Charge Tier ($)",
            color="#d35400",
        )
        st.pyplot(fig4)
        st.caption(
            "💡 *Insight:* Higher monthly charge brackets (particularly $60–90 and Above $90) demonstrate noticeably higher observed churn rates compared to low-tier plans (Below $30)."
        )

    st.markdown("---")

    col_summary, col_interp = st.columns([1, 1])

    with col_summary:
        st.markdown("### 📌 Key Findings")
        st.markdown(
            """
            * **Contract Structure:** Month-to-month contracts have an observed churn rate of **42.71%**, compared to **11.27%** for 1-year and **2.83%** for 2-year contracts.
            * **Customer Tenure Lifecycle:** Subscribers in their first year (**0–12 months**) display the highest churn rate (**47.44%**), which steadily declines as tenure increases beyond 24 months.
            * **Internet Product Category:** Fiber optic subscribers experience an observed churn rate of **41.89%**, compared to **18.96%** for DSL and **7.40%** for non-internet accounts.
            * **Monthly Billing Brackets:** Higher bill tiers ($60–90 and Above $90) show elevated churn rates (**36.84%** and **32.22%**), whereas plans below $30 exhibit low turnover (**8.86%**).
            """
        )

    with col_interp:
        st.markdown("### 💼 Business Interpretation")
        st.info(
            """
            **Operational Value of Observed Historical Patterns:**
            
            The historical churn patterns identified above represent key descriptive traits associated with customer turnover across the business. 
            
            * **Prioritization:** These findings allow retention teams to isolate specific high-turnover segments—such as new Fiber optic users on month-to-month plans—for deeper operational analysis.
            * **Causation Disclaimer:** These figures reflect **observed historical associations** in the dataset and do not imply direct cause-and-effect relationships. They serve as descriptive signals to guide targeted retention strategy and model feature selection.
            """
        )

# ==========================================
# 6. SECTION 3: PREDICTION & RISK
# ==========================================
elif selected_section == "3. Prediction & Risk":
    st.title("🤖 Prediction & Risk")
    st.subheader(
        "Machine Learning Performance & Customer Risk Segmentation"
    )
    st.markdown("---")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", "80.48%")
    with m2:
        st.metric("Precision", "65.62%")
    with m3:
        st.metric("Recall", "55.61%")

    st.markdown(
        """
        * **Accuracy (80.48%):** Overall proportion of correct churn/non-churn predictions.
        * **Precision (65.62%):** Among customers predicted as churners, the proportion who actually churned.
        * **Recall (55.61%):** Among customers who actually churned, the proportion correctly identified by the model.
        """
    )

    st.markdown("---")

    col_cm, col_risk = st.columns(2)

    with col_cm:
        st.markdown("### 🔲 Confusion Matrix (Test Set)")

        cm_data = np.array([[926, 109], [166, 208]])

        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4))
        sns.heatmap(
            cm_data,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Predicted No Churn", "Predicted Churn"],
            yticklabels=["Actual No Churn", "Actual Churn"],
        )
        ax_cm.set_title(
            "Model Evaluation Matrix", fontsize=10, fontweight="bold", pad=8
        )
        plt.tight_layout()
        st.pyplot(fig_cm)

        st.markdown(
            """
            * **True Negative (926):** Customers correctly predicted as retained.
            * **False Positive (109):** Retained customers incorrectly flagged as churners (false alarms).
            * **False Negative (166):** Churning customers missed by the model.
            * **True Positive (208):** Churning customers correctly identified in advance.
            """
        )

    with col_risk:
        st.markdown("### 📊 Company-Wide Risk Distribution (N=7,043)")

        risk_labels = ["Low Risk", "Medium Risk", "High Risk"]
        risk_counts = [4384, 1616, 1043]
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        fig_donut, ax_donut = plt.subplots(figsize=(5.5, 4))
        wedges, texts, autotexts = ax_donut.pie(
            risk_counts,
            labels=risk_labels,
            autopct="%1.2f%%",
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
            textprops=dict(fontsize=9, fontweight="bold"),
        )

        for autotext in autotexts:
            autotext.set_color("white")

        ax_donut.set_title(
            "Overall Risk Classification",
            fontsize=10,
            fontweight="bold",
            pad=8,
        )
        plt.tight_layout()
        st.pyplot(fig_donut)

        st.markdown(
            """
            * **Low Risk (< 0.30):** 4,384 customers (62.25%)
            * **Medium Risk (0.30 to < 0.60):** 1,616 customers (22.94%)
            * **High Risk ($\ge$ 0.60):** 1,043 customers (14.81%)
            """
        )

    st.markdown("---")

    col_interp_score, col_val = st.columns(2)

    with col_interp_score:
        st.markdown("### 💡 How to Interpret the Risk Scores")
        st.markdown(
            """
            * **Low Risk:** Predicted churn probability is **below 30%**.
            * **Medium Risk:** Predicted churn probability is **30% to below 60%**.
            * **High Risk:** Predicted churn probability is **60% or higher**.

            > **Note:** A risk category is a model-based statistical probability scoring, not a absolute guarantee that an individual customer will churn.
            """
        )

    with col_val:
        st.markdown("### 🎯 High-Risk Group Validation")
        st.markdown(
            """
            * **Total High-Risk Customers:** 1,043
            * **Actual Churn = Yes:** 752 (**72.10%**)
            * **Actual Churn = No:** 291 (**27.90%**)

            *Among the customers classified as High Risk, 72.10% have an actual historical churn status of Yes.*
            """
        )

    st.markdown("---")

    st.markdown("### 💼 Business Takeaway")
    st.info(
        """
        The predictive model converts historical customer information into individual churn-risk estimates. This allows retention teams to prioritize customers for intervention instead of treating all customers equally.
        
        High-risk customers can be prioritized for proactive retention actions, while medium-risk customers can be monitored and engaged preventively.
        """
    )

# ==========================================
# 7. SECTION 4: BUSINESS ACTION PLAN
# ==========================================
elif selected_section == "4. Business Action Plan":
    # --- 1. SECTION HEADER ---
    st.title("💼 Business Action Plan")
    st.subheader(
        "Translate churn risk into prioritized customer retention actions."
    )
    st.markdown("---")

    # --- 2. PRIORITY OVERVIEW ---
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric(
            "High Risk Priority",
            "1,043 Customers",
            "14.81% of total",
            delta_color="inverse",
        )
    with p2:
        st.metric(
            "Medium Risk Priority",
            "1,616 Customers",
            "22.94% of total",
            delta_color="off",
        )
    with p3:
        st.metric(
            "Low Risk Priority",
            "4,384 Customers",
            "62.25% of total",
            delta_color="normal",
        )

    st.caption(
        "💡 *Retention effort can be prioritized using predicted churn risk rather than treating all customers equally.*"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 3. PRIORITY SEGMENT TABLE ---
    st.markdown("### 📋 Risk Segment Prioritization Matrix")

    segment_data = {
        "Risk Level": ["High Risk", "Medium Risk", "Low Risk"],
        "Customer Count": ["1,043", "1,616", "4,384"],
        "Recommended Priority": [
            "Highest priority",
            "Preventive priority",
            "Maintenance priority",
        ],
        "Suggested Business Response": [
            "Proactive retention intervention",
            "Engagement and value reinforcement",
            "Loyalty and ongoing service engagement",
        ],
    }
    df_matrix = pd.DataFrame(segment_data)
    st.table(df_matrix)

    st.markdown("---")

    # --- 4. HIGH-RISK ACTION PLAN ---
    st.markdown("### 🚨 High-Risk Customer Strategy")
    st.markdown(
        """
        **Observed High-Risk Profile Characteristics (N=1,043):**
        * **Contract Structure:** 100.00% Month-to-month contracts
        * **Internet Service:** 93.58% Fiber optic subscribers
        * **Tenure Range:** 74.50% in early tenure (0–12 months)
        * **Payment Method:** 82.17% Electronic check payments
        * **Average Billing:** $82.11 average monthly charge
        """
    )

    st.markdown("#### Recommended High-Risk Initiatives:")

    hr_col1, hr_col2 = st.columns(2)

    with hr_col1:
        st.markdown(
            """
            **A. Contract Transition Incentives**
            * Offer suitable incentives or benefits for eligible month-to-month customers to consider longer-term plans.
            * Examples include contract-transition discounts, added service benefits, or flexible plan options.

            **B. Early-Tenure Onboarding**
            * Prioritize customers in their first year of service.
            * Implement structured 30/60/90-day check-ins.
            * Provide setup assistance, service guidance, and proactive support.
            """
        )

    with hr_col2:
        st.markdown(
            """
            **C. Fiber Customer Value Support**
            * For high-risk Fiber customers, communicate relevant service features and available support.
            * Consider bundled value propositions involving services such as TechSupport or OnlineSecurity where appropriate.

            **D. Payment Experience Review**
            * Review the payment and billing experience for electronic check users.
            * Provide convenient payment guidance or alternative payment options where appropriate. *(Note: Payment method represents an observed association, not a proven direct cause of churn).*
            """
        )

    st.markdown("---")

    # --- 5. MEDIUM-RISK & LOW-RISK ACTION PLANS ---
    mr_col, lr_col = st.columns(2)

    with mr_col:
        st.markdown("### ⚠️ Medium-Risk Preventive Strategy")
        st.markdown(
            """
            * **Target Group:** 1,616 Customers (Predicted probability 30% to <60%)
            * **Key Actions:**
                * Send personalized engagement messages.
                * Highlight useful service features and value add-ons.
                * Encourage adoption of relevant add-on features (e.g., Online Security, Device Protection).
                * Provide helpful billing/payment reminders and support.
                * Monitor changes in predicted risk scoring over time.

            > *Medium risk indicates a model-predicted probability range, not a confirmed future churn event.*
            """
        )

    with lr_col:
        st.markdown("### 🛡️ Low-Risk Relationship Strategy")
        st.markdown(
            """
            * **Target Group:** 4,384 Customers (Predicted probability <30%)
            * **Key Actions:**
                * Maintain consistent core service quality and support.
                * Deploy standard loyalty and positive engagement communications.
                * Encourage continued feature usage and feature adoption.
                * Avoid unnecessary or aggressive promotional retention offers for stable accounts.
            """
        )

    st.markdown("---")

    # --- 6. RETENTION PRIORITY FLOW ---
    st.markdown("### 🔄 Retention Priority Workflow")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.error(
            """
            **HIGH RISK**
            
            ↓
            
            **Proactive Intervention**
            
            ↓
            
            Contract + Onboarding + Support Review
            """
        )

    with f2:
        st.warning(
            """
            **MEDIUM RISK**
            
            ↓
            
            **Preventive Engagement**
            
            ↓
            
            Feature/Value Communication + Monitoring
            """
        )

    with f3:
        st.success(
            """
            **LOW RISK**
            
            ↓
            
            **Relationship Maintenance**
            
            ↓
            
            Loyalty + Service Engagement
            """
        )

    st.markdown("---")

    # --- 7. KEY BUSINESS INSIGHT & METHODOLOGY NOTE ---
    st.markdown("### 📌 Executive Takeaway")
    st.info(
        """
        The model identifies 1,043 customers as high risk. This group is concentrated around month-to-month contracts, early tenure, Fiber optic service, and electronic-check payment. These patterns can be used to prioritize retention outreach, while recognizing that the model identifies risk rather than proving the cause of churn.
        """
    )

    st.caption(
        "📝 **Methodology Note:** These recommendations represent decision-support actions based on observed customer patterns and model-predicted risk. They should be tested through controlled retention campaigns and monitored using future churn and customer-response data."
    )

# ==========================================
# 8. SECTION 5: CUSTOMER RISK EXPLORER
# ==========================================
elif selected_section == "5. Customer Risk Explorer":
    st.title("🔎 Customer Risk Explorer")
    st.subheader(
        "Explore model-identified customer risk segments and prioritize retention outreach."
    )
    st.markdown("---")

    # --- 1. OVERVIEW EXPLANATION ---
    st.info(
        "💡 **Overview:** This section allows users to filter customers by predicted churn risk and customer characteristics. "
        "It is intended for decision support and prioritization, not as a guarantee that an individual customer will churn."
    )

    # Standardize predictions dataset columns for mapping
    df_exp = df_predictions.copy()

    # Identify probability column
    prob_col = None
    for col in ["Churn_Probability", "churn_probability", "Predicted_Probability", "predicted_probability", "Probability"]:
        if col in df_exp.columns:
            prob_col = col
            break
    if prob_col is None:
        # Fallback to any float column that looks like probability
        float_cols = df_exp.select_dtypes(include=[np.number]).columns
        prob_col = float_cols[0] if len(float_cols) > 0 else "tenure"

    # Identify actual churn column
    actual_col = "Actual_Churn" if "Actual_Churn" in df_exp.columns else ("Churn" if "Churn" in df_exp.columns else None)

    # Ensure Tenure_Group exists
    if "Tenure_Group" not in df_exp.columns and "tenure" in df_exp.columns:
        tenure_bins = [-1, 12, 24, 48, 60, float("inf")]
        tenure_labels = ["0–12 months", "13–24 months", "25–48 months", "49–60 months", "61+ months"]
        df_exp["Tenure_Group"] = pd.cut(df_exp["tenure"], bins=tenure_bins, labels=tenure_labels, right=True)

    # --- 2. FILTER PANEL ---
    st.markdown("### 🎛️ Filter Panel")
    
    filter_c1, filter_c2, filter_c3 = st.columns(3)
    
    with filter_c1:
        risk_options = ["All", "High Risk", "Medium Risk", "Low Risk"]
        selected_risk = st.selectbox("Risk Level", risk_options, index=0)

        contract_opts = ["All"] + list(df_exp["Contract"].dropna().unique()) if "Contract" in df_exp.columns else ["All"]
        selected_contract = st.selectbox("Contract Type", contract_opts, index=0)

    with filter_c2:
        internet_opts = ["All"] + list(df_exp["InternetService"].dropna().unique()) if "InternetService" in df_exp.columns else ["All"]
        selected_internet = st.selectbox("Internet Service", internet_opts, index=0)

        tenure_opts = ["All", "0–12 months", "13–24 months", "25–48 months", "49–60 months", "61+ months"]
        selected_tenure = st.selectbox("Tenure Group", tenure_opts, index=0)

    with filter_c3:
        payment_opts = ["All"] + list(df_exp["PaymentMethod"].dropna().unique()) if "PaymentMethod" in df_exp.columns else ["All"]
        selected_payment = st.selectbox("Payment Method", payment_opts, index=0)

        max_display = st.slider("Maximum customers to display in table", min_value=10, max_value=500, value=50, step=10)

    # --- APPLY FILTERS ---
    filtered_df = df_exp.copy()

    if selected_risk != "All" and "Risk_Level" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Risk_Level"] == selected_risk]

    if selected_contract != "All" and "Contract" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]

    if selected_internet != "All" and "InternetService" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["InternetService"] == selected_internet]

    if selected_tenure != "All" and "Tenure_Group" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Tenure_Group"].astype(str) == selected_tenure]

    if selected_payment != "All" and "PaymentMethod" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["PaymentMethod"] == selected_payment]

    # Sort descending by Churn Probability
    filtered_df = filtered_df.sort_values(by=prob_col, ascending=False)

    st.markdown("---")

    # --- 3. FILTER RESULTS KPI CARDS ---
    st.markdown("### 📊 Selection Summary")

    total_filtered = len(filtered_df)

    if total_filtered == 0:
        st.warning("⚠️ No customers match the selected filter criteria. Please broaden your filter selections.")
    else:
        avg_prob = filtered_df[prob_col].mean() * 100 if filtered_df[prob_col].max() <= 1.0 else filtered_df[prob_col].mean()
        high_risk_filtered = (filtered_df["Risk_Level"] == "High Risk").sum() if "Risk_Level" in filtered_df.columns else 0
        avg_charges = filtered_df["MonthlyCharges"].mean() if "MonthlyCharges" in filtered_df.columns else 0.0

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.metric("Customers in Selection", f"{total_filtered:,}")
        with kpi_col2:
            st.metric("Avg Churn Probability", f"{avg_prob:.2f}%")
        with kpi_col3:
            st.metric("High Risk in Selection", f"{high_risk_filtered:,}")
        with kpi_col4:
            st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- 4. CUSTOMER TABLE ---
        st.markdown("### 📋 Filtered Customer List")
        st.caption("Sorted by Churn Probability (Descending) — Highest predicted risk shown first.")

        # Prepare display dataframe
        display_df = filtered_df.head(max_display).copy()

        # Format probability display
        if display_df[prob_col].max() <= 1.0:
            display_df["Formatted_Prob"] = (display_df[prob_col] * 100).map("{:.2f}%".format)
        else:
            display_df["Formatted_Prob"] = display_df[prob_col].map("{:.2f}%".format)

        # Column renaming and ordering map
        column_map = {
            "customerID": "Customer ID",
            "Risk_Level": "Risk Level",
            "Formatted_Prob": "Churn Probability",
            "Contract": "Contract",
            "tenure": "Tenure (m)",
            "InternetService": "Internet Service",
            "PaymentMethod": "Payment Method",
            "MonthlyCharges": "Monthly Charges ($)",
        }
        if actual_col:
            column_map[actual_col] = "Actual Churn"

        existing_display_cols = [c for c in column_map.keys() if c in display_df.columns]
        render_df = display_df[existing_display_cols].rename(columns=column_map)

        st.dataframe(render_df, use_container_width=True)

        st.markdown("---")

        # --- 5. HIGH-RISK CUSTOMER SPOTLIGHT ---
        st.markdown("### 🚨 High-Risk Customer Spotlight")
        st.caption("Top 10 highest predicted-risk accounts from current selection.")

        high_risk_spotlight = filtered_df[filtered_df["Risk_Level"] == "High Risk"] if "Risk_Level" in filtered_df.columns else filtered_df
        
        if len(high_risk_spotlight) == 0:
            st.info("No High-Risk customers present in the current filter selection.")
        else:
            spotlight_df = high_risk_spotlight.head(10).copy()
            if spotlight_df[prob_col].max() <= 1.0:
                spotlight_df["Formatted_Prob"] = (spotlight_df[prob_col] * 100).map("{:.2f}%".format)
            else:
                spotlight_df["Formatted_Prob"] = spotlight_df[prob_col].map("{:.2f}%".format)

            spotlight_render = spotlight_df[existing_display_cols].rename(columns=column_map)
            st.dataframe(spotlight_render, use_container_width=True)

        st.markdown("---")

        # --- 6. CUSTOMER ACTION GUIDANCE ---
        st.markdown("### 💡 Recommended Business Action Guidance")

        if selected_risk == "High Risk":
            st.error(
                "**Suggested action:** Prioritize proactive retention outreach. Review contract status, "
                "early-tenure experience, service needs, and payment experience before selecting an appropriate intervention."
            )
        elif selected_risk == "Medium Risk":
            st.warning(
                "**Suggested action:** Use preventive engagement, value communication, and monitoring to reduce "
                "the chance of the customer moving into a higher-risk segment."
            )
        elif selected_risk == "Low Risk":
            st.success(
                "**Suggested action:** Maintain service quality and use normal loyalty and engagement activities "
                "rather than aggressive retention intervention."
            )
        else:
            st.info(
                "**Suggested action:** Use predicted risk to prioritize outreach, beginning with customers showing "
                "higher predicted churn probability."
            )

        st.markdown("---")

        # --- 7. RISK DISTRIBUTION OF FILTERED CUSTOMERS ---
        st.markdown("### 📊 Risk Distribution of Filtered Customers")

        col_fig, col_fig_text = st.columns([1.2, 1])

        with col_fig:
            filtered_risk_counts = (
                filtered_df["Risk_Level"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
                if "Risk_Level" in filtered_df.columns
                else pd.Series([0, 0, 0], index=["Low Risk", "Medium Risk", "High Risk"])
            )

            fig_f_risk, ax_f_risk = plt.subplots(figsize=(6, 3.5))
            colors = ["#2ecc71", "#f39c12", "#e74c3c"]

            bars_f = ax_f_risk.bar(
                filtered_risk_counts.index,
                filtered_risk_counts.values,
                color=colors,
                width=0.5,
                edgecolor="black",
                linewidth=0.8,
            )

            for bar in bars_f:
                height = bar.get_height()
                ax_f_risk.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (max(filtered_risk_counts.values) * 0.02 if max(filtered_risk_counts.values) > 0 else 0.1),
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            ax_f_risk.set_ylabel("Customer Count", fontsize=9, fontweight="bold")
            ax_f_risk.set_ylim(0, max(filtered_risk_counts.values) * 1.25 if max(filtered_risk_counts.values) > 0 else 10)
            ax_f_risk.spines["top"].set_visible(False)
            ax_f_risk.spines["right"].set_visible(False)
            ax_f_risk.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()

            st.pyplot(fig_f_risk)

        with col_fig_text:
            st.markdown("#### Selection Composition Breakdown")
            for r_lvl in ["High Risk", "Medium Risk", "Low Risk"]:
                cnt = int(filtered_risk_counts.get(r_lvl, 0))
                pct = (cnt / total_filtered * 100) if total_filtered > 0 else 0.0
                st.write(f"• **{r_lvl}:** {cnt:,} accounts ({pct:.1f}%)")

        st.markdown("---")

        # --- 8. OPTIONAL CUSTOMER DETAIL INSPECTOR ---
        st.markdown("### 🔍 Individual Customer Profile Inspector")

        customer_list = filtered_df["customerID"].tolist() if "customerID" in filtered_df.columns else []
        if customer_list:
            selected_cust_id = st.selectbox("Select a customer to inspect", customer_list)

            cust_row = filtered_df[filtered_df["customerID"] == selected_cust_id].iloc[0]

            cust_p_val = cust_row[prob_col]
            cust_p_str = f"{cust_p_val * 100:.2f}%" if cust_p_val <= 1.0 else f"{cust_p_val:.2f}%"
            cust_risk = cust_row.get("Risk_Level", "N/A")

            ci1, ci2, ci3, ci4 = st.columns(4)
            with ci1:
                st.write(f"**Customer ID:** `{cust_row.get('customerID', 'N/A')}`")
                st.write(f"**Risk Level:** `{cust_risk}`")
            with ci2:
                st.write(f"**Churn Probability:** `{cust_p_str}`")
                st.write(f"**Contract:** `{cust_row.get('Contract', 'N/A')}`")
            with ci3:
                st.write(f"**Tenure:** `{cust_row.get('tenure', 'N/A')} months`")
                st.write(f"**Internet Service:** `{cust_row.get('InternetService', 'N/A')}`")
            with ci4:
                st.write(f"**Payment Method:** `{cust_row.get('PaymentMethod', 'N/A')}`")
                st.write(f"**Monthly Charges:** `${cust_row.get('MonthlyCharges', 0.0):.2f}`")
                if actual_col in cust_row:
                    st.write(f"**Actual Churn:** `{cust_row[actual_col]}`")

            st.markdown("**Recommended Next Step:**")
            if cust_risk == "High Risk":
                st.error("Priority proactive retention: Review onboarding setup, offer contract conversion options, and assign support touchpoints.")
            elif cust_risk == "Medium Risk":
                st.warning("Preventive maintenance: Provide service feature guidance, check-in on billing satisfaction, and promote add-on value.")
            else:
                st.success("Standard engagement: Maintain service quality and standard loyalty touchpoints.")

    st.markdown("---")

    # --- 9. IMPORTANT BUSINESS NOTE ---
    st.info(
        "📝 **Important Business Note:** Risk scores are model predictions, not confirmed outcomes. "
        "Customer-level actions should be validated through appropriate retention processes and future campaign results."
    )

    # --- 10. FINAL EXECUTIVE TAKEAWAY ---
    st.markdown("### 📌 Executive Takeaway")
    st.info(
        "Customer-level risk exploration converts the predictive model into an actionable workflow: "
        "identify higher-risk customers, review their characteristics, prioritize appropriate outreach, "
        "and use future customer responses to evaluate the effectiveness of retention actions."
    )
