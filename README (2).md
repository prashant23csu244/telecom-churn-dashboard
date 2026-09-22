# Telecom Customer Churn \& Retention Intelligence

An end-to-end Business Intelligence and Machine Learning project that analyzes telecom customer churn, identifies high-risk customers, and converts predictive insights into actionable customer-retention strategies.

\---

## 📌 Project Overview

Customer churn is a major challenge for telecom companies. This project uses customer-level telecom data to understand churn patterns, identify important customer segments, predict churn risk, and recommend suitable retention actions.

The project follows the business intelligence workflow:

**Data → Information → Insights → Decision → Action**

The solution includes:

* Exploratory Data Analysis (EDA)
* Business Intelligence insights
* Hypothesis formulation
* Logistic Regression churn prediction
* Customer risk classification
* Business action recommendations
* Interactive Streamlit dashboard

\---

## 🎯 Project Objectives

1. Analyze customer churn patterns.
2. Identify customer segments with higher observed churn.
3. Create meaningful visualizations and business insights.
4. Formulate hypotheses for further investigation.
5. Build a machine-learning model to estimate churn probability.
6. Classify customers into Low, Medium, and High risk groups.
7. Develop retention strategies based on predicted risk.
8. Provide an interactive Streamlit dashboard for decision support.

\---

## 📊 Dataset

The project uses the **IBM Telco Customer Churn** dataset.

### Dataset Information

* **Customers:** 7,043
* **Columns:** 21
* **Churned Customers:** 1,869
* **Overall Churn Rate:** 26.54%
* **Tenure:** 0–72 months
* **Average Monthly Charges:** $64.76

### Original Dataset Source

IBM Telco Customer Churn dataset:

https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

Raw CSV:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The project uses a cleaned version of the original dataset for analysis and dashboard development.

\---

## 🧹 Data Preparation

The following preprocessing steps were performed:

* Cleaned and standardized column names.
* Checked for missing values.
* Checked for duplicate records.
* Prepared categorical variables for analysis.
* Excluded `customerID` from machine-learning features.
* Encoded categorical variables for model training.
* Converted the `Churn` target into binary values:

  * `Yes = 1`
  * `No = 0`

\---

## 🔎 Exploratory Data Analysis

Five major areas were analyzed:

### 1\. Contract Type

Month-to-month customers show an observed churn rate of approximately **42.71%**.

### 2\. Tenure

Customers with **0–12 months** of tenure show approximately **47.44%** churn, while customers with **61+ months** show approximately **6.61%** churn.

### 3\. Internet Service

Observed churn rates:

* Fiber optic: **41.89%**
* DSL: **18.96%**
* No internet service: **7.40%**

### 4\. Monthly Charges

Customers with monthly charges between **$60–$90** show approximately **33.74%** churn.

### 5\. Payment Method

Electronic-check customers show approximately **45.28%** churn.

These are observed patterns in the dataset and do not by themselves establish causation.

\---

## 💡 Key Insights

### Insight 1 — Month-to-Month Customers

Month-to-month customers have a substantially higher observed churn rate than customers on longer contracts.

### Insight 2 — Early-Tenure Customers

Customers in their first year show considerably higher observed churn than long-tenure customers.

### Insight 3 — Fiber Customers

The Fiber optic segment has a higher observed churn rate than DSL and customers without internet service.

\---

## 🧪 Hypotheses

### H1 — Contract Commitment

Month-to-month customers may be more price-sensitive or less committed to long-term service arrangements.

### H2 — Early Customer Experience

New customers may experience onboarding or early-service issues that could contribute to higher churn during the first year.

### H3 — Fiber Customer Experience

Fiber customers may have combinations of higher charges, month-to-month contracts, or fewer add-on services that could be associated with higher churn.

These hypotheses are not treated as proven causal explanations.

\---

## 🤖 Machine Learning Model

### Algorithm

**Logistic Regression**

The model predicts whether a customer is likely to churn.

### Train-Test Split

* Training set: 80%
* Testing set: 20%
* Stratified split used to preserve the churn-class distribution.

### Model Evaluation

|Metric|Result|
|-|-:|
|Accuracy|80.48%|
|Precision|65.62%|
|Recall|55.61%|

### Confusion Matrix

||Predicted No|Predicted Yes|
|-|-:|-:|
|Actual No|926|109|
|Actual Yes|166|208|

The model is intended for **risk prioritization**, not as a guarantee that an individual customer will churn.

\---

## ⚠️ Customer Risk Classification

Customers are classified according to predicted churn probability.

|Risk Level|Probability|
|-|-|
|Low Risk|< 30%|
|Medium Risk|30% – <60%|
|High Risk|≥ 60%|

### Risk Distribution

|Risk Level|Customers|Share|
|-|-:|-:|
|High|1,043|14.81%|
|Medium|1,616|22.94%|
|Low|4,384|62.25%|

These figures are based on model predictions for all **7,043 customers**.

\---

## 🚨 High-Risk Customer Profile

The model identified **1,043 high-risk customers**.

Key characteristics of this group include:

* **100%** are on month-to-month contracts.
* **93.58%** use Fiber optic internet.
* **74.50%** have 0–12 months of tenure.
* **82.17%** use electronic check.
* **72.10%** have an actual recorded churn value of Yes.
* Average monthly charge is approximately **$82.11**.

These characteristics describe the model-identified high-risk segment and should not be interpreted as proof of causation.

\---

## 🎯 Business Recommendations

### 1\. Contract Transition Incentives

Target high-risk month-to-month customers with appropriate incentives to consider longer-term contracts.

### 2\. First-Year \& Fiber Onboarding

Provide structured customer engagement through 30/60/90-day touchpoints, particularly for early-tenure Fiber customers.

### 3\. Preventive Medium-Risk Engagement

Use proactive communication, feature education, and value reminders for medium-risk customers before their risk increases.

\---

## 🖥️ Streamlit Dashboard

The interactive Streamlit dashboard contains five sections:

### 1\. Executive Overview

* Total customers
* Overall churn rate
* High-risk customers
* Medium-risk customers
* Average monthly charges
* Risk distribution
* Contract churn patterns
* Key business insights

### 2\. Churn Analysis

Includes visualizations for:

* Contract type
* Tenure
* Internet service
* Monthly charges

### 3\. Prediction \& Risk

Displays:

* Model accuracy
* Precision
* Recall
* Confusion matrix
* Risk distribution
* High-risk customer validation
* Business takeaway

### 4\. Business Action Plan

Provides recommended actions for:

* High Risk
* Medium Risk
* Low Risk

### 5\. Customer Risk Explorer

Allows users to:

* Filter customers by risk level.
* Explore customer-level predictions.
* View high-risk customer information.
* Review risk-based guidance.

\---

## 🗂️ Project Structure

```text
telecom-churn-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── telecom\_churn\_cleaned.csv
└── customer\_risk\_predictions.csv
```

\---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib

\---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR\_GITHUB\_REPOSITORY\_URL
```

Move into the project folder:

```bash
cd telecom-churn-dashboard
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

\---

## ▶️ Run the Dashboard

Run the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your browser.

\---

## 📁 Required Project Files

The project should contain:

* `app.py`
* `requirements.txt`
* `README.md`
* `telecom\_churn\_cleaned.csv`
* `customer\_risk\_predictions.csv`

The CSV files are required by the dashboard if the application loads the saved datasets.

\---

## ⚠️ Project Limitations

* The model uses historical customer data and does not guarantee future customer behavior.
* Observed relationships do not establish causality.
* Risk thresholds are project-level thresholds and may require business validation.
* Retention recommendations should be evaluated against operational cost and customer response.
* The dataset is a sample and may not represent every telecom market or customer population.

\---

## 👨‍💻 Author

**Prashant Kumar**

B.Tech Computer Science — Artificial Intelligence \& Machine Learning

\---

## 📌 Note

This project is intended as a Business Intelligence and Machine Learning decision-support solution. Model predictions should be used to prioritize customer-retention efforts and should be combined with business knowledge and customer-level context.

