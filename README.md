# 🏥 Healthcare Cost & Utilisation Analyser

An end-to-end healthcare analytics project analysing CMS Medicare inpatient data to identify cost drivers, regional utilisation patterns, and predict Medicare payments — built to demonstrate life sciences analytics skills for healthcare consulting roles.

---

## 📌 Project Overview

This project analyses a CMS Medicare-style inpatient dataset to:
- Identify the costliest diagnosis groups and procedures
- Analyse regional cost variation across 30 US states
- Predict average Medicare payments using ML regression models
- Surface KPIs and cost intelligence via an interactive Streamlit dashboard

---

## 🗂️ Project Structure

```
healthcare-cost-analyser/
│
├── data/
│   ├── generate_data.py        ← CMS-style dataset generator
│   └── cms_inpatient.csv       ← Generated dataset (3,000 records)
│
├── models/
│   ├── train_model.py          ← Model training (Random Forest + Linear Regression)
│   ├── rf_model.pkl            ← Saved Random Forest model
│   ├── lr_model.pkl            ← Saved Linear Regression model
│   ├── results.pkl             ← Evaluation metrics & feature importance
│   └── encoders.pkl            ← Label encoders for categorical variables
│
├── app/
│   └── dashboard.py            ← Streamlit dashboard (5 pages)
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
cd data
python generate_data.py
```

### 3. Train the models
```bash
cd models
python train_model.py
```

### 4. Launch the dashboard
```bash
cd app
streamlit run dashboard.py
```

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 📊 Overview | KPI cards, cost by category, discharge volumes |
| 🗺️ Regional Analysis | State-level cost heatmap, covered charges vs payments |
| 🔬 Diagnosis Insights | Top procedures by cost & volume, cost-to-charge ratio |
| 🤖 Model Performance | R², RMSE, actual vs predicted, feature importance |
| 🔮 Cost Predictor | Input provider profile → predicted Medicare payment |

---

## 🤖 Models Used

| Model | R² Score | RMSE |
|---|---|---|
| Random Forest | 0.9983 | ~$107 |
| Linear Regression | 0.9874 | ~$293 |

Random Forest significantly outperforms Linear Regression, capturing non-linear relationships between covered charges, diagnosis type, and actual Medicare payments.

---

## 🔬 Key Findings

- **Orthopedic** procedures (joint replacements, spinal fusion) are the costliest category (~$9,500 avg)
- **NY and CA** show 35-40% higher costs than national average
- **Covered charges** are the strongest predictor of actual payment (feature importance: ~0.85)
- **Cost-to-charge ratios** average 0.33 — Medicare pays ~33 cents per dollar billed
- High-volume categories (Cardiovascular) don't necessarily correlate with highest cost per case

---

## 🛠️ Tech Stack

- **Python** — Pandas, NumPy, Scikit-Learn
- **ML Models** — Random Forest Regressor, Linear Regression
- **Visualisation** — Matplotlib, Seaborn
- **Dashboard** — Streamlit

---

## 📝 Resume Bullet

> *Analysed 3,000+ CMS Medicare inpatient records across 30 states to identify high-cost diagnosis groups and regional utilisation patterns; built a Random Forest regression model (R²: 0.998) to predict Medicare payments and deployed a 5-page Streamlit dashboard surfacing cost intelligence for simulated payer decision-making.*

---

## 📡 Data Source

Dataset modelled after CMS Medicare Inpatient Hospital data.
Real data available at: [data.cms.gov](https://data.cms.gov)

---

*Built as a portfolio project targeting healthcare analytics and life sciences consulting roles.*
