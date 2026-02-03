# ⚡ EV-Charging-Load-Prediction  
Predicting residential EV-charging energy (kWh) from real-world Norwegian apartment data with PyTorch.

---

## 📌 What & Why
Accurate forecasts of **how much energy every charging session will actually draw** help operators size transformers, schedule load, and estimate energy-cost revenue before installing new stations.  
We train a small feed-forward net on 6 833 real sessions from apartment garages in Norway and beat a linear baseline by **≈ 12 %** (MSE ↓ from 131.4 → 115.2 kWh²).

---

## 🗃️ Data
Mendeley open dataset  
*“Residential electric vehicle charging datasets from apartment buildings”*  
[doi:10.17632/jbks2rcwyj.1](https://data.mendeley.com/datasets/jbks2rcwyj/1)

| File | Rows | Description |
|---|---|---|
| `EV charging reports.csv` | 6 833 sessions | plug-in/out times, garage ID, user ID, kWh delivered, private/public flag |
| `Local traffic distribution.csv` | 8 784 h | hourly vehicle traffic counts around the buildings |

---

## 🧪 Features used (26)

---

## 🏗️ Model
3-layer fully-connected network built in PyTorch (26→56→26→1 architecture)
- **Training**: 3000-4500 epochs with Adam optimizer (lr=0.0007)
- **Loss Function**: Mean Squared Error (MSE)
- **Results**: Test MSE = 115.2 kWh² (12% improvement over linear baseline)

---

## 💼 Professional Context

This project demonstrates **analytical and energy-related experience** relevant to roles in:
- Energy analytics and forecasting
- EV infrastructure planning
- Utility operations and grid management
- Data science in the energy sector

### Key Competencies Showcased
✅ **Analytical Skills**: Statistical modeling, feature engineering, model evaluation, performance optimization  
✅ **Energy Domain**: EV charging infrastructure, load forecasting, utility planning, energy consumption analysis  
✅ **Technical Skills**: Python, PyTorch, scikit-learn, pandas, regression analysis, neural networks  
✅ **Business Impact**: Supporting infrastructure decisions, revenue forecasting, capacity planning

📄 **See [EXPERIENCE_SHOWCASE.md](EXPERIENCE_SHOWCASE.md) for detailed breakdown of skills and experience demonstrated by this project**

---

## 🎯 Business Applications
- **Transformer Sizing**: Predict peak loads to size electrical infrastructure appropriately
- **Revenue Forecasting**: Estimate energy costs before installing charging stations
- **Load Management**: Schedule and balance electrical loads across the distribution network
- **Investment Planning**: Data-driven decisions for EV infrastructure deployments
