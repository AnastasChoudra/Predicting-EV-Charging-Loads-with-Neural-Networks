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
3-layer fully-connected network built in PyTorch  
