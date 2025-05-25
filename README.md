# Electric Vehicle Population Analysis - Washington State

This project performs comprehensive data cleaning, exploration, and visualization on the Electric Vehicle Population dataset published by the Washington State Department of Licensing. It identifies trends in EV registrations, brand popularity, electric range statistics, and assesses infrastructure fairness across counties.

**Source**: [WA State EV Registration Dataset]
**File used**: `Electric_Vehicle_Population_Data.csv`

# Features

-  Robust data cleaning (handle missing values, rename columns, drop outliers, convert types)
-  Outlier removal using IQR method
-  Imputation of key fields using medians and modes
-  Visualizations of:
  - Top EV brands
  - EV type distribution (BEV vs PHEV)
  - Electric range comparison by model
  - County-wise registration trends
  - County infrastructure fairness (charging utility coverage)
  - Business insight generation (growth trend & resource allocation logic)

# How to Run

# Install Required Packages

pip install pandas numpy matplotlib seaborn

