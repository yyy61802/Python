# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display format
pd.set_option('display.float_format', '{:.0f}'.format)

# Load dataset with exception handling
try:
    file_path = r"/content/Electric_Vehicle_Population_Data.csv"
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
except Exception as e:
    print("An error occurred while loading the file:", e)

for col in df.columns:
    print("-", col)

# Replace 'unknown' and empty strings with NaN
df.replace(['unknown', ''], np.nan, inplace=True)
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

print("\nAfter replacement:")
print(df.isnull().sum().to_string())

# Rename columns to meaningful names
df.rename(columns={
    'VIN (1-10)': 'vin_10',
    'DOL Vehicle ID': 'dol_vehicle_id',
    'Model Year': 'model_year',
    'Electric Vehicle Type': 'ev_type',
    'Clean Alternative Fuel Vehicle (CAFV) Eligibility': 'cafv_eligibility',
    'Electric Range': 'electric_range',
    'Base MSRP': 'base_msrp',
    'Legislative District': 'legislative_district',
    'Vehicle Location': 'vehicle_location',
    'Electric Utility': 'electric_utility',
    '2020 Census Tract': 'census_tract_2020'
}, inplace=True)

# Convert incorrect data types
print("\nColumn data types:\n", df.dtypes.to_string())

if df['Postal Code'].dtype != 'object':
    df['Postal Code'] = df['Postal Code'].astype(str)

print("\nDataset shape:", df.shape)
print('\nAfter converting the data type:')
print("\nColumn data types:\n", df.dtypes.to_string())

# Sort before filling missing values
df.sort_values(by=['Model', 'Make', 'model_year'], inplace=True)

print("\nPreview of the dataset:")
print("Top 3 rows:\n", df.head(3))

# Drop unnecessary columns
columns_to_drop = ['vin_10', 'base_msrp', 'vehicle_location']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

print("\nDataset colums after dropping specified columns:")
for col in df.columns:
    print("-", col)

# Flexible IQR filtering function with relaxed upper boundary
def remove_outliers_iqr(df, column):
    Q1 = df[column].dropna().quantile(0.25)
    Q3 = df[column].dropna().quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column].isna()) | ((df[column] >= lower) & (df[column] <= upper))]

# Apply relaxed filtering
df_no_outliers = remove_outliers_iqr(df, 'electric_range')
print("Dataset shape:", df_no_outliers.shape)

# Calculate fill values based on clean data
median_range = df_no_outliers['electric_range'].median()
modes_cafv = df_no_outliers['cafv_eligibility'].mode()
mode_cafv = modes_cafv.iloc[0] if not modes_cafv.empty else np.nan

# Fill in missing values in the original data df based on clean data
print("\nMissing values BEFORE filling:")
print(df[['electric_range', 'cafv_eligibility']].isnull().sum().to_string())

df['electric_range'] = df['electric_range'].fillna(median_range)
df['cafv_eligibility'] = df['cafv_eligibility'].fillna(mode_cafv)

print("\nMissing values AFTER filling:")
print(df[['electric_range', 'cafv_eligibility']].isnull().sum().to_string())

duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")
df = df.drop_duplicates()

# Fill census tract special case
if 'census_tract_2020' in df.columns:
    df['census_tract_2020'] = df['census_tract_2020'].fillna(-1)

# Business problem analysis and visualization

# Distribution of numbers of registration for each brand (top 10)
plt.figure(figsize=(10, 6))
top_brands = df_no_outliers['Make'].value_counts().head(10)
sns.barplot(x=top_brands.values, y=top_brands.index)
plt.title("Top 10 Registered EV Brands")
plt.xlabel("Number of Registrations")
plt.ylabel("Brand")
plt.show()

# Distribution of vehicle types (BEV/PHEV)
plt.figure(figsize=(8, 6))
sns.countplot(data=df_no_outliers, x='ev_type', order=df_no_outliers['ev_type'].value_counts().index)
plt.title("Distribution of EV Types (BEV/PHEV)")
plt.xlabel("EV Type")
plt.ylabel("Count")
plt.show()

# Comparison of battery life of various models among the most registered brands
# Identify the brand with the highest registration volume
most_common_brand = df['Make'].value_counts().idxmax()
print(f"Most Registered Brand: {most_common_brand}")

brand_data = df[df['Make'] == most_common_brand]

# Draw the endurance box line diagram by model
plt.figure(figsize=(12, 6))
sns.boxplot(data=brand_data, x='Model', y='electric_range')
plt.title(f"Electric Range Distribution by Model - {most_common_brand}")
plt.xlabel("Model")
plt.ylabel("Electric Range (Miles)")
plt.xticks(rotation=45)
plt.show()

# Analysis of the top 5 counties with the fastest growth rate in electric vehicle registration in recent years (based on Model Year)
# Select the top 5 counties with the highest registration volume
top_counties = df['County'].value_counts().head(5).index
df_top_counties = df[df['County'].isin(top_counties)]

# Count quantities by County and model_year
trend_data = df_top_counties.groupby(['model_year', 'County']).size().unstack(fill_value=0)

# Draw line chart
trend_data.plot(figsize=(12, 6), marker='o')
plt.title("EV Registration Trends by Model Year (Top 5 Counties)")
plt.xlabel("Model Year")
plt.ylabel("Number of Registrations")
plt.xticks(rotation=45)
plt.legend(title="County")
plt.tight_layout()
plt.show()

# Evaluate the rationality of charging infrastructure
# Calculate the number of electric vehicle registrations in each county
county_ev_counts = df['County'].value_counts()

# 2. Calculate the number of supplier types for each county
county_utility_counts = df.groupby('County')['electric_utility'].nunique()

# Calculate how many countries there are
total_counties = df['County'].nunique()
print(f"Total number of unique counties: {total_counties}")

# Merge two indicators
county_analysis = pd.DataFrame({
    'EV_Registrations': county_ev_counts,
    'Utility_Providers': county_utility_counts
}).dropna().sort_values(by='EV_Registrations', ascending=False)

# Display the top 10 counties
print(county_analysis.head(10))

# Visualization
ax = county_analysis.head(10).plot(kind='bar', figsize=(12, 6), logy=True)
plt.title("Top 10 Counties - EV Registrations vs. Utility Providers (Log Scale)")
plt.ylabel("Log Count")
plt.xlabel("County")
plt.xticks(rotation=45)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()
