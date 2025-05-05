import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import numpy as np
from datetime import datetime

# ||------DATA PREPROCESSING------||

# Load the dataset
df = pd.read_csv("weatherdata.csv")

# Drop unnamed/unnecessary column
new_df = df.drop("Unnamed:0", axis=1)

# check rows with any null values
l=new_df.isnull().sum()
print(l,"\n")    # there is no null value in our dataset 

# Convert 'Date' column to datetime and format as DD/MM/YYYY
new_df['Date'] = pd.to_datetime(new_df['Date'], errors='coerce')
new_df.dropna(subset=['Date'], inplace=True)  # Drop rows where date parsing failed

# Display dataset info and statistics
print(new_df.info(),"\n")
pd.set_option('display.max_columns', None)    # to see all columns in output
print(new_df.describe(),"\n")

# # Example: Calculate IQR for MaxT and MinT
# a = np.percentile(new_df["MaxT"], 25)
# b = np.percentile(new_df["MinT"], 75)
# iqr = b - a
# print(f"IQR: {iqr}")
# lb = a - (1.5 * iqr)
# ub = b + (1.5 * iqr)
# print(f"Range is: [{lb}, {ub}]")
# # Optional: Boxplot of MaxT
# new_df.boxplot(column="MaxT")
# plt.show()

# print(new_df.info())
# print(new_df.head())
# print(new_df.tail())


# ||-----EXPLORATORY DATA ANALYSIS-----||

# Extract day, month, year for trend analysis
new_df['Day'] = new_df['Date'].dt.day
new_df['Month'] = new_df['Date'].dt.month
new_df['Year'] = new_df['Date'].dt.year

# Convert temperature columns to numeric (in case they were read as strings)
new_df['MaxT'] = pd.to_numeric(new_df['MaxT'], errors='coerce')
new_df['MinT'] = pd.to_numeric(new_df['MinT'], errors='coerce')

# Average temperature (mean of MaxT and MinT)
new_df['AvgT'] = (new_df['MaxT'] + new_df['MinT']) / 2

# Print basic stats
print("\n--- BASIC TEMPERATURE STATISTICS ---")
print(f"Average Temperature: {new_df['AvgT'].mean():.2f}°","\n")
print(f"Maximum Temperature: {new_df['MaxT'].max()}°","\n")
print(f"Minimum Temperature: {new_df['MinT'].min()}°","\n")

# ||-----MONTHLY TEMPERATURE TRENDS------||
# Group by Month and calculate mean
monthly_trends = new_df.groupby('Month')[['MaxT', 'MinT', 'AvgT']].mean()
print("\n--- MONTHLY AVERAGE TEMPERATURES ---")
print(monthly_trends,"\n")

# VISUALIZATION 
# Line plot of monthly trends
plt.figure(figsize = (10, 6))
sns.lineplot(data = monthly_trends)
plt.title('Monthly Temperature Trends')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(ticks = range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.show()

# BAR CHART: AVERAGE MONTHLY TEMPERATURE 

plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_trends.index, y=monthly_trends['AvgT'], hue=monthly_trends.index, palette='coolwarm')
plt.title('Average Monthly Temperature')
plt.xlabel('Month')
plt.ylabel('Average Temperature (°C)')
plt.xticks(ticks=range(0, 12), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# |-----YEARLY TEMPERATURE TRENDS-----|
# Group by year and compute mean of temperature columns
yearly_trends = new_df.groupby('Year')[['MaxT', 'MinT', 'AvgT']].mean()
print("\n--- YEARLY AVERAGE TEMPERATURES ---")
print(yearly_trends,"\n")

# Plot yearly temperature trends
plt.figure(figsize = (10, 6))
sns.lineplot(data = yearly_trends)
plt.title('Yearly Temperature Trends')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()


# ||-------MONTHLY RAINFALL TREND-------||
# Group by Month
monthly_rain = new_df.groupby('Month')['Rain'].mean()
print(monthly_rain,"\n")

# Plot Monthly Rainfall
plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_rain.index, y=monthly_rain.values, hue=monthly_rain.index, palette='Blues', legend = False)
plt.title('Average Monthly Rainfall')
plt.xlabel('Month')
plt.ylabel('Rainfall (mm)')
plt.xticks(ticks=range(0, 12), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ||------YEARLY RAINFALL TREND-----||
# Group by year
yearly_rain = new_df.groupby('Year')['Rain'].mean()
print(yearly_rain,"\n")

# Plot Yearly Rainfall
plt.figure(figsize=(10, 6))
sns.lineplot(x=yearly_rain.index, y=yearly_rain.values, marker='o', color='green')
plt.title('Average Yearly Rainfall')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.grid(True)
plt.tight_layout()
plt.show()