import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Aryan\Downloads\archive\Unemployment in India.csv")
print("First 5 Rows of Dataset:\n")
print(data.head())

print("\nDataset Information:\n")
print(data.info())

print("\nMissing Values:\n")
print(data.isnull().sum())

data = data.dropna()

data.columns = [
    'States',
    'Date',
    'Frequency',
    'Estimated_Unemployment_Rate',
    'Estimated_Employed',
    'Estimated_Labour_Participation_Rate',
    'Region'
]
data['Date'] = pd.to_datetime(data['Date'])

print("\nStatistical Summary:\n")
print(data.describe())

state_avg = data.groupby('States')['Estimated_Unemployment_Rate'].mean()

print("\nAverage Unemployment Rate by State:\n")
print(state_avg)

plt.figure(figsize=(12,5))

plt.plot(
    data['Date'],
    data['Estimated_Unemployment_Rate']
)

plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate")
plt.xticks(rotation=45)

plt.show()

covid_data = data[data['Date'].dt.year == 2020]

plt.figure(figsize=(12,5))

plt.plot(
    covid_data['Date'],
    covid_data['Estimated_Unemployment_Rate']
)

plt.title("Covid-19 Impact on Unemployment")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate")
plt.xticks(rotation=45)

plt.show()

data['Month'] = data['Date'].dt.month

monthly_avg = data.groupby('Month')[
    'Estimated_Unemployment_Rate'
].mean()

plt.figure(figsize=(10,5))

monthly_avg.plot(kind='bar')

plt.title("Monthly Average Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate")

plt.show()
region_avg = data.groupby('Region')[
    'Estimated_Unemployment_Rate'
].mean()
plt.figure(figsize=(10,5))
region_avg.plot(kind='bar')
plt.title("Region-wise Average Unemployment Rate")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate")
plt.show()
