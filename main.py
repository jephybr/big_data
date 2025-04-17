import dask.dataframe as dd
import matplotlib.pyplot as plt

# Explicit column dtypes to avoid ValueError
dtype_dict = {
    'County Name': 'object',
    'Number of Trips': 'float64',
    'Number of Trips 1-3': 'float64',
    'Number of Trips 10-25': 'float64',
    'Number of Trips 100-250': 'float64',
    'Number of Trips 25-50': 'float64',
    'Number of Trips 250-500': 'float64',
    'Number of Trips 3-5': 'float64',
    'Number of Trips 5-10': 'float64',
    'Number of Trips 50-100': 'float64',
    'Number of Trips <1': 'float64',
    'Number of Trips >=500': 'float64',
    'Population Not Staying at Home': 'float64',
    'Population Staying at Home': 'float64',
    'State Postal Code': 'object',
    'Week': 'int64'
}

#load dataset with correct datatypes
df = dd.read_csv("Trips_by_Distance.csv", dtype=dtype_dict)
df_full = dd.read_csv("Trips_Full_Data.csv", dtype=dtype_dict)

#clean up columns
df.columns = df.columns.str.strip()
df_full.columns = df_full.columns.str.strip()

df_full['Date'] = dd.to_datetime(df_full['Date'], errors='coerce')
df_full['Week'] = df_full['Date'].dt.isocalendar().week

#counting unique value of the week column
df['Week'].nunique().compute()
df_full['Week'].nunique().compute()

#group and compute average population staying at home
weekly_home = df.groupby('Week')['Population Staying at Home'].mean().compute()
weekly_travellers = df_full.groupby('Week')['Trips 1-25 Miles'].mean().compute()

#Barplot of people staying at home
plt.figure(figsize=(10,7))
plt.bar(weekly_home.index, weekly_home.values, color='orange', width=0.4)
plt.xlabel("Week")
plt.ylabel("Avg People Staying at Home")
plt.title("Weekly Average of People Staying at Home")
plt.grid(False)
plt.show()


#Average people traveling vs distance
distance_column = [
    'Number of Trips <1', 
    'Number of Trips 1-3', 
    'Number of Trips 3-5',
    'Number of Trips 5-10', 
    'Number of Trips 10-25', 
    'Number of Trips 25-50',
    'Number of Trips 50-100', 
    'Number of Trips 100-250',
    'Number of Trips 250-500', 
    'Number of Trips >=500'
]

#Calculate average trips
average_trips = df[distance_column].mean().compute()

#2nd barplot: people traveling vs distance
plt.figure(figsize=(12,6))
plt.bar(average_trips.index, average_trips.values, color='teal')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Trip Distance Range (miles)')
plt.ylabel('Average Number of Trips')
plt.title('Average People Traveling vs Distance')
plt.grid(False)
plt.tight_layout()
plt.show



#Scatter plot for 10-25 miles >10M

import plotly.express as px

#data filtering
df_10_25 = df[df['Number of Trips 10-25'] > 10_000_000].compute()
df_50_100 = df[df['Number of Trips 50-100'] > 10_000_000].compute()

#get dates
dates_10_25 = set(df_10_25['Date'])
dates_50_100 = set(df_50_100['Date'])

#compare dates
common_dates = dates_10_25.intersection(dates_50_100)
only_10_25 = dates_10_25 - dates_50_100
only_50_100 = dates_50_100 - dates_10_25

#print("Dates where both trips > 10M:", sorted(common_dates))
#print("Dates ONLY for 10-25 mile trips > 10M:", sorted(only_10_25))
#print("Dates ONLY for 50-100 mile trips > 10M:", sorted(only_50_100)) 

#scatter plot for 10-25 mile trips
fig1 = px.scatter(
    x=df_10_25["Date"],
    y=df_10_25["Number of Trips 10-25"],
    title="Trips 10-25 Miles > 10M",
    labels={
        "x": "Date",
        "y": "Number of Trips (10-25 Miles)"
    }
)
fig1.show()



#scatter plot for 50-100 mile trips
fig2 = px.scatter(
    x=df_50_100["Date"],
    y=df_50_100["Number of Trips 50-100"],
    title="Trips 50-100 Miles > 10M",
    labels={
        "x": "Dates",
        "y": "Number of Trips (50-100 Miles)"
    }
)
fig2.show()



#comparison visualization

import plotly.graph_objects as go

#scatter plot with both 10-25 miles and 50-100 miles trips
fig = go.Figure()

#Add trace to 10-25 mile trips
fig.add_trace(go.Scatter(
    x=df_10_25["Date"],
    y=df_10_25["Number of Trips 10-25"],
    mode="markers",
    name="10-25 Miles Trips",
    marker=dict(color="blue")
))

#Add trace to 50-100 mile trips
fig.add_trace(go.Scatter(
    x=df_50_100["Date"],
    y=df_50_100["Number of Trips 50-100"],
    mode="markers",
    name="50-100 Miles Trips",
    marker=dict(color="orange")
))

#update layout with titles
fig.update_layout(
    title="Comparison: Days with >10M Trips (10–25 Miles vs. 50–100 Miles)",
    xaxis_title="Date",
    yaxis_title="Number of Trips",
    legend_title="Trip Distance",
    template="plotly_dark"
)

fig.show()




#c. parallel computing using dask vs pandas
import time
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client

#filepath
file_path = "Trips_by_Distance.csv"

#loading dataset with dask
dtype_map = {
    'County Name': 'object',
    'Number of Trips': 'float64',
    'Number of Trips 1-3': 'float64',
    'Number of Trips 10-25': 'float64',
    'Number of Trips 100-250': 'float64',
    'Number of Trips 25-50': 'float64',
    'Number of Trips 250-500': 'float64',
    'Number of Trips 3-5': 'float64',
    'Number of Trips 5-10': 'float64',
    'Number of Trips 50-100': 'float64',
    'Number of Trips <1': 'float64',
    'Number of Trips >=500': 'float64',
    'Population Not Staying at Home': 'float64',
    'Population Staying at Home': 'float64',
    'State Postal Code': 'object',
    'Week': 'int64'
}

df = dd.read_csv(file_path, blocksize="16MB", dtype=dtype_map)
df.columns = df.columns.str.strip()

#define number preprocessors to compare
n_processors = [10,20]
n_processors_time = {}

for processor in n_processors:
    print(f"\n\nStarting computation with {processor} processors...\n")

    #start dask client
    from dask.distributed import default_client
    try: 
        default_client().close()
    except ValueError:
        pass
    
    client = Client(n_workers=processor, memory_limit="2GB", dashboard_address=":0")

    #track time
    start = time.time()

    #question a
    weekly_home = df.groupby("Week")["Population Staying at Home"].mean().compute()

    #question b
    df_10_25 = df[df["Number of Trips 10-25"] > 10000000][["Date", "Number of Trips 10-25"]].compute()
    df_50_100 = df[df['Number of Trips 50-100'] > 10000000][['Date', 'Number of Trips 50-100']].compute()

    #stop timer
    dask_time = time.time() - start
    n_processors_time[processor] = dask_time

    print(f"Time with {processor} processors: {dask_time:.2f} seconds\n")

    #close dask client
    client.close()

#print results
print("\n=== Time Taken with Parallel Processors ===")
for proc, secs in n_processors_time.items():
    print(f"{proc} processors: {secs:.2f} seconds")





#1d.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#load dataset
df = pd.read_csv("Trips_By_Distance.csv")
df_full = pd.read_csv("Trips_Full_Data.csv")

#strip whitespaces from column
df.columns = df.columns.str.strip()
df_full.columns = df_full.columns.str.strip()

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
df['Week'] = df['Date'].dt.isocalendar().week
df_full['Week'] = df_full['Date'].dt.isocalendar().week

df_week32 = df[df['Week'] == 32]
df_full_week32 = df_full[df_full['Week'] == 32]

print("Rows in df_week32:", len(df_week32))
print("Rows in df_full_week32:", len(df_full_week32))

#feature dataset
y = df_week32['Number of Trips 10-25']
x = df_full_week32[['Trips 1-25 Miles', 'Trips 25-100 Miles']]

#reset indices
x = x.reset_index(drop=True)
y = y.reset_index(drop=True)

#drop NaN
combined_df = pd.concat([x, y], axis=1).dropna()

#extract final x and y
x_clean = combined_df[['Trips 1-25 Miles', 'Trips 25-100 Miles']].values
y_clean = combined_df['Number of Trips 10-25'].values

#check shape
print("Shape of x_clean:", x_clean.shape)
print("Shape of y_clean:", y_clean.shape)

#proceed only if data is valid
if len(x_clean) > 0 and len(y_clean) > 0:
    x_train, x_test, y_train, y_test = train_test_split(x_clean, y_clean, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Evaluate
    r_sq = model.score(x_test, y_test)
    print(f"R²: {r_sq}")
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")

    # Predict
    y_pred = model.predict(x_test)
    print("\nPredictions:\n", y_pred)

else:
    print("\nNo data left after filtering — check the input CSVs or Week 32 availability.\n")


#drop missing values
x = x.dropna()
y = y.loc[x.index]  # Align indices

#transform input data with polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

#fit model
model = LinearRegression()
model.fit(x_poly, y)

#get results
r_sq = model.score(x_poly, y)
intercept = model.intercept_
coefficients = model.coef_
y_pred = model.predict(x_poly)

#print results
print(f"R²: {r_sq}")
print(f"Intercept: {intercept}")
print(f"Coefficients: {coefficients}")
print(f"\nPredictions (first 5):\n{y_pred[:5]}\n")

plt.figure(figsize=(8,5))
plt.plot(y.values, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.title("Polynomial Regression Fit (Week 32)")
plt.xlabel("Data Point")
plt.ylabel("Number of Trips (10-25 miles)")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()




#e.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_full = pd.read_csv("Trips_Full_Data.csv")

#distance columns to analyze
distance_columns = [
    'Trips 1-25 Miles',
    'Trips 25-100 Miles',
    'Trips 100-250 Miles',
    'Trips 250-500 Miles',
    'Trips 500+ Miles'
]

#drop missing values
df_full_clean = df_full[distance_columns].dropna()

#sum of trip for each distance category
trip_totals = df_full_clean.sum()

#barplot
plt.figure(figsize=(10,6))
sns.barplot(x=trip_totals.index, y=trip_totals.values)

plt.title("Total Number of Travellers by Trip Distance", fontsize=16)
plt.xlabel("Trip Distance", fontsize=14)
plt.ylabel("Number of Travellers", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(False)
plt.show()




#2a
import pandas as pd

df = pd.read_csv("Trips_By_Distance.csv")

#fill missing values
df["Population Staying at Home"] = df["Population Staying at Home"].fillna(0)

#data exploration
print("\nColumn Data Types:\n")
print(df.dtypes)

#count non-null values
print("\nNon-null Count per Column:\n")
print(df.notnull().sum())

#null and missing values
print("\nNull Values per Column:\n")
print(df.isna().sum())

#descriptive statistics
print("\nDescriptive Statistics (All Columns):\n")
print(df.describe())

#descriptive statistics for staying at home
print("\nDescriptive Statistics for 'Population Staying at Home':\n")
print(df['Population Staying at Home'].describe())



#2b(i).

import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import time

#Sequential processing
start_seq = time.time()

df_seq = pd.read_csv("Trips_Full_Data.csv")

df_seq['Travelled'] = df_seq['Trips'] > 0
traveled_df_seq = df_seq[df_seq['Travelled'] == True]

#average miles traveled
avg_trip = traveled_df_seq[['Trips 1-25 Miles', 'Trips 25-100 Miles']].mean()

end_seq = time.time()
print("\nSequential Processing Time:", end_seq - start_seq, "seconds")

#Parallel processing
start_par = time.time()

df_par = dd.read_csv("Trips_Full_Data.csv")
df_par['Travelled'] = df_par['Trips'] > 0
traveled_df_par = df_par[df_par['Travelled'] == True]

#average travel distances using dask
avg_trip_par = traveled_df_par[['Trips 1-25 Miles', 'Trips 25-100 Miles']].mean().compute()

end_par = time.time()
print("\nParallel Processing Time:", end_par - start_par, "seconds\n\n\n")

#total miles by trip segment
trip_counts = traveled_df_seq[['Trips 1-25 Miles', 'Trips 25-100 Miles', 'Trips 100-250 Miles']].mean()

# Bar plot
plt.figure(figsize=(10, 6))
trip_counts.plot(kind='bar', color='skyblue')
plt.title('Average Distance Traveled by People Who Left Home')
plt.xlabel('Trip Distance Category')
plt.ylabel('Average Number of Trips')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#2b(ii)

import dask.dataframe as dd
import matplotlib.pyplot as plt

dtype_map = {
    'County Name': 'object',
    'Number of Trips': 'float64',
    'Number of Trips 1-3': 'float64',
    'Number of Trips 10-25': 'float64',
    'Number of Trips 100-250': 'float64',
    'Number of Trips 25-50': 'float64',
    'Number of Trips 250-500': 'float64',
    'Number of Trips 3-5': 'float64',
    'Number of Trips 5-10': 'float64',
    'Number of Trips 50-100': 'float64',
    'Number of Trips <1': 'float64',
    'Number of Trips >=500': 'float64',
    'Population Not Staying at Home': 'float64',
    'Population Staying at Home': 'float64',
    'State Postal Code': 'object',
    'Week': 'int64'
}

df = dd.read_csv("Trips_By_Distance.csv", dtype=dtype_map)
df.columns = df.columns.str.strip()

#count how many unique weeks are available
num_weeks = df['Week'].nunique().compute()
print(f"Number of unique weeks: {num_weeks}\n\n")

#staying at home week average
weekly_home_avg = df.groupby('Week')['Population Staying at Home'].mean().compute()

#plot for average weekly population staying at home
plt.figure(figsize=(12, 6))
weekly_home_avg.plot(kind='line', marker='o', color='orange')
plt.title('Weekly Average of Population Staying at Home')
plt.xlabel('Week Number')
plt.ylabel('Average Population Staying at Home')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



#2b(iii)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("Trips_By_Distance.csv")
df_full = pd.read_csv("Trips_Full_Data.csv")

#strip whitespaces from column
df.columns = df.columns.str.strip()
df_full.columns = df_full.columns.str.strip()

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
df['Week'] = df['Date'].dt.isocalendar().week
df_full['Week'] = df_full['Date'].dt.isocalendar().week

#filter week32 data
df_week32 = df[df['Week'] == 32]
df_full_week32 = df_full[df_full['Week'] == 32]

#define feature columns
x = df_full_week32[['Trips 1-25 Miles', 'Trips 25-100 Miles']]
y = df_week32['Number of Trips 5-10']

#reset indices
x = x.reset_index(drop=True)
y = y.reset_index(drop=True)

#drop NaN
combined_df = pd.concat([x, y], axis=1).dropna()

#extract final x and y
x_clean = combined_df[['Trips 1-25 Miles', 'Trips 25-100 Miles']].values
y_clean = combined_df['Number of Trips 5-10'].values

#linear regression
x_np = np.array(x_clean)
y_np = np.array(y_clean)

#train model
model = LinearRegression()
model.fit(x_np, y_np)

#evaluate model
r_sq = model.score(x_np, y_np)
print(f"R² (coefficient of determination): {r_sq}")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

#predict y
y_pred = model.predict(x_np)
print(f"\nPredicted response:\n{y_pred}\n")
