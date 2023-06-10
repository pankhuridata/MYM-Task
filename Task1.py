#Author: Pankhuri
#Description: Data Analysis and Insights Generation

#import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df1 = pd.read_csv('account.csv')
df2 = pd.read_csv('account_device.csv')
df3 = pd.read_csv('transactions.csv')

df = df1.merge(df2, on='account_id').merge(df3, on='account_id')        #inner join all 3 dataset

columns_to_drop = ['birth_year','last_login_city', 'gender' ]
df = df.drop(columns=columns_to_drop)                                 #dropping unneccessary columns

df['created_date'] = pd.to_datetime(df['created_date'])                # Convert columns to the correct data types
df['last_login_date'] = pd.to_datetime(df['last_login_date'])
df['created_time'] = pd.to_datetime(df['created_time'])
df['age_seconds'] = df['age_seconds'].astype(float)
df['session_count'] = df['session_count'].astype(float)

null_values = df.isnull().sum()                                            #checking the null values in dataset
df['last_login_country'].fillna('No_Country', inplace=True)
df['create_country'].fillna('No_Country', inplace=True)

df.rename(columns={'in_game_currency_amoung': 'in_game_currency_amount'}, inplace=True)   #proper name for columns

df.drop_duplicates(inplace=True)                                            #dropping the duplicates
df.info()
print('\n')

mean_amt = df['in_game_currency_amount'].mean()
median_amt = df['in_game_currency_amount'].median()
std_amt = df['in_game_currency_amount'].std()

print("Mean Amount in game :", mean_amt)
print("Median Amount in game:", median_amt)
print("Standard deviation of Amount in game:", std_amt)
print('\n')

ipad_users = df[df['device'] == 'iPad']['account_id'].unique()                 # Filter for users who use ipads & iPhones
iphone_users = df[df['device'] == 'iPhone']['account_id'].unique()
users_with_both = set(ipad_users) & set(iphone_users)

# Check if there are any users with both devices
if users_with_both:
    print("Yes, there are users who use both iPads and iPhones.")
    print("Number of users:", len(users_with_both))
    print("User IDs with both devices:", users_with_both)
else:
    print("No, there are no users who use both iPads and iPhones.")

print('\n')

revenue_by_country = df.groupby('last_login_country')['cash_amount'].sum()
top_countries = revenue_by_country.nlargest(5).reset_index()

# Bar plot
plt.figure(figsize=(10, 6))
plt.bar(top_countries['last_login_country'], top_countries['cash_amount'])
plt.title('Top 5 Countries by Revenue')
plt.xlabel('Country')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.show()

df['created_date'] = pd.to_datetime(df['created_date'])
first_week_revenue = df[df['created_date'] <= df['created_date'].min() + pd.DateOffset(weeks=1)]['cash_amount'].sum()
lifetime_revenue = df['cash_amount'].sum()
proportion_first_week_revenue = first_week_revenue / lifetime_revenue
print("Proportion of lifetime revenue generated in the player's first week:", proportion_first_week_revenue)

usa_users = df.loc[df['last_login_country'] == 'US', :].copy()
usa_users.loc[:, 'device_type'] = usa_users['device'].apply(lambda x: 'iPad' if 'iPad' in x else 'iPhone')

ipad_count = usa_users[usa_users['device_type'] == 'iPad'].shape[0]
iphone_count = usa_users[usa_users['device_type'] == 'iPhone'].shape[0]
print('\n')
# Bar plot
plt.figure(figsize=(8, 6))
devices = ['iPad', 'iPhone']
counts = [ipad_count, iphone_count]
colors = ['#FF6F00', '#1F77B4']

bars = plt.bar(devices, counts, color=colors)

# Add data labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

plt.title('Proportions of iPad and iPhone Users from the United States')
plt.xlabel('Device Type')
plt.ylabel('Count')
plt.show()

sns.pairplot(data=df, vars=['age_seconds', 'session_count', 'in_game_currency_amount', 'cash_amount'])
plt.suptitle('Pair Plot of Numerical Variables')
plt.show()

correlation_matrix = df.corr()                                 # Create the correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

print('\n')

# Select the independent variables
X = df[['age_seconds', 'session_count', 'cash_amount']]

# Add a constant term for the intercept
X = sm.add_constant(X)
y = df['in_game_currency_amount']

# Perform the linear regression
model = sm.OLS(y, X).fit()
print(model.summary())

# Select the features for clustering
X = df[['age_seconds', 'session_count']]

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Add cluster labels to the DataFrame
df['cluster_label'] = kmeans.labels_

# Plot the clusters
plt.scatter(df['age_seconds'], df['session_count'], c=df['cluster_label'], cmap='viridis')
plt.xlabel('Age (seconds)')
plt.ylabel('Session Count')
plt.title('K-means Clustering')
plt.show()



