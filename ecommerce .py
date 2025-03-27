import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('ecommerce_data.csv')




# Display basic information and first few rows
print("Dataset Overview:")
print(data.info())
print(data.head())

# Data Cleaning: Handle missing values
data.dropna(inplace=True)

# Exploratory Data Analysis
# Total Sales by Product Category
category_sales = data.groupby('Category')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title('Total Sales by Category')
plt.xticks(rotation=45)
plt.show()

# Customer Satisfaction Analysis
satisfaction_counts = data['Customer_Satisfaction'].value_counts()
plt.pie(satisfaction_counts, labels=satisfaction_counts.index, autopct='%1.1f%%', startangle=140)


plt.title('Customer Satisfaction Distribution')
plt.show()

# Feature Selection for Sales Prediction
features = ['Quantity', 'Discount', 'Shipping_Time']
target = 'Sales'

# Split Data into Training and Test Sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Prediction
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

# Delivery Performance Analysis
delivery_performance = data.groupby('Delivery_Status')['Order_ID'].count()
delivery_performance.plot(kind='bar', figsize=(8, 5), color='skyblue')
plt.title('Delivery Performance Overview')
plt.ylabel('Number of Orders')
plt.show()

print("Ecommerce data analysis completed successfully!")
