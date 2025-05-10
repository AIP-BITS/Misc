#%%
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
df = pd.read_csv('Housing.csv')
df = df.dropna()  

weights = {
    'ISLAND': 5,
    'NEAR OCEAN': 4,
    'NEAR BAY': 3,
    '<1H OCEAN': 2,
    'INLAND': 1
}

df['ocean_proximity_encoded'] = df['ocean_proximity'].map(weights)
df.drop(columns=['ocean_proximity'], inplace=True) 

df['avg_rooms'] = df['total_rooms'] / df['households']
df['avg_bedrooms'] = df['total_bedrooms'] / df['households']

df.drop(columns=['total_rooms','total_bedrooms'], inplace=True) 

correlation_matrix = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
print(df.columns)

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
regressor = DecisionTreeRegressor(random_state=42,
                                  max_depth=10,
                                  max_features=None,
                                  min_samples_leaf=4,
                                  min_samples_split=10
                                  )

X_train_scaled = scaler.fit_transform(X_train)
regressor.fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(X_test)

y_pred = regressor.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
sns.barplot(y=X_train.columns, x=regressor.feature_importances_)

# %%
