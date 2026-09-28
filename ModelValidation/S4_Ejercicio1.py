import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

file_path = '../train.csv'
home_data = pd.read_csv(file_path)

y = home_data.SalePrice
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

print("--- Verificación de la división de datos ---")
print(f"Número total de muestras: {len(X)}")
print(f"Muestras para entrenamiento (train_X): {len(train_X)}")
print(f"Muestras para validación (val_X): {len(val_X)}")

iowa_model = DecisionTreeRegressor(random_state=1)

iowa_model.fit(train_X, train_y)

print("¡Modelo entrenado exitosamente con el conjunto de entrenamiento!")

val_predictions = iowa_model.predict(val_X)

print("Primeras 5 predicciones en los datos de validación:")
print(val_predictions[:5])

print("\nPrimeros 5 valores reales de validación (val_y):")
print(val_y.head().values)

val_mae = mean_absolute_error(val_y, val_predictions)

print(f"Validation MAE: {val_mae:,.2f}")