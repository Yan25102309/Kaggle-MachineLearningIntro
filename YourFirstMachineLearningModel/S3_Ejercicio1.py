import pandas as pd
from sklearn.tree import DecisionTreeRegressor

file_path = "train.csv"
home_data = pd.read_csv(file_path)

print("Columnas del dataset:")
print(home_data.columns)

y = home_data.SalePrice

print("\n--- Verificación ---")
print("Primeras filas de la variable 'y':")
print(y.head())

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[feature_names]

print("\n--- Verificación de X ---")
print("Descripción estadística básica:")
print(X.describe())

print("\nPrimeras filas de X:")
print(X.head())

iowa_model = DecisionTreeRegressor(random_state=1)

iowa_model.fit(X, y)

print("¡Modelo entrenado con éxito!")
print("Instancia del modelo:", iowa_model)

predictions = iowa_model.predict(X)

print("Primeras 5 predicciones realizadas por el modelo:")
print(predictions[:5])

print("\nPrimeros 5 valores reales de y (SalePrice) para comparar:")
print(y.head().values)