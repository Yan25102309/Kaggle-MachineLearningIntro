import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train_data_path = '../train.csv'
home_data = pd.read_csv(train_data_path)

y = home_data.SalePrice
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_names]

rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X, y)
print("¡Modelo entrenado exitosamente con el 100% de los datos!")

test_data_path = '../test.csv'
test_data = pd.read_csv(test_data_path)

test_X = test_data[feature_names]

test_preds = rf_model_on_full_data.predict(test_X)

print(f"\nNúmero de muestras en test: {len(test_X)}")
print(f"Número de predicciones generadas: {len(test_preds)}")

output = pd.DataFrame({
    'Id': test_data.Id,
    'SalePrice': test_preds
})

output.to_csv('submission.csv', index=False)
print("\n¡Archivo 'submission.csv' generado correctamente en tu directorio actual!")