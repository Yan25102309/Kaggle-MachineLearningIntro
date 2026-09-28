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

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

print("Resultados de MAE por tamaño:")
for size, mae in scores.items():
    print(f"Max leaf nodes: {size:3d} | MAE: {mae:,.2f}")

print(f"\nEl mejor tamaño de árbol es: {best_tree_size}")

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

final_model.fit(X, y)

print(f"Mejor tamaño de árbol seleccionado: {best_tree_size}")
print("¡Modelo final reentrenado con éxito utilizando todos los datos!")