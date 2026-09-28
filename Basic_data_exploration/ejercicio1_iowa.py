import pandas as pd
import datetime

iowa_file_path = '../train.csv'

home_data = pd.read_csv(iowa_file_path)

print("--- Datos de Iowa cargados con éxito ---")
print(home_data.describe())


# Cargar los datos (asumiendo que ya leíste el archivo)
# home_data = pd.read_csv('train.csv')

# Print summary statistics in next line
print(home_data.describe())

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = round(home_data['LotArea'].mean())

# As of today, how old is the newest home (current year - the date in which it was built)
current_year = datetime.datetime.now().year
newest_home_age = current_year - home_data['YearBuilt'].max()

# Comprobación de resultados en consola (reemplazo de step_2.check())
print(f"Average lot size: {avg_lot_size}")
print(f"Newest home age: {newest_home_age}")

