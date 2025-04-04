import pandas as pd

# Wczytanie danych
file_path = 'C:\\Users\\shind\\Documents\\GitHub\\s26102-Air-Quality-And-Pollution\\data\\pollution_dataset.csv'
data = pd.read_csv(file_path)

# Tworzenie podsumowania tekstowego z kodowaniem UTF-8
with open("../results/eda_text_results.txt", "w", encoding="utf-8") as file:
    # Rozkład zmiennych numerycznych (opis statystyczny)
    file.write("=== Rozkład zmiennych numerycznych ===\n")
    file.write(data.describe().to_string())

    # Rozkład zmiennej kategorycznej
    file.write("\n\n=== Rozkład zmiennej kategorycznej ===\n")
    file.write(data['Air Quality'].value_counts().to_string())

    # Brakujące wartości
    file.write("\n\n=== Brakujące wartości ===\n")
    file.write(data.isnull().sum().to_string())

    # Macierz korelacji (tylko numeryczne kolumny)
    file.write("\n\n=== Macierz korelacji ===\n")
    numerical_data = data.select_dtypes(include=["number"])
    file.write(numerical_data.corr().to_string())
