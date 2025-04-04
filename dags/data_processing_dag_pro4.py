from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Google Sheets API imports
from google.oauth2 import service_account
from googleapiclient.discovery import build

# -------------------------------------------------
# Konfiguracja połączenia z Google Sheets
# -------------------------------------------------

# Ścieżka do pliku z poświadczeniami Google Service Account
GOOGLE_SHEETS_CREDENTIALS = '/opt/airflow/config/credentials.json'

# Zakresy uprawnień dla Google Sheets API
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Inicjalizacja poświadczeń z pliku JSON
credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_SHEETS_CREDENTIALS, scopes=SCOPES
)

# Budowanie serwisu do komunikacji z Google Sheets API
service = build('sheets', 'v4', credentials=credentials)

# -------------------------------------------------
# Definicje ID arkuszy Google Sheets oraz zakresów
# -------------------------------------------------

# ID arkusza z przetworzonymi danymi
PROCESSED_SPREADSHEET_ID = '1vaPyOWJxX5E1OWi28PB2vC_NGS_pGckIgtrmpKM-40E'
PROCESSED_RANGE_NAME = 'Arkusz1!A1:Z'

# ID arkuszy treningowego i testowego
TRAIN_SPREADSHEET_ID = '1kSu8UOd8ggldQ5Syt5xIDtIAT1-MTnbMWjrUHEyD_vI'
TEST_SPREADSHEET_ID = '1X1BdV5dEqj6b4WHEvST1xXELdupUZIeQsZ4Z6t98qC0'

# Zakresy w arkuszach treningowym i testowym
TRAIN_RANGE_NAME = 'Arkusz1!A1:Z'
TEST_RANGE_NAME = 'Arkusz1!A1:Z'

# -------------------------------------------------
# Funkcje pomocnicze do operacji na Google Sheets
# -------------------------------------------------

def clear_google_sheet(spreadsheet_id, range_name=TRAIN_RANGE_NAME):
    """
    Czyści zawartość określonego zakresu w danym arkuszu Google Sheets.

    :param spreadsheet_id: ID arkusza Google Sheets.
    :param range_name: Zakres do wyczyszczenia (domyślnie TRAIN_RANGE_NAME).
    """
    print(f"[clear_google_sheet] Czyszczenie arkusza {spreadsheet_id} w zakresie {range_name}.")
    sheet = service.spreadsheets()
    sheet.values().clear(spreadsheetId=spreadsheet_id, range=range_name).execute()

def update_google_sheet(df, spreadsheet_id, range_name=TRAIN_RANGE_NAME):
    """
    Aktualizuje dane w określonym zakresie arkusza Google Sheets na podstawie DataFrame.

    :param df: Pandas DataFrame z danymi do zapisania.
    :param spreadsheet_id: ID arkusza Google Sheets.
    :param range_name: Zakres do aktualizacji (domyślnie TRAIN_RANGE_NAME).
    """
    print(f"[update_google_sheet] Aktualizacja arkusza {spreadsheet_id} w zakresie {range_name}. "
          f"Rozmiar DataFrame: {df.shape}")
    clear_google_sheet(spreadsheet_id, range_name)  # Czyści wcześniej dane
    sheet = service.spreadsheets()
    body = {
        'values': [df.columns.tolist()] + df.values.tolist()  # Łączy nagłówki z danymi
    }
    sheet.values().update(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption='RAW',
        body=body
    ).execute()
    print("[update_google_sheet] Zakończono aktualizację arkusza.")

def read_data_from_sheets(spreadsheet_id, range_name):
    """
    Odczytuje dane z określonego zakresu arkusza Google Sheets i zwraca je jako DataFrame.

    :param spreadsheet_id: ID arkusza Google Sheets.
    :param range_name: Zakres do odczytu danych.
    :return: Pandas DataFrame z odczytanymi danymi.
    :raises ValueError: Jeśli arkusz jest pusty.
    """
    print(f"[read_data_from_sheets] Odczyt danych z arkusza {spreadsheet_id}, zakres: {range_name}")
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])

    if not values:
        raise ValueError(f"[read_data_from_sheets] Brak danych w arkuszu: {spreadsheet_id}")

    headers = values[0]  # Pierwszy wiersz jako nagłówki kolumn
    rows = values[1:]    # Reszta wierszy jako dane
    df = pd.DataFrame(rows, columns=headers)
    print(f"[read_data_from_sheets] Odczytano DataFrame rozmiaru: {df.shape}")
    return df

# -------------------------------------------------
# Funkcje zadań DAG
# -------------------------------------------------

def fetch_processed_data(**kwargs):
    """
    Pobiera przetworzone dane z Google Sheets, konwertuje je do odpowiedniego formatu,
    oraz zapisuje je do XCom dla dalszego przetwarzania.

    :param kwargs: Argumenty przekazywane przez Airflow.
    """
    print("[fetch_processed_data] Start.")
    try:
        # Odczytuje dane z arkusza przetworzonych danych
        df = read_data_from_sheets(PROCESSED_SPREADSHEET_ID, PROCESSED_RANGE_NAME)
    except Exception as e:
        print("[fetch_processed_data] Błąd podczas odczytu danych:", str(e))
        raise

    # Zamiana przecinków na kropki w całym DataFrame (np. dla wartości liczbowych zapisanych jako string)
    df = df.replace(',', '.', regex=True)

    # Lista kolumn, które powinny być numeryczne
    numeric_cols = [
        "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
        "Proximity_to_Industrial_Areas", "Population_Density"
    ]
    for col in numeric_cols:
        if col in df.columns:
            # Konwersja kolumny do typu numerycznego, wartości niekonwertowalne zostają NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Przetwarzanie kolumny 'Air Quality'
    if 'Air Quality' in df.columns:
        # Ujednolicenie tekstu do małych liter
        df['Air Quality'] = df['Air Quality'].str.lower()
        # Zamiana kategorii jakości powietrza na wartości numeryczne
        df['Air Quality'] = df['Air Quality'].replace({
            'poor': 0,
            'hazardous': 0,
            'good': 1,
            'moderate': 1
        })

    print(f"[fetch_processed_data] Zapisuję DataFrame do XCom. Rozmiar: {df.shape}")
    # Zapisuje przetworzone dane do XCom, aby były dostępne dla kolejnych zadań
    kwargs['ti'].xcom_push(key='processed_data', value=df.to_dict(orient='list'))

def split_data_and_save(**kwargs):
    """
    Dzieli dane na zbiory treningowy i testowy, zapisuje je do Google Sheets oraz do XCom.

    :param kwargs: Argumenty przekazywane przez Airflow.
    """
    ti = kwargs['ti']
    print("[split_data_and_save] Pobieram 'processed_data' z XCom.")
    # Pobiera przetworzone dane z XCom
    data_dict = ti.xcom_pull(key='processed_data', task_ids='fetch_processed_data')
    if not data_dict:
        raise ValueError("[split_data_and_save] Brak danych w XCom (processed_data).")

    # Konwertuje słownik na DataFrame
    df = pd.DataFrame(data_dict)
    print(f"[split_data_and_save] Otrzymano DataFrame: {df.shape}")
    if df.empty:
        raise ValueError("[split_data_and_save] DataFrame jest pusty!")

    # Sprawdza, czy kolumna 'Target' istnieje, jeśli nie, dodaje losowe wartości
    if 'Target' not in df.columns:
        print("[split_data_and_save] Kolumna 'Target' nie istnieje. Dodaję sztuczną kolumnę.")
        df['Target'] = np.random.randint(2, size=len(df))  # Generuje wartości 0 lub 1

    # Sprawdza minimalną liczbę wierszy do podziału
    if len(df) < 2:
        raise ValueError("[split_data_and_save] Za mało wierszy, by dzielić na train/test.")

    # Dzieli dane na zbiory treningowy (70%) i testowy (30%)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    print(f"[split_data_and_save] Zbiór treningowy: {train_df.shape}, testowy: {test_df.shape}")

    # Aktualizuje odpowiednie arkusze Google Sheets z danymi treningowymi i testowymi
    update_google_sheet(train_df, TRAIN_SPREADSHEET_ID, TRAIN_RANGE_NAME)
    update_google_sheet(test_df, TEST_SPREADSHEET_ID, TEST_RANGE_NAME)

    print("[split_data_and_save] Zapisuję zbiory do XCom.")
    # Zapisuje zbiory treningowy i testowy do XCom
    ti.xcom_push(key='train_data', value=train_df.to_dict(orient='list'))
    ti.xcom_push(key='test_data', value=test_df.to_dict(orient='list'))

# -------------------------------------------------
# Definicja domyślnych argumentów DAG
# -------------------------------------------------

default_args = {
    'owner': 'airflow',                    # Właściciel zadania
    'depends_on_past': False,              # Zadanie nie zależy od poprzednich uruchomień
    'email_on_failure': False,             # Nie wysyła e-maili w przypadku błędu
    'retries': 0                            # Liczba prób ponowienia zadania w razie niepowodzenia
}

# -------------------------------------------------
# Definicja DAG
# -------------------------------------------------

with DAG(
    'data_processing_dag',  # Unikalna nazwa DAG
    default_args=default_args,  # Używa wcześniej zdefiniowanych argumentów
    description='DAG do pobierania i przetwarzania danych z Google Sheets',  # Opis DAG
    schedule_interval=None,  # Brak harmonogramu – DAG jest uruchamiany ręcznie lub zależny od innego DAG
    start_date=datetime(2025, 1, 1),  # Data początkowa (ustawiona w przyszłości)
    catchup=False  # Nie uruchamia zaległych uruchomień DAG
) as dag:
    """
    DAG (Directed Acyclic Graph) do pobierania, przetwarzania i dzielenia danych na zbiory treningowy i testowy.
    Składa się z dwóch głównych zadań:
    1. Pobranie i przetworzenie danych.
    2. Podział danych na zbiory treningowy i testowy oraz zapisanie ich.
    """

    # -------------------------------------------------
    # Definicje zadań DAG
    # -------------------------------------------------

    # Zadanie 1: Pobranie i przetworzenie danych
    fetch_processed_data_task = PythonOperator(
        task_id='fetch_processed_data',  # Unikalna nazwa zadania
        python_callable=fetch_processed_data  # Funkcja do wykonania
    )

    # Zadanie 2: Podział danych i zapisanie zbiorów
    split_data_and_save_task = PythonOperator(
        task_id='split_data_and_save',
        python_callable=split_data_and_save
    )

    # -------------------------------------------------
    # Definicja kolejności wykonywania zadań
    # -------------------------------------------------

    # Ustala kolejność: fetch_processed_data_task → split_data_and_save_task
    fetch_processed_data_task >> split_data_and_save_task
