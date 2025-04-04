from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# -------------------------------------------------------------------------
# Konfiguracja Google Sheets
# -------------------------------------------------------------------------
GOOGLE_SHEETS_CREDENTIALS = '/opt/airflow/config/credentials.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Arkusz wejściowy - pobieramy dane do przetwarzania
SOURCE_SPREADSHEET_ID = '1L3fis1zcZ9eARUmgI_3-p9GnoDJP5zHGDqhmy38Q4Yw'
SOURCE_RANGE_NAME = 'Arkusz1!A1:Z'

# Arkusz wyjściowy - zapis przetworzonych danych
TARGET_SPREADSHEET_ID = '1vaPyOWJxX5E1OWi28PB2vC_NGS_pGckIgtrmpKM-40E'
TARGET_RANGE_NAME = 'Arkusz1!A1:Z'

credentials = Credentials.from_service_account_file(
    GOOGLE_SHEETS_CREDENTIALS, scopes=SCOPES)
service = build('sheets', 'v4', credentials=credentials)

# -------------------------------------------------------------------------
# Funkcja: Pobranie danych z Google Sheets
# -------------------------------------------------------------------------
# Pobiera dane z arkusza źródłowego, konwertuje do DataFrame i przekazuje do XCom.
# Spełnia to wymóg operatora do pobrania danych w drugim DAG-u.
def fetch_data_from_sheets(**kwargs):
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SOURCE_SPREADSHEET_ID, range=SOURCE_RANGE_NAME).execute()
    values = result.get('values', [])

    if not values:
        raise ValueError("Brak danych w arkuszu Google Sheets!")

    # Pierwszy wiersz - nagłówki
    headers = values[0]
    # Pozostałe wiersze - dane
    rows = values[1:]
    data = pd.DataFrame(rows, columns=headers)

    if data.empty:
        raise ValueError("Dane są puste po pobraniu z arkusza Google Sheets!")

    numeric_cols = [
        "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
        "Proximity_to_Industrial_Areas", "Population_Density"
    ]

    # Zamiana przecinków na kropki i konwersja do typów numerycznych
    data[numeric_cols] = data[numeric_cols].replace(",", ".", regex=True).apply(pd.to_numeric, errors='coerce')

    # Logowanie informacji o danych do debugowania
    print("Dane po wczytaniu z Google Sheets:")
    print(data.info())
    print(data.head())

    kwargs['ti'].xcom_push(key='data', value=data.to_dict(orient='list'))

# -------------------------------------------------------------------------
# Funkcja: Czyszczenie danych
# -------------------------------------------------------------------------
# Usuwanie duplikatów i uzupełnianie braków średnimi wartościami.
# Pozwala to na uzyskanie pełnego, spójnego zestawu danych do dalszej analizy.
def clean_data(**kwargs):
    ti = kwargs['ti']
    data_dict = ti.xcom_pull(key='data', task_ids='fetch_data')

    if data_dict is None or not data_dict:
        raise ValueError("Brak danych w XCom! Sprawdź task `fetch_data`.")

    data = pd.DataFrame(data_dict)

    # Przed czyszczeniem - logowanie stanu danych
    print("Przed czyszczeniem danych:")
    print(data.info())
    print(data.head())

    # Sprawdzenie brakujących wartości
    print("Brakujące wartości przed czyszczeniem:")
    print(data.isnull().sum())

    # Czyszczenie:
    # 1. Usunięcie duplikatów
    # 2. Wypełnienie braków wartościami średnimi dla kolumn numerycznych
    try:
        data.drop_duplicates(inplace=True)
        data.fillna(data.mean(numeric_only=True), inplace=True)
    except Exception as e:
        print("Błąd podczas czyszczenia danych:", e)
        raise e

    # Po czyszczeniu - logowanie stanu danych
    print("Po czyszczeniu danych:")
    print(data.info())
    print(data.head())

    ti.xcom_push(key='cleaned_data', value=data.to_dict(orient='list'))

# -------------------------------------------------------------------------
# Funkcja: Standaryzacja i normalizacja danych
# -------------------------------------------------------------------------
# Standaryzujemy dane (średnia=0, odchylenie=1) a następnie normalizujemy 
# (przeskalowanie do zakresu [0,1]) cechy numeryczne.
# Zapewnia to, że cechy mają porównywalne skale i poprawia stabilność modeli.
def scale_and_normalize_data(**kwargs):
    ti = kwargs['ti']
    data_dict = ti.xcom_pull(key='cleaned_data', task_ids='clean_data')
    data = pd.DataFrame(data_dict)

    numeric_cols = [
        "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
        "Proximity_to_Industrial_Areas", "Population_Density"
    ]

    scaler = StandardScaler()
    normalizer = MinMaxScaler()

    # Najpierw standaryzacja:
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    # Następnie normalizacja:
    data[numeric_cols] = normalizer.fit_transform(data[numeric_cols])

    ti.xcom_push(key='processed_data', value=data.to_dict(orient='list'))

# -------------------------------------------------------------------------
# Funkcja: Zapisanie przetworzonych danych do Google Sheets
# -------------------------------------------------------------------------
# Po zakończeniu czyszczenia, standaryzacji i normalizacji zapisujemy
# przetworzone dane do docelowego arkusza. Pozwala to na dalszą analizę
# lub wykorzystanie w modelach.
def save_processed_data_to_sheets(**kwargs):
    ti = kwargs['ti']
    processed_data_dict = ti.xcom_pull(key='processed_data', task_ids='scale_and_normalize_data')
    processed_data = pd.DataFrame(processed_data_dict)

    # Czyszczenie i aktualizacja w Google Sheets
    sheet = service.spreadsheets()
    body = {
        'values': [processed_data.columns.tolist()] + processed_data.values.tolist()
    }

    sheet.values().clear(spreadsheetId=TARGET_SPREADSHEET_ID, range=TARGET_RANGE_NAME).execute()
    sheet.values().update(
        spreadsheetId=TARGET_SPREADSHEET_ID, range=TARGET_RANGE_NAME,
        valueInputOption='RAW', body=body
    ).execute()

    ti.xcom_push(key='processed_data', value=processed_data.to_dict(orient='list'))

# -------------------------------------------------------------------------
# Definicja DAG
# -------------------------------------------------------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Ten DAG obsługuje proces:
# 1. Pobranie danych z arkusza Google Sheets (zbiór modelowy z poprzedniego DAG-a)
# 2. Czyszczenie danych: usunięcie duplikatów i braków
# 3. Skalowanie i normalizacja cech numerycznych
# 4. Zapis przetworzonych danych z powrotem do Google Sheets
with DAG(
    'data_processing_dag',
    default_args=default_args,
    description='DAG do przetwarzania danych',
    schedule_interval=None,
    start_date=datetime(2023, 12, 1),
    catchup=False,
) a
