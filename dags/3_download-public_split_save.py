from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import kagglehub
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

# -------------------------------------------------------------------------
# Konfiguracja Google Sheets
# -------------------------------------------------------------------------
GOOGLE_SHEETS_CREDENTIALS = '/opt/airflow/config/credentials.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Dwa osobne arkusze docelowe: jeden na zbiór modelowy (0)
# i jeden na zbiór douczeniowy (1).
SPREADSHEET_ID = [
    '1L3fis1zcZ9eARUmgI_3-p9GnoDJP5zHGDqhmy38Q4Yw',  # Zbiór modelowy
    '1uvP0Zn8Q3ihDPO3EN9-nlQrChnSr6qZ5QJkayQhWtNk'   # Zbiór douczeniowy
]

RANGE_NAME = 'Arkusz1!A1:Z'

credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_SHEETS_CREDENTIALS, scopes=SCOPES)
service = build('sheets', 'v4', credentials=credentials)

# -------------------------------------------------------------------------
# Funkcja: Pobieranie danych
# -------------------------------------------------------------------------
# Pobiera dane z Kaggle (lub lokalnie), ładuje do DataFrame
# i umieszcza w XCom. To spełnia wymóg operatora pobrania danych.
def fetch_data(**kwargs):
    # Pobranie przykładowego datasetu z Kaggle
    path = kagglehub.dataset_download("mujtabamatin/air-quality-and-pollution-assessment")
    data_path = os.path.join(path, "updated_pollution_dataset.csv")
    data = pd.read_csv(data_path)
    # Zapis danych do XCom w postaci słownika
    kwargs['ti'].xcom_push(key='data', value=data.to_dict(orient='list'))

# -------------------------------------------------------------------------
# Funkcja: Podział danych
# -------------------------------------------------------------------------
# Dzieli dane na dwa zbiory: 70% modelowy i 30% douczeniowy.
# Korzysta z train_test_split i ustawia random_state w celu reprodukowalności.
def split_data(**kwargs):
    ti = kwargs['ti']
    data_dict = ti.xcom_pull(key='data', task_ids='fetch_data')
    data = pd.DataFrame(data_dict)
    # Podział na zbiór treningowy (70%) i testowy (30%)
    train, test = train_test_split(data, test_size=0.3, random_state=42)
    # Zapis podzielonych danych do XCom
    ti.xcom_push(key='train_data', value=train.to_dict(orient='list'))
    ti.xcom_push(key='test_data', value=test.to_dict(orient='list'))

# -------------------------------------------------------------------------
# Funkcje pomocnicze do zapisu w Google Sheets
# -------------------------------------------------------------------------
def clear_google_sheet(spreadsheet_id):
    # Czyszczenie poprzedniej zawartości arkusza
    sheet = service.spreadsheets()
    sheet.values().clear(spreadsheetId=spreadsheet_id, range=RANGE_NAME).execute()

def update_google_sheet(df_to_update, spreadsheet_id):
    # Aktualizacja arkusza o nowe dane
    clear_google_sheet(spreadsheet_id)
    sheet = service.spreadsheets()
    body = {
        'values': [df_to_update.columns.tolist()] + df_to_update.values.tolist()
    }
    sheet.values().update(
        spreadsheetId=spreadsheet_id, range=RANGE_NAME,
        valueInputOption='RAW', body=body).execute()

# -------------------------------------------------------------------------
# Funkcja: Zapisanie danych do Google Sheets
# -------------------------------------------------------------------------
# Pobiera podzielone dane z XCom i zapisuje je do dwóch osobnych arkuszy:
# - Pierwszy arkusz: zbiór modelowy
# - Drugi arkusz: zbiór douczeniowy
def save_data_to_sheets(**kwargs):
    ti = kwargs['ti']
    train_data = pd.DataFrame(ti.xcom_pull(key='train_data', task_ids='split_data'))
    test_data = pd.DataFrame(ti.xcom_pull(key='test_data', task_ids='split_data'))
    
    # Zapis zbioru modelowego
    update_google_sheet(train_data, SPREADSHEET_ID[0])
    # Zapis zbioru douczeniowego
    update_google_sheet(test_data, SPREADSHEET_ID[1])

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
# 1. Pobranie danych
# 2. Podział danych
# 3. Zapis podzielonych danych do Google Sheets
with DAG(
    'data_download_and_split_dag',
    default_args=default_args,
    description='DAG do pobierania, podziału i zapisu danych',
    schedule_interval=None,
    start_date=datetime(2023, 11, 1),
    catchup=False,
) as dag:

    fetch_data_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_data,
    )

    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
    )

    save_data_task = PythonOperator(
        task_id='save_data',
        python_callable=save_data_to_sheets,
    )

    # Kolejność wykonywania zadań:
    # Pobranie danych -> Podział danych -> Zapis do arkuszy
    fetch_data_task >> split_data_task >> save_data_task
