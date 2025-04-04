from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -------------------------------------------------
# Ścieżki do zapisu modelu i raportu
# -------------------------------------------------

MODEL_PATH = '/opt/airflow/models/gaussian_nb_model.joblib'
REPORT_PATH = '/opt/airflow/models/evaluation_report.txt'  # Zapis raportu w katalogu 'models/'

# -------------------------------------------------
# Funkcje zadań DAG
# -------------------------------------------------

def train_and_evaluate_model(**kwargs):
    """
    Trenuje model Gaussian Naive Bayes na danych treningowych, ocenia go na zbiorze testowym,
    zapisuje model oraz raport z oceną.

    :param kwargs: Argumenty przekazywane przez Airflow.
    :raises RuntimeError: Jeśli wystąpi błąd podczas zapisu modelu lub raportu.
    """
    print("[train_and_evaluate_model] Start.")
    ti = kwargs['ti']
    # Pobiera zbiory treningowy i testowy z XCom (z DAG 'data_processing_dag')
    train_dict = ti.xcom_pull(key='train_data', task_ids='data_processing_dag_pro4.split_data_and_save')
    test_dict = ti.xcom_pull(key='test_data', task_ids='data_processing_dag_pro4.split_data_and_save')

    if not train_dict or not test_dict:
        raise ValueError("[train_and_evaluate_model] Brak train_data lub test_data w XCom!")

    # Konwertuje słowniki na DataFrame
    train_df = pd.DataFrame(train_dict)
    test_df = pd.DataFrame(test_dict)
    print(f"[train_and_evaluate_model] train_df={train_df.shape}, test_df={test_df.shape}")

    # Sprawdza obecność kolumny 'Target'
    if 'Target' not in train_df.columns or 'Target' not in test_df.columns:
        raise ValueError("[train_and_evaluate_model] Kolumna 'Target' nie istnieje w zbiorze train/test!")
    if train_df.empty or test_df.empty:
        raise ValueError("[train_and_evaluate_model] Zbiory treningowy/testowy są puste!")

    # Przygotowuje dane wejściowe (X) i etykiety (y) dla modelu
    X_train = train_df.drop(columns=['Target']).values
    y_train = train_df['Target'].values
    X_test = test_df.drop(columns=['Target']).values
    y_test = test_df['Target'].values

    print("[train_and_evaluate_model] Pierwszy wiersz X_train:", X_train[0] if len(X_train) > 0 else "BRAK")

    print("[train_and_evaluate_model] Trenuję model GaussianNB.")

    # Inicjalizacja i trenowanie modelu Gaussian Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)

    print("[train_and_evaluate_model] Predykcja na zbiorze testowym.")
    # Dokonuje predykcji na zbiorze testowym
    y_pred = model.predict(X_test)

    # Oblicza metryki oceny modelu
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print("[train_and_evaluate_model] Metryki:")
    print(f" - Accuracy:  {acc:.4f}")
    print(f" - Precision: {prec:.4f}")
    print(f" - Recall:    {rec:.4f}")

    # Zapisuje wytrenowany model do pliku
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Tworzy katalog, jeśli nie istnieje
        joblib.dump(model, MODEL_PATH)  # Serializuje i zapisuje model
        print(f"[train_and_evaluate_model] Zapisano model do: {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"[train_and_evaluate_model] Błąd zapisu modelu: {str(e)}")

    # Zapisuje raport z oceną modelu do pliku
    try:
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)  # Tworzy katalog, jeśli nie istnieje
        with open(REPORT_PATH, 'w') as f:
            f.write("=== Model Evaluation Report ===\n")
            f.write("Model: GaussianNB\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")

        print(f"[train_and_evaluate_model] Zapisano raport do: {REPORT_PATH}")
    except Exception as e:
        raise RuntimeError(f"[train_and_evaluate_model] Błąd zapisu raportu: {str(e)}")

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
    'model_training_dag',  # Unikalna nazwa DAG
    default_args=default_args,  # Używa wcześniej zdefiniowanych argumentów
    description='DAG do trenowania modelu GaussianNB na danych z Google Sheets',  # Opis DAG
    schedule_interval=None,  # Brak harmonogramu – DAG jest uruchamiany ręcznie lub zależny od innego DAG
    start_date=datetime(2025, 1, 1),  # Data początkowa (ustawiona w przyszłości)
    catchup=False  # Nie uruchamia zaległych uruchomień DAG
) as dag:
    """
    DAG (Directed Acyclic Graph) do trenowania modelu Gaussian Naive Bayes na danych pobranych z Google Sheets.
    Składa się z jednego głównego zadania:
    1. Trenowanie modelu, ocena jego wydajności oraz zapisanie wyników.
    """

    # -------------------------------------------------
    # Definicje zadań DAG
    # -------------------------------------------------

    # Zadanie: Trenowanie modelu i ocena
    train_and_evaluate_model_task = PythonOperator(
        task_id='train_and_evaluate_model',
        python_callable=train_and_evaluate_model
    )

    # -------------------------------------------------
    # Definicja kolejności wykonywania zadań
    # -------------------------------------------------

    # Ponieważ jest tylko jedno zadanie, nie ma potrzeby definiowania kolejności
    train_and_evaluate_model_task
