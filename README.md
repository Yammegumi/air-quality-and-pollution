# 🌍 Air Quality and Pollution Assessment

**Author:** Mateusz Łaski  
**Student ID:** s26102
**Subject:** ASI (Architecture of Solutions and Methodology of AI Implementation)

---

## 🧠 Project Overview

### 🔎 Topic
Assessment of air quality based on selected environmental and demographic factors.

### ❗ Problem
The project aims to develop a machine learning model that can predict air quality levels across various regions using publicly available data. The goal is to identify high-risk pollution areas and support decision-making for environmental improvement.

---

## 📊 Dataset

- **Source:** Public dataset from Kaggle: *Air Quality and Pollution Assessment*  
- **Size:** 5000 records  
- **Features:** 9 numerical attributes + 1 target variable (air quality level)

### 📈 Attributes

- **Temperature (°C):** Average regional temperature
- **Humidity (%):** Relative humidity
- **PM2.5 Concentration (µg/m³):** Fine particulate matter
- **PM10 Concentration (µg/m³):** Coarser particulate matter
- **NO2 Concentration (ppb):** Nitrogen dioxide level
- **SO2 Concentration (ppb):** Sulfur dioxide level
- **CO Concentration (ppm):** Carbon monoxide level
- **Proximity to Industrial Areas (km):** Distance to the nearest industrial zone
- **Population Density (people/km²):** People per square kilometer

### 🎯 Target Variable – Air Quality Levels:
- **Good:** Low pollution
- **Moderate:** Acceptable pollution
- **Poor:** Noticeable pollution, risky for sensitive groups
- **Hazardous:** Very high pollution, a serious health risk

---

## 🎯 Project Goals

1. **Build a predictive model:**
   - Classify air quality levels as: Good, Moderate, Poor, Hazardous

2. **Analyze feature influence:**
   - Identify key attributes impacting air quality

3. **Develop actionable insights:**
   - Recommend environmental improvement strategies for specific regions

---

## 📆 Timeline & Workflow

- Short iterations (1–2 days per goal)
- Workflow:
  1. Data acquisition
  2. Preprocessing
  3. Exploratory Data Analysis (EDA)
  4. Model training
  5. Evaluation & optimization
  6. Results & recommendations

---

## 🗂️ Repository Structure

- `README.md` – Project documentation  
- `data/` – Raw dataset files  
- `notebooks/` – Jupyter Notebooks  
- `models/` – Trained ML models  
- `results/` – Analysis outputs and visualizations  
- `.idea/` – IDE config files (e.g. PyCharm)  
- `dags/` – Apache Airflow DAG definitions  
- `raports/` – Project reports and findings  
- `screenshots/` – Screenshots showing DAG functionality (as required by PRO4)  
- `scripts/` – Helper/automation scripts  
- `.gitignore` – Git ignore rules  
- `LICENSE` – License info  
- `docker-compose.yaml` – Docker Compose config for Airflow  
- `requirements.txt` – Python dependencies

---

## 🧪 Project Backlog

- Acquire data and set up the environment
- Data preprocessing and EDA
- Build and train the prediction model
- Test and tune the model
- Present the results

---

## 🐳 How to Run Apache Airflow with Docker

**Make sure Docker and Docker Compose are installed.**

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/s26102-Air-Quality-And-Pollution.git
cd s26102-Air-Quality-And-Pollution
```

### 2. Install Docker & Docker Compose

If not installed, download them from the official [Docker](https://www.docker.com) site.

### 3. Start Airflow

In the root project directory, run:

```bash
docker-compose up -d
```

This will pull the images, create containers, and launch Airflow in the background.

### 4. Check container status

```bash
docker-compose ps
```

### 5. Open Airflow web interface

Visit: [http://localhost:8080](http://localhost:8080)

---

