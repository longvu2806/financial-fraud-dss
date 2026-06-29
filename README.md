# Financial Fraud Detection & Transactional Behavioral Modelling

> End-to-end Machine Learning and Data Engineering pipeline for fraud detection on large-scale transactional data (13.3M+ records).

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-336791) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project implements a full ML pipeline to detect fraudulent financial transactions. It covers:

- **ETL pipeline** — extract, clean, and load 13.3M+ transaction records into PostgreSQL
- **Behavioral modelling** — feature engineering based on transactional patterns
- **Fraud classification** — supervised ML model (XGBoost) trained on labeled data
- **Interactive dashboard** — Streamlit app for real-time monitoring and analysis

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Database | PostgreSQL 14+ |
| ML Framework | XGBoost, scikit-learn |
| Dashboard | Streamlit |
| Data Processing | pandas, NumPy |

---

## Project Structure

```
.
├── data/                   # Raw and processed data
├── notebooks/              # Exploratory analysis
├── src/
│   ├── etl/
│   │   └── run_pipeline.py # Main ETL entry point
│   └── app/
│       └── main_app.py     # Streamlit dashboard
├── .env                    # Environment variables (not committed)
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.9+
- PostgreSQL 14+
- Git

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/longvu2806/financial-fraud-dss.git
cd financial-fraud-dss
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the database

Open your PostgreSQL client and create the target database:

```sql
CREATE DATABASE caixabank_db;
```

Create a `.env` file in the root directory (same level as `/data`, `/src`, `/notebooks`) with the following content:

```env
DB_USER=postgres
DB_PASSWORD=your_actual_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=caixabank_db
```

> **Note:** Never commit `.env` to version control. Make sure it is listed in `.gitignore`.

---

## Running the Project

### Run the ETL pipeline

Execute from the root directory to trigger extraction, cleaning, and loading:

```bash
python src/etl/run_pipeline.py
```

### Launch the Streamlit dashboard

```bash
python -m streamlit run src/app/main_app.py
```

The dashboard connects to the PostgreSQL database and provides interactive fraud monitoring and analysis.

---

