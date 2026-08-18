# 🚗 Vehicle Insurance Prediction | End-to-End MLOps Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Azure](https://img.shields.io/badge/Azure-Web_App-0078D4)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-2088FF)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2)
![License](https://img.shields.io/badge/License-MIT-success)

![image alt](https://github.com/masudrr72/Vehicle-Insurance-Domain-MLops/blob/main/vehicle_insurance_mlops_banner.png)

🎥 Project Demo

Watch the full project walkthrough on YouTube:
[https://youtube.com/...](https://youtu.be/tFf4Try6CCM?si=ifyW1UZHHDNbH71l)

An industry-standard **End-to-End Machine Learning MLOps Project** that predicts whether a customer is interested in purchasing vehicle insurance. This project demonstrates a complete production ML workflow from data ingestion to cloud deployment using **FastAPI, Docker, Azure, GitHub Actions, and MLflow**.

---

# 🌐 Live Demo

**🚀 Web Application:** https://vehicle-insurance-api-a6d3hjazeqf9aqha.southeastasia-01.azurewebsites.net/

**📄 Swagger API:** https://vehicle-insurance-api-a6d3hjazeqf9aqha.southeastasia-01.azurewebsites.net/docs

---

# ✨ Project Highlights

- 🚀 End-to-End Machine Learning Pipeline
- 🤖 XGBoost Classification Model
- 📊 Automated Data Validation & Transformation
- 🧪 MLflow Experiment Tracking (params, metrics, model comparison)
- ☁️ Azure Blob Storage Model Registry
- ⚡ FastAPI REST API
- 🐳 Docker Containerization
- 🔄 GitHub Actions CI/CD
- 🌍 Azure Web App Deployment
- 📝 Logging & Exception Handling
- 🏗️ Production-Ready Modular Architecture

---

# 🏗️ System Architecture

```text
MongoDB Atlas
      │
      ▼
Data Ingestion
      │
      ▼
Data Validation
      │
      ▼
Data Transformation
      │
      ▼
Model Training ─────────► MLflow Tracking
      │                   (params, metrics,
      ▼                    model artifact)
Model Evaluation ───────► MLflow Tracking
      │                   (candidate vs. production
      ▼                    ROC-AUC, accept/reject)
Azure Blob Storage
(Model Registry)
      │
      ▼
Prediction Pipeline
      │
      ▼
FastAPI
      │
      ▼
Docker
      │
      ▼
Azure Web App
```

> MLflow tracks every training/evaluation run for experiment visibility and reproducibility. Azure Blob Storage remains the single source of truth for the deployed production model — MLflow observes the pipeline, it doesn't decide what gets deployed.

---

# 📊 Dataset

| Property | Value |
|-----------|-------|
| Total Records | 381,109 |
| Features | 11 |
| Target Variable | Response |
| Problem Type | Binary Classification |
| Primary Metric | ROC-AUC |

---

# 🔍 Key Insights

- Customers without previous insurance are much more likely to purchase vehicle insurance.
- Older vehicles show higher purchase probability.
- Previously damaged vehicles have significantly higher purchase interest.
- Gender has minimal influence on customer response.
- Tree-based models handled premium outliers effectively.

---

# ⚙️ Machine Learning Pipeline

- Data Ingestion
- Data Validation
- Data Transformation
- Feature Engineering
- Model Training (tracked via MLflow)
- Model Evaluation (tracked via MLflow)
- Model Registry
- Prediction Pipeline

---

# 🤖 Model Information

### Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- KNN
- XGBoost
- CatBoost

### Final Model

**XGBoost**

**Evaluation Metrics**

- ROC-AUC (~0.858, primary metric)
- Accuracy
- Precision
- Recall
- F1-Score

---

# 🧪 Experiment Tracking with MLflow

Every training run is logged to MLflow as a single, complete lifecycle record — from hyperparameters through to the final accept/reject decision:

- **Parameters:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `scale_pos_weight`, `random_state`, and the class-balancing strategy
- **Metrics:** `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`, `training_time_seconds`, dataset size and class ratios
- **Model comparison:** `candidate_roc_auc` vs `production_roc_auc` and the resulting `score_difference`
- **Tags:** `is_model_accepted`, `model_stage`, `git_commit`
- **Artifacts:** the trained XGBoost model, logged via MLflow's native `xgboost` flavor

**Design principle — separation of concerns:**

| | Responsibility |
|---|---|
| **MLflow** | Experiment tracking, run comparison, reproducibility — development-time observability |
| **Azure Blob Storage** | Production model registry — the actual deployment source of truth |

MLflow tracking runs locally (`file:` / `sqlite:` backend by default, configurable via env vars) and is designed to be fail-safe: if experiment tracking is unavailable for any reason, model training and the Azure deployment pipeline continue unaffected.

**View the experiment history locally:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) to browse runs, compare hyperparameters, and inspect accepted vs. rejected models.

---

# 🛠️ Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.10 |
| Machine Learning | Scikit-Learn, XGBoost |
| Experiment Tracking | MLflow |
| Database | MongoDB Atlas |
| API Framework | FastAPI |
| Cloud Storage | Azure Blob Storage |
| Deployment | Azure Web App |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Version Control | Git & GitHub |

---

# 📂 Project Structure

```text
Vehicle-Insurance-Domain-MLops/

├── artifact/
├── config/
├── logs/
├── mlruns/                  # MLflow local tracking store (git-ignored)
├── notebook/
├── src/
│   ├── cloud_storage/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py       # starts & logs to MLflow run
│   │   ├── model_evaluation.py    # resumes MLflow run, logs decision
│   │   └── model_pusher.py
│   ├── configuration/
│   ├── constants/
│   ├── data_access/
│   ├── entity/
│   ├── exception/
│   ├── logger/
│   ├── pipeline/
│   ├── templates/
│   └── utils/
│       ├── main_utils.py
│       └── mlflow_utils.py        # lightweight MLflow abstraction
│
├── app.py
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── setup.py
└── README.md
```

---

# 🚀 Run Locally

### Clone Repository

```bash
git clone https://github.com/masudrr72/Vehicle-Insurance-Domain-MLops

cd Vehicle-Insurance-Domain-MLops
```

### Create Environment

```bash
conda create -n vehicle python=3.10 -y

conda activate vehicle
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create `.env`

```env
MONGODB_URL=YOUR_MONGODB_CONNECTION_STRING

DATABASE_NAME=vehicle-project

COLLECTION_NAME=vehicle_project_data

AZURE_STORAGE_CONNECTION_STRING=YOUR_CONNECTION_STRING

AZURE_STORAGE_CONTAINER_NAME=model-registry

# Optional — defaults to a local SQLite store if not set
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

MLFLOW_EXPERIMENT_NAME=vehicle-insurance-prediction
```

### Start API

```bash
uvicorn app:app --reload
```

### View MLflow Experiment Tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

# 🐳 Docker

Build Docker Image

```bash
docker build -t vehicle-insurance-app .
```

Run Docker Container

```bash
docker run -p 8000:8000 vehicle-insurance-app
```

---

# 🔄 CI/CD Pipeline

This project uses **GitHub Actions** for automated deployment.

Every push to the **main** branch automatically:

- ✅ Build Docker Image
- ✅ Push Image to Azure Container Registry
- ✅ Deploy Latest Version to Azure Web App

---

# 🌐 API Endpoints

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/` | GET | Home Page |
| `/predict` | POST | Vehicle Insurance Prediction |
| `/health` | GET | Health Check |
| `/docs` | GET | Swagger Documentation |

---

# 🚀 Future Improvements

- Remote MLflow Tracking Server (Azure-hosted, for team-wide visibility)
- Model Monitoring
- Data Drift Detection
- Automated Model Retraining
- Kubernetes (AKS) Deployment
- Prometheus & Grafana Monitoring

---

# 👨‍💻 Author

**Masudur Rahman**

Machine Learning & MLOps Enthusiast

📧 Email: masudaucb1303@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/masudur-rahman-4b91a3327

💻 GitHub: https://github.com/masudrr72

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
