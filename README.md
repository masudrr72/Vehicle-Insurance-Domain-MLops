# 🚗 Vehicle Insurance Prediction | End-to-End MLOps Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Azure](https://img.shields.io/badge/Azure-Web_App-0078D4)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-2088FF)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![License](https://img.shields.io/badge/License-MIT-success)

![image alt](https://github.com/masudrr72/Vehicle-Insurance-Domain-MLops/blob/main/vehicle_insurance_mlops_banner.png)

An industry-standard **End-to-End Machine Learning MLOps Project** that predicts whether a customer is interested in purchasing vehicle insurance. This project demonstrates a complete production ML workflow from data ingestion to cloud deployment using **FastAPI, Docker, Azure, and GitHub Actions**.

---

# 🌐 Live Demo

**🚀 Web Application:** https://vehicle-insurance-api-a6d3hjazeqf9aqha.southeastasia-01.azurewebsites.net/

**📄 Swagger API:** https://vehicle-insurance-api-a6d3hjazeqf9aqha.southeastasia-01.azurewebsites.net/docs

---

# ✨ Project Highlights

- 🚀 End-to-End Machine Learning Pipeline
- 🤖 XGBoost Classification Model
- 📊 Automated Data Validation & Transformation
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
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
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
- Model Training
- Model Evaluation
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

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-Score

---

# 🛠️ Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.10 |
| Machine Learning | Scikit-Learn, XGBoost |
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
├── notebook/
├── src/
│   ├── cloud_storage/
│   ├── components/
│   ├── configuration/
│   ├── constants/
│   ├── data_access/
│   ├── entity/
│   ├── exception/
│   ├── logger/
│   ├── pipeline/
│   ├── templates/
│   └── utils/
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
```

### Start API

```bash
uvicorn app:app --reload
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

- MLflow Experiment Tracking
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
