# Architechture docs 

This project will live in Docker, and will contain the following amount of containers 


- mongodb 
- postgres 
- mlflow
- triton 
- backend 
- frontend 
- collector 
- training 
- deployment 

This simulates a typical **MLOps** workflow 
The system off folders will be defined as it follows 
```plain
sp500-mlops-platform/
│
├── collector/
│   ├── src/
│   │   ├── providers/
│   │   ├── services/
│   │   ├── storage/
│   │   └── validators/
│   ├── tests/
│   └── Dockerfile
│
├── training/
│   ├── src/
│   │   ├── models/
│   │   ├── preprocessing/
│   │   ├── features/
│   │   ├── evaluation/
│   │   └── experiments/
│   ├── tests/
│   └── Dockerfile
│
├── deployment/
│   ├── src/
│   ├── tests/
│   └── Dockerfile
│
├── inference/
│   ├── model_repository/
│   └── configs/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   ├── schemas/
│   │   └── main.py
│   ├── tests/
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── Dockerfile
│
├── infrastructure/
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   ├── postgres/
│   └── scripts/
│
├── configs/
├── data/
├── manifests/
├── tests/
├── scripts/
├── .env
├── .gitignore
└── README.md
```

The collector, will have the destination of FMP and HTTP API interface,as also tghe mongoDB and Pymongo as well and DVC and a CLI/Python versioning 
The trainer, will hsot the model with MLFLow and DVC, the depployment, will use Triton, and the backend consumes the inference server by triton, http/grpc, and finally a frontend that consumes the backend and display it.

## Datasets 
It will contain two folders, raw, where it goes the raw retrieved data and processed where it goes the data ready for feature engineering and transformations 
The contract for the data will be 
Campo	Tipo	Descripción
date	date32	Fecha de negociación
symbol	string	^GSPC
open	float64	Apertura
high	float64	Máximo
low	float64	Mínimo
close	float64	Cierre
source	string	fmp
retrieved_at	timestamp UTC	Fecha de extracción
And load the manifesto 
```bash 
{
  "dataset_name": "sp500_daily",
  "schema_version": "1.0",
  "provider": "financial_modeling_prep",
  "symbol": "^GSPC",
  "frequency": "1day",
  "date_range": {
    "start": "2020-01-01",
    "end": "2026-11-06"
  },
  "retrieved_at": "2026-11-09T08:00:00Z",
  "format": "parquet"
}
```
The format for configs will be in yaml 
And the metrics will be MAE,RMSE,MSE and directional accuracy, and save these metrics in json

MLFlow will use Model registery qwith candidate and champion
The model lastly will predict until certain time
