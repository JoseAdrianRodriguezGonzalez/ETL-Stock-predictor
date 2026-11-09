# ETL STOCK predictor
The project below consist in a stock predictor, however can work with any stock, but for the first version will just work for S&P500 index, precisely, stock repliers from this fund.
## Installation and preparation

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
