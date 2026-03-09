# FastAPI Project Setup Guide

## Prerequisites

Ensure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)
- **virtualenv** (optional but recommended)

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/your-project.git
cd your-project
```

### 2️⃣ Create a Virtual Environment
#### Windows
```powershell
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## Dependencies

Setup Tika
sudo docker run -d -p 127.0.0.1:9998:9998 apache/tika:latest-full

Setup Neo4j via Docker
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/neo4j123 neo4j:latest
```
---

## 🚢 Docker Deployment (Recommended)

Ensure you have Docker and Docker Compose installed.

### Build and Run
```bash
sudo docker-compose build
sudo docker-compose down
sudo docker-compose up -d
```

Once the containers are up, access the API docs:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

Docker Compose will start the FastAPI app and a Tika server. Environment variables can be provided via a `.env` file in the project root (see the Environment Variables section below).

---

## 💻 Running the Server
```bash
python uvicorn_config.py
```

---

## 🚀 Running in Production

Prefer the Docker-based approach above for production. If you need to run without Docker, a `uvicorn_config.py` file is included:

```bash
python uvicorn_config.py
```

---

## 🔧 Environment Variables
You can configure environment-specific settings using a `.env` file.

Create a **.env** file in the project root:
```ini
APP_NAME=FastAPI App
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

---


## ✅ API Documentation
Once the server is running, access the API docs:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

