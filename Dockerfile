FROM python:3.14-slim

# Install system dependencies required by Pillow (fonts, libjpeg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["sh","-c","streamlit run app.py --server.headless true --server.port $PORT"]
