FROM python:3.10-slim

# Tentukan direktori kerja di dalam server
WORKDIR /code

# Salin requirements dan instal semua library AI
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Salin seluruh file kode ke dalam server
COPY . .

# Jalankan FastAPI di port 7860 (Port wajib untuk Hugging Face Docker)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]