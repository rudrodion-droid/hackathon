# Используем компактный официальный образ Python.
FROM python:3.12-slim

# Системные библиотеки, нужные OpenCV (внутренняя зависимость inference-sdk).
# Без них при импорте cv2 падает: "ImportError: libGL.so.1: cannot open
# shared object file". Ставим их явно через apt-get — так надёжнее, чем
# пытаться подобрать нужные пакеты через Nixpacks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Railway передаёт номер порта через переменную окружения $PORT.
# Если её нет (локальный запуск через "docker run"), используем 8501.
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
