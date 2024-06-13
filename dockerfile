# Gunakan image Python 3.10
FROM python:3.10-slim

# Set lingkungan kerja
WORKDIR /app

# Copy requirements.txt dan install dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy seluruh source code ke dalam container
COPY . .

# Pindah ke folder theme/static_src dan install npm dependencies
WORKDIR /app/myproject/theme/static_src
RUN npm install

# Pindah ke folder web
WORKDIR /app/myproject
RUN python manage.py tailwind build

# crontab

# Expose port yang digunakan oleh Django (misalnya 8000)
EXPOSE 8000

# Tentukan command untuk menjalankan server Django
CMD ["python", "/app/myproject/manage.py", "runserver", "0.0.0.0:8000"]