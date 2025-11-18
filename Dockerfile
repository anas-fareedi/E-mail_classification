FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

RUN pip install --upgrade pip

# Install project dependencies (Torch CPU included)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy ALL project files from root into container
COPY . .

# Streamlit default port
EXPOSE 8501

# Run your Streamlit app (assuming your file is app.py)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
