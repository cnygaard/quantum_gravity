FROM python:3.13-slim-bookworm

# Set the working directory
WORKDIR /app
# Copy the requirements file into the container
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-tk \
    libmpich-dev

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the code into the container
COPY . .

# Run the application
CMD ["python", "demo-blackhole.py"]