FROM public.ecr.aws/lambda/python:3.12

# Install poppler-utils for PDF → image conversion (pdf2image depends on it)
RUN dnf install -y poppler-utils && dnf clean all

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

CMD ["lambda_handler.handler"]
