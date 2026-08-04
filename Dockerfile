FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files (excludes venv, csvs via .dockerignore)
COPY etl.py ai_utils.py dashboard_flows.py taxonomy.py connectors.py ./
COPY configs ./configs
COPY messy_complete_events_simple.csv \
     messy_complete_users.csv \
     stripe_charges.csv \
     stripe_subscriptions.csv \
     stripe_invoices.csv ./

# Expose Dash default port
EXPOSE 8050

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "1", "--timeout", "120", "dashboard_flows:server"]
