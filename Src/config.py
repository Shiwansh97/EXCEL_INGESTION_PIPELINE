# Snowflake connection config
SNOWFLAKE_DB_CONFIG = {
    "user": "SHIV0173",
    "password": "Sparkbrains@9889",
    "account": "MREANEW-OD33745", 
    "warehouse": "COMPUTE_WH",
    "database": "SPARK",
    "schema": "PUBLIC"
}
# PostgreSQL connection config

POSTGRESS_DB_CONFIG = {
    "user": "postgres",
    "password": "spark@1234",
    "host": "localhost",
    "port": 5432,
    "database": "postgres"
}

from datetime import datetime   # add this import at top of config.py if not present

# SLA Settings
SLA_THRESHOLD_MINUTES        = 30     # max gap allowed between loads (business minutes)
SLA_LOOKBACK_HOURS           = 4800     # how many hours of audit history to check

SLA_BUSINESS_HOURS_START      = datetime.strptime("07:30:00", "%H:%M:%S").time()
SLA_BUSINESS_HOURS_END        = datetime.strptime("15:30:00", "%H:%M:%S").time()   # Mon–Thu
SLA_BUSINESS_HOURS_END_FRIDAY = datetime.strptime("12:00:00", "%H:%M:%S").time()   # Friday

print("\n[SLA Config] Business hours loaded:")
print(f"   Mon–Thu : {SLA_BUSINESS_HOURS_START}  –  {SLA_BUSINESS_HOURS_END}")
print(f"   Friday  : {SLA_BUSINESS_HOURS_START}  –  {SLA_BUSINESS_HOURS_END_FRIDAY}")
print(f"   SLA Threshold : {SLA_THRESHOLD_MINUTES} minutes")