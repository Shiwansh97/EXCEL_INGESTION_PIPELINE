import streamlit as st
import pandas as pd
import os
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from datetime import datetime, timedelta, timezone
import altair as alt
from config import SNOWFLAKE_DB_CONFIG
import traceback
import re

# -------------------------------------------------
# TIMEZONE  — IST = UTC + 5:30
# -------------------------------------------------
IST_OFFSET = timezone(timedelta(hours=5, minutes=30))

def now_ist() -> datetime:
    """Return current datetime in IST (naive, for display & comparison)."""
    return datetime.now(tz=IST_OFFSET).replace(tzinfo=None)

def to_ist(utc_dt) -> datetime:
    """
    Convert a UTC datetime (aware or naive) to IST naive datetime.
    Handles both timezone-aware and naive inputs from Snowflake.
    """
    if utc_dt is None:
        return None
    if isinstance(utc_dt, datetime):
        if utc_dt.tzinfo is None:
            # Snowflake returned naive — treat as UTC, convert to IST
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        return utc_dt.astimezone(IST_OFFSET).replace(tzinfo=None)
    return utc_dt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Enterprise Snowflake Ingestion", layout="wide")

# =============================================================
# SLA CONFIGURATION  — edit these to match your requirements
# =============================================================
SLA_THRESHOLD_MINUTES        = 30    # max allowed gap between consecutive loads (minutes)
SLA_LOOKBACK_HOURS           = 48    # how many hours of audit history to analyse

# Use datetime.strptime so there is no ambiguity with integer literals
SLA_BUSINESS_HOURS_START      = datetime.strptime("07:30:00", "%H:%M:%S").time()
SLA_BUSINESS_HOURS_END        = datetime.strptime("15:30:00", "%H:%M:%S").time()  # Mon–Thu
SLA_BUSINESS_HOURS_END_FRIDAY = datetime.strptime("12:00:00", "%H:%M:%S").time()  # Friday

# -------------------------------------------------
# AUTH / ROLE
# -------------------------------------------------
st.sidebar.header("🔐 Access Control")
ROLE = st.sidebar.selectbox("Select Role", ["ANALYST", "ADMIN"])

# -------------------------------------------------
# SNOWFLAKE CONNECTION
# -------------------------------------------------
@st.cache_resource
def get_conn():
    try:
        return snowflake.connector.connect(**SNOWFLAKE_DB_CONFIG)
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {str(e)}")
        st.stop()

conn = get_conn()

# -------------------------------------------------
# HELPERS (original)
# -------------------------------------------------
def normalize_columns(df):
    """Normalize column names to uppercase with underscores"""
    return [c.upper().replace(" ", "_") for c in df.columns]

def sanitize_table_name(name):
    """Sanitize table name to be valid for Snowflake"""
    name = re.sub(r'\.(csv|xlsx)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if name and not name[0].isalpha() and name[0] != '_':
        name = 'TBL_' + name
    return name.upper()

def get_tables():
    """Fetch all tables in current schema"""
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = CURRENT_SCHEMA()
            ORDER BY table_name
        """)
        tables = [r[0] for r in cur.fetchall()]
        cur.close()
        return tables
    except Exception as e:
        st.error(f"Error fetching tables: {str(e)}")
        return []

def get_schema(table):
    """Fetch column names for a given table - SQL injection safe"""
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            AND table_schema = CURRENT_SCHEMA()
            ORDER BY ordinal_position
        """, (table,))
        columns = [r[0] for r in cur.fetchall()]
        cur.close()
        return columns
    except Exception as e:
        st.error(f"Error fetching schema for {table}: {str(e)}")
        return []

def get_table_columns(table_name):
    """Get columns for a specific table"""
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            AND table_schema = CURRENT_SCHEMA()
        """, (table_name,))
        columns = [r[0] for r in cur.fetchall()]
        cur.close()
        return columns
    except Exception as e:
        return []

def create_table_from_dataframe(table_name, df):
    """Create a new table based on dataframe structure"""
    try:
        cur = conn.cursor()
        type_mapping = {
            'int64': 'INTEGER', 'int32': 'INTEGER',
            'float64': 'FLOAT', 'float32': 'FLOAT',
            'bool': 'BOOLEAN', 'datetime64[ns]': 'TIMESTAMP', 'object': 'STRING'
        }
        columns_def = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            snowflake_type = type_mapping.get(dtype, 'STRING')
            columns_def.append(f"{col} {snowflake_type}")
        create_sql = f"CREATE TABLE {table_name} ({', '.join(columns_def)})"
        cur.execute(create_sql)
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        st.error(f"Error creating table: {str(e)}")
        return False

def create_audit_table():
    """Create audit table if it doesn't exist with proper schema"""
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'UPLOAD_AUDIT' AND table_schema = CURRENT_SCHEMA()
        """)
        table_exists = cur.fetchone()[0] > 0

        if table_exists:
            existing_cols = get_table_columns('UPLOAD_AUDIT')
            if 'OPERATION_TYPE' not in existing_cols:
                cur.execute("ALTER TABLE UPLOAD_AUDIT ADD COLUMN OPERATION_TYPE STRING")
                conn.commit()
            if 'UPLOADED_BY' not in existing_cols:
                cur.execute("ALTER TABLE UPLOAD_AUDIT ADD COLUMN UPLOADED_BY STRING")
                conn.commit()
        else:
            cur.execute("""
                CREATE TABLE UPLOAD_AUDIT (
                    FILE_NAME     STRING,
                    TABLE_NAME    STRING,
                    ROW_COUNT     INT,
                    OPERATION_TYPE STRING,
                    UPLOAD_TIME   TIMESTAMP,
                    UPLOADED_BY   STRING
                )
            """)
            conn.commit()
        cur.close()
    except Exception as e:
        st.error(f"Error creating/updating audit table: {str(e)}")

def log_audit(file, table, rows, operation="MERGE"):
    """Log upload audit with IST timestamp — SQL injection safe"""
    try:
        cur = conn.cursor()
        ist_now = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("""
            INSERT INTO UPLOAD_AUDIT (FILE_NAME, TABLE_NAME, ROW_COUNT, OPERATION_TYPE, UPLOAD_TIME, UPLOADED_BY)
            VALUES (%s, %s, %s, %s, %s::TIMESTAMP, %s)
        """, (file, table, rows, operation, ist_now, ROLE))
        conn.commit()
        cur.close()
    except Exception as e:
        st.error(f"Error logging audit: {str(e)}")

def validate_table_exists(table_name):
    tables = get_tables()
    return table_name in tables

def check_schema_compatibility(df_columns, table_columns):
    missing_in_table = [col for col in df_columns if col not in table_columns]
    missing_in_file  = [col for col in table_columns if col not in df_columns]
    return missing_in_table, missing_in_file

# -------------------------------------------------
# SLA HELPERS
# -------------------------------------------------
def is_within_business_hours(check_time) -> bool:
    """Return True if the given datetime falls inside configured business hours."""
    weekday = check_time.weekday()   # 0=Mon … 6=Sun
    if weekday in (5, 6):            # weekend
        return False
    end = SLA_BUSINESS_HOURS_END_FRIDAY if weekday == 4 else SLA_BUSINESS_HOURS_END
    return SLA_BUSINESS_HOURS_START <= check_time.time() <= end


def calculate_business_minutes_between(
    start_timestamp: datetime,
    finish_timestamp: datetime
) -> int:
    """
    Calculate actual business minutes elapsed between two timestamps.
    Skips weekends and honours per-day business-hour windows.
    Returns 0 if start >= finish or inputs are invalid.
    """
    if start_timestamp is None or finish_timestamp is None:
        return 0
    if start_timestamp >= finish_timestamp:
        return 0

    total_business_minutes = 0
    current_date = start_timestamp.date()
    end_date      = finish_timestamp.date()

    while current_date <= end_date:
        weekday = current_date.weekday()

        # Skip weekends
        if weekday in (5, 6):
            current_date += timedelta(days=1)
            continue

        # Determine business-hour window for this day
        business_day_end = (
            SLA_BUSINESS_HOURS_END_FRIDAY if weekday == 4
            else SLA_BUSINESS_HOURS_END
        )
        business_day_start = SLA_BUSINESS_HOURS_START

        # Clamp day start
        if current_date == start_timestamp.date():
            effective_day_start = max(start_timestamp.time(), business_day_start)
        else:
            effective_day_start = business_day_start

        # Clamp day end
        if current_date == finish_timestamp.date():
            effective_day_end = min(finish_timestamp.time(), business_day_end)
        else:
            effective_day_end = business_day_end

        # Convert to datetimes for subtraction
        dt_start = datetime.combine(current_date, effective_day_start)
        dt_end   = datetime.combine(current_date, effective_day_end)

        if dt_end > dt_start:
            diff_seconds = (dt_end - dt_start).total_seconds()
            total_business_minutes += int(diff_seconds // 60)

        current_date += timedelta(days=1)

    return total_business_minutes


def fetch_audit_for_sla() -> pd.DataFrame:
    """
    Pull recent audit records from UPLOAD_AUDIT for SLA analysis.
    Converts UPLOAD_TIME from UTC → IST before returning.
    Returns an empty DataFrame on failure.
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                TABLE_NAME,
                UPLOAD_TIME,
                ROW_COUNT,
                OPERATION_TYPE,
                FILE_NAME
            FROM UPLOAD_AUDIT
            WHERE UPLOAD_TIME >= DATEADD('hour', %s, CURRENT_TIMESTAMP())
            ORDER BY TABLE_NAME, UPLOAD_TIME ASC
        """, (-SLA_LOOKBACK_HOURS,))
        rows = cur.fetchall()
        cur.close()

        if not rows:
            return pd.DataFrame()

        audit_df = pd.DataFrame(rows, columns=[
            "TABLE_NAME", "UPLOAD_TIME", "ROW_COUNT", "OPERATION_TYPE", "FILE_NAME"
        ])

        # Convert UTC → IST so all downstream comparisons use local time
        audit_df["UPLOAD_TIME"] = pd.to_datetime(audit_df["UPLOAD_TIME"])
        audit_df["UPLOAD_TIME"] = audit_df["UPLOAD_TIME"].apply(to_ist)

        return audit_df

    except Exception as fetch_error:
        st.error(f"Error fetching audit data for SLA: {str(fetch_error)}")
        return pd.DataFrame()


def compute_sla_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each table, compute:
      - Total loads in lookback window
      - Last load time
      - Max gap between consecutive loads (business minutes)
      - Whether every gap is within SLA_THRESHOLD_MINUTES
      - Current status (time since last load vs threshold)
    Returns one row per table.
    """
    sla_records = []

    for table_name, table_group in audit_df.groupby("TABLE_NAME"):
        sorted_times = table_group["UPLOAD_TIME"].sort_values().tolist()
        total_loads  = len(sorted_times)
        last_load_ts = sorted_times[-1]

        # ── Gap analysis ──────────────────────────────────────────────
        gap_list_business_minutes = []

        if len(sorted_times) >= 2:
            for gap_index in range(len(sorted_times) - 1):
                gap_start = sorted_times[gap_index]
                gap_end   = sorted_times[gap_index + 1]
                business_gap = calculate_business_minutes_between(gap_start, gap_end)
                gap_list_business_minutes.append(business_gap)

        max_gap_minutes = max(gap_list_business_minutes) if gap_list_business_minutes else 0
        avg_gap_minutes = (
            round(sum(gap_list_business_minutes) / len(gap_list_business_minutes), 1)
            if gap_list_business_minutes else 0
        )
        breached_gaps   = sum(g > SLA_THRESHOLD_MINUTES for g in gap_list_business_minutes)

        # ── Current staleness (time since last ingest) ────────────────
        minutes_since_last_load = calculate_business_minutes_between(
            last_load_ts, now_ist()
        )

        # ── SLA status logic ──────────────────────────────────────────
        # Only flag as stale during business hours; outside hours it's "Off-Hours"
        currently_in_business_hours = is_within_business_hours(now_ist())

        if not currently_in_business_hours:
            current_sla_status = "⚪ Off-Hours"
        elif minutes_since_last_load > SLA_THRESHOLD_MINUTES:
            current_sla_status = "🔴 BREACHED"
        elif minutes_since_last_load > SLA_THRESHOLD_MINUTES * 0.75:
            current_sla_status = "🟡 WARNING"
        else:
            current_sla_status = "🟢 OK"

        overall_sla_met = breached_gaps == 0

        sla_records.append({
            "Table Name":             table_name,
            "Total Loads":            total_loads,
            "Last Load Time":         last_load_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Mins Since Last Load":   minutes_since_last_load,
            "Avg Gap (Biz Mins)":     avg_gap_minutes,
            "Max Gap (Biz Mins)":     max_gap_minutes,
            "Breached Gaps":          breached_gaps,
            "SLA Met (Historical)":   "✅ Yes" if overall_sla_met else "❌ No",
            "Current Status":         current_sla_status,
        })

    return pd.DataFrame(sla_records)


def build_gap_timeline_chart(audit_df: pd.DataFrame, selected_table: str) -> alt.Chart:
    """
    Build an Altair bar chart showing gap size per interval for a single table.
    Bars above the SLA threshold are shown in red, within threshold in green.
    """
    table_times = (
        audit_df[audit_df["TABLE_NAME"] == selected_table]["UPLOAD_TIME"]
        .sort_values()
        .tolist()
    )

    if len(table_times) < 2:
        return None

    gap_chart_rows = []
    for idx in range(len(table_times) - 1):
        gap_label        = f"{table_times[idx].strftime('%H:%M')} → {table_times[idx+1].strftime('%H:%M')}"
        biz_gap_minutes  = calculate_business_minutes_between(table_times[idx], table_times[idx + 1])
        sla_colour_label = "Over SLA" if biz_gap_minutes > SLA_THRESHOLD_MINUTES else "Within SLA"
        gap_chart_rows.append({
            "Interval":  gap_label,
            "Gap (Biz Mins)": biz_gap_minutes,
            "SLA Status": sla_colour_label,
        })

    gap_df = pd.DataFrame(gap_chart_rows)

    threshold_rule = (
        alt.Chart(pd.DataFrame({"threshold": [SLA_THRESHOLD_MINUTES]}))
        .mark_rule(color="orange", strokeDash=[6, 3], size=2)
        .encode(y="threshold:Q")
    )

    bar_chart = (
        alt.Chart(gap_df)
        .mark_bar()
        .encode(
            x=alt.X("Interval:N", sort=None, title="Upload Interval"),
            y=alt.Y("Gap (Biz Mins):Q", title="Business Minutes Between Loads"),
            color=alt.Color(
                "SLA Status:N",
                scale=alt.Scale(
                    domain=["Within SLA", "Over SLA"],
                    range=["#2ecc71", "#e74c3c"]
                )
            ),
            tooltip=["Interval", "Gap (Biz Mins)", "SLA Status"]
        )
        .properties(height=340)
    )

    return threshold_rule + bar_chart


# -------------------------------------------------
# INITIALIZE
# -------------------------------------------------
create_audit_table()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🏢 Enterprise Snowflake Ingestion Platform")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel",
    ["csv", "xlsx"]
)

if not uploaded_file:
    st.info("👈 Please upload a CSV or Excel file to begin")
    st.stop()

# -------------------------------------------------
# READ FILE
# -------------------------------------------------
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = normalize_columns(df)

    if df.empty:
        st.error("The uploaded file is empty")
        st.stop()

except Exception as e:
    st.error(f"Error reading file: {str(e)}")
    st.stop()

# Initialize session state for validated data
if 'valid_df' not in st.session_state:
    st.session_state.valid_df = df.copy()
if 'pk_column' not in st.session_state:
    st.session_state.pk_column = df.columns[0] if len(df.columns) > 0 else None

tabs = st.tabs([
    "📄 Preview",
    "🚦 Validation",
    "📈 Profiling",
    "🔁 Load / Merge",
    "⏱️ SLA Monitor",     # ← NEW
    "🧾 Audit"
])

# -------------------------------------------------
# PREVIEW
# -------------------------------------------------
with tabs[0]:
    st.subheader("Data Preview")
    st.dataframe(df.head(100))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    with st.expander("Column Details"):
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(dtype) for dtype in df.dtypes]
        })
        st.dataframe(dtypes_df, hide_index=True)

# -------------------------------------------------
# ROW-LEVEL VALIDATION
# -------------------------------------------------
with tabs[1]:
    st.subheader("Row-Level Validation")

    pk = st.selectbox("Select Primary Key", df.columns, key='pk_selector')
    st.session_state.pk_column = pk

    null_rows = df[df[pk].isnull()]
    dup_rows  = df[df.duplicated(subset=[pk], keep=False)]

    if not null_rows.empty:
        st.error(f"❌ Found {len(null_rows)} rows with NULL Primary Key")
        with st.expander("View NULL rows"):
            st.dataframe(null_rows)

    if not dup_rows.empty:
        st.error(f"❌ Found {len(dup_rows)} rows with Duplicate Primary Key")
        with st.expander("View duplicate rows"):
            st.dataframe(dup_rows)

    valid_df = df.dropna(subset=[pk]).drop_duplicates(subset=[pk])
    st.session_state.valid_df = valid_df

    invalid_count = len(df) - len(valid_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Valid Rows", len(valid_df))
    with col3:
        st.metric("Invalid Rows", invalid_count, delta=f"-{invalid_count}")

    if invalid_count > 0:
        st.warning(f"⚠️ {invalid_count} invalid rows will be excluded from upload")
    else:
        st.success("✅ All rows passed validation")

# -------------------------------------------------
# COLUMN PROFILING
# -------------------------------------------------
with tabs[2]:
    st.subheader("Column Profiling")

    col = st.selectbox("Select column to profile", df.columns)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        null_count = df[col].isnull().sum()
        null_pct   = (null_count / len(df)) * 100
        st.metric("Null Count", null_count)
    with col2:
        st.metric("Null %", f"{null_pct:.2f}%")
    with col3:
        unique_count = df[col].nunique()
        st.metric("Unique Values", unique_count)
    with col4:
        unique_pct = (unique_count / len(df)) * 100
        st.metric("Unique %", f"{unique_pct:.2f}%")

    st.subheader("Distribution")

    if pd.api.types.is_numeric_dtype(df[col]):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Statistics**")
            stats = df[col].describe()
            st.dataframe(stats.to_frame(name='Value'))
        with col2:
            st.write("**Histogram**")
            try:
                chart = alt.Chart(df.dropna(subset=[col])).mark_bar().encode(
                    alt.X(col, bin=alt.Bin(maxbins=30), title=col),
                    alt.Y('count()', title='Frequency')
                ).properties(height=300)
                st.altair_chart(chart, width='stretch')
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
    else:
        st.write("**Top 10 Values**")
        top_vals = df[col].value_counts().head(10).reset_index()
        top_vals.columns = [col, "COUNT"]
        try:
            chart = alt.Chart(top_vals).mark_bar().encode(
                x=alt.X('COUNT:Q'),
                y=alt.Y(f'{col}:N', sort='-x')
            ).properties(height=300)
            st.altair_chart(chart, width='stretch')
        except Exception:
            st.bar_chart(top_vals.set_index(col))

# -------------------------------------------------
# UPSERT / MERGE
# -------------------------------------------------
with tabs[3]:
    st.subheader("UPSERT / MERGE")

    if ROLE != "ADMIN":
        st.warning("⚠️ Only ADMIN role can load data to Snowflake")
        st.info("Please select ADMIN role from the sidebar to proceed")
        st.stop()

    valid_df = st.session_state.valid_df
    pk       = st.session_state.pk_column

    if valid_df.empty:
        st.error("No valid data to upload. Please check the Validation tab.")
        st.stop()

    st.info(f"📊 Ready to upload {len(valid_df)} valid rows using **{pk}** as primary key")

    available_tables = get_tables()

    table_mode = st.radio(
        "Select Table Mode:",
        ["Use Existing Table", "Create New Table"],
        horizontal=True
    )

    if table_mode == "Use Existing Table":
        if not available_tables:
            st.error("No tables found in the current schema. Please select 'Create New Table' option.")
            st.stop()

        target_table  = st.selectbox("Select Target Table", available_tables)
        schema_mismatch = False

        if target_table:
            target_cols = get_schema(target_table)
            if not target_cols:
                st.error(f"Could not fetch schema for {target_table}")
                st.stop()

            if pk not in target_cols:
                st.error(f"❌ Primary key '{pk}' does not exist in target table {target_table}")
                schema_mismatch = True

            missing_cols, extra_cols = check_schema_compatibility(valid_df.columns, target_cols)

            if missing_cols:
                st.warning(f"⚠️ Columns in file but not in table: {', '.join(missing_cols)}")
                schema_mismatch = True

            if extra_cols:
                st.info(f"ℹ️ Columns in table but not in file: {', '.join(extra_cols)}")

            with st.expander("View Column Mapping"):
                mapping_data = []
                for column_name in valid_df.columns:
                    if column_name in target_cols:
                        mapping_data.append({'File Column': column_name, 'Target Column': column_name, 'Status': '✅ Match'})
                    else:
                        mapping_data.append({'File Column': column_name, 'Target Column': '-', 'Status': '❌ Missing in Table'})
                st.dataframe(pd.DataFrame(mapping_data), hide_index=True)

            if schema_mismatch:
                st.error("⚠️ Schema mismatch detected!")
                if missing_cols:
                    st.warning("The following columns will be ignored during upload:")
                    st.write(", ".join(missing_cols))
                    valid_df = valid_df[[c for c in valid_df.columns if c in target_cols]]
                    st.session_state.valid_df = valid_df
                    if valid_df.empty:
                        st.error("No matching columns found. Please create a new table.")
                        st.stop()

    else:
        st.info("🆕 Creating a new table based on your file structure")
        suggested_name = sanitize_table_name(uploaded_file.name)

        col1, col2 = st.columns([3, 1])
        with col1:
            target_table = st.text_input("New Table Name", value=suggested_name)
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄 Use Filename"):
                target_table = suggested_name
                st.rerun()

        target_table = sanitize_table_name(target_table) if target_table else suggested_name

        if target_table in available_tables:
            st.error(f"❌ Table '{target_table}' already exists!")
            st.stop()

        type_mapping = {
            'int64': 'INTEGER', 'int32': 'INTEGER',
            'float64': 'FLOAT', 'float32': 'FLOAT',
            'bool': 'BOOLEAN', 'datetime64[ns]': 'TIMESTAMP', 'object': 'STRING'
        }
        structure_data = [{
            'Column Name': c,
            'Snowflake Type': type_mapping.get(str(valid_df[c].dtype), 'STRING'),
            'Pandas Type': str(valid_df[c].dtype),
            'Primary Key': '🔑' if c == pk else ''
        } for c in valid_df.columns]
        st.dataframe(pd.DataFrame(structure_data), hide_index=True)

    st.divider()
    st.write("**Operation Details:**")
    if table_mode == "Create New Table":
        st.write("- **Mode:** CREATE TABLE + INSERT")
    else:
        st.write("- **Mode:** MERGE (INSERT new records, UPDATE existing records)")
    st.write(f"- **Table Name:** {target_table}")
    st.write(f"- **Primary Key:** {pk}")
    st.write(f"- **Rows to process:** {len(valid_df)}")

    confirm = st.checkbox(f"I confirm this {'TABLE CREATION and DATA LOAD' if table_mode == 'Create New Table' else 'MERGE'} operation")

    if st.button("🚀 Execute Operation", type="primary", disabled=not confirm):
        with st.spinner("Processing operation..."):
            try:
                cur = conn.cursor()

                if table_mode == "Create New Table":
                    progress = st.progress(0, text="Creating new table...")
                    if not create_table_from_dataframe(target_table, valid_df):
                        st.error("Failed to create table")
                        st.stop()
                    progress.progress(40, text="Loading data to new table...")
                    success, nchunks, nrows, _ = write_pandas(conn, valid_df.reset_index(drop=True), target_table, auto_create_table=False, overwrite=False)
                    if not success:
                        raise Exception("Failed to load data to new table")
                    progress.progress(80, text="Logging audit trail...")
                    log_audit(uploaded_file.name, target_table, len(valid_df), "CREATE + INSERT")
                    conn.commit()
                    cur.close()
                    progress.progress(100, text="Complete!")
                    progress.empty()
                    st.success(f"✅ Table '{target_table}' created and {len(valid_df)} rows inserted successfully!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows Inserted", len(valid_df))
                    with col2:
                        st.metric("Table Created", target_table)
                    st.balloons()

                else:
                    stage_table = f"{target_table}_STAGE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    progress = st.progress(0, text="Creating staging table...")
                    cur.execute(f"DROP TABLE IF EXISTS {stage_table}")
                    cur.execute(f"CREATE TEMPORARY TABLE {stage_table} LIKE {target_table}")
                    progress.progress(25, text="Loading data to staging table...")
                    success, nchunks, nrows, _ = write_pandas(conn, valid_df.reset_index(drop=True), stage_table, auto_create_table=False, overwrite=True)
                    if not success:
                        raise Exception("Failed to write data to staging table")
                    progress.progress(50, text="Calculating merge statistics...")
                    cur.execute(f"SELECT COUNT(*) FROM {target_table} T INNER JOIN {stage_table} S ON T.{pk} = S.{pk}")
                    result = cur.fetchone()
                    rows_to_update = result[0] if result else 0
                    progress.progress(60, text="Executing MERGE...")
                    update_cols = [c for c in valid_df.columns if c != pk]
                    if update_cols:
                        update_set   = ", ".join([f"T.{c} = S.{c}" for c in update_cols])
                        when_matched = f"WHEN MATCHED THEN UPDATE SET {update_set}"
                    else:
                        when_matched = ""
                    insert_cols = ", ".join(valid_df.columns)
                    insert_vals = ", ".join([f"S.{c}" for c in valid_df.columns])
                    merge_sql = f"""
                        MERGE INTO {target_table} T
                        USING {stage_table} S
                        ON T.{pk} = S.{pk}
                        {when_matched}
                        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
                    """
                    cur.execute(merge_sql)
                    rows_inserted = len(valid_df) - rows_to_update
                    rows_updated  = rows_to_update
                    progress.progress(80, text="Logging audit trail...")
                    log_audit(uploaded_file.name, target_table, len(valid_df), "MERGE")
                    progress.progress(90, text="Cleaning up...")
                    cur.execute(f"DROP TABLE IF EXISTS {stage_table}")
                    conn.commit()
                    cur.close()
                    progress.progress(100, text="Complete!")
                    progress.empty()
                    st.success("✅ MERGE completed successfully!")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows Processed", len(valid_df))
                    with col2:
                        st.metric("Rows Inserted", rows_inserted)
                    with col3:
                        st.metric("Rows Updated", rows_updated)
                    st.balloons()

            except Exception as e:
                st.error(f"❌ Error during operation: {str(e)}")
                with st.expander("View detailed error"):
                    st.code(traceback.format_exc())
                try:
                    conn.rollback()
                    st.info("Transaction rolled back successfully")
                except Exception as rollback_error:
                    st.error(f"Rollback failed: {str(rollback_error)}")
            finally:
                try:
                    cur.close()
                except Exception:
                    pass

# =============================================================
# ⏱️  SLA MONITOR TAB  (NEW)
# =============================================================
with tabs[4]:
    st.subheader("⏱️ SLA Monitor — Table-wise Ingestion Health")

    # ── Config banner ─────────────────────────────────────────
    sla_config_col1, sla_config_col2, sla_config_col3 = st.columns(3)
    with sla_config_col1:
        st.info(f"🎯 SLA Threshold: **{SLA_THRESHOLD_MINUTES} minutes**")
    with sla_config_col2:
        st.info(f"🕐 Business Hours: **{SLA_BUSINESS_HOURS_START.strftime('%H:%M')} – {SLA_BUSINESS_HOURS_END.strftime('%H:%M')}**  |  Fri: **– {SLA_BUSINESS_HOURS_END_FRIDAY.strftime('%H:%M')}**")
    with sla_config_col3:
        st.info(f"🔍 Lookback Window: **Last {SLA_LOOKBACK_HOURS} hours**")

    st.divider()

    # ── Refresh button ────────────────────────────────────────
    sla_refresh_col, sla_timestamp_col = st.columns([1, 4])
    with sla_refresh_col:
        sla_refresh_clicked = st.button("🔄 Refresh SLA Data")
    with sla_timestamp_col:
        st.caption(f"Last checked: {now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST")

    # ── Load data ─────────────────────────────────────────────
    raw_audit_df = fetch_audit_for_sla()

    if raw_audit_df.empty:
        st.warning(
            f"⚠️ No upload records found in the last {SLA_LOOKBACK_HOURS} hours. "
            "Run some ingestions first or increase the lookback window."
        )
        st.stop()

    sla_summary_df = compute_sla_summary(raw_audit_df)

    # ── Top-level KPI cards ───────────────────────────────────
    total_tables_monitored  = len(sla_summary_df)
    tables_currently_ok     = (sla_summary_df["Current Status"] == "🟢 OK").sum()
    tables_currently_warn   = (sla_summary_df["Current Status"] == "🟡 WARNING").sum()
    tables_currently_breach = (sla_summary_df["Current Status"] == "🔴 BREACHED").sum()
    tables_historical_fail  = (sla_summary_df["SLA Met (Historical)"] == "❌ No").sum()

    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.metric("📋 Tables Monitored", total_tables_monitored)
    with kpi_col2:
        st.metric("🟢 Currently OK", tables_currently_ok)
    with kpi_col3:
        st.metric("🟡 Warning", tables_currently_warn)
    with kpi_col4:
        st.metric("🔴 Breached Now", tables_currently_breach)
    with kpi_col5:
        st.metric("❌ Historical Fails", tables_historical_fail)

    st.divider()

    # ── SLA Summary Table ─────────────────────────────────────
    st.subheader("📊 Table-wise SLA Summary")

    # Colour-code: highlight breached rows
    def highlight_sla_row(row):
        """Apply background colour based on SLA status."""
        if row["Current Status"] == "🔴 BREACHED":
            return ["background-color: #ffd6d6"] * len(row)
        if row["Current Status"] == "🟡 WARNING":
            return ["background-color: #fff3cd"] * len(row)
        if row["SLA Met (Historical)"] == "❌ No":
            return ["background-color: #fff0e6"] * len(row)
        return [""] * len(row)

    styled_sla_df = sla_summary_df.style.apply(highlight_sla_row, axis=1)
    st.dataframe(styled_sla_df, hide_index=True, width='stretch')

    # ── Download SLA report ───────────────────────────────────
    sla_csv_download = sla_summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download SLA Report (CSV)",
        data=sla_csv_download,
        file_name=f"sla_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.divider()

    # ── Per-table drill-down ──────────────────────────────────
    st.subheader("🔎 Drill-down: Gap Analysis per Table")

    all_monitored_tables = sla_summary_df["Table Name"].tolist()
    selected_drill_table = st.selectbox(
        "Select a table to inspect its upload gap timeline:",
        all_monitored_tables
    )

    if selected_drill_table:
        # Summary row for selected table
        selected_table_row = sla_summary_df[sla_summary_df["Table Name"] == selected_drill_table].iloc[0]

        drill_col1, drill_col2, drill_col3, drill_col4 = st.columns(4)
        with drill_col1:
            st.metric("Total Loads", selected_table_row["Total Loads"])
        with drill_col2:
            st.metric("Last Load", selected_table_row["Last Load Time"])
        with drill_col3:
            st.metric("Max Gap (Biz Mins)", selected_table_row["Max Gap (Biz Mins)"])
        with drill_col4:
            st.metric("Current Status", selected_table_row["Current Status"])

        # Gap timeline chart
        gap_chart = build_gap_timeline_chart(raw_audit_df, selected_drill_table)

        if gap_chart:
            st.markdown(
                f"**Upload Gap Timeline for `{selected_drill_table}`** "
                f"— Orange dashed line = {SLA_THRESHOLD_MINUTES}-min SLA threshold"
            )
            st.altair_chart(gap_chart, width='stretch')
        else:
            st.info("Not enough data points (need at least 2 loads) to draw a gap chart.")

        # Raw upload history for selected table
        with st.expander(f"📋 Raw upload history for {selected_drill_table}"):
            table_raw_history = (
                raw_audit_df[raw_audit_df["TABLE_NAME"] == selected_drill_table]
                .sort_values("UPLOAD_TIME", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(table_raw_history, hide_index=True)

    st.divider()

    # ── Breached tables alert section ─────────────────────────
    breached_table_list = sla_summary_df[
        sla_summary_df["Current Status"] == "🔴 BREACHED"
    ]["Table Name"].tolist()

    if breached_table_list:
        st.error(
            f"🚨 **{len(breached_table_list)} table(s) currently breaching SLA!**\n\n"
            + "\n".join([f"  • `{t}`" for t in breached_table_list])
        )
    elif tables_currently_warn > 0:
        warning_table_list = sla_summary_df[
            sla_summary_df["Current Status"] == "🟡 WARNING"
        ]["Table Name"].tolist()
        st.warning(
            f"⚠️ **{len(warning_table_list)} table(s) approaching SLA threshold:**\n\n"
            + "\n".join([f"  • `{t}`" for t in warning_table_list])
        )
    else:
        st.success("✅ All tables are within SLA during current business hours.")


# -------------------------------------------------
# AUDIT LOG
# -------------------------------------------------
with tabs[5]:
    st.subheader("Upload Audit Log")

    try:
        cur = conn.cursor()
        audit_cols = get_table_columns('UPLOAD_AUDIT')

        if 'OPERATION_TYPE' in audit_cols and 'UPLOADED_BY' in audit_cols:
            cur.execute("""
                SELECT FILE_NAME, TABLE_NAME, ROW_COUNT, OPERATION_TYPE, UPLOAD_TIME, UPLOADED_BY
                FROM UPLOAD_AUDIT ORDER BY UPLOAD_TIME DESC LIMIT 50
            """)
            columns = ["File", "Table", "Rows", "Operation", "Timestamp", "User"]
        elif 'OPERATION_TYPE' in audit_cols:
            cur.execute("""
                SELECT FILE_NAME, TABLE_NAME, ROW_COUNT, OPERATION_TYPE, UPLOAD_TIME
                FROM UPLOAD_AUDIT ORDER BY UPLOAD_TIME DESC LIMIT 50
            """)
            columns = ["File", "Table", "Rows", "Operation", "Timestamp"]
        else:
            cur.execute("""
                SELECT FILE_NAME, TABLE_NAME, ROW_COUNT, UPLOAD_TIME
                FROM UPLOAD_AUDIT ORDER BY UPLOAD_TIME DESC LIMIT 50
            """)
            columns = ["File", "Table", "Rows", "Timestamp"]

        audit_data = cur.fetchall()
        cur.close()

        if audit_data:
            audit_df = pd.DataFrame(audit_data, columns=columns)

            # Convert Timestamp column UTC → IST for correct display
            if 'Timestamp' in audit_df.columns:
                audit_df['Timestamp'] = pd.to_datetime(audit_df['Timestamp']).apply(to_ist)
                audit_df['Timestamp'] = audit_df['Timestamp'].apply(
                    lambda t: t.strftime("%Y-%m-%d %H:%M:%S") if t else ""
                )

            st.dataframe(audit_df, hide_index=True)

            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Uploads", len(audit_df))
            with col2:
                st.metric("Total Rows Uploaded", int(audit_df['Rows'].sum()))
            with col3:
                st.metric("Tables Affected", audit_df['Table'].nunique())

            with st.expander("Upload Trend"):
                try:
                    audit_df['Date'] = pd.to_datetime(audit_df['Timestamp']).dt.date
                    daily_uploads = audit_df.groupby('Date').agg({'Rows': 'sum', 'File': 'count'}).reset_index()
                    daily_uploads.columns = ['Date', 'Total Rows', 'Upload Count']
                    st.line_chart(daily_uploads.set_index('Date'))
                except Exception as e:
                    st.error(f"Error creating trend chart: {str(e)}")
        else:
            st.info("No audit records found")

    except Exception as e:
        st.error(f"Error fetching audit log: {str(e)}")
        with st.expander("View detailed error"):
            st.code(traceback.format_exc())

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption("Enterprise Snowflake Ingestion Platform | Secure & Audited Data Loading")
