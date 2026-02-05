
import streamlit as st
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from datetime import datetime
import altair as alt
from config import SNOWFLAKE_DB_CONFIG
import traceback
import re

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Enterprise Snowflake Ingestion", layout="wide")

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
# HELPERS
# -------------------------------------------------
def normalize_columns(df):
    return [c.upper().replace(" ", "_") for c in df.columns]

def sanitize_table_name(name):
    name = re.sub(r'\.(csv|xlsx)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if name and not name[0].isalpha() and name[0] != '_':
        name = 'TBL_' + name
    return name.upper()

def get_tables():
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
    try:
        cur = conn.cursor()
        type_mapping = {
            'int64': 'INTEGER',
            'int32': 'INTEGER',
            'float64': 'FLOAT',
            'float32': 'FLOAT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'object': 'STRING'
        }
        columns_def = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            snowflake_type = type_mapping.get(dtype, 'STRING')
            columns_def.append(f"{col} {snowflake_type}")
        create_sql = f"""
        CREATE TABLE {table_name} (
            {', '.join(columns_def)}
        )
        """
        cur.execute(create_sql)
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        st.error(f"Error creating table: {str(e)}")
        return False

def create_audit_table():
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'UPLOAD_AUDIT'
            AND table_schema = CURRENT_SCHEMA()
        """)
        table_exists = cur.fetchone()[0] > 0
        if table_exists:
            existing_cols = get_table_columns('UPLOAD_AUDIT')
            if 'OPERATION_TYPE' not in existing_cols:
                cur.execute("""ALTER TABLE UPLOAD_AUDIT ADD COLUMN OPERATION_TYPE STRING""")
                conn.commit()
            if 'UPLOADED_BY' not in existing_cols:
                cur.execute("""ALTER TABLE UPLOAD_AUDIT ADD COLUMN UPLOADED_BY STRING""")
                conn.commit()
        else:
            cur.execute("""
                CREATE TABLE UPLOAD_AUDIT (
                    FILE_NAME STRING,
                    TABLE_NAME STRING,
                    ROW_COUNT INT,
                    OPERATION_TYPE STRING,
                    UPLOAD_TIME TIMESTAMP,
                    UPLOADED_BY STRING
                )
            """)
            conn.commit()
        cur.close()
    except Exception as e:
        st.error(f"Error creating/updating audit table: {str(e)}")

def log_audit(file, table, rows, operation="MERGE"):
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO UPLOAD_AUDIT (FILE_NAME, TABLE_NAME, ROW_COUNT, OPERATION_TYPE, UPLOAD_TIME, UPLOADED_BY)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP(), %s)
        """, (file, table, rows, operation, ROLE))
        conn.commit()
        cur.close()
    except Exception as e:
        st.error(f"Error logging audit: {str(e)}")

def validate_table_exists(table_name):
    tables = get_tables()
    return table_name in tables

def check_schema_compatibility(df_columns, table_columns):
    missing_in_table = [col for col in df_columns if col not in table_columns]
    missing_in_file = [col for col in table_columns if col not in df_columns]
    return missing_in_table, missing_in_file

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

if 'valid_df' not in st.session_state:
    st.session_state.valid_df = df.copy()
if 'pk_column' not in st.session_state:
    st.session_state.pk_column = df.columns[0] if len(df.columns) > 0 else None

tabs = st.tabs([
    "📄 Preview",
    "🚦 Validation",
    "📈 Profiling",
    "🔁 Load / Merge",
    "🧾 Audit"
])

# -------------------- Preview --------------------
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
        dtypes_df = pd.DataFrame({'Column': df.columns,'Data Type': [str(dtype) for dtype in df.dtypes]})
        st.dataframe(dtypes_df, hide_index=True)

# -------------------- Validation --------------------
with tabs[1]:
    st.subheader("Row-Level Validation")
    pk = st.selectbox("Select Primary Key", df.columns, key='pk_selector')
    st.session_state.pk_column = pk
    null_rows = df[df[pk].isnull()]
    dup_rows = df[df.duplicated(subset=[pk], keep=False)]
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
        st.metric("Valid Rows", len(valid_df), delta=None)
    with col3:
        st.metric("Invalid Rows", invalid_count, delta=f"-{invalid_count}")
    if invalid_count > 0:
        st.warning(f"⚠️ {invalid_count} invalid rows will be excluded from upload")
    else:
        st.success("✅ All rows passed validation")

# -------------------- Profiling --------------------
with tabs[2]:
    st.subheader("Column Profiling")
    col = st.selectbox("Select column to profile", df.columns)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
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
        except Exception as e:
            st.bar_chart(top_vals.set_index(col))

# -------------------- Load / Merge --------------------
with tabs[3]:
    st.subheader("UPSERT / MERGE")

    valid_df = st.session_state.valid_df
    pk = st.session_state.pk_column

    if valid_df.empty:
        st.error("No valid data to upload. Please check the Validation tab.")
        st.stop()

    st.info(f"📊 Ready to process {len(valid_df)} valid rows using **{pk}** as primary key")

    available_tables = get_tables()
    table_mode = st.radio(
        "Select Table Mode:",
        ["Use Existing Table", "Create New Table", "Do Not Upload"],
        horizontal=True
    )

    if table_mode == "Do Not Upload":
        st.info("⚠️ Data will not be uploaded or merged. This is view-only mode.")
        st.stop()

    if table_mode != "Do Not Upload" and ROLE != "ADMIN":
        st.warning("⚠️ Only ADMIN role can perform table creation or merge operations")
        st.stop()

    # ---------- Table Handling ----------
    target_table = None
    if table_mode == "Use Existing Table":
        if not available_tables:
            st.error("No existing tables found. Please choose 'Create New Table'.")
            st.stop()
        target_table = st.selectbox("Select Target Table", available_tables)
        if target_table:
            target_cols = get_schema(target_table)
            if pk not in target_cols:
                st.error(f"❌ Primary key '{pk}' does not exist in target table {target_table}")
                st.stop()
            missing_cols, extra_cols = check_schema_compatibility(valid_df.columns, target_cols)
            if missing_cols:
                st.warning(f"⚠️ Columns in file but not in table: {', '.join(missing_cols)}")
                valid_df = valid_df[[col for col in valid_df.columns if col in target_cols]]
                st.session_state.valid_df = valid_df
                if valid_df.empty:
                    st.error("No matching columns to upload. Please create a new table.")
                    st.stop()
    else:
        # Create New Table
        suggested_name = sanitize_table_name(uploaded_file.name)
        target_table = st.text_input("New Table Name", value=suggested_name)
        target_table = sanitize_table_name(target_table)
        if target_table in available_tables:
            st.error(f"Table '{target_table}' already exists. Choose another name.")
            st.stop()
        st.subheader("Table Structure Preview")
        type_mapping = {
            'int64': 'INTEGER',
            'int32': 'INTEGER',
            'float64': 'FLOAT',
            'float32': 'FLOAT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'object': 'STRING'
        }
        structure_data = []
        for col in valid_df.columns:
            dtype = str(valid_df[col].dtype)
            structure_data.append({
                'Column Name': col,
                'Snowflake Type': type_mapping.get(dtype, 'STRING'),
                'Primary Key': '🔑' if col == pk else ''
            })
        st.dataframe(pd.DataFrame(structure_data), hide_index=True)

    st.divider()
    confirm = st.checkbox(f"I confirm this {'TABLE CREATION and DATA LOAD' if table_mode=='Create New Table' else 'MERGE'} operation")

    if st.button("🚀 Execute Operation", type="primary", disabled=not confirm):
        try:
            cur = conn.cursor()
            if table_mode == "Create New Table":
                progress = st.progress(0, text="Creating new table...")
                if not create_table_from_dataframe(target_table, valid_df):
                    st.error("Failed to create table")
                    st.stop()
                progress.progress(40, text="Loading data to new table...")
                success, nchunks, nrows, _ = write_pandas(conn, valid_df, target_table)
                if not success:
                    st.error("Failed to load data")
                    st.stop()
                progress.progress(80, text="Logging audit trail...")
                log_audit(uploaded_file.name, target_table, len(valid_df), "CREATE + INSERT")
                progress.progress(100, text="Complete!")
                st.success(f"✅ Table '{target_table}' created and {len(valid_df)} rows inserted successfully!")
                st.balloons()
            else:
                # MERGE
                stage_table = f"{target_table}_STAGE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                progress = st.progress(0, text="Creating staging table...")
                cur.execute(f"DROP TABLE IF EXISTS {stage_table}")
                cur.execute(f"CREATE TEMPORARY TABLE {stage_table} LIKE {target_table}")
                progress.progress(25, text="Loading data to staging table...")
                success, nchunks, nrows, _ = write_pandas(conn, valid_df, stage_table, overwrite=True)
                if not success:
                    st.error("Failed to load staging table")
                    st.stop()
                progress.progress(50, text="Executing MERGE...")
                update_cols = [col for col in valid_df.columns if col != pk]
                when_matched = f"WHEN MATCHED THEN UPDATE SET {', '.join([f'T.{col}=S.{col}' for col in update_cols])}" if update_cols else ""
                insert_cols = ", ".join(valid_df.columns)
                insert_vals = ", ".join([f"S.{col}" for col in valid_df.columns])
                merge_sql = f"""
                MERGE INTO {target_table} T
                USING {stage_table} S
                ON T.{pk}=S.{pk}
                {when_matched}
                WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
                """
                cur.execute(merge_sql)
                rows_updated = cur.rowcount if cur.rowcount else 0
                rows_inserted = len(valid_df) - rows_updated
                progress.progress(80, text="Logging audit...")
                log_audit(uploaded_file.name, target_table, len(valid_df), "MERGE")
                cur.execute(f"DROP TABLE IF EXISTS {stage_table}")
                conn.commit()
                progress.progress(100, text="Complete!")
                st.success(f"✅ MERGE completed: {rows_inserted} inserted, {rows_updated} updated.")
                st.balloons()
            cur.close()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.code(traceback.format_exc())
            try:
                conn.rollback()
            except:
                pass

# # -------------------- Audit --------------------
# with tabs[4]:
#     st.subheader("Upload Audit Log")
#     try:
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT FILE_NAME, TABLE_NAME, ROW_COUNT, OPERATION_TYPE, UPLOAD_TIME, UPLOADED_BY
#             FROM UPLOAD_AUDIT
#             ORDER BY UPLOAD_TIME DESC
#             LIMIT 50
#         """)
#         audit_df = pd.DataFrame(cur.fetchall(), columns=["File","Table","Rows","Operation","Timestamp","User"])
#         st.dataframe(audit_df)
#         cur.close()
#     except Exception as e:
#         st.error(f"Error fetching audit: {str(e)}")


# -------------------------------------------------
# AUDIT LOG
# -------------------------------------------------
with tabs[4]:
    st.subheader("Upload Audit Log")

    try:
        cur = conn.cursor()
        
        # Check which columns exist in the audit table
        audit_cols = get_table_columns('UPLOAD_AUDIT')
        
        # Build query based on available columns
        if 'OPERATION_TYPE' in audit_cols and 'UPLOADED_BY' in audit_cols:
            cur.execute("""
                SELECT 
                    FILE_NAME,
                    TABLE_NAME,
                    ROW_COUNT,
                    OPERATION_TYPE,
                    UPLOAD_TIME,
                    UPLOADED_BY
                FROM UPLOAD_AUDIT
                ORDER BY UPLOAD_TIME DESC
                LIMIT 50
            """)
            columns = ["File", "Table", "Rows", "Operation", "Timestamp", "User"]
        elif 'OPERATION_TYPE' in audit_cols:
            cur.execute("""
                SELECT 
                    FILE_NAME,
                    TABLE_NAME,
                    ROW_COUNT,
                    OPERATION_TYPE,
                    UPLOAD_TIME
                FROM UPLOAD_AUDIT
                ORDER BY UPLOAD_TIME DESC
                LIMIT 50
            """)
            columns = ["File", "Table", "Rows", "Operation", "Timestamp"]
        else:
            cur.execute("""
                SELECT 
                    FILE_NAME,
                    TABLE_NAME,
                    ROW_COUNT,
                    UPLOAD_TIME
                FROM UPLOAD_AUDIT
                ORDER BY UPLOAD_TIME DESC
                LIMIT 50
            """)
            columns = ["File", "Table", "Rows", "Timestamp"]
        
        audit_data = cur.fetchall()
        cur.close()
        
        if audit_data:
            audit_df = pd.DataFrame(audit_data, columns=columns)
            
            st.dataframe(audit_df, hide_index=True)
            
            # Summary metrics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Uploads", len(audit_df))
            with col2:
                st.metric("Total Rows Uploaded", int(audit_df['Rows'].sum()))
            with col3:
                st.metric("Tables Affected", audit_df['Table'].nunique())
            
            # Upload trend
            with st.expander("Upload Trend"):
                try:
                    audit_df['Date'] = pd.to_datetime(audit_df['Timestamp']).dt.date
                    daily_uploads = audit_df.groupby('Date').agg({
                        'Rows': 'sum',
                        'File': 'count'
                    }).reset_index()
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