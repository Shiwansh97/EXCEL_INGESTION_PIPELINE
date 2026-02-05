



# import streamlit as st
# import pandas as pd
# import os
# import snowflake.connector
# from snowflake.connector.pandas_tools import write_pandas
# from datetime import datetime
# import altair as alt
# from config import SNOWFLAKE_DB_CONFIG
# import traceback

# # -------------------------------------------------
# # CONFIG
# # -------------------------------------------------
# st.set_page_config(page_title="Enterprise Snowflake Ingestion", layout="wide")

# # -------------------------------------------------
# # AUTH / ROLE
# # -------------------------------------------------
# st.sidebar.header("🔐 Access Control")
# ROLE = st.sidebar.selectbox("Select Role", ["ANALYST", "ADMIN"])

# # -------------------------------------------------
# # SNOWFLAKE CONNECTION
# # -------------------------------------------------
# @st.cache_resource
# def get_conn():
#     try:
#         return snowflake.connector.connect(**SNOWFLAKE_DB_CONFIG)
#     except Exception as e:
#         st.error(f"Failed to connect to Snowflake: {str(e)}")
#         st.stop()

# conn = get_conn()

# # -------------------------------------------------
# # HELPERS
# # -------------------------------------------------
# def normalize_columns(df):
#     """Normalize column names to uppercase with underscores"""
#     return [c.upper().replace(" ", "_") for c in df.columns]

# def get_tables():
#     """Fetch all tables in current schema"""
#     try:
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT table_name
#             FROM information_schema.tables
#             WHERE table_schema = CURRENT_SCHEMA()
#             ORDER BY table_name
#         """)
#         tables = [r[0] for r in cur.fetchall()]
#         cur.close()
#         return tables
#     except Exception as e:
#         st.error(f"Error fetching tables: {str(e)}")
#         return []

# def get_schema(table):
#     """Fetch column names for a given table - SQL injection safe"""
#     try:
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT column_name
#             FROM information_schema.columns
#             WHERE table_name = %s
#             AND table_schema = CURRENT_SCHEMA()
#             ORDER BY ordinal_position
#         """, (table,))
#         columns = [r[0] for r in cur.fetchall()]
#         cur.close()
#         return columns
#     except Exception as e:
#         st.error(f"Error fetching schema for {table}: {str(e)}")
#         return []

# def get_table_columns(table_name):
#     """Get columns for a specific table"""
#     try:
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT column_name
#             FROM information_schema.columns
#             WHERE table_name = %s
#             AND table_schema = CURRENT_SCHEMA()
#         """, (table_name,))
#         columns = [r[0] for r in cur.fetchall()]
#         cur.close()
#         return columns
#     except Exception as e:
#         return []

# def create_audit_table():
#     """Create audit table if it doesn't exist with proper schema"""
#     try:
#         cur = conn.cursor()
        
#         # Check if table exists
#         cur.execute("""
#             SELECT COUNT(*)
#             FROM information_schema.tables
#             WHERE table_name = 'UPLOAD_AUDIT'
#             AND table_schema = CURRENT_SCHEMA()
#         """)
#         table_exists = cur.fetchone()[0] > 0
        
#         if table_exists:
#             # Get existing columns
#             existing_cols = get_table_columns('UPLOAD_AUDIT')
            
#             # Add missing columns if needed
#             if 'OPERATION_TYPE' not in existing_cols:
#                 cur.execute("""
#                     ALTER TABLE UPLOAD_AUDIT
#                     ADD COLUMN OPERATION_TYPE STRING
#                 """)
#                 conn.commit()
            
#             if 'UPLOADED_BY' not in existing_cols:
#                 cur.execute("""
#                     ALTER TABLE UPLOAD_AUDIT
#                     ADD COLUMN UPLOADED_BY STRING
#                 """)
#                 conn.commit()
#         else:
#             # Create new table with full schema
#             cur.execute("""
#                 CREATE TABLE UPLOAD_AUDIT (
#                     FILE_NAME STRING,
#                     TABLE_NAME STRING,
#                     ROW_COUNT INT,
#                     OPERATION_TYPE STRING,
#                     UPLOAD_TIME TIMESTAMP,
#                     UPLOADED_BY STRING
#                 )
#             """)
#             conn.commit()
        
#         cur.close()
#     except Exception as e:
#         st.error(f"Error creating/updating audit table: {str(e)}")

# def log_audit(file, table, rows, operation="MERGE"):
#     """Log upload audit - SQL injection safe"""
#     try:
#         cur = conn.cursor()
#         cur.execute("""
#             INSERT INTO UPLOAD_AUDIT (FILE_NAME, TABLE_NAME, ROW_COUNT, OPERATION_TYPE, UPLOAD_TIME, UPLOADED_BY)
#             VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP(), %s)
#         """, (file, table, rows, operation, ROLE))
#         conn.commit()
#         cur.close()
#     except Exception as e:
#         st.error(f"Error logging audit: {str(e)}")

# def validate_table_exists(table_name):
#     """Check if table exists"""
#     tables = get_tables()
#     return table_name in tables

# # -------------------------------------------------
# # INITIALIZE
# # -------------------------------------------------
# create_audit_table()

# # -------------------------------------------------
# # UI
# # -------------------------------------------------
# st.title("🏢 Enterprise Snowflake Ingestion Platform")

# uploaded_file = st.sidebar.file_uploader(
#     "Upload CSV / Excel",
#     ["csv", "xlsx"]
# )

# if not uploaded_file:
#     st.info("👈 Please upload a CSV or Excel file to begin")
#     st.stop()

# # -------------------------------------------------
# # READ FILE
# # -------------------------------------------------
# try:
#     if uploaded_file.name.endswith(".csv"):
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_excel(uploaded_file)
    
#     df.columns = normalize_columns(df)
    
#     if df.empty:
#         st.error("The uploaded file is empty")
#         st.stop()
        
# except Exception as e:
#     st.error(f"Error reading file: {str(e)}")
#     st.stop()

# # Initialize session state for validated data
# if 'valid_df' not in st.session_state:
#     st.session_state.valid_df = df.copy()
# if 'pk_column' not in st.session_state:
#     st.session_state.pk_column = df.columns[0] if len(df.columns) > 0 else None

# tabs = st.tabs([
#     "📄 Preview",
#     "🚦 Validation",
#     "📈 Profiling",
#     "🔁 Load / Merge",
#     "🧾 Audit"
# ])

# # -------------------------------------------------
# # PREVIEW
# # -------------------------------------------------
# with tabs[0]:
#     st.subheader("Data Preview")
#     st.dataframe(df.head(100))
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Rows", len(df))
#     with col2:
#         st.metric("Columns", len(df.columns))
#     with col3:
#         st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
#     with st.expander("Column Details"):
#         # Fix: Convert dtypes to string to avoid Arrow serialization error
#         dtypes_df = pd.DataFrame({
#             'Column': df.columns,
#             'Data Type': [str(dtype) for dtype in df.dtypes]
#         })
#         st.dataframe(dtypes_df, hide_index=True)

# # -------------------------------------------------
# # ROW-LEVEL VALIDATION
# # -------------------------------------------------
# with tabs[1]:
#     st.subheader("Row-Level Validation")

#     pk = st.selectbox("Select Primary Key", df.columns, key='pk_selector')
#     st.session_state.pk_column = pk

#     # Check for null values in primary key
#     null_rows = df[df[pk].isnull()]
    
#     # Check for duplicates in primary key
#     dup_rows = df[df.duplicated(subset=[pk], keep=False)]

#     if not null_rows.empty:
#         st.error(f"❌ Found {len(null_rows)} rows with NULL Primary Key")
#         with st.expander("View NULL rows"):
#             st.dataframe(null_rows)

#     if not dup_rows.empty:
#         st.error(f"❌ Found {len(dup_rows)} rows with Duplicate Primary Key")
#         with st.expander("View duplicate rows"):
#             st.dataframe(dup_rows)

#     # Create valid dataset
#     valid_df = df.dropna(subset=[pk]).drop_duplicates(subset=[pk])
#     st.session_state.valid_df = valid_df
    
#     invalid_count = len(df) - len(valid_df)
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Rows", len(df))
#     with col2:
#         st.metric("Valid Rows", len(valid_df), delta=None)
#     with col3:
#         st.metric("Invalid Rows", invalid_count, delta=f"-{invalid_count}")

#     if invalid_count > 0:
#         st.warning(f"⚠️ {invalid_count} invalid rows will be excluded from upload")
#     else:
#         st.success("✅ All rows passed validation")

# # -------------------------------------------------
# # COLUMN PROFILING
# # -------------------------------------------------
# with tabs[2]:
#     st.subheader("Column Profiling")

#     col = st.selectbox("Select column to profile", df.columns)

#     # Basic statistics
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         null_count = df[col].isnull().sum()
#         null_pct = (null_count / len(df)) * 100
#         st.metric("Null Count", null_count)
    
#     with col2:
#         st.metric("Null %", f"{null_pct:.2f}%")
    
#     with col3:
#         unique_count = df[col].nunique()
#         st.metric("Unique Values", unique_count)
    
#     with col4:
#         unique_pct = (unique_count / len(df)) * 100
#         st.metric("Unique %", f"{unique_pct:.2f}%")

#     # Distribution visualization
#     st.subheader("Distribution")
    
#     if pd.api.types.is_numeric_dtype(df[col]):
#         # Numeric column
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.write("**Statistics**")
#             stats = df[col].describe()
#             st.dataframe(stats.to_frame(name='Value'))
        
#         with col2:
#             st.write("**Histogram**")
#             try:
#                 chart = alt.Chart(df.dropna(subset=[col])).mark_bar().encode(
#                     alt.X(col, bin=alt.Bin(maxbins=30), title=col),
#                     alt.Y('count()', title='Frequency')
#                 ).properties(height=300)
#                 st.altair_chart(chart, width='stretch')
#             except Exception as e:
#                 st.error(f"Error creating chart: {str(e)}")
#     else:
#         # Categorical column
#         st.write("**Top 10 Values**")
#         top_vals = df[col].value_counts().head(10).reset_index()
#         top_vals.columns = [col, "COUNT"]
        
#         try:
#             chart = alt.Chart(top_vals).mark_bar().encode(
#                 x=alt.X('COUNT:Q'),
#                 y=alt.Y(f'{col}:N', sort='-x')
#             ).properties(height=300)
#             st.altair_chart(chart, width='stretch')
#         except Exception as e:
#             st.bar_chart(top_vals.set_index(col))

# # -------------------------------------------------
# # UPSERT / MERGE
# # -------------------------------------------------
# with tabs[3]:
#     st.subheader("UPSERT / MERGE")

#     if ROLE != "ADMIN":
#         st.warning("⚠️ Only ADMIN role can load data to Snowflake")
#         st.info("Please select ADMIN role from the sidebar to proceed")
#         st.stop()

#     # Get valid data from session state
#     valid_df = st.session_state.valid_df
#     pk = st.session_state.pk_column

#     if valid_df.empty:
#         st.error("No valid data to upload. Please check the Validation tab.")
#         st.stop()

#     st.info(f"📊 Ready to upload {len(valid_df)} valid rows using **{pk}** as primary key")

#     # Select target table
#     available_tables = get_tables()
    
#     if not available_tables:
#         st.error("No tables found in the current schema")
#         st.stop()
    
#     target_table = st.selectbox("Target Table", available_tables)

#     # Validate schema compatibility
#     if target_table:
#         target_cols = get_schema(target_table)
        
#         if not target_cols:
#             st.error(f"Could not fetch schema for {target_table}")
#             st.stop()
        
#         # Check if primary key exists in target
#         if pk not in target_cols:
#             st.error(f"❌ Primary key '{pk}' does not exist in target table {target_table}")
#             st.stop()
        
#         # Check column compatibility
#         missing_cols = [col for col in valid_df.columns if col not in target_cols]
#         extra_cols = [col for col in target_cols if col not in valid_df.columns]
        
#         if missing_cols:
#             st.warning(f"⚠️ Columns in file but not in table: {', '.join(missing_cols)}")
#             st.info("These columns will be ignored during upload")
#             # Filter out missing columns
#             valid_df = valid_df[[col for col in valid_df.columns if col in target_cols]]
#             st.session_state.valid_df = valid_df
        
#         if extra_cols:
#             st.info(f"ℹ️ Columns in table but not in file: {', '.join(extra_cols)}")
#             st.info("These columns will use default/NULL values for new rows")
        
#         # Show column mapping
#         with st.expander("View Column Mapping"):
#             mapping_df = pd.DataFrame({
#                 'File Columns': valid_df.columns.tolist(),
#                 'Target Columns': valid_df.columns.tolist(),
#                 'Status': ['✅ Match' for _ in valid_df.columns]
#             })
#             st.dataframe(mapping_df, hide_index=True)

#     st.divider()
    
#     st.write("**Operation Details:**")
#     st.write(f"- **Mode:** MERGE (INSERT new records, UPDATE existing records)")
#     st.write(f"- **Primary Key:** {pk}")
#     st.write(f"- **Rows to process:** {len(valid_df)}")
    
#     # Confirmation
#     confirm = st.checkbox("I confirm this MERGE operation")
    
#     if st.button("🚀 Execute MERGE", type="primary", disabled=not confirm):
#         with st.spinner("Processing MERGE operation..."):
#             try:
#                 cur = conn.cursor()
                
#                 # Create staging table
#                 stage_table = f"{target_table}_STAGE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
#                 progress = st.progress(0, text="Creating staging table...")
                
#                 # Drop if exists
#                 cur.execute(f"DROP TABLE IF EXISTS {stage_table}")
                
#                 # Create staging table with same structure
#                 cur.execute(f"CREATE TEMPORARY TABLE {stage_table} LIKE {target_table}")
                
#                 progress.progress(25, text="Loading data to staging table...")
                
#                 # Load data to staging table
#                 success, nchunks, nrows, _ = write_pandas(
#                     conn, 
#                     valid_df, 
#                     stage_table,
#                     auto_create_table=False,
#                     overwrite=True
#                 )
                
#                 if not success:
#                     raise Exception("Failed to write data to staging table")
                
#                 progress.progress(50, text="Calculating merge statistics...")
                
#                 # Get count before merge for calculating updates
#                 cur.execute(f"""
#                     SELECT COUNT(*)
#                     FROM {target_table} T
#                     INNER JOIN {stage_table} S
#                     ON T.{pk} = S.{pk}
#                 """)
#                 result = cur.fetchone()
#                 rows_to_update = result[0] if result else 0
                
#                 progress.progress(60, text="Executing MERGE...")
                
#                 # Build MERGE statement with explicit column matching
#                 update_cols = [col for col in valid_df.columns if col != pk]
                
#                 if update_cols:
#                     update_set = ", ".join([f"T.{col} = S.{col}" for col in update_cols])
#                     when_matched = f"WHEN MATCHED THEN UPDATE SET {update_set}"
#                 else:
#                     when_matched = ""
                
#                 insert_cols = ", ".join(valid_df.columns)
#                 insert_vals = ", ".join([f"S.{col}" for col in valid_df.columns])
                
#                 merge_sql = f"""
#                 MERGE INTO {target_table} T
#                 USING {stage_table} S
#                 ON T.{pk} = S.{pk}
#                 {when_matched}
#                 WHEN NOT MATCHED THEN 
#                     INSERT ({insert_cols})
#                     VALUES ({insert_vals})
#                 """
                
#                 # Execute MERGE
#                 cur.execute(merge_sql)
                
#                 # Calculate rows inserted
#                 rows_inserted = len(valid_df) - rows_to_update
#                 rows_updated = rows_to_update
                
#                 progress.progress(80, text="Logging audit trail...")
                
#                 # Log audit
#                 log_audit(uploaded_file.name, target_table, len(valid_df), "MERGE")
                
#                 progress.progress(90, text="Cleaning up...")
                
#                 # Cleanup staging table
#                 cur.execute(f"DROP TABLE IF EXISTS {stage_table}")
                
#                 conn.commit()
#                 cur.close()
                
#                 progress.progress(100, text="Complete!")
#                 progress.empty()
                
#                 st.success("✅ MERGE completed successfully!")
                
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Rows Processed", len(valid_df))
#                 with col2:
#                     st.metric("Rows Inserted", rows_inserted)
#                 with col3:
#                     st.metric("Rows Updated", rows_updated)
                
#                 st.balloons()
                
#             except Exception as e:
#                 st.error(f"❌ Error during MERGE operation: {str(e)}")
#                 with st.expander("View detailed error"):
#                     st.code(traceback.format_exc())
#                 try:
#                     conn.rollback()
#                     st.info("Transaction rolled back successfully")
#                 except Exception as rollback_error:
#                     st.error(f"Rollback failed: {str(rollback_error)}")
#             finally:
#                 try:
#                     cur.close()
#                 except:
#                     pass

# # -------------------------------------------------
# # AUDIT LOG
# # -------------------------------------------------
# with tabs[4]:
#     st.subheader("Upload Audit Log")

#     try:
#         cur = conn.cursor()
        
#         # Check which columns exist in the audit table
#         audit_cols = get_table_columns('UPLOAD_AUDIT')
        
#         # Build query based on available columns
#         if 'OPERATION_TYPE' in audit_cols and 'UPLOADED_BY' in audit_cols:
#             cur.execute("""
#                 SELECT 
#                     FILE_NAME,
#                     TABLE_NAME,
#                     ROW_COUNT,
#                     OPERATION_TYPE,
#                     UPLOAD_TIME,
#                     UPLOADED_BY
#                 FROM UPLOAD_AUDIT
#                 ORDER BY UPLOAD_TIME DESC
#                 LIMIT 50
#             """)
#             columns = ["File", "Table", "Rows", "Operation", "Timestamp", "User"]
#         elif 'OPERATION_TYPE' in audit_cols:
#             cur.execute("""
#                 SELECT 
#                     FILE_NAME,
#                     TABLE_NAME,
#                     ROW_COUNT,
#                     OPERATION_TYPE,
#                     UPLOAD_TIME
#                 FROM UPLOAD_AUDIT
#                 ORDER BY UPLOAD_TIME DESC
#                 LIMIT 50
#             """)
#             columns = ["File", "Table", "Rows", "Operation", "Timestamp"]
#         else:
#             cur.execute("""
#                 SELECT 
#                     FILE_NAME,
#                     TABLE_NAME,
#                     ROW_COUNT,
#                     UPLOAD_TIME
#                 FROM UPLOAD_AUDIT
#                 ORDER BY UPLOAD_TIME DESC
#                 LIMIT 50
#             """)
#             columns = ["File", "Table", "Rows", "Timestamp"]
        
#         audit_data = cur.fetchall()
#         cur.close()
        
#         if audit_data:
#             audit_df = pd.DataFrame(audit_data, columns=columns)
            
#             st.dataframe(audit_df, hide_index=True)
            
#             # Summary metrics
#             st.subheader("Summary Statistics")
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 st.metric("Total Uploads", len(audit_df))
#             with col2:
#                 st.metric("Total Rows Uploaded", int(audit_df['Rows'].sum()))
#             with col3:
#                 st.metric("Tables Affected", audit_df['Table'].nunique())
            
#             # Upload trend
#             with st.expander("Upload Trend"):
#                 try:
#                     audit_df['Date'] = pd.to_datetime(audit_df['Timestamp']).dt.date
#                     daily_uploads = audit_df.groupby('Date').agg({
#                         'Rows': 'sum',
#                         'File': 'count'
#                     }).reset_index()
#                     daily_uploads.columns = ['Date', 'Total Rows', 'Upload Count']
                    
#                     st.line_chart(daily_uploads.set_index('Date'))
#                 except Exception as e:
#                     st.error(f"Error creating trend chart: {str(e)}")
#         else:
#             st.info("No audit records found")
            
#     except Exception as e:
#         st.error(f"Error fetching audit log: {str(e)}")
#         with st.expander("View detailed error"):
#             st.code(traceback.format_exc())

# # -------------------------------------------------
# # FOOTER
# # -------------------------------------------------
# st.divider()
# st.caption("Enterprise Snowflake Ingestion Platform | Secure & Audited Data Loading")




import streamlit as st
import pandas as pd
import os
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
    """Normalize column names to uppercase with underscores"""
    return [c.upper().replace(" ", "_") for c in df.columns]

def sanitize_table_name(name):
    """Sanitize table name to be valid for Snowflake"""
    # Remove extension if present
    name = re.sub(r'\.(csv|xlsx)$', '', name, flags=re.IGNORECASE)
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it starts with a letter or underscore
    if name and not name[0].isalpha() and name[0] != '_':
        name = 'TBL_' + name
    # Convert to uppercase
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
        
        # Map pandas dtypes to Snowflake types
        type_mapping = {
            'int64': 'INTEGER',
            'int32': 'INTEGER',
            'float64': 'FLOAT',
            'float32': 'FLOAT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'object': 'STRING'
        }
        
        # Build CREATE TABLE statement
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
    """Create audit table if it doesn't exist with proper schema"""
    try:
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'UPLOAD_AUDIT'
            AND table_schema = CURRENT_SCHEMA()
        """)
        table_exists = cur.fetchone()[0] > 0
        
        if table_exists:
            # Get existing columns
            existing_cols = get_table_columns('UPLOAD_AUDIT')
            
            # Add missing columns if needed
            if 'OPERATION_TYPE' not in existing_cols:
                cur.execute("""
                    ALTER TABLE UPLOAD_AUDIT
                    ADD COLUMN OPERATION_TYPE STRING
                """)
                conn.commit()
            
            if 'UPLOADED_BY' not in existing_cols:
                cur.execute("""
                    ALTER TABLE UPLOAD_AUDIT
                    ADD COLUMN UPLOADED_BY STRING
                """)
                conn.commit()
        else:
            # Create new table with full schema
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
    """Log upload audit - SQL injection safe"""
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
    """Check if table exists"""
    tables = get_tables()
    return table_name in tables

def check_schema_compatibility(df_columns, table_columns):
    """Check if dataframe columns are compatible with table"""
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
        # Fix: Convert dtypes to string to avoid Arrow serialization error
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

    # Check for null values in primary key
    null_rows = df[df[pk].isnull()]
    
    # Check for duplicates in primary key
    dup_rows = df[df.duplicated(subset=[pk], keep=False)]

    if not null_rows.empty:
        st.error(f"❌ Found {len(null_rows)} rows with NULL Primary Key")
        with st.expander("View NULL rows"):
            st.dataframe(null_rows)

    if not dup_rows.empty:
        st.error(f"❌ Found {len(dup_rows)} rows with Duplicate Primary Key")
        with st.expander("View duplicate rows"):
            st.dataframe(dup_rows)

    # Create valid dataset
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

# -------------------------------------------------
# COLUMN PROFILING
# -------------------------------------------------
with tabs[2]:
    st.subheader("Column Profiling")

    col = st.selectbox("Select column to profile", df.columns)

    # Basic statistics
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

    # Distribution visualization
    st.subheader("Distribution")
    
    if pd.api.types.is_numeric_dtype(df[col]):
        # Numeric column
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
        # Categorical column
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

# -------------------------------------------------
# UPSERT / MERGE
# -------------------------------------------------
with tabs[3]:
    st.subheader("UPSERT / MERGE")

    if ROLE != "ADMIN":
        st.warning("⚠️ Only ADMIN role can load data to Snowflake")
        st.info("Please select ADMIN role from the sidebar to proceed")
        st.stop()

    # Get valid data from session state
    valid_df = st.session_state.valid_df
    pk = st.session_state.pk_column

    if valid_df.empty:
        st.error("No valid data to upload. Please check the Validation tab.")
        st.stop()

    st.info(f"📊 Ready to upload {len(valid_df)} valid rows using **{pk}** as primary key")

    # Select target table or create new
    available_tables = get_tables()
    
    # Table selection mode
    table_mode = st.radio(
        "Select Table Mode:",
        ["Use Existing Table", "Create New Table"],
        horizontal=True
    )
    
    if table_mode == "Use Existing Table":
        if not available_tables:
            st.error("No tables found in the current schema. Please select 'Create New Table' option.")
            st.stop()
        
        target_table = st.selectbox("Select Target Table", available_tables)
        schema_mismatch = False
        
        # Validate schema compatibility
        if target_table:
            target_cols = get_schema(target_table)
            
            if not target_cols:
                st.error(f"Could not fetch schema for {target_table}")
                st.stop()
            
            # Check if primary key exists in target
            if pk not in target_cols:
                st.error(f"❌ Primary key '{pk}' does not exist in target table {target_table}")
                schema_mismatch = True
            
            # Check column compatibility
            missing_cols, extra_cols = check_schema_compatibility(valid_df.columns, target_cols)
            
            if missing_cols:
                st.warning(f"⚠️ Columns in file but not in table: {', '.join(missing_cols)}")
                schema_mismatch = True
            
            if extra_cols:
                st.info(f"ℹ️ Columns in table but not in file: {', '.join(extra_cols)}")
                st.info("These columns will use default/NULL values for new rows")
            
            # Show column mapping
            with st.expander("View Column Mapping"):
                matching_cols = [col for col in valid_df.columns if col in target_cols]
                
                mapping_data = []
                for col in valid_df.columns:
                    if col in target_cols:
                        mapping_data.append({
                            'File Column': col,
                            'Target Column': col,
                            'Status': '✅ Match'
                        })
                    else:
                        mapping_data.append({
                            'File Column': col,
                            'Target Column': '-',
                            'Status': '❌ Missing in Table'
                        })
                
                mapping_df = pd.DataFrame(mapping_data)
                st.dataframe(mapping_df, hide_index=True)
            
            # Offer to create new table if schema mismatch
            if schema_mismatch:
                st.error("⚠️ Schema mismatch detected!")
                st.info("💡 You can switch to 'Create New Table' mode to create a table matching your file structure.")
                
                if missing_cols:
                    st.warning("The following columns will be ignored during upload:")
                    st.write(", ".join(missing_cols))
                    # Filter valid_df to only include matching columns
                    valid_df = valid_df[[col for col in valid_df.columns if col in target_cols]]
                    st.session_state.valid_df = valid_df
                    
                    if valid_df.empty:
                        st.error("No matching columns found. Please create a new table.")
                        st.stop()
    
    else:  # Create New Table mode
        st.info("🆕 Creating a new table based on your file structure")
        
        # Suggest table name based on file name
        suggested_name = sanitize_table_name(uploaded_file.name)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            target_table = st.text_input(
                "New Table Name",
                value=suggested_name,
                help="Table name will be converted to uppercase and sanitized"
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄 Use Filename"):
                target_table = suggested_name
                st.rerun()
        
        # Sanitize the entered table name
        target_table = sanitize_table_name(target_table) if target_table else suggested_name
        
        # Check if table already exists
        if target_table in available_tables:
            st.error(f"❌ Table '{target_table}' already exists! Please choose a different name or use 'Use Existing Table' mode.")
            st.stop()
        
        # Show the table structure that will be created
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
            snowflake_type = type_mapping.get(dtype, 'STRING')
            is_pk = '🔑' if col == pk else ''
            structure_data.append({
                'Column Name': col,
                'Snowflake Type': snowflake_type,
                'Pandas Type': dtype,
                'Primary Key': is_pk
            })
        
        structure_df = pd.DataFrame(structure_data)
        st.dataframe(structure_df, hide_index=True)
        
        st.info(f"💾 Table '{target_table}' will be created with {len(valid_df.columns)} columns")

    st.divider()
    
    st.write("**Operation Details:**")
    if table_mode == "Create New Table":
        st.write(f"- **Mode:** CREATE TABLE + INSERT")
        st.write(f"- **Table Name:** {target_table}")
    else:
        st.write(f"- **Mode:** MERGE (INSERT new records, UPDATE existing records)")
        st.write(f"- **Table Name:** {target_table}")
    st.write(f"- **Primary Key:** {pk}")
    st.write(f"- **Rows to process:** {len(valid_df)}")
    
    # Confirmation
    confirm = st.checkbox(f"I confirm this {'TABLE CREATION and DATA LOAD' if table_mode == 'Create New Table' else 'MERGE'} operation")
    
    if st.button("🚀 Execute Operation", type="primary", disabled=not confirm):
        with st.spinner("Processing operation..."):
            try:
                cur = conn.cursor()
                
                if table_mode == "Create New Table":
                    # CREATE NEW TABLE MODE
                    progress = st.progress(0, text="Creating new table...")
                    
                    # Create the table
                    if not create_table_from_dataframe(target_table, valid_df):
                        st.error("Failed to create table")
                        st.stop()
                    
                    progress.progress(40, text="Loading data to new table...")
                    
                    # Insert data directly
                    success, nchunks, nrows, _ = write_pandas(
                        conn,
                        valid_df,
                        target_table,
                        auto_create_table=False,
                        overwrite=False
                    )
                    
                    if not success:
                        raise Exception("Failed to load data to new table")
                    
                    progress.progress(80, text="Logging audit trail...")
                    
                    # Log audit
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
                    # MERGE TO EXISTING TABLE MODE
                    stage_table = f"{target_table}_STAGE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    progress = st.progress(0, text="Creating staging table...")
                    
                    # Drop if exists
                    cur.execute(f"DROP TABLE IF EXISTS {stage_table}")
                    
                    # Create staging table with same structure
                    cur.execute(f"CREATE TEMPORARY TABLE {stage_table} LIKE {target_table}")
                    
                    progress.progress(25, text="Loading data to staging table...")
                    
                    # Load data to staging table
                    success, nchunks, nrows, _ = write_pandas(
                        conn,
                        valid_df,
                        stage_table,
                        auto_create_table=False,
                        overwrite=True
                    )
                    
                    if not success:
                        raise Exception("Failed to write data to staging table")
                    
                    progress.progress(50, text="Calculating merge statistics...")
                    
                    # Get count before merge for calculating updates
                    cur.execute(f"""
                        SELECT COUNT(*)
                        FROM {target_table} T
                        INNER JOIN {stage_table} S
                        ON T.{pk} = S.{pk}
                    """)
                    result = cur.fetchone()
                    rows_to_update = result[0] if result else 0
                    
                    progress.progress(60, text="Executing MERGE...")
                    
                    # Build MERGE statement with explicit column matching
                    update_cols = [col for col in valid_df.columns if col != pk]
                    
                    if update_cols:
                        update_set = ", ".join([f"T.{col} = S.{col}" for col in update_cols])
                        when_matched = f"WHEN MATCHED THEN UPDATE SET {update_set}"
                    else:
                        when_matched = ""
                    
                    insert_cols = ", ".join(valid_df.columns)
                    insert_vals = ", ".join([f"S.{col}" for col in valid_df.columns])
                    
                    merge_sql = f"""
                    MERGE INTO {target_table} T
                    USING {stage_table} S
                    ON T.{pk} = S.{pk}
                    {when_matched}
                    WHEN NOT MATCHED THEN 
                        INSERT ({insert_cols})
                        VALUES ({insert_vals})
                    """
                    
                    # Execute MERGE
                    cur.execute(merge_sql)
                    
                    # Calculate rows inserted
                    rows_inserted = len(valid_df) - rows_to_update
                    rows_updated = rows_to_update
                    
                    progress.progress(80, text="Logging audit trail...")
                    
                    # Log audit
                    log_audit(uploaded_file.name, target_table, len(valid_df), "MERGE")
                    
                    progress.progress(90, text="Cleaning up...")
                    
                    # Cleanup staging table
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
                except:
                    pass

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