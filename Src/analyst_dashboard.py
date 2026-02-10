import streamlit as st
import pandas as pd
import snowflake.connector
import altair as alt
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from config import SNOWFLAKE_DB_CONFIG

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Analyst SLA Dashboard",
    layout="wide"
)

st.title("📊 Analyst SLA Monitoring Dashboard")

# -------------------------------------------------
# AUTO REFRESH (60 sec)
# -------------------------------------------------
st_autorefresh(interval=60 * 1000, key="sla_refresh")

# -------------------------------------------------
# SNOWFLAKE CONNECTION
# -------------------------------------------------
@st.cache_resource
def get_conn():
    return snowflake.connector.connect(**SNOWFLAKE_DB_CONFIG)

conn = get_conn()

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------
SLA_GREEN_MIN = 60      # <= 60 min
SLA_AMBER_MIN = 180    # 60–180 min
# >180 min = RED

# -------------------------------------------------
# FETCH AUDIT DATA
# -------------------------------------------------
@st.cache_data(ttl=60)
def load_audit_data():
    query = """
        SELECT
            TABLE_NAME,
            MAX(UPLOAD_TIME) AS LAST_UPLOAD
        FROM UPLOAD_AUDIT
        GROUP BY TABLE_NAME
    """
    cur = conn.cursor()
    cur.execute(query)
    df = cur.fetch_pandas_all()
    cur.close()
    return df

audit_df = load_audit_data()

if audit_df.empty:
    st.warning("No audit data available yet")
    st.stop()

# -------------------------------------------------
# SLA CALCULATION
# -------------------------------------------------
now = datetime.utcnow()

audit_df["LAST_UPLOAD"] = pd.to_datetime(audit_df["LAST_UPLOAD"])
audit_df["MINUTES_SINCE_LOAD"] = (
    now - audit_df["LAST_UPLOAD"]
).dt.total_seconds() / 60

def rag_status(minutes):
    if minutes <= SLA_GREEN_MIN:
        return "GREEN"
    elif minutes <= SLA_AMBER_MIN:
        return "AMBER"
    else:
        return "RED"

audit_df["SLA_STATUS"] = audit_df["MINUTES_SINCE_LOAD"].apply(rag_status)

# -------------------------------------------------
# KPI METRICS
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tables", audit_df["TABLE_NAME"].nunique())

with col2:
    st.metric("🟢 Green", (audit_df["SLA_STATUS"] == "GREEN").sum())

with col3:
    st.metric("🟠 Amber", (audit_df["SLA_STATUS"] == "AMBER").sum())

with col4:
    st.metric("🔴 Red", (audit_df["SLA_STATUS"] == "RED").sum())

st.divider()

# -------------------------------------------------
# RAG SUMMARY
# -------------------------------------------------
rag_summary = (
    audit_df["SLA_STATUS"]
    .value_counts()
    .reset_index()
)

rag_summary.columns = ["STATUS", "COUNT"]

# -------------------------------------------------
# BAR CHART
# -------------------------------------------------
bar_chart = (
    alt.Chart(rag_summary)
    .mark_bar()
    .encode(
        x=alt.X("STATUS:N", title="SLA Status"),
        y=alt.Y("COUNT:Q", title="Number of Tables"),
        tooltip=["STATUS", "COUNT"]
    )
    .properties(
        title="📊 SLA Status Distribution (Bar Chart)",
        height=300
    )
)

# -------------------------------------------------
# DONUT CHART
# -------------------------------------------------
donut_chart = (
    alt.Chart(rag_summary)
    .mark_arc(innerRadius=70)
    .encode(
        theta=alt.Theta("COUNT:Q"),
        tooltip=["STATUS", "COUNT"]
    )
    .properties(
        title="🍩 SLA Health Distribution (Donut Chart)",
        height=300
    )
)

# -------------------------------------------------
# CHART LAYOUT
# -------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.altair_chart(bar_chart, use_container_width=True)

with c2:
    st.altair_chart(donut_chart, use_container_width=True)

st.divider()

# -------------------------------------------------
# TABLE LEVEL DETAIL (ANALYST VIEW)
# -------------------------------------------------
st.subheader("📋 Table-Level SLA Details")

audit_df_display = audit_df.copy()
audit_df_display["LAST_UPLOAD"] = audit_df_display["LAST_UPLOAD"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

st.dataframe(
    audit_df_display[
        ["TABLE_NAME", "LAST_UPLOAD", "MINUTES_SINCE_LOAD", "SLA_STATUS"]
    ],
    hide_index=True,
    use_container_width=True
)

# -------------------------------------------------
# ALERT SECTION
# -------------------------------------------------
red_tables = audit_df[audit_df["SLA_STATUS"] == "RED"]

if not red_tables.empty:
    st.error("🚨 SLA BREACH DETECTED")
    st.write("Following tables have crossed SLA:")
    st.dataframe(
        red_tables[["TABLE_NAME", "MINUTES_SINCE_LOAD"]],
        hide_index=True
    )
else:
    st.success("✅ All tables are within SLA")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.caption("Analyst Dashboard | Auto-refreshed SLA Monitoring | Snowflake")