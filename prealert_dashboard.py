import streamlit as st
import pandas as pd
from prealert_jfk import get_jfk_data
from prealert_ord import get_ord_data
from transfer_prealert_atl import get_atl_data

now = pd.Timestamp.now()

yesterday_end = (now - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
ten_days_ago_start = (now - pd.Timedelta(days=10)).normalize()
end = yesterday_end.strftime("%Y-%m-%d %H:%M:%S")
start = ten_days_ago_start.strftime("%Y-%m-%d %H:%M:%S")

jfk_mawb, jfk_container_channel = get_jfk_data(start, end)
ord_mawb, ord_container_channel = get_ord_data(start, end)
atl_mawb, atl_container_channel = get_atl_data(start, end)


jfk_container_channel.dropna(subset=['container_no'],inplace=True)
ord_container_channel.dropna(subset=['container_no'],inplace=True)
atl_container_channel.dropna(subset=['container_no'],inplace=True)

#overall_data_=overall_data.sort_values(by='操作日期',ascending=False)

st.set_page_config(page_title="KPI", layout="wide")
st.title("JFK KPI")

with st.expander("一、两天内超时主单", expanded=False):
    st.dataframe(jfk_mawb)

st.markdown("---")
with st.expander("二、两天内超时托盘", expanded=False):
    st.dataframe(jfk_container_channel)

st.title("ORD KPI")

with st.expander("一、两天内超时主单", expanded=False):
    st.dataframe(ord_mawb)

st.markdown("---")
with st.expander("二、两天内超时托盘", expanded=False):
    st.dataframe(ord_container_channel)

st.title("ATL卡转 KPI")

with st.expander("一、两天内超时主单", expanded=False):
    st.dataframe(atl_mawb)

st.markdown("---")
with st.expander("二、两天内超时托盘", expanded=False):
    st.dataframe(atl_container_channel)


