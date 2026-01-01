import streamlit as st
import pandas as pd
from analysis import get_data,get_transfer_data

now = pd.Timestamp.now()

yesterday_end = (now - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
two_weeks_ago_start = (now - pd.Timedelta(days=14)).normalize()
end = yesterday_end.strftime("%Y-%m-%d %H:%M:%S")
start = two_weeks_ago_start.strftime("%Y-%m-%d %H:%M:%S")

ord_mawb, ord_container_channel, jfk_mawb, jfk_container_channel = get_data(start, end)
ord_atl_mawb, ord_atl_container_channel, ord_jfk_mawb, ord_jfk_container_channel = get_transfer_data(start, end)


jfk_container_channel.dropna(subset=['container_no'],inplace=True)
ord_container_channel.dropna(subset=['container_no'],inplace=True)
ord_atl_container_channel.dropna(subset=['container_no'],inplace=True)
ord_jfk_container_channel.dropna(subset=['container_no'],inplace=True)

#overall_data_=overall_data.sort_values(by='操作日期',ascending=False)

st.set_page_config(page_title="Temu KPI", layout="wide")

st.title("JFK KPI【直入】")
with st.expander("一、3天内超时主单", expanded=False):
    st.dataframe(jfk_mawb)

with st.expander("二、3天内超时托盘", expanded=False):
    st.dataframe(jfk_container_channel)

st.markdown("---")
st.title("JFK KPI【ORD卡转】")
with st.expander("一、3天内超时主单", expanded=False):
    st.dataframe(ord_jfk_mawb)

with st.expander("二、3天内超时托盘", expanded=False):
    st.dataframe(ord_jfk_container_channel)

st.markdown("---")
st.title("ORD KPI【直入】")
with st.expander("一、3天内超时主单", expanded=False):
    st.dataframe(ord_mawb)

with st.expander("二、3天内超时托盘", expanded=False):
    st.dataframe(ord_container_channel)

st.markdown("---")
st.title("ATL KPI【ORD卡转】")
with st.expander("一、3天内超时主单", expanded=False):
    st.dataframe(ord_atl_mawb)

with st.expander("二、3天内超时托盘", expanded=False):
    st.dataframe(ord_atl_container_channel)




