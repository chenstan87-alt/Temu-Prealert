# coding=utf-8
import pymysql
import pandas as pd
import warnings
import numpy as np
from pandas.tseries.offsets import BDay
from datetime import timedelta
import streamlit as st

# ----------------------
# 密码管理
# ----------------------
db_wms = st.secrets["db_wms"]
db_cbs = st.secrets["db_cbs"]

warnings.filterwarnings("ignore")  # 忽略警告

# ----------------------
# 数据查询函数
# ----------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_mawb(start, end):
    """获取主单数据"""
    conn = pymysql.connect(
        host=db_cbs["host"],
        port=int(db_cbs["port"]),
        user=db_cbs["user"],
        password=db_cbs["password"],
        database=db_cbs["database"],
        charset="utf8"
    )
    sql_ = """
    SELECT
	a.*,
	c.channel_info,
IF
	( a.outbound IS NULL, 'N', 'Y' ) outbound_status,
CASE
		a.pod_code 
		WHEN 'LAX' THEN
		DATE_SUB( a.ata, INTERVAL 8 HOUR ) 
		WHEN 'JFK' THEN
		DATE_SUB( a.ata, INTERVAL 5 HOUR ) ELSE DATE_SUB( a.ata, INTERVAL 6 HOUR ) 
	END AS ata_local,
CASE
		a.pod_code 
		WHEN 'LAX' THEN
		DATE_SUB( a.full_release, INTERVAL 8 HOUR ) 
		WHEN 'JFK' THEN
		DATE_SUB( a.full_release, INTERVAL 5 HOUR ) ELSE DATE_SUB( a.full_release, INTERVAL 6 HOUR ) 
	END AS full_release_local,
CASE
		a.pod_code 
		WHEN 'LAX' THEN
		DATE_SUB( a.cbp_release, INTERVAL 8 HOUR ) 
		WHEN 'JFK' THEN
		DATE_SUB( a.cbp_release, INTERVAL 5 HOUR ) ELSE DATE_SUB( a.cbp_release, INTERVAL 6 HOUR ) 
	END AS cbp_release_local,
CASE
		a.pod_code 
		WHEN 'LAX' THEN
		DATE_SUB( a.pga_release, INTERVAL 8 HOUR ) 
		WHEN 'JFK' THEN
		DATE_SUB( a.pga_release, INTERVAL 5 HOUR ) ELSE DATE_SUB( a.pga_release, INTERVAL 6 HOUR ) 
	END AS pga_release_local 
FROM
	(
	SELECT
		o.id,
		e.mawb_no,
		o.customer_code,
		e.is_destination_transfer scf_type,
		e.transfer_status,
		o.pod_code,
		(
		SELECT
			ev.operate_date 
		FROM
			tl_order_event ev 
		WHERE
			ev.order_id = o.id 
			AND ev.event_type_id = 18 
			AND ev.is_delete = 0 
			LIMIT 1 
		) ata,
		(
		SELECT
			ev.operate_date 
		FROM
			tl_order_event ev 
		WHERE
			ev.order_id = o.id 
			AND ev.event_type_id = 10 
			AND ev.is_delete = 0 
			LIMIT 1 
		) full_release,
		(
		SELECT
			ev.operate_date 
		FROM
			tl_order_event ev 
		WHERE
			ev.order_id = o.id 
			AND ev.event_type_id = 57 
			AND ev.is_delete = 0 
			LIMIT 1 
		) cbp_release,
		(
		SELECT
			ev.operate_date 
		FROM
			tl_order_event ev 
		WHERE
			ev.order_id = o.id 
			AND ev.event_type_id = 59 
			AND ev.is_delete = 0 
			LIMIT 1 
		) pga_release,
		(
		SELECT
			ev.operate_date 
		FROM
			tl_order_event ev 
		WHERE
			ev.order_id = o.id 
			AND ev.event_type_id = 16 
			AND ev.is_delete = 0 
			AND ev.remark IS NULL 
			LIMIT 1 
		) outbound 
	FROM
		tl_order o,
		tl_order_extra e 
	WHERE
		o.create_time >= '2025-11-10'
		AND o.create_time <= '2025-11-15'
		AND o.is_delete = 0 
		AND o.id = e.order_id 
		AND e.business_type = '17' 
		AND e.is_delete = 0 
	) a,
	tl_order_customs c 
WHERE
	a.id = c.order_id 
	AND c.is_delete = 0
	group by c.order_id,c.channel_info;
    """
    data = pd.read_sql(sql_, conn, params=(start, end))
    conn.close()
    return data


@st.cache_data(ttl=60, show_spinner=False)
def get_no_outbound_carton(start_date):
    """获取未出库大包"""
    conn = pymysql.connect(
        host=db_wms["host"],
        port=int(db_wms["port"]),
        user=db_wms["user"],
        password=db_wms["password"],
        database=db_wms["database"],
        charset="utf8"
    )
    sql_ = """
    SELECT w.warehouse_code, m.mawb_no, m.pod, b.bag_no, b.channel
    FROM ifm_warehouse_bag b
    LEFT JOIN ifm_warehouse_mawb m ON b.warehouse_mawb_id = m.id AND m.mark=1
    LEFT JOIN sys_warehouse w ON b.warehouse_id = w.id
    WHERE b.mark=1 AND b.status=0 AND m.is_transfer=1 AND m.status<90 AND m.ata>%s;
    """
    data = pd.read_sql(sql_, conn, params=[start_date])
    conn.close()
    return data


@st.cache_data(ttl=60, show_spinner=False)
def get_carton_container_selected(carton_list):
    """根据大包列表获取托盘/货载"""
    if not carton_list:
        return pd.DataFrame(columns=["bag_no", "container_no"])

    placeholders = ",".join(["%s"] * len(carton_list))
    conn = pymysql.connect(
        host=db_wms["host"],
        port=int(db_wms["port"]),
        user=db_wms["user"],
        password=db_wms["password"],
        database=db_wms["database"],
        charset="utf8"
    )
    sql_ = f"""
    SELECT iwb.bag_no, iwg.gayload_no, iwpl.pallet_no
    FROM ifm_warehouse_bag iwb
    LEFT JOIN ifm_warehouse_pallet iwpl ON iwb.pallet_id = iwpl.id
    LEFT JOIN ifm_warehouse_gayload iwg ON iwb.gayload_id = iwg.id
    WHERE iwb.bag_no IN ({placeholders});
    """
    data = pd.read_sql(sql_, conn, params=carton_list)
    conn.close()
    data['container_no'] = data.apply(lambda r: r['gayload_no'] if pd.notna(r['gayload_no']) else r['pallet_no'], axis=1)
    return data[["bag_no", "container_no"]]


# ----------------------
# 主函数
# ----------------------

def get_atl_data(start, end):
    """获取 ATL 需要预警的主单和容器数据"""
    n_mawb = get_mawb(start, end)
    n_mawb = n_mawb[n_mawb['customer_code'].isin(['Temu-ORD'])]
    n_mawb['weekday'] = n_mawb["ata_local"].dt.weekday + 1

    # 计算 base_time（周末调整）
    def compute_base_time(row):
        a, b = row["ata_local"], row["weekday"]
        if b in [6, 7]:
            next_monday = a + pd.offsets.Week(weekday=0)
            return next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return a

    n_mawb["base_time"] = n_mawb.apply(compute_base_time, axis=1)
    n_mawb = n_mawb[n_mawb['channel_info'].str.endswith('ATL', na=False)]
    n_mawb.dropna(subset=['ata'], inplace=True)

    # 108小时工作日算法
    def add_working_hours_skip_weekends(start_time, hours):
        current = pd.Timestamp(start_time)
        remaining_hours = hours
        while remaining_hours > 0:
            if current.weekday() < 5:
                hours_until_day_end = 24 - current.hour - current.minute/60 - current.second/3600
                if remaining_hours <= hours_until_day_end:
                    current += pd.Timedelta(hours=remaining_hours)
                    remaining_hours = 0
                else:
                    current += pd.Timedelta(hours=hours_until_day_end)
                    remaining_hours -= hours_until_day_end
                    current = current.normalize()
            else:
                current += pd.Timedelta(days=(7-current.weekday()))
                current = current.normalize()
        return current

    n_mawb["basetime_delivery_ddl"] = n_mawb['base_time'].apply(lambda r: add_working_hours_skip_weekends(r, 108))
    n_mawb.dropna(subset=['full_release_local'], inplace=True)

    # full release 周末调整
    n_mawb['release_weekday'] = n_mawb["full_release_local"].dt.weekday + 1

    def compute_release_base(row):
        a, b = row["full_release_local"], row["release_weekday"]
        if b in [6, 7]:
            next_monday = a + pd.offsets.Week(weekday=0)
            return next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return a

    n_mawb["release_base_time"] = n_mawb.apply(compute_release_base, axis=1)

    # 2.5 工作日差计算
    def workday_diff_decimal(start: pd.Timestamp, end: pd.Timestamp) -> float:
        if pd.isna(start) or pd.isna(end):
            return pd.NA
        sign = 1
        if end < start:
            start, end = end, start
            sign = -1
        full_days = np.busday_count(start.date(), end.date())
        seconds_per_day = 24*3600
        start_sec = (start - start.normalize()).total_seconds()
        end_sec = (end - end.normalize()).total_seconds()
        extra = (end_sec - start_sec)/seconds_per_day
        return round(sign * (full_days + extra), 2)

    n_mawb["release_diff"] = n_mawb.apply(lambda r: workday_diff_decimal(r['base_time'], r['release_base_time']), axis=1)
    n_mawb["over_2_5_bdays"] = n_mawb["release_diff"] > 2.5

    # 真实出库ddl
    def compute_real_ddl(row):
        return row['full_release_local'] + BDay(2) if row['over_2_5_bdays'] else row['basetime_delivery_ddl']

    n_mawb['real_delivery_ddl'] = n_mawb.apply(compute_real_ddl, axis=1)
    ord_prealert_mawb = n_mawb[n_mawb['pod_code']=='ORD']

    # 仓库实际最晚出库时间
    def adjust_time(t):
        if t.hour >= 19:
            return t.replace(hour=19, minute=0, second=0, microsecond=0)
        elif t.hour < 8:
            return (t - pd.Timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)
        else:
            return t

    ord_prealert_mawb["adjusted_ddl"] = ord_prealert_mawb["real_delivery_ddl"].apply(adjust_time)

    # 48小时内预警
    now = pd.Timestamp.now()
    next_day_start = (now + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    x = next_day_start + pd.Timedelta(days=2)
    ord_prealert_mawb['prealert'] = ord_prealert_mawb["adjusted_ddl"] < x

    atl_mawb_alert = ord_prealert_mawb[ord_prealert_mawb['prealert']].copy()
    atl_mawb_alert = atl_mawb_alert[['mawb_no', 'ata_local', 'full_release_local', 'adjusted_ddl']].drop_duplicates()

    no_outbound_carton = get_no_outbound_carton(start)
    atl_overdue_carton = pd.merge(no_outbound_carton, atl_mawb_alert, on="mawb_no", how="inner")
    atl_overdue_carton.rename(columns={"packet_no": "bag_no", "channel": "channel_info"}, inplace=True)

    carton_list = atl_overdue_carton["bag_no"].to_list()
    carton_container_selected = get_carton_container_selected(carton_list)

    atl_overdue_carton_container = pd.merge(atl_overdue_carton, carton_container_selected, on="bag_no", how="inner")
    atl_overdue_carton_container = atl_overdue_carton_container[
        ~(atl_overdue_carton_container['container_no'].fillna('').astype(str).str.startswith('EPORD'))
    ]

    # 主单预警
    mawb_channel = atl_overdue_carton_container.drop_duplicates(subset=["adjusted_ddl", "mawb_no", "channel_info"])
    mawb_channel_ = mawb_channel[["adjusted_ddl", "mawb_no", "channel_info"]].reset_index(drop=True)

    # 容器预警
    container_channel = atl_overdue_carton_container.drop_duplicates(subset=["adjusted_ddl", "container_no", "channel_info"])
    container_channel_ = container_channel[["adjusted_ddl", "container_no", "channel_info"]].reset_index(drop=True)

    return mawb_channel_, container_channel_
