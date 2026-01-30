# coding=utf-8
import os
import warnings
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import pymysql
import streamlit as st
from pandas.tseries.offsets import BDay

warnings.filterwarnings("ignore")

# ---------- helper: read data files relative to this module ----------
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_data_file(filename: str) -> pd.DataFrame:
    """
    Try to read an Excel file from repository (data/ or same folder). If not found, raise FileNotFoundError.
    """
    candidates = [
        os.path.join(MODULE_DIR, filename),
        os.path.join(MODULE_DIR, "data", filename),
        filename
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_excel(p)
    raise FileNotFoundError(f"Could not find {filename} in {candidates}")


# ---------- Secrets (read at runtime) ----------
# Ensure you have [db_wms] and [db_cbs] set in Streamlit secrets (Advanced settings)
def _get_db_conf(section: str):
    conf = st.secrets.get(section)
    if conf is None:
        raise RuntimeError(f"Secrets section '{section}' not found. Please set it in Streamlit Advanced settings.")
    return conf


# ---------- Cached DB fetch functions ----------
@st.cache_data(ttl=60, show_spinner=False)
def get_no_outbound_mawb_from_db(start: str, end: str) -> pd.DataFrame:
    """
    Fetch large-mawb dataset from CBS (cached).
    """
    db_cbs = _get_db_conf("db_cbs")
    conn = pymysql.connect(
        host=db_cbs["host"],
        port=int(db_cbs.get("port", 3306)),
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
		o.create_time >= %s
		AND o.create_time <= %s
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
    df = pd.read_sql(sql_, conn, params=(start, end))
    conn.close()

    # normalize datetime columns to pandas datetime if present
    for col in ["ata", "full_release", "cbp_release", "pga_release"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df

"""从cbs获取未出库大包"""
@st.cache_data(ttl=60, show_spinner=False)
def get_no_outbound_carton_from_db(start: str, end: str) -> pd.DataFrame:
    db_cbs = _get_db_conf("db_cbs")
    conn = pymysql.connect(
        host=db_cbs["host"],
        port=int(db_cbs.get("port", 3306)),
        user=db_cbs["user"],
        password=db_cbs["password"],
        database=db_cbs["database"],
        charset="utf8"
    )
    sql_ = """
    SELECT
        e.mawb_no,
        c.packet_no,
        c.channel_info
    FROM
        tl_order o,
        tl_order_extra e ,
        tl_order_customs c
    WHERE
        o.create_time >= %s
        AND o.create_time <= %s
        AND o.is_delete = 0
        AND o.id = e.order_id
        AND e.business_type = '17'
        AND e.is_delete = 0
        and c.order_id = o.id
        and c.is_delete= 0
        and not exists (select 1 from tl_customers_so_detail d where d.order_id = c.order_id and d.bag_no = c.packet_no and d.is_delete = 0)
    GROUP BY c.order_id ,c.packet_no
    ;
    """
    df = pd.read_sql(sql_, conn, params=(start, end))
    conn.close()
    return df

"""从wms获取卡转主单的未出库大包"""
@st.cache_data(ttl=60, show_spinner=False)
def get_no_outbound_carton_transfer(start_date):
    db_wms = _get_db_conf("db_wms")
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
def get_carton_container_selected_from_db(carton_list: list) -> pd.DataFrame:
    """
    Given a list of bag_no (carton_list), return bag_no -> container_no mapping (cached).
    """
    if not carton_list:
        return pd.DataFrame(columns=["bag_no", "container_no"])

    db_wms = _get_db_conf("db_wms")
    conn = pymysql.connect(
        host=db_wms["host"],
        port=int(db_wms.get("port", 3306)),
        user=db_wms["user"],
        password=db_wms["password"],
        database=db_wms["database"],
        charset="utf8"
    )

    placeholders = ",".join(["%s"] * len(carton_list))
    sql_ = f"""
    SELECT iwb.bag_no, iwg.gayload_no, iwpl.pallet_no
    FROM ifm_warehouse_bag iwb
    LEFT JOIN ifm_warehouse_pallet iwpl ON iwb.pallet_id = iwpl.id
    LEFT JOIN ifm_warehouse_gayload iwg ON iwb.gayload_id = iwg.id
    WHERE iwb.bag_no IN ({placeholders});
    """
    df = pd.read_sql(sql_, conn, params=carton_list)
    conn.close()

    if not df.empty:
        df["container_no"] = df.apply(lambda r: r["gayload_no"] if pd.notna(r.get("gayload_no")) else r.get("pallet_no"), axis=1)
        return df[["bag_no", "container_no"]]
    else:
        return pd.DataFrame(columns=["bag_no", "container_no"])


# ---------- core processing function (no top-level DB calls) ----------
def get_data(start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Main function to return (mawb_channel_, container_channel_, debug_dict)
    - start, end: strings like "YYYY-MM-DD HH:MM:SS"
    - returns two DataFrames ready to be used by main app.
    """
    # 1) fetch raw tables (these are cached)
    no_outbound_mawb = get_no_outbound_mawb_from_db(start, end)
    no_outbound_carton = get_no_outbound_carton_from_db(start, end)

    # Normalize and rename columns to match downstream logic
    # original code expected columns: ata, full_release, cbp_release, pga_release, channel_info, is_destination_transfer (scf_type)
    # adapt names if necessary:

    # Convert some columns and filter Temu customers as original code
    if "scf_type" in no_outbound_mawb.columns:
        # ensure scf_type numeric
        try:
            no_outbound_mawb["scf_type"] = no_outbound_mawb["scf_type"].astype(int)
        except Exception:
            # if values are '0'/'1' string style
            no_outbound_mawb["scf_type"] = pd.to_numeric(no_outbound_mawb["scf_type"], errors="coerce").fillna(0).astype(int)

    # 2) compute local times like in original (apply timezone offsets)
    # We'll compute ata_local, full_release_local, cbp_release_local, pga_release_local similar to original logic
    def _apply_local_offset(df: pd.DataFrame, colname: str) -> pd.Series:
        if colname not in df.columns:
            return pd.Series([pd.NaT] * len(df))
        s = pd.to_datetime(df[colname], errors="coerce")
        def _localize(row):
            pod = row.get("pod_code")
            t = row[colname]
            if pd.isna(t):
                return pd.NaT
            if pod == "LAX":
                return t - pd.Timedelta(hours=8)
            elif pod == "JFK":
                return t - pd.Timedelta(hours=5)
            else:
                return t - pd.Timedelta(hours=6)
        return df.apply(_localize, axis=1)

    no_outbound_mawb["ata_local"] = _apply_local_offset(no_outbound_mawb, "ata")
    no_outbound_mawb["full_release_local"] = _apply_local_offset(no_outbound_mawb, "full_release")
    no_outbound_mawb["cbp_release_local"] = _apply_local_offset(no_outbound_mawb, "cbp_release")
    no_outbound_mawb["pga_release_local"] = _apply_local_offset(no_outbound_mawb, "pga_release")

    # 3) filter customer and outbound status like your original script
    if "customer_code" in no_outbound_mawb.columns:
        no_outbound_mawb = no_outbound_mawb[no_outbound_mawb["customer_code"].isin(["Temu","Temu-JFK","Temu-ORD","Temu-DFW"])]

    # create outbound_status
    if "outbound" in no_outbound_mawb.columns:
        no_outbound_mawb["outbound_status"] = no_outbound_mawb["outbound"].apply(lambda x: "N" if pd.isna(x) else "Y")
    else:
        # fallback if you already had outbound_status column
        if "outbound_status" not in no_outbound_mawb.columns:
            no_outbound_mawb["outbound_status"] = "N"

    no_outbound_mawb = no_outbound_mawb[no_outbound_mawb["outbound_status"].isin(["N"])]

    # 4) compute weekday and base_time (if ata on weekend -> next Monday 00:00)
    no_outbound_mawb["weekday"] = no_outbound_mawb["ata_local"].dt.weekday + 1

    def compute_base_time_for_ata(row):
        a = row["ata_local"]
        if pd.isna(a):
            return pd.NaT
        weekday = int(a.weekday()) + 1
        if weekday in (6, 7):
            next_monday = (a + pd.offsets.Week(weekday=0)).replace(hour=0, minute=0, second=0, microsecond=0)
            return next_monday
        else:
            return a

    no_outbound_mawb["base_time"] = no_outbound_mawb.apply(compute_base_time_for_ata, axis=1)

    # 5) load US holidays file (try reading from repo data)
    try:
        us_holiday_df = _read_data_file("USPS美国法定假日.xlsx")
        if "日期" in us_holiday_df.columns:
            us_holiday_list = pd.to_datetime(us_holiday_df["日期"], errors="coerce").dt.date.tolist()
        else:
            # try first column
            us_holiday_list = pd.to_datetime(us_holiday_df.iloc[:, 0], errors="coerce").dt.date.tolist()
    except FileNotFoundError:
        # fallback to empty list if file missing (you should add this file to repo/data/)
        us_holiday_list = []

    # 6) compute customs_del = base_time + 3 business days

    ##USPSGRR 3.5个工作日时效
    def add_3_5_business_days(t):
        # 先加 3 个工作日
        if t['channel_info'] == "USPSGRR":
            # 1️⃣ 先加 3 个工作日
            t['base_time'] = t['base_time'] + BDay(3)

            # 2️⃣ 再加 12 小时
            dt_plus_12h = t['base_time'] + pd.Timedelta(hours=12)

            # 3️⃣ 如果落在周末，顺延到下一个周一，保留时间
            if dt_plus_12h.weekday() >= 5:  # 5=周六，6=周日
                days_to_monday = 7 - dt_plus_12h.weekday()
                dt_plus_12h = dt_plus_12h + pd.Timedelta(days=days_to_monday)
            return dt_plus_12h
        else:
            return t['base_time'] + BDay(3)

    no_outbound_mawb["customs_del"] = no_outbound_mawb.apply(add_3_5_business_days, axis=1)
    no_outbound_mawb = no_outbound_mawb.dropna(subset=["base_time", "customs_del"])

    # robust holiday check per row
    def is_holiday_between(base_time, customs_del):
        if pd.isna(base_time) or pd.isna(customs_del) or not us_holiday_list:
            return False
        base_date = base_time.date()
        customs_date = customs_del.date()
        # simple loop; list is small (US holidays)
        return any(base_date <= h <= customs_date for h in us_holiday_list)

    no_outbound_mawb["holiday"] = no_outbound_mawb.apply(
        lambda t: "Y" if is_holiday_between(t["base_time"], t["customs_del"]) else "N",
        axis=1
    )

    # 7) compute basetime_delivery_ddl
    try:
        scf_delivery_time_df = _read_data_file("scf_trip_time.xlsx")
        scf_delivery_time_df.columns = [c.strip() for c in scf_delivery_time_df.columns]
        scf_delivery_time_dict = scf_delivery_time_df.set_index("channel")["transfertime"].to_dict()
    except FileNotFoundError:
        scf_delivery_time_dict = {}
    no_outbound_mawb["scf_delivery_time"] = no_outbound_mawb["channel_info"].map(scf_delivery_time_dict).fillna(0)
    def compute_basetime_delivery_ddl(row):
        if row["holiday"] == "N":
            base = row["customs_del"]
        else:
            # if holiday, shift by 1 business day
            base = row["customs_del"] + BDay(1)

        if row.get("scf_type", 0) != 0:
            base = base + pd.Timedelta(hours=row["scf_delivery_time"])
        return base

    no_outbound_mawb["basetime_delivery_ddl"] = no_outbound_mawb.apply(compute_basetime_delivery_ddl, axis=1)

    # 8) scf trip time mapping from file scf_trip_time.xlsx (repo/data/)
    try:
        scf_route_time_df = _read_data_file("scf_trip_time.xlsx")
        scf_route_time_df.columns = [c.strip() for c in scf_route_time_df.columns]
        scf_route_time_dict = scf_route_time_df.set_index("channel")["route_time"].to_dict()
    except FileNotFoundError:
        scf_route_time_dict = {}

    no_outbound_mawb["scf_route_time"] = no_outbound_mawb["channel_info"].map(scf_route_time_dict).fillna(0)

    # compute basetime_outbound_ddl
    no_outbound_mawb['basetime_outbound_ddl'] = no_outbound_mawb.apply(
        lambda r: r["basetime_delivery_ddl"] if r['scf_type'] == 0 else r["basetime_delivery_ddl"] - pd.Timedelta(
            hours=r['scf_route_time']), axis=1)

    # 9) drop rows missing full_release_local as original
    no_outbound_mawb = no_outbound_mawb.dropna(subset=["full_release_local"])

    # 10) adjust full_release_local for weekend -> next Monday 00:00
    def compute_release_base_time(row):
        a = row["full_release_local"]
        if pd.isna(a):
            return pd.NaT
        weekday = int(a.weekday()) + 1
        if weekday in (6, 7):
            next_monday = (a + pd.offsets.Week(weekday=0)).replace(hour=0, minute=0, second=0,
                                                                    microsecond=0)
            return next_monday
        else:
            return a

    no_outbound_mawb["release_base_time"] = no_outbound_mawb.apply(compute_release_base_time, axis=1)

    # 11) workday_diff_decimal function (copied from your original, robust)
    def workday_diff_decimal(start: pd.Timestamp, end: pd.Timestamp) -> float:
        if pd.isna(start) or pd.isna(end):
            return pd.NA
        sign = 1
        if end < start:
            start, end = end, start
            sign = -1
        full_days = np.busday_count(start.date(), end.date())
        seconds_per_day = 24 * 3600
        start_sec = (start - start.normalize()).total_seconds()
        end_sec = (end - end.normalize()).total_seconds()
        extra = (end_sec - start_sec) / seconds_per_day
        return round(sign * (full_days + extra), 2)

    no_outbound_mawb["release_diff"] = no_outbound_mawb.apply(
        lambda r: workday_diff_decimal(r["base_time"], r["release_base_time"]), axis=1
    )
    no_outbound_mawb["over_2_5_bdays"] = no_outbound_mawb["release_diff"] > 2.5

    # 12) compute real delivery ddl and real outbound ddl
    def compute_real_ddl(row):
        if row["over_2_5_bdays"]:
            base_time = row["full_release_local"] + BDay(1)
            release_date = row["full_release_local"].date()
            # holiday detection between release_date and base_time.date()
            holiday_flag = any(release_date <= h <= base_time.date() for h in us_holiday_list)
            row["holiday_release"] = "Y" if holiday_flag else "N"
            if holiday_flag:
                base_time = row["full_release_local"] + BDay(1)
            if row.get("scf_type", 0) != 0:
                base_time = base_time + pd.Timedelta(hours=row["scf_delivery_time"])
        else:
            base_time = row["basetime_delivery_ddl"]
        return base_time

    no_outbound_mawb["real_delivery_ddl"] = no_outbound_mawb.apply(compute_real_ddl, axis=1)
    no_outbound_mawb["real_outbound_ddl"] = no_outbound_mawb.apply(
        lambda r: r["real_delivery_ddl"] if r.get("scf_type", 0) == 0 else r["real_delivery_ddl"] - pd.Timedelta(hours=r['scf_route_time']) , axis=1)

    # 13) select pod_code == 'JFK' (your original selected ORD? you had pod_code=='JFK' for ord_prealert)
    ord_prealert_mawb = no_outbound_mawb[no_outbound_mawb.get("pod_code") == "ORD"].copy()
    jfk_prealert_mawb = no_outbound_mawb[no_outbound_mawb.get("pod_code") == "JFK"].copy()

    # 14) compute adjusted_ddl according to daily warehouse operation (18:00 close, 09:00 open)
    def jfk_adjust_time(t):
        if pd.isna(t):
            return pd.NaT
        if t.hour >= 18:
            return t.replace(hour=18, minute=0, second=0, microsecond=0)
        elif t.hour < 9:
            return (t - pd.Timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        else:
            return t
    jfk_prealert_mawb["adjusted_ddl"] = jfk_prealert_mawb["real_outbound_ddl"].apply(jfk_adjust_time)

    def ord_adjust_time(t):
        if pd.isna(t):
            return pd.NaT
        if t.hour >= 20:
            return t.replace(hour=20, minute=0, second=0, microsecond=0)
        elif t.hour < 8:
            return (t - pd.Timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
        else:
            return t
    ord_prealert_mawb["adjusted_ddl"] = ord_prealert_mawb["real_outbound_ddl"].apply(ord_adjust_time)



    # IB pickup special case: if adjusted_ddl.hour > 15 and channel_info == "IB", set hour to 15
    if "channel_info" in ord_prealert_mawb.columns:
        ord_prealert_mawb["adjusted_ddl"] = ord_prealert_mawb.apply(
            lambda r: r["adjusted_ddl"].replace(hour=15, minute=0, second=0, microsecond=0)
            if (pd.notna(r["adjusted_ddl"]) and r["adjusted_ddl"].hour > 15 and r["channel_info"] == "IB")
            else r["adjusted_ddl"],
            axis=1
        )

    # 15) compute next_day_start + 2 day threshold for the "48 hours" rule (original logic)
    now = pd.Timestamp.now()
    next_day_start = (now + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    threshold_x = next_day_start + pd.Timedelta(days=2)
    ord_prealert_mawb["prealert"] = ord_prealert_mawb["adjusted_ddl"] < threshold_x
    ord_overdue_mawb = ord_prealert_mawb[ord_prealert_mawb["prealert"] == True]
    jfk_prealert_mawb["prealert"] = jfk_prealert_mawb["adjusted_ddl"] < threshold_x
    jfk_overdue_mawb = jfk_prealert_mawb[jfk_prealert_mawb["prealert"] == True]
    #ord_overdue_mawb_list = ord_overdue_mawb["mawb_no"].unique().tolist()

    # 16) cartons not outbound in CBS
    no_outbound_carton_list = no_outbound_carton["mawb_no"].unique().tolist()
    # difference = set(ord_overdue_mawb_list) - set(no_outbound_carton_list)
    # print("CBS需要更更改outbound status的主单:", difference)

    # 17) join to get ord_overdue_carton (only those cartons that match mawb channel)
    # ensure ord_overdue_mawb has channel_info (or delivery_channel)
    join_col = "channel_info" if "channel_info" in no_outbound_carton.columns else "delivery_channel"
    if join_col not in no_outbound_carton.columns:
        no_outbound_carton[join_col] = no_outbound_carton.get("channel_info", "")

    ord_overdue_carton = pd.merge(no_outbound_carton, ord_overdue_mawb, left_on=["mawb_no", join_col], right_on=["mawb_no", "channel_info"], how="inner")
    ord_overdue_carton.rename(columns={"packet_no": "bag_no"}, inplace=True)

    jfk_overdue_carton = pd.merge(no_outbound_carton, jfk_overdue_mawb, left_on=["mawb_no", join_col],
                                  right_on=["mawb_no", "channel_info"], how="inner")
    jfk_overdue_carton.rename(columns={"packet_no": "bag_no"}, inplace=True)

    # 18) get container mapping for the bags
    carton_list_ord = ord_overdue_carton["bag_no"].astype(str).tolist()
    carton_container_selected_ord = get_carton_container_selected_from_db(carton_list_ord)
    ord_overdue_carton_container = pd.merge(ord_overdue_carton, carton_container_selected_ord, left_on="bag_no", right_on="bag_no", how="inner")

    carton_list_jfk = jfk_overdue_carton["bag_no"].astype(str).tolist()
    carton_container_selected_jfk = get_carton_container_selected_from_db(carton_list_jfk)
    jfk_overdue_carton_container = pd.merge(jfk_overdue_carton, carton_container_selected_jfk, left_on="bag_no",
                                            right_on="bag_no", how="inner")

    # 19) compute mawb_channel_ and container_channel_
    ord_overdue_carton_container = ord_overdue_carton_container.sort_values("adjusted_ddl")
    mawb_channel_ord = ord_overdue_carton_container.drop_duplicates(subset=["adjusted_ddl", "mawb_no", "channel_info"])
    mawb_channel_ord_ = mawb_channel_ord[["adjusted_ddl", "mawb_no", "channel_info"]].reset_index(drop=True)

    container_channel_ord = ord_overdue_carton_container.drop_duplicates(subset=["adjusted_ddl", "container_no", "channel_info"])
    container_channel_ord_ = container_channel_ord[["adjusted_ddl", "container_no", "channel_info"]].reset_index(drop=True)

    jfk_overdue_carton_container = jfk_overdue_carton_container.sort_values("adjusted_ddl")
    mawb_channel_jfk = jfk_overdue_carton_container.drop_duplicates(subset=["adjusted_ddl", "mawb_no", "channel_info"])
    mawb_channel_jfk_ = mawb_channel_jfk[["adjusted_ddl", "mawb_no", "channel_info"]].reset_index(drop=True)

    container_channel_jfk = jfk_overdue_carton_container.drop_duplicates(
        subset=["adjusted_ddl", "container_no", "channel_info"])
    container_channel_jfk_ = container_channel_jfk[["adjusted_ddl", "container_no", "channel_info"]].reset_index(
        drop=True)


    return mawb_channel_ord_, container_channel_ord_,mawb_channel_jfk_,container_channel_jfk_


def get_transfer_data(start, end):
    """获取 ATL/JFK 需要预警的主单和容器数据"""
    n_mawb = get_no_outbound_mawb_from_db(start, end)
    n_mawb_ord = n_mawb[n_mawb['customer_code'].isin(['Temu-ORD'])]
    n_mawb_ord['weekday'] = n_mawb_ord["ata_local"].dt.weekday + 1

    # 计算 base_time（周末调整）
    def compute_base_time(row):
        a, b = row["ata_local"], row["weekday"]
        if b in [6, 7]:
            next_monday = a + pd.offsets.Week(weekday=0)
            return next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return a

    n_mawb_ord["base_time"] = n_mawb_ord.apply(compute_base_time, axis=1)
    n_mawb_ord_atl = n_mawb_ord[n_mawb_ord['channel_info'].str.endswith('ATL', na=False)]
    n_mawb_ord_atl.dropna(subset=['ata'], inplace=True)

    n_mawb_ord_jfk = n_mawb_ord[n_mawb_ord['channel_info'].str.endswith('JFK', na=False)]
    n_mawb_ord_jfk.dropna(subset=['ata'], inplace=True)

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

    n_mawb_ord_atl["basetime_delivery_ddl"] = n_mawb_ord_atl['base_time'].apply(lambda r: add_working_hours_skip_weekends(r, 108))
    n_mawb_ord_atl.dropna(subset=['full_release_local'], inplace=True)
    # full release 周末调整
    n_mawb_ord_atl['release_weekday'] = n_mawb_ord_atl["full_release_local"].dt.weekday + 1

    n_mawb_ord_jfk["basetime_delivery_ddl"] = n_mawb_ord_jfk['base_time'].apply(lambda r: add_working_hours_skip_weekends(r, 108))
    n_mawb_ord_jfk.dropna(subset=['full_release_local'], inplace=True)
    # full release 周末调整
    n_mawb_ord_jfk['release_weekday'] = n_mawb_ord_jfk["full_release_local"].dt.weekday + 1

    def compute_release_base(row):
        a, b = row["full_release_local"], row["release_weekday"]
        if b in [6, 7]:
            next_monday = a + pd.offsets.Week(weekday=0)
            return next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return a

    n_mawb_ord_atl["release_base_time"] = n_mawb_ord_atl.apply(compute_release_base, axis=1)
    n_mawb_ord_jfk["release_base_time"] = n_mawb_ord_jfk.apply(compute_release_base, axis=1)

    # 2.5 工作日差计算
    def workday_diff_decimal(start: pd.Timestamp, end: pd.Timestamp) -> float:
        if pd.isna(start) or pd.isna(end):
            return np.nan
           #return pd.NA
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

    n_mawb_ord_atl["release_diff"] = n_mawb_ord_atl.apply(lambda r: workday_diff_decimal(r['base_time'], r['release_base_time']), axis=1)
    n_mawb_ord_atl["over_2_5_bdays"] = n_mawb_ord_atl["release_diff"] > 2.5

    n_mawb_ord_jfk["release_diff"] = n_mawb_ord_jfk.apply(lambda r: workday_diff_decimal(r['base_time'], r['release_base_time']), axis=1)
    n_mawb_ord_jfk["over_2_5_bdays"] = n_mawb_ord_jfk["release_diff"] > 2.5

    # 真实出库ddl
    # def compute_real_ddl(row):
    #     return row['full_release_local'] + BDay(2) if row['over_2_5_bdays'] else row['basetime_delivery_ddl']
    #
    # n_mawb_ord_atl['real_delivery_ddl'] = n_mawb_ord_atl.apply(compute_real_ddl, axis=1)
    # mawb_ord_atl = n_mawb_ord_atl[n_mawb_ord_atl['pod_code']=='ORD'].copy()
    #
    # n_mawb_ord_jfk['real_delivery_ddl'] = n_mawb_ord_jfk.apply(compute_real_ddl, axis=1)
    # mawb_ord_jfk = n_mawb_ord_jfk[n_mawb_ord_jfk['pod_code'] == 'ORD'].copy()
    def compute_real_ddl(row):
        # 增加对布尔判断列的空值检查
        if pd.isna(row['over_2_5_bdays']):
            return pd.NaT

        if row['over_2_5_bdays']:
            # 确保时间列不为空再加 BDay
            if pd.isna(row['full_release_local']):
                return pd.NaT
            return row['full_release_local'] + BDay(2)
        else:
            return row['basetime_delivery_ddl']

    # 处理 ATL
    if not n_mawb_ord_atl.empty:
        n_mawb_ord_atl['real_delivery_ddl'] = n_mawb_ord_atl.apply(compute_real_ddl, axis=1)
    else:
        n_mawb_ord_atl['real_delivery_ddl'] = pd.Series(dtype='datetime64[ns]')
    mawb_ord_atl = n_mawb_ord_atl[n_mawb_ord_atl['pod_code'] == 'ORD'].copy()

    # 处理 JFK
    if not n_mawb_ord_jfk.empty:
        n_mawb_ord_jfk['real_delivery_ddl'] = n_mawb_ord_jfk.apply(compute_real_ddl, axis=1)
    else:
        n_mawb_ord_jfk['real_delivery_ddl'] = pd.Series(dtype='datetime64[ns]')
    mawb_ord_jfk = n_mawb_ord_jfk[n_mawb_ord_jfk['pod_code'] == 'ORD'].copy()

    # 仓库实际最晚出库时间
    def adjust_time(t):
        if t.hour >= 20:
            return t.replace(hour=20, minute=0, second=0, microsecond=0)
        elif t.hour < 8:
            return (t - pd.Timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
        else:
            return t

    mawb_ord_atl["adjusted_ddl"] = mawb_ord_atl["real_delivery_ddl"].apply(adjust_time)
    mawb_ord_jfk["adjusted_ddl"] = mawb_ord_jfk["real_delivery_ddl"].apply(adjust_time)

    # 48小时内预警
    now = pd.Timestamp.now()
    next_day_start = (now + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    x = next_day_start + pd.Timedelta(days=2)
    mawb_ord_atl['prealert'] = mawb_ord_atl["adjusted_ddl"] < x
    mawb_ord_jfk['prealert'] = mawb_ord_jfk["adjusted_ddl"] < x

    mawb_alert_ord_atl = mawb_ord_atl[mawb_ord_atl['prealert']].copy()
    mawb_alert_ord_atl = mawb_alert_ord_atl[['mawb_no', 'ata_local', 'full_release_local', 'adjusted_ddl']].drop_duplicates()
    mawb_alert_ord_jfk = mawb_ord_jfk[mawb_ord_jfk['prealert']].copy()
    mawb_alert_ord_jfk = mawb_alert_ord_jfk[
        ['mawb_no', 'ata_local', 'full_release_local', 'adjusted_ddl']].drop_duplicates()


    no_outbound_carton_transfer = get_no_outbound_carton_transfer(start)
    ord_atl_overdue_carton = pd.merge(no_outbound_carton_transfer, mawb_alert_ord_atl, on="mawb_no", how="inner")
    ord_atl_overdue_carton.rename(columns={"packet_no": "bag_no", "channel": "channel_info"}, inplace=True)
    ord_jfk_overdue_carton = pd.merge(no_outbound_carton_transfer, mawb_alert_ord_jfk, on="mawb_no", how="inner")
    ord_jfk_overdue_carton.rename(columns={"packet_no": "bag_no", "channel": "channel_info"}, inplace=True)

    ord_atl_carton_list_transfer = ord_atl_overdue_carton["bag_no"].to_list()
    ord_atl_carton_container_selected = get_carton_container_selected_from_db(ord_atl_carton_list_transfer)
    ord_jfk_carton_list_transfer = ord_jfk_overdue_carton["bag_no"].to_list()
    ord_jfk_carton_container_selected = get_carton_container_selected_from_db(ord_jfk_carton_list_transfer)

    ord_atl_overdue_carton_container = pd.merge(ord_atl_overdue_carton, ord_atl_carton_container_selected, on="bag_no", how="inner")
    ord_atl_overdue_carton_container = ord_atl_overdue_carton_container[
        ~(ord_atl_overdue_carton_container['container_no'].fillna('').astype(str).str.startswith('EPORD'))]

    ord_jfk_overdue_carton_container = pd.merge(ord_jfk_overdue_carton, ord_jfk_carton_container_selected, on="bag_no",how="inner")
    ord_jfk_overdue_carton_container = ord_jfk_overdue_carton_container[
        ~(ord_jfk_overdue_carton_container['container_no'].fillna('').astype(str).str.startswith('EPORD'))]

    # 主单预警
    ord_atl_mawb_channel = ord_atl_overdue_carton_container.drop_duplicates(subset=["adjusted_ddl", "mawb_no", "channel_info"])
    ord_atl_mawb_channel_ = ord_atl_mawb_channel[["adjusted_ddl", "mawb_no", "channel_info"]].reset_index(drop=True)
    ord_jfk_mawb_channel = ord_jfk_overdue_carton_container.drop_duplicates(subset=["adjusted_ddl", "mawb_no", "channel_info"])
    ord_jfk_mawb_channel_ = ord_jfk_mawb_channel[["adjusted_ddl", "mawb_no", "channel_info"]].reset_index(drop=True)

    # 容器预警
    ord_atl_container_channel = ord_atl_overdue_carton_container.drop_duplicates(subset=["adjusted_ddl", "container_no", "channel_info"])
    ord_atl_container_channel_ = ord_atl_container_channel[["adjusted_ddl", "container_no", "channel_info"]].reset_index(drop=True)
    ord_jfk_container_channel = ord_jfk_overdue_carton_container.drop_duplicates(subset=["adjusted_ddl", "container_no", "channel_info"])
    ord_jfk_container_channel_ = ord_jfk_container_channel[["adjusted_ddl", "container_no", "channel_info"]].reset_index(drop=True)

    return ord_atl_mawb_channel_, ord_atl_container_channel_,ord_jfk_mawb_channel_, ord_jfk_container_channel_