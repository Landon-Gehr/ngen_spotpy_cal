import sys
import os
import shutil
import argparse
import subprocess
from datetime import datetime
import requests
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

def check_path(path):
    try:
        p = Path(path).expanduser().resolve()
        return p.exists()
    except:
        return False

def clean_path(path):
    return str(Path(path).expanduser().resolve())

def parse_args():
 
    parser = argparse.ArgumentParser(description="NGen calibration data prep")
 
    # Positional (required) arguments
    parser.add_argument("gage_id",       type=str, help="gage_id to perform data collection on")
    parser.add_argument("start_date", type=str, help="Singularity image path with compiled ngen binary")
    parser.add_argument("end_date", type=str, help="Singularity image path with compiled ngen binary")
    parser.add_argument("--method", type=str, nargs="?", default=f"local", help="[remote | local] prefer gathering data from remote or local sources (default: local)")
    parser.add_argument("--forcings", type=bool, nargs="?", default=True, help="Create forcing files (default: True)")
    parser.add_argument("--gpkg", type=bool, nargs="?", default=True, help="Fetch hydrofabric (default: True)")
    parser.add_argument("--save_dir", type=str, nargs="?", default=f".", help="Save directory for gathered data (default: .)")
    
    args = parser.parse_args()
    args.start_date = pd.to_datetime(args.start_date)
    args.end_date = pd.to_datetime(args.end_date)

    return args

def ngiab_data_retrieval(site,start,end,save_dir,tries=1):
    cmd = ["uvx",
           "ngiab-prep",
           "-i",f"{site}",
           "--start", f"{start}",
           "--end", f"{end}",
           "-sfr",
           "--source", "aorc", 
           "--output_root", f"{save_dir}"]
    
    for _ in range(tries):
        try:
            subprocess.run(cmd, check=True)
        except:
            continue
        break

def obs_data_retrieval(gage_num, start_dt, end_dt, save=True, save_as=None):
    site = gage_num
    start = (start_dt - pd.Timedelta(hours=14)).strftime("%Y-%m-%dT%H:%M")
    end = (end_dt   + pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M")

    # official USGS timezone offsets, offsets from UTC: https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-zone-codes/items?limit=100
    tz_offsets = {
        "ACST": "+09:30",
        "ACSST": "+10:30",
        "AEST": "+10:00",
        "AESST": "+11:00",
        "AFT": "+04:30",
        "AKST": "-09:00",
        "AKDT": "-08:00",
        "AST": "-04:00",
        "ADT": "-03:00",
        "AWST": "+08:00",
        "AWSST": "+09:00",
        "BT": "+03:00",
        "CAST": "+09:30",
        "CADT": "+10:30",
        "CCT": "+08:00",
        "CET": "+01:00",
        "CETDST": "+02:00",
        "CST": "-06:00",
        "CDT": "-05:00",
        "DNT": "+01:00",
        "DST": "+01:00",
        "EAST": "+10:00",
        "EASST": "+11:00",
        "EET": "+02:00",
        "EETDST": "+03:00",
        "EST": "-05:00",
        "EDT": "-04:00",
        "FST": "+01:00",
        "FWT": "+02:00",
        "GST": "+10:00",
        "HST": "-10:00",
        "HDT": "-09:00",
        "IDLE": "+12:00",
        "IDLW": "-12:00",
        "IST": "+02:00",
        "IT": "+03:30",
        "JST": "+09:00",
        "JT": "+07:30",
        "KST": "+09:00",
        "LIGT": "+10:00",
        "MET": "+01:00",
        "METDST": "+02:00",
        "MEWT": "+01:00",
        "MEST": "+02:00",
        "MEZ": "+01:00",
        "MST": "-07:00",
        "MDT": "-06:00",
        "MT": "+08:30",
        "NFT": "-03:30",
        "NDT": "-02:30",
        "NOR": "+01:00",
        "NST": "-03:30",
        "NZST": "+12:00",
        "NZDT": "+13:00",
        "NZT": "+12:00",
        "PST": "-08:00",
        "PDT": "-07:00",
        "SAT": "+09:30",
        "SADT": "+10:30",
        "SET": "+01:00",
        "SWT": "+01:00",
        "SST": "+02:00",
        "UTC": "+00:00",
        "WAST": "+07:00",
        "WADT": "+08:00",
        "WAT": "-01:00",
        "WET": "+00:00",
        "WETDST": "+01:00",
        "WST": "+08:00",
        "WDT": "+09:00",
        "ZP-11": "-11:00",
        "ZP-2": "-02:00",
        "ZP-3": "-03:00",
        "ZP11": "+11:00",
        "ZP4": "+04:00",
        "ZP5": "+05:00",
        "ZP6": "+06:00",
    }

#need to shift start/end to make it land on desired times after UTC conversion
    url = (
        "https://nwis.waterservices.usgs.gov/nwis/iv/"
        f"?format=rdb"
        f"&sites={site}"
        f"&startDT={start}"
        f"&endDT={end}"
        f"&siteStatus=all"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    text = response.text

    station_name = None
    lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            if (station_name is None) and (site in line):
                parts = line.split(site, 1)
                if len(parts) == 2:
                    remainder = parts[1].strip()
                    station_name = remainder.split(",", 1)[0].strip()
        else:
            lines.append(line)
    lines = [lines[0]] + lines[2:] # drop data description row

    df = pd.read_csv(StringIO("\n".join(lines)),sep="\t")
    flow_col = [c for c in df.columns if "00060" in c and not c.endswith("_cd")][0] # discharge columns marked with code 00060
    df["datetime_with_offset"] = (
        df["datetime"].astype(str)
        + df["tz_cd"].map(tz_offsets)
    )
    df["value_date"] = pd.to_datetime(
        df["datetime_with_offset"],
        utc=True
    ).dt.tz_localize(None)

    df = df[(df["value_date"] >= start_dt) & (df["value_date"] <= end_dt)]
    df["obs_flow"] = pd.to_numeric(df[flow_col],errors="coerce")
    df["obs_flow"] = df["obs_flow"] / 35.3146667215
    df = df[["value_date", "obs_flow"]]
    df = df.groupby(level=0).mean()

    if save or (save_as != None):
        default_save_dir = station_name if station_name else f"{site}"
        save_as = save_as if save_as else clean_path(os.path.join(".",default_save_dir,"obs_hourly_discharge_cms.csv"))
        Path(clean_path(save_as)).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            save_as,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S"
        )

    return station_name, df


if __name__ == "__main__":
    args = parse_args()
    obs_data_retrieval(args.gage_id, args.start_date, args.end_date)