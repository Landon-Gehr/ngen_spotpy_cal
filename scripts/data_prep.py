import sys
import os
import shutil
import argparse
import subprocess
from datetime import datetime
import requests
from io import StringIO
from pathlib import Path
import json
import yaml

import numpy as np
import pandas as pd
import xarray as xr

# official USGS timezone offsets, offsets from UTC: https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-zone-codes/items?limit=100
TZ_OFFSETS = {
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

DEFAULT_CFE_PARAMS = [
    {
        "name": "b",
        "min": 2.0,
        "max": 15.0,
        "init": 4.05,
    },
    {
        "name": "satpsi",
        "min": 0.03,
        "max": 0.955,
        "init": 0.355,
    },
    {
        "name": "satdk",
        "min": 1.0e-07,
        "max": 0.000726,
        "init": 3.38e-06,
    },
    {
        "name": "maxsmc",
        "min": 0.16,
        "max": 0.59,
        "init": 0.439,
    },
    {
        "name": "refkdt",
        "min": 0.1,
        "max": 4.0,
        "init": 1.0,
    },
    {
        "name": "slope",
        "min": 0.0,
        "max": 1.0,
        "init": 0.1,
    },
    {
        "name": "max_gw_storage",
        "min": 0.01,
        "max": 0.25,
        "init": 0.05,
    },
    {
        "name": "expon",
        "min": 1.0,
        "max": 8.0,
        "init": 3.0,
    },
    {
        "name": "Cgw",
        "min": 1.8e-06,
        "max": 0.0018,
        "init": 1.8e-05,
    },
    {
        "name": "Klf",
        "min": 0.0,
        "max": 1.0,
        "init": 0.01,
    },
    {
        "name": "Kn",
        "min": 0.0,
        "max": 1.0,
        "init": 0.03,
    },
]

DEFAULT_NOAH_PARAMS = [
    {
        "name": "RSURF_EXP",
        "min": 1.0,
        "max": 6.0,
        "init": 5.0,
    },
    {
        "name": "CWP",
        "min": 0.09,
        "max": 0.36,
        "init": 0.18,
    },
    {
        "name": "MP",
        "min": 3.6,
        "max": 12.6,
        "init": 9.0,
    },
    {
        "name": "VCMX25",
        "min": 24.0,
        "max": 112.0,
        "init": 52.2,
    },
    {
        "name": "MFSNO",
        "min": 0.5,
        "max": 4.0,
        "init": 2.0,
    },
    {
        "name": "RSURF_SNOW",
        "min": 0.136,
        "max": 100.0,
        "init": 50.0,
    },
    {
        "name": "SCAMAX",
        "min": 0.7,
        "max": 1.0,
        "init": 0.9,
    },
]

def load_params(path, key=None, default=None):
    if (not path) or (path == ""):
        if default is not None:
            return default
        else:
            return {}

    path = clean_path(path)

    with open(path, "r") as f:
        params = json.load(f)
        if key:
            params = params[key]
        return params

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

    # Required arguments
    parser.add_argument("gage_id",type=str,help="USGS gage ID to perform data collection on")
    parser.add_argument("start_date",type=str,help="Start date of the data collection/calibration period")
    parser.add_argument("end_date",type=str,help="End date of the data collection/calibration period")

    # Optional data selection
    parser.add_argument("--cat-id",type=str,default="",help="Catchment ID to perform data collection on")
    parser.add_argument("--forcings",action=argparse.BooleanOptionalAction,default=True,help="Create forcing files (default: True)")
    parser.add_argument("--gpkg",action=argparse.BooleanOptionalAction,default=True,help="Fetch hydrofabric (default: True)")
    parser.add_argument("--save-dir",type=str,default=".",help="Save directory for gathered data (default: .)")
    parser.add_argument("--obs-save-path",type=str,default="",help="Save path for gathered observed flow data (default: <save_dir>/obs_quarterly_discharge_cms.csv)")

    # Calibration configuration
    parser.add_argument("--image",type=str,default="",help="Path to the Singularity ngen image")
    parser.add_argument("--iterations",type=int,default=1000,help="Number of calibration iterations (default: 1000)")
    parser.add_argument("--cfe-params",type=str,default="",help="Path to JSON file containing CFE calibration parameters")
    parser.add_argument("--noah-params",type=str,default="",help="Path to JSON file containing NoahOWP calibration parameters")

    # Optional realization/troute templates
    parser.add_argument("--template-realization",type=str,default="",help="Path to template realization JSON")
    parser.add_argument("--template-troute",type=str,default="",help="Path to template t-route YAML")

    # sbatch creation args
    parser.add_argument("--nodes", type=int, default=2, help="Number of SLURM nodes (default: 2)")
    parser.add_argument("--cpus-per-node", type=int, default=4, help="Number of tasks/CPUs per node (default: 4)")
    parser.add_argument("--time", type=str, default="00:10:00", help="SLURM time limit (default: 00:10:00)")
    parser.add_argument("--restarts", type=int, default=0, help="Maximum number of automatic restarts (default: 0)")

    args = parser.parse_args()

    args.start_date = pd.to_datetime(args.start_date)
    args.end_date = pd.to_datetime(args.end_date)

    args.save_dir = clean_path(args.save_dir)
    args.image = clean_path(args.image)
    args.template_realization = clean_path(args.template_realization)
    args.template_troute = clean_path(args.template_troute)

    args.cfe_params = load_params(args.cfe_params, default=DEFAULT_CFE_PARAMS)
    args.noah_params = load_params(args.noah_params, default=DEFAULT_NOAH_PARAMS)

    if args.obs_save_path == "":
        args.obs_save_path = str(args.save_dir) + "<SITE_NAME>" + "obs_quarterly_discharge_cms.csv"

    return args

def ngiab_data_retrieval(site,start,end,save_dir, output_name=None, use_gage_id=False, tries=1):
    # https://github.com/CIROH-UA/NGIAB_data_preprocess#installation-and-running
    start_time = start.strftime("%Y-%m-%d")
    end_time = end.strftime("%Y-%m-%d")

    cmd = ["uvx",
           "ngiab-prep",
           "-i"]
    if use_gage_id:
        cmd = cmd + [f"{site}","-g"]
    else:
        cmd = cmd + [f"cat-{site}"]
    cmd = cmd + [
           "--start", f"{start_time}",
           "--end", f"{end_time}",
           "-s", # subset nexus to given feature
           "-f", # generate forcings
           "-r", # create a realization file
           "--source", "aorc", 
           "--output_root", f"{save_dir}"
           ]
    if output_name:
        cmd = cmd + ["--output_name",f"{output_name}"]
    
    for n in range(tries):
        try:
            print(f"Attempt {n}: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            # subprocess.run(cmd, input="y\n", text=True, check=True, stdout=subprocess.DEVNULL)
        except:
            continue
        break

def obs_data_retrieval(gage_num, start_dt, end_dt, save=True, save_as=None, tries=1):
    site = gage_num
    start = (start_dt - pd.Timedelta(hours=14)).strftime("%Y-%m-%dT%H:%M")
    end = (end_dt   + pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M")

    #need to shift start/end to make it land on desired times after UTC conversion
    url = (
        "https://nwis.waterservices.usgs.gov/nwis/iv/"
        f"?format=rdb"
        f"&sites={site}"
        f"&startDT={start}"
        f"&endDT={end}"
        f"&siteStatus=all"
    )
    for _ in range(tries):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
        except:
            continue
        break

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

    if station_name is None:
        station_name = f"gage-{args.gage_id}"

    df = pd.read_csv(StringIO("\n".join(lines)),sep="\t")
    flow_col = [c for c in df.columns if "00060" in c and not c.endswith("_cd")][0] # discharge columns marked with code 00060
    df["datetime_with_offset"] = (
        df["datetime"].astype(str)
        + df["tz_cd"].map(TZ_OFFSETS)
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
        default_save_dir = default_save_dir.replace(" ","_")
        save_as = save_as if save_as else clean_path(os.path.join(".",default_save_dir,"obs_quarterly_discharge_cms.csv"))
        Path(clean_path(save_as)).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            save_as,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S"
        )

    return station_name, df

def postprocess_ngen_data(output_dir,output_name,gage_id,cat_id,start_date,end_date,image,iterations,template_realization,template_troute,cfe_params,noah_params,obs_csv_path):
    
    # ngiab prep cli generates data in the following structure:
    # STATION_NAME/ 
    #     config/
    #         cat_config/
    #         STATION_NAME_subset.gpkg
    #         realization.json
    #         troute.yaml
    #     forcings/
    #     metadata/
    #     outputs/ 
    #     obs_quarterly_discharge_cms.csv # generated by obs_data_retrieval not ngiab_prep cli

    output_dir = Path(output_dir)
    config_dir = output_dir / "config"

    for directory in ["metadata", "outputs"]: # remove empty/useless directories
        path = output_dir / directory

        if path.exists():
            shutil.rmtree(path)

    # Replace generated realization/troute with templates if supplied
    realization_path = config_dir / "realization.json"
    troute_path = config_dir / "troute.yaml"

    if template_realization:
        template_realization = Path(template_realization).expanduser().resolve()

        if not template_realization.exists():
            raise FileNotFoundError(f"Template realization file does not exist: {template_realization}")

        realization_path.unlink(missing_ok=True)

        # Preserve the template filename.
        shutil.copy2(template_realization,config_dir / template_realization.name)
        realization_path = config_dir / template_realization.name

    if template_troute:
        template_troute = Path(template_troute).expanduser().resolve()

        if not template_troute.exists():
            raise FileNotFoundError(f"Template t-route file does not exist: {template_troute}")

        troute_path.unlink(missing_ok=True)

        # Preserve the template filename.
        shutil.copy2(template_troute, config_dir / template_troute.name)
        troute_path = config_dir / template_troute.name

    # Generate crosswalk if both cat-id and gage-id were supplied
    crosswalk_path = ""

    if cat_id and gage_id:
        crosswalk_path = config_dir / "crosswalk.json"
        crosswalk = {f"cat-{cat_id}": {"Gage_no": str(gage_id)}}
        with open(crosswalk_path, "w") as f:
            json.dump(crosswalk, f)

    # Construct ngen_cal_conf.yaml
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Default evaluation start is one year after calibration start.
    # If that goes past the end date, use midpoint rounded to nearest day.
    eval_start_date = start_date + pd.DateOffset(years=1)
    if eval_start_date > end_date:
        midpoint = start_date + (end_date - start_date) / 2
        eval_start_date = midpoint.round("D")

    # Station name is represented with spaces in the YAML.
    site_name = output_name.replace("_", " ")

    gpkg_path = config_dir / f"{output_name}_subset.gpkg"
    forcing_path = output_dir / "forcings" / "forcings.nc"
    obsflow_path = Path(obs_csv_path)
    ngen_cal_conf_path = config_dir / "ngen_cal_conf.yaml"

    # used for creaing yaml ids
    cfe_params = cfe_params
    noah_params = noah_params

    config = {
        "general": {
            "strategy": {
                "type": "estimation",
                "algorithm": "scuea",
            },
            "name": "calib",
            "log": True,
            "workdir": str(output_dir),
            "yaml_file": str(ngen_cal_conf_path),
            "image": image,
            "start_iteration": 0,
            "iterations": iterations,
            "restart": 0,
        },

        "CFE": cfe_params,

        "NoahOWP": noah_params,

        "model": {
            "type": "ngen",
            "binary": "/dmod/bin/ngen-parallel",
            "realization": str(realization_path),
            "troute": str(troute_path),
            "gpkg": str(gpkg_path),
            "config": str(config_dir),
            "forcing": str(forcing_path),
            "crosswalk": str(crosswalk_path),
            "obsflow": str(obsflow_path),
            "strategy": "uniform",

            "params": {
                "CFE": cfe_params,
                "NoahOWP": noah_params,
            },

            "eval_params": {
                "objective": "kge",
                "start_time": start_date.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_date.strftime("%Y-%m-%d %H:%M:%S"),
                "eval_start_time": eval_start_date.strftime("%Y-%m-%d %H:%M:%S"),
                "eval_end_time": end_date.strftime("%Y-%m-%d %H:%M:%S"),
                "save_output_iteration": 0,
                "save_plot_iteration": 0,
                "save_plot_iter_freq": 50,
                "basinID": str(gage_id),
                "threshold": None,
                "site_name": site_name,
                "user": "",
            },
        },
    }

    with open(ngen_cal_conf_path, "w") as f:
        yaml.safe_dump(config,f,sort_keys=False,default_flow_style=False)

    return ngen_cal_conf_path

def create_slurm_script(config_file, output_dir, singularity_image_path, nodes=2, cpus_per_node=4, time="00:10:00", restarts=0):
    output_dir = Path(output_dir)

    script_path = output_dir / "scuea_optimization.sbatch"

    python_script = "/home/lsgehr/scratch/NextGen/ngen_spotpy_cal/scripts/spotpy_calibration.py"

    user = os.environ.get("USER", "")
    scratch_bind = f"/scratch/{user}:/scratch/{user}"

    conda_env = "ngen"

    script = f"""#!/bin/bash
#SBATCH --job-name="scuea-optimization"
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={cpus_per_node}
#SBATCH --cpus-per-task=1
#SBATCH --output="script_outputs/scuea_optimization-%j.out"
#SBATCH --time={time}
#SBATCH --signal=SIGTERM@90

module purge
module load compilers/gcc/12.2.1
module load mpi/openmpi/gcc/4.1.6

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {conda_env}

export OMPI_MCA_btl_vader_single_copy_mechanism=none

python_script="{python_script}"
config_file="{config_file}"
scratch_bind="{scratch_bind}"

export PYTHONUNBUFFERED=1
RESTARTS=${{RESTARTS:-0}}
MAX_RESTARTS=${{MAX_RESTARTS:-{restarts}}}

srun --nodes=1 \\
     --ntasks-per-node=$SLURM_NTASKS_PER_NODE \\
     --cpu-bind=NONE \\
     --exclusive \\
     python $python_script $config_file \\
        --ngs=20 \\
        --bind=$scratch_bind \\
        --verbosity=warning

EXIT_CODE=$?

if [ $EXIT_CODE -eq 42 ]; then
    if [ $RESTARTS -lt $MAX_RESTARTS ]; then
        NEW_RESTARTS=$((RESTARTS + 1))
        echo "Restarting ($NEW_RESTARTS/$MAX_RESTARTS)..."
        sbatch --export=ALL,RESTARTS=$NEW_RESTARTS $0
    else
        echo "Max restarts ($MAX_RESTARTS) reached, not resubmitting."
    fi
else
    echo "Job finished with exit code $EXIT_CODE."
fi
"""

    script_path.write_text(script)
    script_path.chmod(0o755)

    return script_path


if __name__ == "__main__":
    args = parse_args()
    args.site_id = args.cat_id if args.cat_id != "" else args.gage_id
    station_name, obs_df = obs_data_retrieval(args.gage_id, args.start_date, args.end_date, save=False)
    output_name=station_name.replace(" ","_")
    ngiab_data_retrieval(args.site_id, args.start_date, args.end_date, args.save_dir, output_name=output_name, tries=1)
    output_dir = Path(args.save_dir) / output_name
    if (not check_path(args.obs_save_path)) or ("<SITE_NAME>" in args.obs_save_path):
        args.obs_save_path = output_dir / "obs_quarterly_discharge_cms.csv"
    obs_df.to_csv(args.obs_save_path,index=False,date_format="%Y-%m-%d %H:%M:%S")
    print("Generating conf yaml")
    config_file = postprocess_ngen_data(output_dir,output_name,
                          args.gage_id,args.cat_id,
                          args.start_date,args.end_date,
                          args.image,args.iterations,args.template_realization,args.template_troute,
                          args.cfe_params,args.noah_params,
                          args.obs_save_path
                        )
    slurm_script = create_slurm_script(
        config_file=config_file,
        output_dir=output_dir,
        singularity_image_path=args.image,
        nodes=args.nodes,
        cpus_per_node=args.cpus_per_node,
        time=args.time,
        restarts=args.restarts
    )
    print(f"Finished creating data at {output_dir}")