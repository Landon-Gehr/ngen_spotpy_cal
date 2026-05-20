import sys
import os
import yaml
import json
import shutil
import subprocess
import argparse
from datetime import datetime
import time

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

def check_path(path):
    try:
        p = Path(path).expanduser().resolve()
        return p.exists()
    except:
        return False

def clean_path(path):
    return str(Path(path).expanduser().resolve())

def parse_args():
    parser = argparse.ArgumentParser(description="NGen runner")
    parser.add_argument("--run-config-path", type=str, default=None)
    args, _ = parser.parse_known_args()

    defaults = {}
    if args.run_config_path:
        defaults = yaml.safe_load(open(args.run_config_path, 'r'))

    path_args = [
        "singularity-image-path",
        "run-dir",
        "data-config-dir",
        "gpkg-file-path",
        "forcings-file-path",
        "observed-flow-path",
        "template-realization",
        "template-routing",
        "runscript-path",
        "params-json"
    ]
    non_path_args = [
        "start-date",
        "eval-start-date",
        "end-date",
        "bind",
        "feature-id",
    ]
    tasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE", 1)
    optional_args = {
        "ngen-parallel":tasks_per_node,
        "figure-output-dir":clean_path("./figures"),
        "sim-verbosity":"none",
        "gage-id":"",
        "site-name":"",
        "stream-output-frequency":15,
    }

    all_args = path_args + non_path_args + list(optional_args.keys())

    for arg in all_args:
        key = arg.replace("-", "_")
        if key in defaults:
            parser.add_argument(f"--{arg}", type=str, default=defaults[key])
        elif arg in list(optional_args.keys()):
            parser.add_argument(f"--{arg}", type=str, default=optional_args[arg])
        else:
            parser.add_argument(f"--{arg}", type=str, required=True)

    args = parser.parse_args()

    for p in path_args:
        p = p.replace("-", "_")
        val = getattr(args, p)
        if not check_path(val):
            raise ValueError(f"Invalid path received: {val}")
        setattr(args, p, clean_path(val))

    args.start_date = pd.to_datetime(args.start_date)
    args.eval_start_date = pd.to_datetime(args.eval_start_date)
    args.end_date   = pd.to_datetime(args.end_date)
    args.feature_id = int(args.feature_id)
    args.stream_output_frequency = int(args.stream_output_frequency)
    args.ngen_parallel = int(args.ngen_parallel)

    return args


def directory_setup(args):
    args.image_run_dir = os.path.join("/ngen/ngen/data","run")
    os.makedirs(args.run_dir, exist_ok=True)
 
    input_path = os.path.join(args.run_dir, "inputs")
    args.image_input_path = os.path.join(args.image_run_dir,"inputs")
    os.makedirs(input_path, exist_ok=True)
 
    args.root_output_dir = os.path.join(args.run_dir, "outputs")
    
    args.ngen_output_path = os.path.join(args.root_output_dir, "ngen")
    args.troute_output_path = os.path.join(args.root_output_dir, "troute")
    os.makedirs(args.root_output_dir, exist_ok=True)
    os.makedirs(args.ngen_output_path, exist_ok=True)
    os.makedirs(args.troute_output_path, exist_ok=True)

    args.image_output_path = os.path.join(args.image_run_dir, "outputs")
    args.image_ngen_output_path = os.path.join(args.image_output_path, "ngen")
    args.image_troute_output_path = os.path.join(args.image_output_path, "troute")
 
    args.realization_path = os.path.join(input_path, "realization.json")
    shutil.copyfile(args.template_realization, args.realization_path)

    args.routing_path = os.path.join(input_path, "routing.yaml")
    shutil.copyfile(args.template_routing, args.routing_path)

    binds = args.bind
    args.image_config_path = os.path.join("/ngen/ngen/data","config")
    args.image_forcing_path = os.path.join("/ngen/ngen/data","forcings","forcings.nc")
    args.realization_image_path = os.path.join(args.image_input_path, "realization.json")
    args.routing_image_path = os.path.join(args.image_input_path, "routing.yaml")
    args.gpkg_file = os.path.basename(args.gpkg_file_path)
    args.gpkg_image_path = os.path.join("/ngen/ngen/data","config",args.gpkg_file)
    args.image_runscript_path = os.path.join("/ngen/ngen/data",os.path.basename(args.runscript_path))
    binds += (
        f",{args.run_dir}:{args.image_run_dir}"
        f",{args.data_config_dir}:{args.image_config_path}:ro"
        f",{args.forcings_file_path}:{args.image_forcing_path}:ro"
        f",{args.gpkg_file_path}:{args.gpkg_image_path}:ro"
        f",{args.root_output_dir}:{args.image_output_path}"
        f",{args.runscript_path}:{args.image_runscript_path}:ro"
        )
    args.bind = binds


class NgenRun():
    MODULE_PARAM_MAP = {
            "CFE":["b","satpsi","satdk","maxsmc","refkdt","slope","max_gw_storage","expon","Cgw","Klf","Kn"],
            "NoahOWP":["RSURF_EXP","CWP","MP","VCMX25","MFSNO","RSURF_SNOW","SCAMAX"]
        }
    def __init__(self, args):
        self.args = args
        self.realization = self.load_realization(self.args.realization_path)
        self.update_realization()
        self.troute_yaml = self.read_yaml(self.args.routing_path)
        self.observed = self.read_observed(self.args.observed_flow_path, self.args.start_date, self.args.end_date)

    def load_realization(self, realization_file):
        jsondata = json.load(open(realization_file,"r"))
        jsondata["time"]["start_time"] = self.args.start_date.strftime("%Y-%m-%d %H:%M:%S")
        jsondata["time"]["end_time"] = self.args.end_date.strftime("%Y-%m-%d %H:%M:%S")
        jsondata["routing"]["t_route_config_file_with_path"] = self.args.routing_image_path
        jsondata["output_root"] = self.args.image_ngen_output_path
        json.dump(jsondata, open(realization_file, "w"),indent=4)
        return jsondata
    
    def update_realization(self, wipe_prev=True):
        modules = self.realization["global"]["formulations"][0]["params"]["modules"] # assumes formulations is array of size 1
        if wipe_prev:
            for i, module in enumerate(modules): # Wipe past params
                if self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_type_name"] in list(self.MODULE_PARAM_MAP.keys()):
                    self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_params"] = {}
        for i, module in enumerate(modules): # create if not exists
            if (self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_type_name"] in list(self.MODULE_PARAM_MAP.keys())) and ("model_params" not in self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]):
                self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_params"] = {}
        for module_name in list(self.MODULE_PARAM_MAP.keys()):
            for param in self.MODULE_PARAM_MAP[module_name]:
                for i, module in enumerate(modules): # interate across all modules and all params per module
                    if module["params"]["model_type_name"] == module_name:
                        self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_params"][param] = self.args.params[param]

        json.dump(self.realization, open(self.args.realization_path, "w"),indent=4)
    
    def read_yaml(self, yaml_file):
        ymldata = yaml.safe_load(open(yaml_file, 'r'))
        ymldata["compute_parameters"]["restart_parameters"]["start_datetime"] = self.args.start_date.strftime("%Y-%m-%d %H:%M:%S")
        time_diff = self.args.end_date - self.args.start_date
        self.args.nts = (time_diff.total_seconds() / 60) / 5 # in 5 minute intervals
        ymldata["compute_parameters"]["forcing_parameters"]["nts"] = self.args.nts
        ymldata["compute_parameters"]["forcing_parameters"]["max_loop_size"] = self.args.nts
        ymldata["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"] = self.args.gpkg_image_path
        ymldata["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = self.args.image_ngen_output_path
        ymldata["output_parameters"]["stream_output"]["stream_output_directory"] = self.args.image_troute_output_path
        ymldata["output_parameters"]["stream_output"]["stream_output_internal_frequency"] = self.args.stream_output_frequency
        yaml.dump(ymldata, open(yaml_file, 'w'))
        return ymldata
    
    def read_observed(self, observed_file, start_date, end_date, freq_minutes=15):
        df = pd.read_csv(observed_file, parse_dates=["value_date"])
        df["value_date"] = df["value_date"].dt.round(f"{freq_minutes}min")
        df = df.groupby("value_date")["obs_flow"].mean().reset_index()
        df = df.set_index("value_date").sort_index()
        df = df.loc[start_date:end_date]
        full_index = pd.date_range(start=start_date, end=end_date, freq=f"{freq_minutes}min")[:-1]
        df = df.reindex(full_index)
        obs = df["obs_flow"].to_numpy()
        return obs

    def run_ngen(self):

        ngen_command = ["srun",
                        "--nodes=1",
                        f"--ntasks-per-node={self.args.ngen_parallel}",
                        "--cpus-per-task=1",
                        "--cpu-bind=NONE", 
                        "--exclusive",
                        # "--overlap",
                        "--exact",
                        "singularity","exec",
                        "--bind",f"{self.args.bind}",
                        self.args.singularity_image_path,
                        self.args.image_runscript_path,
                        self.args.gpkg_image_path,
                        self.args.realization_image_path,
                        os.environ.get("SLURM_NTASKS"),
                        self.args.image_run_dir,
                        "/dmod/bin/ngen-parallel",
                        self.args.sim_verbosity,
                        str(self.args.rank)
                        ]

        
        start = time.time()
        subprocess.run(ngen_command, check=True)
        runtime = time.time() - start
        print(f"subprocess finished in {runtime} seconds")

def kge(evaluation, simulation, sim_start, sim_stop, eval_start, freq=15, return_values=False): 
    full_index = pd.date_range(start=sim_start, end=sim_stop, freq=f"{freq}min")[:-1]

    clean_mask = (~(np.isnan(evaluation)))
    full_index_clean = full_index[clean_mask]
    eval_mask = (
        (full_index_clean >= eval_start) &
        (full_index_clean <= sim_stop)
    )
    # eval_index = full_index_clean[eval_mask]

    evaluation_clean = evaluation[clean_mask]
    simulation_clean = simulation[clean_mask]
    evaluation_eval = evaluation_clean[eval_mask]
    simulation_eval = simulation_clean[eval_mask]

    r = np.corrcoef(evaluation_eval,simulation_eval)[0,1]
    a = np.std(evaluation_eval) / np.std(simulation_eval)
    b = np.mean(evaluation_eval) / np.mean(simulation_eval)
    kge_score = 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2)

    if return_values:
        return kge_score, (full_index_clean, evaluation_clean, simulation_clean)
    return kge_score

if __name__ == "__main__":

    args = parse_args()
    args.rank=0
    args.params = json.load(open(args.params_json))
    directory_setup(args)
    runner = NgenRun(args)
    runner.run_ngen()
    filename = args.start_date.strftime("troute_output_%Y%m%d%H%M") + ".nc"
    ds = xr.open_dataset(os.path.join(args.troute_output_path, filename))
    sim_res = ds['flow'].sel(feature_id=args.feature_id).values

    kge_score, (indices, obs_vals, sim_vals) = kge(runner.observed, sim_res, args.start_date, args.end_date, args.eval_start_date, args.stream_output_frequency, True)

    print(f"KGE: {kge_score}")

    plt.figure()
    plt.title(f"{args.site_name} ({args.gage_id})\nsimulated vs observed values\nKGE:{kge_score}")
    plt.plot(indices, obs_vals, label="observed flow")
    plt.plot(indices, sim_vals, label="simulated flow")
    plt.axvline(
        args.eval_start_date,
        linestyle="--",
        color="black",
        label="evaluation start"
    )
    plt.xticks(rotation=45) 
    plt.legend()
    plt.savefig(os.path.join(args.figure_output_dir,"run_ngen_obs_vs_sim.png"))