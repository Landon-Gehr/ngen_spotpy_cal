import sys
import os
import yaml
import json
import shutil
import subprocess
import traceback
import argparse
from pprint import pprint
from datetime import datetime
import logging
import time
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import spotpy
from mpi4py import MPI
WORLD = None

def out_of_time_cleanup(args, signum, frame):
    rank = args.rank
    args.logger.info(f"[rank {rank}] SIGTERM received", flush=True)
    MPI.Finalize()
    sys.exit(42)

class NgenLogger:
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
        "none": logging.CRITICAL + 10
    }

    LEVELS_REV = {v: k for k, v in LEVELS.items()}

    def no_log(*args, **kwargs):
        pass

    def __init__(self, name="ngen_logger", default_level="info", stream=sys.stdout, prefix=""):
        
        self.logger = logging.getLogger(name)
        self.default_level = "none" if default_level == None else default_level.lower()
        self.default_logging = self.LEVELS[default_level]
        self.default_logging_call = getattr(self.logger, default_level, self.no_log)
        self.logger_prefix = prefix

        if not self.logger.handlers:
            handler = logging.StreamHandler(stream)
            formatter = logging.Formatter(
                fmt=f"{self.logger_prefix} : %(levelname)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.propagate = False
        self.set_level(default_level)

        self.debug = lambda msg, flush=False: self.log(msg, "debug", flush)
        self.info = lambda msg, flush=False: self.log(msg, "info", flush)
        self.warning = lambda msg, flush=False: self.log(msg, "warning", flush)
        self.error = lambda msg, flush=False: self.log(msg, "error", flush)
        self.critical = lambda msg, flush=False: self.log(msg, "critical", flush)
        self.none = lambda msg, flush=False: self.log(msg, "none", flush)


    def set_level(self, level):
        level = level.lower()
        self.logger.setLevel(self.LEVELS.get(level, self.default_logging))
        self.logging_level = self.LEVELS_REV[self.logger.level]

    def log(self, msg, level=None, flush=False):
        level = level.lower() if level else self.default_level.lower()
        log_fn = getattr(self.logger, level, self.default_logging_call)
        log_fn(msg)

        if flush:
            for handler in self.logger.handlers:
                handler.flush()

def check_path(path):
    try:
        p = Path(path).expanduser().resolve()
        return p.exists()
    except:
        return False

def clean_path(path):
    return str(Path(path).expanduser().resolve())

def parse_args():
 
    parser = argparse.ArgumentParser(description="NGen calibration runner")
 
    # Positional (required) arguments
    parser.add_argument("calibration_conf",       type=str, help="location of calibration config yaml")
    args, _ = parser.parse_known_args()

    with open(args.calibration_conf, 'r') as f:
        defaults = yaml.safe_load(f)
    crosswalk = defaults["model"].get("crosswalk",None)
    feature_id = None
    gage_id = None
    if check_path(crosswalk):
        with open(crosswalk,"r") as f:
            crosswalk = json.load(f)
        feature_id = int(((list(crosswalk.keys())[0]).split("-"))[1])
        gage_id = crosswalk[list(crosswalk.keys())[0]]["Gage_no"]    

    path_args = [
        {"name": "gpkg-file-path",        "type": str, "default":defaults["model"].get("gpkg", None), "help": "Path to gpkg file with catchment and nexus data (default: read by conf file)"},
        {"name": "observed-flow-path",    "type": str, "default":defaults["model"].get("obsflow", None), "help": "Path to observed flow data (default: read by conf file)"},
        {"name": "template-realization",  "type": str, "default":defaults["model"].get("realization", None), "help": "Path to realization template JSON (default: read by conf file)"},
        {"name": "template-routing",      "type": str, "default":defaults["model"].get("troute", None), "help": "Path to routing template YAML (default: read by conf file)"},
        {"name": "data-config-path",      "type": str, "default":defaults["model"].get("config", None), "help": "Path to directory containing /cat_config with CFE and NOAH-OWP-M (default: read by conf file)"},
        {"name": "data-forcing-path",     "type": str, "default":defaults["model"].get("forcing", None), "help": "Path to aorc forcings nc file (default: read by conf file)"},
        {"name": "workdir-path",          "type": str, "default":defaults["general"].get("workdir", None), "help": "Path to use as working directory (default: read by conf file)"},
        {"name": "singularity-image-path","type": str, "default":defaults["general"].get("image", None), "help": "Path to Singularity image path with compiled ngen binary (default: read by conf file)"},
    ]

    non_path_args = [
        {"name": "feature-id",      "type": int, "default":defaults["model"]["eval_params"].get("featureID",feature_id), "help": "Feature id to be used for flow evaluation (default: read by crosswalk file)"},
        {"name": "gage-id",         "type": str, "default":defaults["model"]["eval_params"].get("basinID",gage_id), "help": "Gage ID (default: read by crosswalk file)"},
        {"name": "start-time",      "type": str, "default":defaults["model"]["eval_params"].get("start_time",None), "help": "date to start simulation (default: read by conf file)"},
        {"name": "end-time",        "type": str, "default":defaults["model"]["eval_params"].get("end_time",None), "help": "date to stop simulation (default: read by conf file)"},
        {"name": "eval-start-time", "type": str, "default":defaults["model"]["eval_params"].get("eval_start_time",None), "help": "date to start evaluation (default: read by conf file)"},
        {"name": "eval-end-time",   "type": str, "default":defaults["model"]["eval_params"].get("eval_end_time",None), "help": "date to stop evaluation (default: read by conf file)"},
        {"name": "sampling-reps",   "type": int, "default":defaults["general"].get("iterations",None), "help": "Number of sampling repetitions (default: read by conf file)"},
    ]

    non_opt_args = path_args + non_path_args
    for arg in non_opt_args:
        if arg["default"] != None:
            parser.add_argument(f"--{arg['name']}", type=arg["type"], default=arg["default"], help=arg["help"])
        else:
            parser.add_argument(f"--{arg['name']}", type=arg["type"], required=True, help=arg["help"])
    args, _ = parser.parse_known_args()
    tasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE", 1)
    run_path = os.path.join(args.workdir_path,"run")
    dbpath = os.path.join(args.workdir_path,"ngen_param_tuning.csv")
    best_save_dir = os.path.join(args.workdir_path,"best")

    optional_path_args = [
        {"name": "model-binary",             "type": str, "default": "/dmod/bin/ngen-parallel",   "help": "Path to binary to execute inside the singularity (default: /dmod/bin/ngen-parallel)"},
        {"name": "singularity-run-script",   "type": str, "default": "./ngen_image_runscript.sh", "help": "Path to executable to be called by singularity exec (default: scripts/ngen_image_runscript.sh)"},
        {"name": "image-data-path",          "type": str, "default": "/ngen/ngen/data",           "help": "Image data path (default: /ngen/ngen/data)"},
        {"name": "input-path",               "type": str, "default": None,                        "help": "Input path (default: run-path/rank_N/inputs)"},
        {"name": "output-path",              "type": str, "default": None,                        "help": "Output path (default: run-path/rank_N/outputs)"},
        {"name": "run-path",                 "type": str, "default": run_path,                        "help": "Root path for each rank (default: workdir/run)"},
        {"name": "dbpath",                   "type": str, "default": dbpath,                          "help": "Path to optimization storage csv (default: workdir/ngen_param_tuning.csv)"},
        {"name": "best-save-dir",           "type": str, "default": best_save_dir,                          "help": "Directory to store best params and plot (default: workdir/best)"},
    ]

    optional_non_path_args = [
        {"name": "bind",                     "type": str, "default": "",                          "help": "Additional binds to singularity (default: '')"},
        {"name": "ngen-parallel",            "type": int, "default": int(tasks_per_node),         "help": f"NGen parallel tasks (default: SLURM_NTASKS_PER_NODE, currently {tasks_per_node})"},
        {"name": "ngs",                      "type": int, "default": 20,                          "help": "Number of SCE-UA complexes (default: 20)"},
        {"name": "routing-output-frequency", "type": int, "default": 15,                          "help": "Routing output frequency in minutes (default: 15)"},
        {"name": "verbosity",                "type": str, "default": "none",                      "help": "Control text output levels [debug, info, warning, error, critical, none] (default: none)"},
        {"name": "site-name",                "type": str, "default": defaults["model"]["eval_params"].get("site_name",""),                      "help": "Name of site being optimized (default: read from conf file)"},
    ]

    opt_args = optional_path_args + optional_non_path_args
    for arg in opt_args:
        parser.add_argument(f"--{arg['name']}", type=arg["type"], default=arg["default"], help=arg["help"])

    flag_args = [
        {"name": "no-auto-bind",    "const": True, "default": False, "help": "Disable auto binding to singularity. Omit = False, bare flag = True, explicit true/false also accepted"},
        {"name": "serial-sampling", "const": True, "default": False, "help": "Enable serial sampling. Omit = False, bare flag = True"},
        {"name": "clean",           "const": True, "default": False, "help": "Enable cleaning on finish. Omit = False, bare flag = True"},
        {"name": "save-best",       "const": True, "default": True,  "help": "Save and plot the best params (default: True)"},
    ]

    for arg in flag_args:
        parser.add_argument(f"--{arg['name']}",nargs="?",const=arg["const"],default=arg["default"], help=arg["help"])

    
    args = parser.parse_args()

    for p in path_args + optional_path_args:
        p = p["name"].replace("-", "_")
        val = getattr(args, p)
        if val:
            setattr(args, p, clean_path(val))

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    args.sim_start_dt = pd.to_datetime(args.start_time)
    args.sim_stop_dt = pd.to_datetime(args.end_time)
    args.eval_start_dt = pd.to_datetime(args.eval_start_time)
    args.eval_stop_dt = pd.to_datetime(args.eval_end_time)

    args.dbname = str(Path(args.dbpath).with_suffix(""))
    args.verbosity = args.verbosity.lower() if args.verbosity.lower() in NgenLogger.LEVELS else "info"

    return args

def directories_setup(args, rank):
    args.rank = rank
    args.rank_dir = os.path.join(args.run_path, f"rank_{rank}")
    args.image_run_dir = os.path.join(args.image_data_path,"run")
    os.makedirs(args.rank_dir, exist_ok=True)
 
    if args.input_path is None:
        input_path = os.path.join(args.rank_dir, "inputs")
    else:
        args.input_path = os.path.normpath(args.input_path)
        input_path = os.path.join(args.input_path, f"rank_{rank}")
    args.rank_input_dir = input_path
    args.image_input_path = os.path.join(args.image_run_dir,"inputs")
    os.makedirs(input_path, exist_ok=True)
 
    if args.output_path is None:
        rank_root_output_dir = os.path.join(args.rank_dir, "outputs")
    else:
        args.output_path = os.path.normpath(args.output_path)
        rank_root_output_dir = os.path.join(args.output_path, f"rank_{rank}")
    
    args.rank_root_output_dir = rank_root_output_dir 
    args.ngen_output_path = os.path.join(rank_root_output_dir, "ngen")
    args.troute_output_path = os.path.join(rank_root_output_dir, "troute")
    os.makedirs(rank_root_output_dir, exist_ok=True)
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
    if args.no_auto_bind == False:
        args.image_config_path = os.path.join(args.image_data_path,"config")
        args.image_forcing_path = os.path.join(args.image_data_path,"forcings","forcings.nc")
        # args.image_forcing_path = os.path.join(args.image_data_path,"forcings")
        args.realization_image_path = os.path.join(args.image_input_path, "realization.json")
        args.routing_image_path = os.path.join(args.image_input_path, "routing.yaml")
        args.gpkg_file = os.path.basename(args.gpkg_file_path)
        args.gpkg_image_path = os.path.join(args.image_data_path,"config",args.gpkg_file)
        args.image_runscript_path = os.path.join(args.image_data_path,os.path.basename(args.singularity_run_script))
        binds += (
            f",{args.rank_dir}:{args.image_run_dir}"
            f",{args.data_config_path}:{args.image_config_path}:ro"
            f",{args.data_forcing_path}:{args.image_forcing_path}:ro"
            f",{args.gpkg_file_path}:{args.gpkg_image_path}:ro"
            f",{args.rank_root_output_dir}:{args.image_output_path}"
            f",{args.singularity_run_script}:{args.image_runscript_path}:ro"
        )
        args.bind = binds

class NgenRun():
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        self.realization = self.load_realization(self.args.realization_path)
        self.troute_yaml = self.read_yaml(self.args.routing_path)
        self.observed = self.read_observed(self.args.observed_flow_path, self.args.sim_start_dt, self.args.sim_stop_dt, freq_minutes=self.args.routing_output_frequency)

    def load_realization(self, realization_file):
        with open(realization_file,"r") as f:
            jsondata = json.load(f)
        jsondata["time"]["start_time"] = self.args.start_time
        jsondata["time"]["end_time"] = self.args.end_time
        jsondata["routing"]["t_route_config_file_with_path"] = self.args.routing_image_path
        jsondata["output_root"] = self.args.image_ngen_output_path
        with open(realization_file,"w") as f:
            json.dump(jsondata, f,indent=4)
        return jsondata
    
    def update_realization(self, sim_params, module_map, wipe_prev=False):
        modules = self.realization["global"]["formulations"][0]["params"]["modules"] # assumes formulations is array of size 1
        if wipe_prev:
            for i, module in enumerate(modules): # Wipe past params
                if self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_type_name"] in list(module_map.keys()):
                    self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_params"] = {}
        for i, module in enumerate(modules): # create if not exists
            if (self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_type_name"] in list(module_map.keys())) and ("model_params" not in self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]):
                self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_params"] = {}
        for module_name in list(module_map.keys()):
            for param in module_map[module_name]:
                for i, module in enumerate(modules): # interate across all modules and all params per module
                    if module["params"]["model_type_name"] == module_name:
                        self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_params"][param] = sim_params[param]

        with open(self.args.realization_path, "w") as f:
            json.dump(self.realization, f,indent=4)
    
    def read_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            ymldata = yaml.safe_load(f)
        ymldata["compute_parameters"]["restart_parameters"]["start_datetime"] = self.args.start_time
        time_diff = self.args.sim_stop_dt - self.args.sim_start_dt
        self.args.nts = (time_diff.total_seconds() / 60) / 5 # in 5 minute intervals
        ymldata["compute_parameters"]["forcing_parameters"]["nts"] = self.args.nts
        ymldata["compute_parameters"]["forcing_parameters"]["max_loop_size"] = self.args.nts
        ymldata["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"] = self.args.gpkg_image_path
        ymldata["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = self.args.image_ngen_output_path
        ymldata["output_parameters"]["stream_output"]["stream_output_directory"] = self.args.image_troute_output_path
        ymldata["output_parameters"]["stream_output"]["stream_output_internal_frequency"] = self.args.routing_output_frequency
        with open(yaml_file, 'w') as f:
            yaml.dump(ymldata, f)
        return ymldata
    
    def read_observed(self, observed_file, start_date, end_date, freq_minutes=15):
        df = pd.read_csv(observed_file, parse_dates=["value_date"])
        self.logger.debug(f"df read len {len(df)}")
        df["value_date"] = df["value_date"].dt.round(f"{freq_minutes}min")
        df = df.groupby("value_date")["obs_flow"].mean().reset_index()
        df = df.set_index("value_date").sort_index()
        obs_start = df.index.min()
        obs_end   = df.index.max()
        if (start_date < obs_start) or (end_date > obs_end):
            self.logger.warning(f"date range {start_date} to {end_date} is outside observed {obs_start} to {obs_end}")
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
                        str(self.args.ngen_parallel),
                        self.args.image_run_dir,
                        self.args.model_binary,
                        self.args.verbosity,
                        str(self.args.rank)
                        ]

        
        run_dir = os.path.abspath(self.args.rank_dir)
        self.logger.debug(f"running ngen {' '.join(ngen_command)} from working directory {run_dir}", flush=True)
        try:
            start = time.time()
            subprocess.run(ngen_command, check=True)
            self.args.runtime = time.time() - start
            self.args.timings += [self.args.runtime]
            # self.args.average_runtime = np.mean(self.args.timings)
            self.args.average_runtime = self.args.average_runtime + (self.args.runtime - self.args.average_runtime)/self.args.trial_num
            self.args.logger.info(f"subprocess with {self.args.ngen_parallel} cpus finished in {self.args.runtime} seconds (avg: {self.args.average_runtime})", flush=True)
            self.args.logger.info(f"timings: {self.args.timings}")
        except Exception as e:
            traceback.print_exc()
            sys.stdout.flush()
            MPI.COMM_WORLD.Abort(1)


class SpotPySetup:
    def __init__(self, args, Ngen):
        self.logger = args.logger
        self.args = args
        self.ngen = Ngen
        self.config_yaml_path = args.calibration_conf
        self.params = self.load_params(self.config_yaml_path)
        self.save_best = self.args.save_best  
        if self.save_best:
            self.best = np.inf      

    def load_params(self,yaml_file):
        training_params_dict = {}
        with open(yaml_file) as f:
            ymldata = yaml.safe_load(f)
        self.cfe_noah_map = {'CFE':[],"NoahOWP":[]}
        for param in ymldata['CFE']:
            self.cfe_noah_map["CFE"] += [param["name"]]
        for param in ymldata['NoahOWP']:
            self.cfe_noah_map["NoahOWP"] += [param["name"]]
        
        for param in ymldata['CFE'] + ymldata['NoahOWP']:
            training_params_dict[param["name"]] = spotpy.parameter.Uniform( param["name"],
                                                                            low=param["min"],
                                                                            high=param["max"],
                                                                            optguess=param["init"])
            
        return training_params_dict

    def update_params(self,sim_params):
        self.ngen.update_realization(sim_params, self.cfe_noah_map)
        # self.ngen.update_routing()

    # return parameters list
    def parameters(self):
        return spotpy.parameter.generate(list(self.params.values()))

    # returns the optimal value simulation could achieve (the observed ground truth)
    def evaluation(self):
        return self.ngen.observed

    # performs a round of running the model with the next guessed optimal parameters
    def simulation(self,sim_params):
        self.logger.info(f"run simulation with guess {type(sim_params)} {sim_params}")
        try:
            self.args.trial_num += 1

            # # Enforce rule: Cgw <= gw_storage
            # if sim_params.get("Cgw",1.8e-05) > sim_params.get("max_gw_storage",0.05):
            #     return [np.inf]

            self.update_params(sim_params)
            self.ngen.run_ngen()
            dt = datetime.strptime(self.args.start_time, "%Y-%m-%d %H:%M:%S")
            filename = dt.strftime("troute_output_%Y%m%d%H%M") + ".nc"
            ds = xr.open_dataset(os.path.join(self.args.troute_output_path, filename), engine="netcdf4")
            sim_res = ds['flow'].sel(feature_id=self.args.feature_id).values
            # sim_res = ds['flow'].sel(feature_id=self.args.feature_id).values * 35.3147 # 35.3147 ft^3/s == 1.0 m^3/s
            self.logger.debug(f"sim len {sim_res.shape}")
        except Exception as e:
            traceback.print_exc()
            sys.stdout.flush()
            MPI.COMM_WORLD.Abort(1)

        return sim_res

    # perform loss calculation with simulated results: Kling-Gupta Efficiency (KGE)
    # 1 - sqrt((r-1)^2 + (a-1)^2 + (b-1)^2)
    # r = corr coef
    # a = std dev pred / std dev obs
    # b = mean pred / mean obs
    def objectivefunction(self, evaluation, simulation, params=None):
        self.logger.debug(f"observed data {len(evaluation)} ({type(evaluation)})")
        self.logger.debug(f"simulated data {len(simulation)} ({type(simulation)})")
        full_index = pd.date_range(start=self.args.sim_start_dt, end=self.args.sim_stop_dt, freq=f"{self.args.routing_output_frequency}min")[:-1]
        self.logger.debug(f"date range {len(full_index)} ({full_index[0]} to {full_index[-1]})")

        eval_mask = (full_index >= self.args.eval_start_dt) & (full_index <= self.args.eval_stop_dt) & (~(np.isnan(evaluation)))
        evaluation_clean = evaluation[eval_mask]
        simulation_clean = simulation[eval_mask]

        r = np.corrcoef(evaluation_clean,simulation_clean)[0,1]
        a = np.std(evaluation_clean) / np.std(simulation_clean)
        b = np.mean(evaluation_clean) / np.mean(simulation_clean)
        kge = 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2)

        if self.save_best:
            if self.best > -kge:
                self.best = -kge
                self.logger.info(f"new best {self.best}: {params}")

                self.args.best_dict = {
                    "site": self.args.site_name,
                    "gage_id": self.args.gage_id,
                    "feature_id": self.args.feature_id,
                    "simulation": {
                        "start": self.args.start_time,
                        "end": self.args.end_time
                    },
                    "evaluation": {
                        "start": self.args.eval_start_time,
                        "end": self.args.eval_end_time
                    },
                    "kge": kge,
                    "params": dict(zip(params[1], params[0]))
                }

                Path(self.args.best_save_dir).mkdir(parents=True,exist_ok=True)
                best_save_path = os.path.join(self.args.best_save_dir,f"best_params_{self.args.gage_id}.json")
                with open(best_save_path, "w") as f:
                    json.dump(self.args.best_dict, f,indent=4)

                plt.figure()
                plt.plot(full_index, evaluation, label="observed flow")
                plt.plot(full_index, simulation, label="simulated flow")
                plt.xticks(rotation=45) 
                plt.axvline(x=pd.to_datetime(self.args.eval_start_time), color="k", linestyle="--", linewidth=1, label="eval start")
                plt.title(f"{self.args.site_name} ({self.args.gage_id})\nsimulated vs observed values\nKGE:{kge}")
                plt.legend()
                savename = os.path.join(self.args.best_save_dir, "best_params.png")
                os.makedirs(self.args.best_save_dir, exist_ok=True)
                plt.savefig(f"{savename}")
                self.logger.info(f"save figure to {savename}")

        self.logger.info(f"trial {self.args.trial_num} on rank {self.args.rank} complete with kge {kge} in {self.args.runtime} seconds")
        return -kge





def run_spotpy(args):

    parallel_sampling = "seq" if args.serial_sampling else "mpi"

    args.logger.debug(f"run spotpy in {args.workdir_path} feature {args.feature_id} for {args.sampling_reps} reps")

    args.logger.debug(f"setup ngen model")
    NgenModel = NgenRun(args)
    args.logger.debug(f"setup spotpy")
    setup = SpotPySetup(args, NgenModel)
    completed = 0
    if args.rank == 0:
        if check_path(args.dbpath):
            try:
                results = spotpy.analyser.load_csv_results(args.dbname)
                completed = len(results)
            except FileNotFoundError:
                completed = 0
    completed = args.sampling_comm.bcast(completed, root=0)
    
    args.logger.debug(f"[rank {args.rank}] create sampler with database {args.dbpath} with {completed} completed trials and in mode {parallel_sampling}")
    args.sampler = spotpy.algorithms.sceua(setup, dbname=args.dbname, dbformat="csv", parallel=parallel_sampling, save_sim=False, dbappend=(completed > 0))

    args.sampling_reps = max(args.sampling_reps - completed, 0)
    args.logger.debug(f"begin optimization with {args.sampling_reps} reps")
    args.sampler.sample(args.sampling_reps, ngs=args.ngs)

    args.logger.info(f"sampling finished, average simulation runtime {args.average_runtime}")
    results = args.sampler.getdata()
    results = spotpy.analyser.load_csv_results(args.dbname)
    best_params = spotpy.analyser.get_best_parameterset(results, maximize=False)
    


if __name__ == "__main__":
    WORLD = MPI.COMM_WORLD
    rank = WORLD.Get_rank()
    size = WORLD.Get_size()
    try:
        args = parse_args()
        args.logger = NgenLogger(default_level=args.verbosity, prefix=f"[rank {rank}]")
        signal.signal(signal.SIGTERM,lambda signum, frame: out_of_time_cleanup(args, signum, frame))

        args.logger.debug(f"[rank {rank}/{size-1}] serial_sampling={args.serial_sampling}, rank_ignored={args.serial_sampling and rank > 0}")
        if args.serial_sampling and rank > 0:
            args.logger.info(f"rank {rank} exiting")
            WORLD = None
            size = 1
            MPI.Finalize()
            sys.exit(0)
        args.sampling_comm = WORLD
        args.sampling_size = size

        directories_setup(args, rank)

        args.timings = []
        args.average_runtime = 0
        args.runtime = None
        args.trial_num = 0

        run_spotpy(args)
    except Exception as e:
        args.logger.warning(f"\nERROR ON RANK {rank}:\n")
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
        raise e
    
    args.logger.info(f"rank {rank} sampling finished, exiting")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        args.logger.info(f"cancelling slurm job {job_id}")
        subprocess.run(["scancel", job_id])
    MPI.COMM_WORLD.Abort(0)
    sys.exit(0)