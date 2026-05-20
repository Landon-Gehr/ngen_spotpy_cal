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
    parser.add_argument("singularity_image_path", type=str, help="Singularity image path with compiled ngen binary")
    args, _ = parser.parse_known_args()

    defaults = yaml.safe_load(open(args.calibration_conf, 'r'))
    crosswalk = json.load(open(defaults["model"]["crosswalk"],"r"))

    reps = defaults["general"]["iterations"]
    root_data_path = defaults["general"]["workdir"]
    model_binary = defaults["model"]["binary"]
    realization_template = defaults["model"]["realization"]
    routing_template = defaults["model"]["troute"]
    gpkg_file = defaults["model"]["catchments"]
    observed_flow = defaults["model"]["obsflow"]

    feature_id = int(((list(crosswalk.keys())[0]).split("-"))[1])
    gage_id = crosswalk[list(crosswalk.keys())[0]]["Gage_no"]

    parser.add_argument(
        "--root-data-path",
        type=str,
        default=root_data_path,
        dest="root_data_path",
        help=("Path to directory containing /config and /forcings (default: read by conf file)"
        ),
    )
    parser.add_argument(
        "--feature-id",
        type=int,
        default=feature_id,
        dest="feature_id",
        help=("Feature id to be used for flow evaluation (default: read by crosswalk file)"
        ),
    )
    parser.add_argument(
        "--gpkg-file-path",
        type=str,
        default=gpkg_file,
        dest="gpkg_file_path",
        help=("Path to gpkg file to with catchment and nexus data (default: read by conf file)"
        ),
    )
    parser.add_argument(
        "--observed-flow",
        type=str,
        default=observed_flow,
        dest="observed_flow_path",
        help=("Observed flow data used as ground truth (default: read by conf file)"
        ),
    )
    parser.add_argument(
        "--model-binary",
        type=str,
        default=model_binary,
        dest="model_binary",
        help=("Path to binary to execute inside the singularity (default: read by conf file)"
        ),
    )
    parser.add_argument(
        "--template-realization",
        type=str,
        default=realization_template,
        dest="template_realization",
        help="Path to realization template JSON (default: read by conf file)",
    )
    parser.add_argument(
        "--template-routing",
        type=str,
        default=routing_template,
        dest="template_routing",
        help="Path to routing template YAML (default: read by template json)",
    )
    parser.add_argument(
        "--singularity-run-script",
        type=str,
        default="./ngen_image_runscript.sh",
        dest="singularity_run_script",
        help="Path to executable to be called by singularity exec (default: ./ngen_image_runscript.sh)",
    )
    parser.add_argument(
        "--image-data-path",
        type=str,
        default="/ngen/ngen/data",
        dest="image_data_path",
        help="Image data path (default: /ngen/ngen/data)",
    )
    parser.add_argument(
        "--gage-id",
        type=int,
        default=gage_id,
        dest="gage_id",
        help="Gage ID (default read by conf file)",
    )
    parser.add_argument(
        "--sampling-reps",
        type=int,
        default=reps,
        dest="sampling_reps",
        help="Number of sampling repetitions (default read by conf file)",
    )
    parser.add_argument(
        "--ngs",
        type=int,
        default=20,
        dest="ngs",
        help=f"number scuea complexes (default: 20)",
    )
    parser.add_argument(
        "--bind",
        type=str,
        default="",
        dest="bind",
        help="additional binds to singularity (i.e. /scratch/user:/scratch/user = local_path:container_path) (default: ''), automatic binds (root_data_path:image_data_path,rank_input_path:image_data_path/inputs,rank_output_path:image_data_path/outputs) can be disabled with --no-auto-bind",
    )
    parser.add_argument(
        "--no-auto-bind",
        nargs="?",           
        const=True,         
        default=False,
        type=lambda x: x.lower() == "true", 
        dest="no_auto_bind",
        help=(
            "Disable auto binding to singularity. Omit flag = False. "
            "Flag with no value = True. "
            "Flag with value = evaluated as bool string (e.g. 'true'/'false')."
        ),
    )
    parser.add_argument(
        "--serial-sampling",
        nargs="?",           
        const=True,         
        default=False,
        type=lambda x: x.lower() == "true", 
        dest="serial_sampling",
        help=(
            "Enable serial sampling. Omit flag = False. "
            "Flag with no value = True. "
            "Flag with value = evaluated as bool string (e.g. 'true'/'false')."
        ),
    )
    tasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE", 1)
    parser.add_argument(
        "--ngen-parallel",
        type=int,
        default=int(tasks_per_node),
        dest="ngen_parallel",
        help=f"NGen parallel tasks (default: SLURM_NTASKS_PER_NODE, currently {tasks_per_node})",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        dest="input_path",
        help="Input path (default: None = auto-created under run/rank_N/input)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        dest="output_path",
        help="Output path (default: None = auto-created under run/rank_N/output)",
    )
    parser.add_argument(
        "--run-path",
        type=str,
        default=None,
        dest="run_path",
        help="root path for each rank to set up ngen simulation runs (default: root_data_path/run)",
    )
    parser.add_argument(
        "--routing-output-frequency",
        type=int,
        default=15,
        dest="routing_output_frequency",
        help="Routing output frequency in minutes (default: 15)",
    )
    parser.add_argument(
        "--clean",
        nargs="?",           
        const=True,         
        default=False,
        type=lambda x: x.lower() == "true", 
        dest="clean",
        help=(
            "Enable cleaning on finish. Omit flag = False. "
            "Flag with no value = True. "
            "Flag with value = evaluated as bool string (e.g. 'true'/'false')."
        ),
    )
    parser.add_argument(
        "--verbosity",
        nargs="?",       
        default="none",
        type=str, 
        dest="verbosity",
        help=(
            "Control text output levels. [debug, info, warning, error, critical, none] (default: none)"
        ),
    )
    parser.add_argument(
        "--dbpath",
        nargs="?",       
        default=f"./ngen_param_tuning.csv",
        type=str, 
        dest="dbpath",
        help=(
            "Path to optimization storage csv (default: ./ngen_param_tuning.csv)"
        ),
    )
    
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if args.run_path == None:
        args.run_path = os.path.join(args.root_data_path,"run")
    args.root_data_path           = clean_path(args.root_data_path)
    args.data_config_path         = os.path.join(args.root_data_path,"config")
    args.data_forcing_path        = os.path.join(args.root_data_path,"forcings")
    args.gpkg_file_path           = clean_path(args.gpkg_file_path)
    args.observed_flow_path       = clean_path(args.observed_flow_path)
    args.singularity_image_path   = clean_path(args.singularity_image_path)
    args.template_realization     = clean_path(args.template_realization)
    args.template_routing         = clean_path(args.template_routing)
    args.calibration_conf         = clean_path(args.calibration_conf)
    args.singularity_run_script   = clean_path(os.path.abspath(args.singularity_run_script))
    args.image_data_path          = clean_path(args.image_data_path)
    args.run_path                 = clean_path(args.run_path)
    args.dbpath                   = clean_path(args.dbpath)

    args.config = yaml.safe_load(open(args.calibration_conf, 'r'))
    args.sim_start = args.config["model"]["eval_params"]["valid_start_time"]
    args.sim_stop = args.config["model"]["eval_params"]["valid_end_time"]
    args.eval_start = args.config["model"]["eval_params"]["valid_eval_start_time"]
    args.eval_stop = args.config["model"]["eval_params"]["valid_eval_end_time"]
    args.sim_start_dt = pd.to_datetime(args.sim_start)
    args.sim_stop_dt = pd.to_datetime(args.sim_stop)
    args.eval_start_dt = pd.to_datetime(args.eval_start)
    args.eval_stop_dt = pd.to_datetime(args.eval_stop)

    args.dbname = os.path.basename(args.dbpath).split(".")[0]
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
        args.image_forcing_path = os.path.join(args.image_data_path,"forcings")
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
        jsondata = json.load(open(realization_file,"r"))
        jsondata["time"]["start_time"] = self.args.sim_start
        jsondata["time"]["end_time"] = self.args.sim_stop
        jsondata["routing"]["t_route_config_file_with_path"] = self.args.routing_image_path
        jsondata["output_root"] = self.args.image_ngen_output_path
        json.dump(jsondata, open(realization_file, "w"),indent=4)
        return jsondata
    
    def update_realization(self, sim_params, module_map, wipe_prev=False):
        self.logger.info(f"update param realization {sim_params}, {module_map}")
        modules = self.realization["global"]["formulations"][0]["params"]["modules"] # assumes formulations is array of size 1
        if wipe_prev:
            for i, module in enumerate(modules): # Wipe past params
                if self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_type_name"] in list(module_map.keys()):
                    self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_params"] = {}
        for i, module in enumerate(modules): # create if not exists
            if (self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_type_name"] in list(module_map.keys())) and ("model_params" not in self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]):
                self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_params"] = {}
        for module_name in list(self.module_map.keys()):
            for param in self.module_map[module_name]:
                for i, module in enumerate(modules): # interate across all modules and all params per module
                    if module["params"]["model_type_name"] == module_name:
                        self.realization["global"]["formulations"][0]["params"]["modules"][i]["params"]["model_params"][param] = self.args.params[param]

# # Enforce rule: Cgw <= gw_storage
# if param_map['Cgw'] > param_map['gw_storage']:
#     # Return NaNs or a poor score (depending on your objective function)
#     return np.full_like(self.observed.values.squeeze(), np.nan)

        json.dump(self.realization, open(self.args.realization_path, "w"),indent=4)
    
    def read_yaml(self, yaml_file):
        ymldata = yaml.safe_load(open(yaml_file, 'r'))
        ymldata["compute_parameters"]["restart_parameters"]["start_datetime"] = self.args.sim_start
        time_diff = self.args.sim_stop_dt - self.args.sim_start_dt
        self.args.nts = (time_diff.total_seconds() / 60) / 5 # in 5 minute intervals
        ymldata["compute_parameters"]["forcing_parameters"]["nts"] = self.args.nts
        ymldata["compute_parameters"]["forcing_parameters"]["max_loop_size"] = self.args.nts
        ymldata["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"] = self.args.gpkg_image_path
        ymldata["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = self.args.image_ngen_output_path
        ymldata["output_parameters"]["stream_output"]["stream_output_directory"] = self.args.image_troute_output_path
        ymldata["output_parameters"]["stream_output"]["stream_output_internal_frequency"] = self.args.routing_output_frequency
        yaml.dump(ymldata, open(yaml_file, 'w'))
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
        self.logger.debug(f"rank {self.args.rank} running ngen '{" ".join(ngen_command)}' from working directory {run_dir}")
        try:
            start = time.time()
            subprocess.run(ngen_command, check=True)
            self.args.runtime = time.time() - start
            self.args.timings += [self.args.runtime]
            # self.args.average_runtime = np.mean(self.args.timings)
            self.args.average_runtime = self.args.average_runtime + (self.args.runtime - self.args.average_runtime)/self.args.trial_num
            self.args.logger.info(f"subprocess with {self.args.ngen_parallel} cpus finished in {self.args.runtime} seconds (avg: {self.args.average_runtime})")
            self.args.logger.info(f"[rank {self.args.rank}] subprocess finished in {self.args.runtime} seconds (avg: {self.args.average_runtime})")
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

    def load_params(self,yaml_file):
        training_params_dict = {}
        ymldata = yaml.safe_load(open(yaml_file))
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
            self.update_params(sim_params)
            self.ngen.run_ngen()
            dt = datetime.strptime(self.args.sim_start, "%Y-%m-%d %H:%M:%S")
            filename = dt.strftime("troute_output_%Y%m%d%H%M") + ".nc"
            ds = xr.open_dataset(os.path.join(self.args.troute_output_path, filename))
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

        # plt.figure()
        # plt.plot(full_index, evaluation, label="observed flow")
        # plt.plot(full_index, simulation, label="simulated flow")
        # plt.xticks(rotation=45) 
        # plt.axvline(x=pd.to_datetime(self.args.eval_start), color="k", linestyle="--", linewidth=1, label="eval start")
        # plt.title(f"{self.args.sim_start} {self.args.sim_stop} (output every {self.args.routing_output_frequency}min) \n kge={kge}")
        # plt.legend()
        # plt.savefig(f"./figures/kge_{kge:.4f}.png")
        # self.logger.info(f"save figure to ./figures/kge_{kge:.4f}.png")

        self.logger.info(f"trial {self.args.trial_num} on rank {self.args.rank} complete with kge {kge} in {self.args.runtime} seconds")
        return -kge





def run_spotpy(args):

    parallel_sampling = "seq" if args.serial_sampling else "mpi"

    args.logger.debug(f"run spotpy in {args.root_data_path} feature {args.feature_id} for {args.sampling_reps} reps")

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
    
    args.logger.debug(f"[rank {args.rank}] create sampler with database {args.dbname}.csv with {completed} completed trials and in mode {parallel_sampling}")
    args.sampler = spotpy.algorithms.sceua(setup, dbname=args.dbname, dbformat="csv", parallel=parallel_sampling, save_sim=False, dbappend=(completed > 0))

    args.sampling_reps = max(args.sampling_reps - completed, 0)
    args.logger.debug(f"begin optimization with {args.sampling_reps} reps")
    args.sampler.sample(args.sampling_reps, ngs=args.ngs)

    args.logger.info(f"sampling finished, average simulation runtime {args.average_runtime}")
    results = args.sampler.getdata()
    results = spotpy.analyser.load_csv_results(args.dbname)
    best_params = spotpy.analyser.get_best_parameterset(results, maximize=False)
    

#TODO: 
#      verify objective scoring is correct
#      run on one gage
#      perform data setup to all other gages and run
#      documentation

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