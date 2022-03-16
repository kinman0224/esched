from distutils.command.config import config
import logging
import os
from sched import scheduler
import sys
import argparse
import zmq
import threading

from typing import Union, Dict, Any, Set

import logging

from common import event_loop, log
from common.types import *
from utils import misc

import ray
from ray import tune
from ray.tune.schedulers import FIFOScheduler, TrialScheduler, ResourceChangingScheduler
from ray.tune.utils.placement_groups import PlacementGroupFactory
from ray.tune.trial import Trial
from ray.tune import trial_runner
from ray.tune.resources import Resources
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer

TERM_STATES = (Trial.ERROR, Trial.TERMINATED)

L = logging.getLogger("Scheduler")
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class ElasticScheduler(FIFOScheduler):
    
    def __init__(self, wid, scheduler_endpoint: str, max_num_workers: int):
        FIFOScheduler.__init__(self)
        self.wid = wid
        self.host = misc.getHostname()
        self.max_num_workers = max_num_workers

        # Event Loop
        self.event_loop = event_loop.EventLoop(0)
        
        # Socket
        self.conn: zmq.Socket = zmq.Context.instance().socket(zmq.PUSH)
        self.conn.connect(scheduler_endpoint)
        
        self.report_new_worker()

        self._trials_to_reallocate: Dict[Trial, Union[
            None, dict, PlacementGroupFactory]] = {}
        self._reallocated_trial_ids: Set[str] = set()

    @property
    def endpoint(self):
        return self.event_loop.endpoint

    # <--- Information report - BEGIN --->
    
    def report_new_worker(self):
        # L.info("report_new_worker")
        self.conn.send_json({
            'type': MsgType.NEW_WORKER,
            'wid': self.wid,
            'endpoint': self.endpoint,
        })

    def report_new_trial(self, trial_runner, trial):
        self.conn.send_json({
            'type': MsgType.NEW_TRIAL,
            'trial_id': trial.trial_id,
            'driver_resources': trial.placement_group_factory.bundles[0],
            'worker_resources' : trial.placement_group_factory.bundles[1],
            'num_workers' : trial.config['num_workers'],
            'max_num_workers': self.max_num_workers,
            'endpoint': self.endpoint,
            'stopping_criteria': trial_runner._stopping_criteria,
            'trainable_name': trial._trainable_name(),
        })
        
    def report_train_intermediate_result(self, trial, result):
        # TODO: construct result metrics
        self.conn.send_json({
            'type': MsgType.IMD_RES_TRAIN,
            'trial_id': trial.trial_id,
            'result': {
                'timesteps_total': result['timesteps_total'],
                'episode_reward_mean': result['episode_reward_mean'],
                },
        })
        
    def report_trial_completed(self, trial, result):
        self.conn.send_json({
            'type': MsgType.COMPLETED_TRIAL,
            'trial_id': trial.trial_id,
        })

    # <--- Information report - END --->
    
    # <--- Job Worker Management - BEGIN --->
        
    # <--- Job Worker Management - END --->
    
    # <--- ElasticScheduler Part - Begin --->

    def on_trial_add(self, trial_runner, trial):
        pass

    def on_trial_result(
        self, trial_runner, trial, result
    ) -> str:
        decision = TrialScheduler.CONTINUE
        # print(result['episode_reward_mean'])
        # Ask global scheduler whether have to reallocate trial resource
        self.report_train_intermediate_result(trial, result)
        new_resources = self.reallocate_trial_resources_if_needed(
            trial_runner, trial, result)
        if new_resources:
            self._trials_to_reallocate[trial] = new_resources
            trial.config["num_workers"] = len(new_resources.bundles) - 1
            return TrialScheduler.PAUSE
        
        return decision
    
    def on_trial_complete(self, trial_runner, trial, result):
        self.report_trial_completed(trial, result)
        
    def choose_trial_to_run(self, trial_runner): 
        any_resources_changed = False

        new_trials_to_reallocate = {}
        for trial, new_resources in self._trials_to_reallocate.items():
            if trial.status == Trial.RUNNING:
                new_trials_to_reallocate[trial] = new_resources
                L.debug(f"{trial} is still running, skipping for now")
                continue
            any_resources_changed = (any_resources_changed
                                     or self.set_trial_resources(
                                         trial, new_resources))
        self._trials_to_reallocate = new_trials_to_reallocate

        if any_resources_changed:
            # force reconcilation to ensure resource changes
            # are implemented right away
            trial_runner.trial_executor.force_reconcilation_on_next_step_end()

        trial = FIFOScheduler.choose_trial_to_run(self, trial_runner)
        if trial:
            # print(f"Choosing trial {trial.config} to run from trial runner.")
            if "node_ids" not in trial.placement_group_factory._kwargs:
                self.report_new_trial(trial_runner, trial)
                print(f'wait global scheduler to allocate resources.')
                # self.resources_allocation_event.wait()
                msg = self.event_loop.recv()
                if msg['type'] == MsgType.RESOURCES_ALLOCATION:
                    trial.placement_group_factory._kwargs["node_ids"] = msg["node_ids"]
                    trial.placement_group_factory._bind()
        return trial

    def set_trial_resources(
            self, trial: Trial,
            new_resources: Union[Dict, PlacementGroupFactory]) -> bool:
        """Returns True if new_resources were set."""
        if new_resources:
            L.info(f"Setting trial {trial} resource to {new_resources}")
            trial.placement_group_factory = None
            trial.update_resources(new_resources)
            # keep track of all trials which had their resources changed
            self._reallocated_trial_ids.add(trial.trial_id)
            return True
        return False

    def _are_resources_the_same(
            self,
            trial: Trial,
            new_resources,
    ) -> bool:
        """Returns True if trial's resources are value equal to new_resources.

        Only checks for PlacementGroupFactories at this moment.
        """
        if (isinstance(new_resources, PlacementGroupFactory)
                and trial.placement_group_factory == new_resources):
            L.debug(f"{trial} PGF "
                         f"{trial.placement_group_factory.required_resources}"
                         f" and {new_resources.required_resources}"
                         f" are the same, skipping")
            return True
        else:
            return False
        
    def reallocate_trial_resources_if_needed(
            self, trial_runner: "trial_runner.TrialRunner", trial: Trial,
            result: Dict) -> Union[None, dict, PlacementGroupFactory]:

        new_resources = self.conn_to_global_scheduler(trial_runner, trial, result)

        # if we can check if the new resources are the same,
        # we do that here and skip resource allocation
        if new_resources and not self._are_resources_the_same(
                trial, new_resources):
            return new_resources
        return None
    
    def conn_to_global_scheduler(self, trial_runner, trial, result):
        msg = self.event_loop.recv()
        if msg['type'] == MsgType.RESOURCES_REALLOCATION:
            new_bundles = msg['bundles']
            new_node_ids = msg['node_ids']
            new_pgf = PlacementGroupFactory(bundles=new_bundles, node_ids=new_node_ids)
            return new_pgf
        return None

    # def get_live_trials(self, runner):
    #     return [
    #         t for t in runner.get_trials() if t.status not in TERM_STATES
    #     ]

    # def get_pending(self, trial_runner):
    #     return [
    #         t for t in trial_runner.get_trials() if t.status == Trial.PENDING
    #     ]

    # def get_paused(self, trial_runner):
    #     return [
    #         t for t in trial_runner.get_trials() if t.status == Trial.PAUSED
    #     ]

    # <--- ElasticScheduler Part - End --->

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Worker', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wid', required=True, type=str,
                        help='Worker ID')
    parser.add_argument('-s', '--scheduler', required=True, type=str,
                        help='Endpoint of scheduler')
    args = parser.parse_args()
    
    ray.init(address="auto")

    # search_space = {
    #         "framework": "torch",
    #         "env": "QbertNoFrameskip-v4",
    #         # "kl_coeff": 1.0,
    #         "num_workers": 16,
    #         "num_gpus": 2,
    #         "num_envs_per_worker": 5,
    #         # These params are tuned from a fixed starting value.
    #         "rollout_fragment_length": 50,
    #         "broadcast_interval": 5,
    #         "max_sample_requests_in_flight_per_worker": 1,
    #         "model": {
    #             "dim": 42,
    #         },
    #         # "lr": 1e-4,
    #         "train_batch_size": 1000,
    #         # "num_multi_gpu_tower_stacks": 4,
    #         # "clip_rewards": True,
    #         # These params start off randomly drawn from a set.
    #         # "num_sgd_iter": tune.choice([10, 20, 30]),
    #         # "sgd_minibatch_size": tune.choice([32, 64]),
    #         # "train_batch_size": tune.choice([500, 1000]),
    #         # "lr":  tune.choice([1e-4, 1e-5]),
    #         "lr": 1e-4,
    #     }
    search_space = {
        "framework": "torch",
        "env": "BeamRiderNoFrameskip-v4",
        "lambda": 0.95,
        "kl_coeff": 0.5,
        "clip_rewards": True,
        "clip_param": 0.1,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "train_batch_size": 5000,
        "rollout_fragment_length": 100,
        "sgd_minibatch_size": 500,
        "num_sgd_iter": 10,
        "num_workers": 32,
        "num_envs_per_worker": 10,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "model": {
            "vf_share_layers": True,
        },
        "num_gpus": 1,
    }


    base_scheduler = FIFOScheduler()
    elastic_scheduler = ElasticScheduler(wid=args.wid, 
                                         scheduler_endpoint=args.scheduler,
                                         max_num_workers=32)

    analysis = tune.run(PPOTrainer,
                        name='exp',
                        num_samples=1,
                        stop={'timesteps_total': 100000000, 'episode_reward_mean': 5000.0},
                        config=search_space,
                        scheduler=elastic_scheduler,
                        # scheduler=base_scheduler,
                        reuse_actors=False
                        )