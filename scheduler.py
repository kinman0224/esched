from platform import node
import signal
import logging
import argparse
import time
from threading import Thread
from collections import OrderedDict, Counter
from venv import create

import ray

from trial import Trial
from common import event_loop, log
from common.utils import random_token
from common.types import *
from worker_conn import WorkerConn
from policy import *

L = logging.getLogger("Scheduler")
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class RayElasticScheduler:
    
    def __init__(self, port: str, ray_address: str):
        # Fetch the node information from ray
        L.info("Get Ray cluster INFO:")
        ray.init(address=ray_address)        
        self.nodes = self.get_node_infos()
        
        self.event_loop = event_loop.EventLoop(port)

        self.event_loop.register(MsgType.NEW_WORKER, self.on_new_worker)
        self.event_loop.register(MsgType.NEW_TRIAL, self.on_new_trial)
        self.event_loop.register(MsgType.IMD_RES_TRAIN, self.on_train_intermiate_result)
        self.event_loop.register(MsgType.COMPLETED_TRIAL, self.on_completed_trial)
        
        self.trials = OrderedDict()

        self.allocations = {}

        self.policy = OptimusPolicy()

        self._toExit = False
        

    @property
    def endpoint(self):
        return self.event_loop.endpoint
    
    @property
    def toExit(self):
        return self._toExit and self.event_loop.toExit

    def shutdown(self):
        L.info("shutdown...")
        self._toExit = True
        self.event_loop.shutdown()
        self.event_loop_thread.join()

    def run(self):
        # L.info("Serve at {ep}".format(ep=self.endpoint))
        self.event_loop_thread = Thread(target=self.event_loop.run, name='EventLoop')
        self.event_loop_thread.start()

        self.schedule_loop()

    def schedule_loop(self):
        while not self._toExit:
            # L.info("Schedule..., and sleep 10 sec")
            time.sleep(10)
            job_infos = self.get_trial_infos()
            
            if job_infos:
                L.info(f"{len(job_infos)} job schedule...")
                # update the node infos
                node_infos = self.get_node_infos()
                self.allocations = {k: v for k, v in self.allocations.items() if k in job_infos}
                allocations = self.policy.optimize(job_infos, node_infos, self.allocations)
                # L.info(allocations)
                self.allocations = allocations
            else:
                L.info("No job schedule...")
                
        L.info('Exit')

    def save(self):
        pass
    
    # <--- Job Worker Management - Begin --->

    def on_new_worker(self, msg):
        assert msg['type'] == MsgType.NEW_WORKER
        L.info("A new worker from {}".format(msg['endpoint']))

        # TODO: unuse
        worker = WorkerConn(msg['endpoint'])

    def on_new_trial(self, msg):
        assert msg['type'] == MsgType.NEW_TRIAL
        
        trial_id = msg['trial_id']
        if trial_id in self.trials:
            L.info("Job {jid} already exist.".format(jid=trial_id))

        self.trials[trial_id] = Trial(
                trial_id,
                msg['driver_resources'],
                msg['worker_resources'],
                msg['num_workers'],
                msg['max_num_workers'],
                msg['stopping_criteria'],
                msg['endpoint'],
                msg['trainable_name'],
            )
        self.trials[trial_id].add_event('submit')

        self.schedule()

    def on_train_intermiate_result(self, msg):
        assert msg['type'] == MsgType.IMD_RES_TRAIN

        id = msg['trial_id']
        trial =  self.trials.get(id, None)
        if trial is None:
            L.info("Trial {id} not found.".format(id=id))
            return
        result = msg['result']

        trial.current_step = result['timesteps_total']

        allocations = self.allocations[id]
        
        new_num_workers = len(allocations) - 1
        new_bundles = [trial.driver_resources] + [trial.worker_resources] * new_num_workers
        new_node_ids = allocations
        # self.allocations[id] = new_node_ids
        msg = {
            'type': MsgType.RESOURCES_REALLOCATION,
            'trial_id': trial.id,
            'bundles': new_bundles,
            'node_ids': new_node_ids,
        }
        trial.conn.send_json(msg)
    
    def on_completed_trial(self, msg):
        assert msg['type'] == MsgType.COMPLETED_TRIAL

        id = msg['trial_id']
        trial =  self.trials.get(id, None)
        if trial is None:
            L.info("Trial {id} not found.".format(id=id))
            return
        self.trials.pop(id)
        
    # <--- Job Worker Management - End --->

    def schedule(self):
        L.debug("schedule")
        for trial in self.trials.values():
            if not trial.has_allocation():
                resources_request = sum(
                    [Counter(trial.worker_resources)] * trial.num_workers + [Counter(trial.driver_resources)], Counter()
                    )
                total_cpus = {idx: int(node.resources["CPU"]) for idx, node in self.nodes.items()}
                free_cpus = Counter(total_cpus)
                node_id, count = free_cpus.most_common(1)[0]
                if count > resources_request["CPU"]:
                    trial.add_event('start')
                    self.allocations[trial.id] = [node_id] + [node_id] * trial.num_workers
                    msg = {
                        'type': MsgType.RESOURCES_ALLOCATION,
                        'trial_id': trial.id,
                        'node_ids': self.allocations[trial.id],
                    }
                    # print(msg)
                    trial.conn.send_json(msg)

    def get_node_infos(self):
        class NodeInfo(object):
            def __init__(self, resources) -> None:
                self.resources = resources
        nodes = {}
        for node in ray.state.nodes():
            id, resources = node['NodeID'], node['Resources']
            nodes[id] = NodeInfo(resources=resources)
            # L.info(f"NodeID: {id}, Resources: {resources}")
        return nodes
    
    def get_trial_infos(self):
        trial_infos = {}
        for id, trial in self.trials.items():
            trial_infos[id] = TrialInfo(
                creation_timestamp=None,
                driver_resources=trial.driver_resources,
                worker_resources=trial.worker_resources,
                min_workers=0,
                max_workers=trial.max_num_workers, # the max record workers
                total_step=trial.total_step,
                current_step=trial.current_step,
            )
            trial_infos[id].application = trial.application
        return trial_infos


class TrialInfo(object):
    def __init__(self, creation_timestamp, driver_resources, worker_resources, min_workers, max_workers, total_step=None, current_step=None):
        self.creation_timestamp = creation_timestamp
        self.driver_resources = driver_resources
        self.worker_resources = worker_resources

        self.min_workers = min_workers
        self.max_workers = max_workers

        self.total_step = total_step
        self.current_step = current_step
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scheduler', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--port', default=5050, type=int,
                        help='Port of scheduler')
    args = parser.parse_args()
    
    scheduler = RayElasticScheduler(
        port=args.port, 
        ray_address="auto"
        )
    def sigint(sig, frame):
        if not scheduler.toExit:
            scheduler.save()
            scheduler.shutdown()

    signal.signal(signal.SIGINT, sigint)

    L.info("Scheduler serve at {ep}".format(ep=scheduler.endpoint))
    
    scheduler.run()