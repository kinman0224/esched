from typing import Dict, List
from utils import misc
import zmq
from application import APPLICATIONS

class Trial:
    
    def __init__(self, 
                 trial_id: str, 
                 driver_resources: Dict[str, float], 
                 worker_resources: Dict[str, float], 
                 num_workers: int,
                 max_num_workers: int,
                 stopping_criteria: Dict,
                 endpoint: str,
                 trainable_name: str):
        self.id = trial_id
        self.driver_resources = driver_resources
        self.worker_resources = worker_resources
        self.num_workers = num_workers
        self.max_num_workers = max_num_workers
        self.stopping_criteria = stopping_criteria
        self.endpoint = endpoint
        self.trainable_name = trainable_name

        self.total_step = stopping_criteria['timesteps_total']        
        self.current_step = 0
        self.application = self.get_application()

        if self.endpoint:
            self.conn: zmq.Socket = zmq.Context.instance().socket(zmq.PUSH)
            self.conn.connect(self.endpoint)
        else:
            self.conn = None

        self.event = []
        
        self.placement = None
        

    def add_event(self, event):
        self.event.append((event, misc.now()))
        
    def has_allocation(self):
        return self.placement is not None
    
    def get_env(self):
        return self.trainable_name.split('_')[1]
    
    def get_trainable(self):
        return self.trainable_name.split('_')[0]
    
    def get_application(self):
        application = None
        if self.trainable_name in ["ImpalaTrainer_PongNoFrameskip-v4", "ImpalaTrainer_QbertNoFrameskip-v4", "PPOTrainer_QbertNoFrameskip-v4", "PPOTrainer_BeamRiderNoFrameskip-v4"]:
            application = APPLICATIONS['impala-atari']
        else:
            raise NotImplementedError

        return application
    
    def is_atari(self, env):
        import atari_py as ap
        game_list = ap.list_games()
        for game in game_list:
            if game in env.lower():
                return True