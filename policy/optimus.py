import collections
import copy
import math


class OptimusPolicy(object):
    def __init__(self):
        pass

    def optimize(self, jobs, nodes, prev_allocations):
        allocations = {k: v for k, v in prev_allocations.items() if k in jobs}

        # # get remain step?
        for job in jobs.values():
            job.remaining = job.total_step - job.current_step

        # assign min replicas
        min_replicas = {}
        for key, job in jobs.items():
            min_replicas[key] = 1  # math.ceil(job.target_batch_size / job.application.max_local_bsz)

        # calculate marignal gain, here is use cpu to measure
        num_cpus = sum(node.resources["CPU"] for node in nodes.values())
        num_replicas = {}
        gain = {}
        # TODO: sort by what
        for key, job in sorted(jobs.items(), key=lambda item: min_replicas[item[0]]):
            req_cpus = min_replicas[key] * job.worker_resources["CPU"] + job.driver_resources["CPU"]
            if req_cpus > num_cpus:
                num_replicas[key] = 0
                gain[key] = 0
                continue
            num_replicas[key] = min_replicas[key]
            num_cpus -= req_cpus
            if num_replicas[key] + 1 > job.max_workers or num_cpus < job.worker_resources["CPU"]:
                gain[key] = 0
            else:
                gain[key] = (job.remaining / job.application.get_throughput(num_replicas[key], 1) - 
                        job.remaining / job.application.get_throughput(num_replicas[key]+1, 1))

        # Add resources in order of maximum marginal gain.
        while num_cpus > 0 and max(gain.values()) > 0:
            key = max(gain, key=lambda k: gain[k])
            job = jobs[key]
            num_replicas[key] += 1
            if num_replicas[key] + 1 > job.max_workers or num_cpus < job.worker_resources["CPU"]:
                gain[key] = 0
            else:
                gain[key] = (job.remaining / job.application.get_throughput(num_replicas[key], 1) - 
                        job.remaining / job.application.get_throughput(num_replicas[key]+1, 1))
            num_cpus -= job.worker_resources["CPU"]

        # Placements.
        # TODO: 
        #       1. driver are consider with placement
        #       2. which job are allocated first, which would affect the placement

        # The jobs which do not need to be reallocated.
        allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas[k]}
        job_keys = sorted(jobs, key=lambda k: num_replicas[k])
        total_cpus = {idx: int(node.resources["CPU"]) for idx, node in nodes.items()}
        total_gpus = {idx: int(node.resources["GPU"]) for idx, node in nodes.items()}
        free_cpus = collections.Counter(total_cpus) - collections.Counter(sum(allocations.values(), []))
        free_gpus = collections.Counter(total_cpus) - collections.Counter(
            [allocation[0] for key, allocation in allocations.items() if jobs[key].driver_resources["GPU"] >= 1]
        )
        for key in job_keys:
            have_allocated_driver = False
            if num_replicas[key] > 0 and not allocations.get(key):
                num_replicas[key] += 1 #jobs[key].driver_resources["CPU"]
                # Allocate resources.
                allocations[key] = []
                while len(allocations[key]) < num_replicas[key]:
                    # check GPU, since driver would need GPU.
                    for node_id, count in free_cpus.most_common():
                        if not have_allocated_driver and free_gpus[node_id] == 0:
                            continue
                        num = min(count, num_replicas[key] - len(allocations[key]))
                        allocations[key].extend([node_id] * num)
                        free_cpus[node_id] -= num
                        have_allocated_driver |= True
                        break
        return allocations

