
from doubly_robust_method.utils import *
from generate_job_parameters import *
import os
if __name__ == '__main__':

    fold = 'all_gpu_baselines_2'
    jobs = os.listdir(fold)
    jobs.sort()
    print(jobs)
    # job = jobs[140]
    for job in [jobs[570],jobs[560]]:
        experiment_params = load_obj(job,folder=f'{fold}/')
        print(experiment_params)
        c = experiment_object(**experiment_params)
        # c.debug_mode=True
        c.run_experiments()
