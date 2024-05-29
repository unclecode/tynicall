
from lightning_sdk import Studio, Machine

# reference to the current studio
studio = Studio()

# use the jobs plugin
studio.install_plugin('jobs')
job_plugin = studio.installed_plugins['jobs']

# do a sweep over learning rates
learning_rates = [1e-4, 1e-3, 1e-2]

# start all jobs on an A100 GPU with names containing an index
for index, lr in enumerate(learning_rates):
    cmd = f'python main.py --lr {lr} --max_steps 100'
    job_name = f'run-{index}'
    job_plugin.run(cmd, machine=Machine.A100, name=job_name)
