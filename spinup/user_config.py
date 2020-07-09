import os
import os.path as osp
import socket
import pwd

# Default neural network backend for each algo
# (Must be either 'tf1' or 'pytorch')
DEFAULT_BACKEND = {
    'vpg': 'pytorch',
    'trpo': 'tf1',
    'ppo': 'pytorch',
    'ddpg': 'pytorch',
    'td3': 'pytorch',
    'sac': 'pytorch',
    'egl': 'pytorch'
}

project_name = 'spinningup'
username = pwd.getpwuid(os.geteuid()).pw_name

if "gpu" in socket.gethostname():
    root_path = os.path.join('/mnt/dsi_vol1/users', 'data', project_name)
elif "root" == username:
    root_path = os.path.join('/data/data', project_name)
else:
    root_path = os.path.join('/data/', username, project_name)

if not os.path.exists(root_path):
    os.makedirs(root_path)

# Where experiment outputs are saved by default:
# DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')
DEFAULT_DATA_DIR = osp.join(root_path, 'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = True

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching 
# experiments.
WAIT_BEFORE_LAUNCH = .5
