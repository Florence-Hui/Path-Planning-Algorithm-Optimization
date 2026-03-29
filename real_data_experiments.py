# !/usr/bin/python

'''
Script for running myopic experiments using the run_sim bash script.
Generally a function of convenience in the event of parallelizing simulation runs.
Note: some of the parameters may need to be set prior to running the bash script.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''
import os
import time
import sys
import logging
import numpy as np
import random

import aq_library as aqlib
import mcts_library as mctslib
import gpmodel_library as gplib 
import evaluation_library as evalib 
import paths_library as pathlib 
import envmodel_library as envlib 
import robot_library as roblib
import obstacles as obslib

#mport bag_utils as baglib


print ("User specified options: SEED, REWARD_FUNCTION, PATHSET, USE_COST, NONMYOPIC, GOAL_ONLY, TREE_TYPE, RUN_REAL")
# Allow selection of seed world to be consistent, and to run through reward functions
SEED =  0
# control the randomness
#np.random.seed(SEED)
#random.seed(SEED)
# SEED = 0 
REWARD_FUNCTION = "mean"
PATHSET = "dubins"
USE_COST = False
NONMYOPIC = True
GOAL_ONLY = False
TREE_TYPE = "dpw"
RUN_REAL_EXP =False

if RUN_REAL_EXP:
    MAX_COLOR = 1.50
    MIN_COLOR = -1.80
else:
    MAX_COLOR = 25.0
    MIN_COLOR = -25.0
# MAX_COLOR = None
# MIN_COLOR = None

# Parameters for plotting based on the seed world information
# Set up paths for logging the data from the simulation run
if not os.path.exists('./figures/' + str(REWARD_FUNCTION)): 
    os.makedirs('./figures/' + str(REWARD_FUNCTION))
logging.basicConfig(filename = './figures/'+ REWARD_FUNCTION + '/robot.log', level = logging.INFO)
logger = logging.getLogger('robot')

# Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters, mean function 0 
# The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS
ranges = (0.0, 10.0, 0.0, 10.0)

# Create obstacle world
ow = obslib.FreeWorld()
# ow = obslib.ChannelWorld(ranges, (3.5, 7.), 3., 0.3)
# ow = obslib.BugTrap(ranges, (2.2, 3.0), 4.6, orientation = 'left', width = 5.0)
# ow = obslib.BlockWorld(ranges,12, dim_blocks=(1., 1.), centers=[(2.5, 2.5), (7.,4.), (5., 8.), (8.75, 6.), (3.5,6.), (6.,1.5), (1.75,5.), (6.2,6.), (8.,8.5), (4.2, 3.8), (8.75,2.5), (2.2,8.2)])


if RUN_REAL_EXP:
    ''' Bagging '''
    xfull, zfull = baglib.read_fulldataset()

    # Add subsampled data from a previous bagifle
    seed_bag = '/home/genevieve/mit-whoi/barbados/rosbag_15Jan_slicklizard/slicklizard_2019-01-15-20-22-16.bag'
    xobs, zobs = baglib.read_bagfile(seed_bag)
    print (xobs.shape)
    print (zobs.shape)
    # Create the GP model
    gp_world = gplib.GPModel(ranges, lengthscale = 4.0543111858072445, variance = 0.3215773006606948, noise = 0.0862445597387173)
    gp_world.add_data(xfull[::5], zfull[::5])

    VAR = 0.3215773006606948
    LEN = 4.0543111858072445
    NOISE = 0.0862445597387173
else:
    gp_world = None
    # VAR = 50.0
    # LEN = 5.0
    # NOISE = 0.1
    VAR = 100.0
    LEN = 1.0
    NOISE = 1.0

# Create the evaluation class used to quantify the simulation metrics
# evaluation = evalib.Evaluation(world = world, reward_function = REWARD_FUNCTION)

# Generate a prior dataset
'''
x1observe = np.linspace(ranges[0], ranges[1], 5)
x2observe = np.linspace(ranges[2], ranges[3], 5)
x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
observations = world.sample_value(data)
'''


# Create the point robot
NUM_RUNS = 5
T = 40

all_regrets = []

for seed in range(NUM_RUNS):
    print("\n===== Running seed {} =====".format(seed))

    #np.random.seed(seed)
    #random.seed(seed)

    # --- world ---
    world = envlib.Environment(
        ranges = ranges,
        NUM_PTS = 20, 
        variance = VAR,
        lengthscale = LEN,
        noise = NOISE,
        visualize = False, 
        seed = seed,
        MAX_COLOR = MAX_COLOR,
        MIN_COLOR = MIN_COLOR,
        model = gp_world,
        obstacle_world = ow
    )

    evaluation = evalib.Evaluation(world = world, reward_function = REWARD_FUNCTION)

    robot = roblib.Robot(
        sample_world = world.sample_value,
        start_loc = (1.0, 1.0,0.0),
        extent = ranges,
        MAX_COLOR = MAX_COLOR,
        MIN_COLOR = MIN_COLOR,
        kernel_file = None,
        kernel_dataset = None,
        prior_dataset = None,
        init_lengthscale = LEN,
        init_variance = VAR,
        noise = NOISE,
        path_generator = PATHSET,
        goal_only = GOAL_ONLY,
        frontier_size = 10,
        horizon_length = 1,
        turning_radius = 0.11,
        sample_step = 0.1,
        evaluation = evaluation,
        f_rew = REWARD_FUNCTION,
        create_animation = False, 
        learn_params = False,
        nonmyopic = NONMYOPIC,
        discretization = (20, 20),
        use_cost = USE_COST,
        computation_budget = 250,
        rollout_length = 4,
        obstacle_world = ow,
        dimension = 2,
        start_time = 0,
        tree_type = TREE_TYPE
    )

    robot.planner(T = T)
    robot.visualize_trajectory(screen = False) #creates a summary trajectory image
    robot.plot_information() #plots all of the metrics of interest

    regret_dict = robot.eval.metrics['simple_regret']
    regret_list = [regret_dict[t] for t in sorted(regret_dict.keys())]

    all_regrets.append(regret_list)
    

import matplotlib.pyplot as plt

all_regrets = np.array(all_regrets)

mean = np.mean(all_regrets, axis=0)
std = np.std(all_regrets, axis=0)

time = np.arange(len(mean))

plt.figure(figsize=(8,6))
plt.plot(time, mean, label="Mean Simple Regret")
plt.fill_between(time, mean-std, mean+std, alpha=0.2)

plt.xlabel("Time step")
plt.ylabel("Simple Regret")
plt.title("Average Simple Regret over {} runs".format(NUM_RUNS))
plt.legend()

plt.savefig('./figures/avg_simple_regret.png')
plt.show()

