import jax

from hurd.internal_datasets import load_c13k_data
import numpy as np
import pprint
from datetime import datetime
import math

from hurd.models.psychophysical import ExpectedUtilityModel
from hurd.models.utility_functions import NeuralNetworkUtil
from hurd.optimizers import Adam

import wandb as wandb

WANDB_PROJECT_NAME = "choices13k-mattarlab"
WANDB_MODE = "online"
DEBUG = True
INFO = True

def pretrain_model(train_data, val_data, human_val_data, experiment_config, model_config, EV_config, data_percentage):
    EV_train_data, EV_val_data = init_data_split(fb_filter="EV_B_highest_target")

    if INFO:
        print("======================\n    PRETRAIN MODEL\n======================")
        print("DATA PERCENTAGE:", data_percentage)
        pprint.pprint(model_config)

    # initialize optimizer and utility function
    optimizer = Adam(lr=EV_config['learning_rate'], n_iters=EV_config['n_iters'])
    util_function = NeuralNetworkUtil(activation=model_config['activation'], n_units=model_config['hidden_units'])

    if DEBUG:
        print("optimizer:", optimizer.id)
        print("util function:", util_function.id)

    # initialize the expected utility model
    model = ExpectedUtilityModel(optimizer=optimizer, util_func=util_function, loss_function=model_config['loss_function'])
    model.set_optimizer(optimizer=optimizer, batch_size=model_config['batch_size'])
    if DEBUG:
        print("model: ", model)

    if INFO:
        print("======================\n  PRETRAIN WITH EV\n======================")
    model.fit(dataset=EV_train_data,
              val_dataset=EV_val_data,
              batch_size=EV_config['batch_size'],
              init=True)
    if INFO:
        print("Pretrain Accuracy:", model.compute_accuracy(dataset=EV_val_data))
        print("Precise Pretrain Accuracy:", model.compute_precise_accuracy(dataset=EV_val_data, range=experiment_config["accuracy_range"]))

    if INFO:
        print("======================\n  EV + HUMAN MODEL\n======================")
        # check that the dataset is larger than 0
    if len(train_data._problems) > 0:
        optimizer.n_iters = model_config["n_iters"]
        optimizer.lr = model_config["learning_rate"]
        model.fit(dataset=train_data,
                  val_dataset=val_data,
                  batch_size=model_config['batch_size'],
                  init=True)
        model_results = model.results
    else:
        model_results = None

    model_accuracy = jax.device_get(model.compute_accuracy(dataset=human_val_data))
    model_precise_accuracy = jax.device_get(model.compute_precise_accuracy(dataset=human_val_data, range=experiment_config["accuracy_range"]))
    model_loss = jax.device_get(model.evaluate(dataset=human_val_data))

    plt = model.plot(show=False)

    if INFO:
        print("Human Data Percentage:", data_percentage)
        print("Trial Accuracy:", model_accuracy)
        print("Precise Accuracy:", model_precise_accuracy)
        print("Loss:", model_loss)

    wandb.log({
        # "Results": model_results,
        "Trial Accuracy": model_accuracy,
        "Precise Accuracy": model_precise_accuracy,
        "Human Data Percentage": data_percentage,
        "Loss": model_loss,
        "Utility Function Plot": plt,
        "Utility Function Figure": wandb.Image(plt)
    })


def baseline_model(train_data, val_data, human_val_data, experiment_config, model_config, data_percentage):
    if INFO:
        print("======================\n    BASELINE MODEL\n======================")
        print("DATA PERCENTAGE:", data_percentage)
        pprint.pprint(model_config)

    # initialize optimizer and utility function
    optimizer = Adam(lr=model_config['learning_rate'], n_iters=model_config['n_iters'])
    util_function = NeuralNetworkUtil(activation=model_config['activation'], n_units=model_config['hidden_units'])
    if DEBUG:
        print("optimizer:", optimizer.id)
        print("util function:", util_function.id)

    # initialize the expected utility model
    model = ExpectedUtilityModel(optimizer=optimizer, util_func=util_function, loss_function=model_config['loss_function'])

    if DEBUG:
        print("model: ", model)

    # TODO: should the val dataset here be the human data or the val_data?
    # check that the dataset is larger than 0
    if len(train_data._problems) > 0:
        model.fit(dataset=train_data,
                  val_dataset=val_data,
                  batch_size=model_config['batch_size'],
                  init=model_config['reinitialize_model'])
        model_results = model.results
    else:
        model_results = None

    model_accuracy = jax.device_get(model.compute_accuracy(dataset=human_val_data))
    model_precise_accuracy = jax.device_get(model.compute_precise_accuracy(dataset=human_val_data, range=experiment_config["accuracy_range"]))
    model_loss = jax.device_get(model.evaluate(dataset=human_val_data))

    plt = model.plot(show=False)

    if INFO:
        print("Human Data Percentage:", data_percentage)
        print("Trial Accuracy:", model_accuracy)
        print("Precise Accuracy:", model_precise_accuracy)
        print("Loss:", model_loss)

    wandb.log({
        # "Results": model_results,
        "Trial Accuracy": model_accuracy,
        "Precise Accuracy": model_precise_accuracy,
        "Human Data Percentage": data_percentage,
        "Loss": model_loss,
        "Utility Function Plot": plt,
        "Utility Function Figure": wandb.Image(plt)
    })


def initialize_databases(num_trials=1, lowerbound=0, upperbound=0.175, increment=0.025):
    if INFO:
        print("initializing trial databases...")
    rows, cols = (num_trials, int(math.ceil(upperbound / increment)))
    training_data_arr = [[None] * cols] * rows
    validation_data_arr = [[None] * cols] * rows
    if DEBUG:
        print("num trials: ", num_trials)
        print("num data increments: ", int(math.ceil(upperbound / increment)))
        print("rows, cols:", rows, ",", cols)
        print(training_data_arr)

    human_train_dataset, human_val_data = init_data_split()
    if INFO:
        print("total number of training problems:", len(human_train_dataset._problems))

    for i, trial in enumerate(range(num_trials)):
        if INFO:
            print("#########\nTrial", trial, "\n#########")
        for j, data_percentage in enumerate(np.arange(lowerbound, upperbound, increment)):
            reduced_dataset = reduce_dataset(human_train_dataset, data_percentage)
            if INFO:
                print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(get_data_percentage(data_percentage), "% of the human dataset:", len(reduced_dataset._problems))
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            train_data, val_data = shuffle_and_get_test_train_split(reduced_dataset, trial)

            training_data_arr[i][j] = train_data
            validation_data_arr[i][j] = val_data

    if INFO:
        print("completed initializing trial databases...")
        print(training_data_arr)
        print(validation_data_arr)

    return training_data_arr, validation_data_arr, human_val_data


def run_experiments(name, training_data_arr, validation_data_arr, human_val_data, uuid, experiment_config, model_config, EV_config=None):
    for i, trial in enumerate(range(experiment_config["num_trials"])):
        run_name = generate_run_name(name, trial)
        if INFO:
            print("#########\nTrial", trial, "\n#########")
            print("RUN NAME:", run_name)
        # init wandb run
        wandb.init(project=WANDB_PROJECT_NAME,
                   group=uuid,
                   mode=WANDB_MODE,
                   name=run_name,
                   config=model_config)
        for j, data_percentage in enumerate(np.arange(experiment_config["humandata_lowerbound"], experiment_config["humandata_upperbound"], experiment_config["humandata_increment"])):
            if INFO:
                print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(get_data_percentage(data_percentage), "% of the human dataset")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            train_data = training_data_arr[i][j]
            val_data = validation_data_arr[i][j]
            if name == "BASELINE":
                baseline_model(train_data,
                               val_data,
                               human_val_data,
                               experiment_config,
                               model_config,
                               get_data_percentage(data_percentage))
            elif name == "PRETRAIN":
                pretrain_model(train_data,
                               val_data,
                               human_val_data,
                               experiment_config,
                               model_config,
                               EV_config,
                               get_data_percentage(data_percentage))
        wandb.finish()


#####################################
#         Utility Functions         #
#####################################
def generate_UUID(optional=""):
    uuid = datetime.now().strftime("%m_%d_%y_%H%M%S")
    if optional:
        uuid += "_" + optional
    if INFO:
        print("`````````````````````````````")
        print("GROUP NAME: ", uuid)
        print("`````````````````````````````")
    return uuid


def generate_run_name(name, trial_num):
    """
    Generates a run name for wandb logging.
    @param name: Model name (BASELINE or PRETRAIN)
    @param data_percentage: percentage of the total human data being used
    @param trial_num: Which trial number
    @return: run name of the form TRIALNUM_MODEL_DATAPERCENTAGE
    """
    # run_name = str(trial_num) + "_" + name + "_" + str(data_percentage) + "%"
    run_name = str(trial_num) + "_" + name
    if DEBUG:
        print(run_name)
    return run_name


def get_data_percentage(data_percentage):
    return round(data_percentage * 100, 1)


def init_data_split(fb_filter="only_fb"):
    """
    Shuffles the data and returns a 90/10 train/test split. This is just used for initializing the validation set
    and training set (which will later be split into further increments).
    @param fb_filter: default is "None" which means no ambiguity and yes feedback.
    @return: training and validation datasets.
    """
    dataset = load_c13k_data(fb_filter)
    splitter = dataset.split(p=0.9, n_splits=1, shuffle=True, random_state=None)
    (train_data, val_data) = list(splitter)[0]
    return train_data, val_data


def reduce_dataset(train_data, data_percentage):
    """
    This function takes the training data from init_data_split() and reduces it into a new dataset that is
    data_percentage% of the original size.

    @param train_data: training data from init_data_split()
    @param data_percentage: percent of the original size we want to take
    @return: reduced dataset
    """
    # reduce train_data to be size data_percentage
    splitter = train_data.split(p=data_percentage, n_splits=1, shuffle=False, random_state=None)
    (new_dataset, _) = list(splitter)[0]
    return new_dataset


def shuffle_and_get_test_train_split(reduced_dataset, trial):
    splitter = reduced_dataset.split(p=.90, n_splits=1, shuffle=True, random_state=trial)
    train_data, val_data = list(splitter)[0]
    return train_data, val_data


if __name__ == '__main__':
    experiment_config = {
        "num_trials": 4,
        "humandata_lowerbound": 0,
        "humandata_upperbound": 0.01,
        "humandata_increment": 0.001,
        "accuracy_range": 0.1
    }

    # group name for wandb
    uuid = generate_UUID()

    training_data_arr, validation_data_arr, human_val_data = initialize_databases(
        num_trials=experiment_config["num_trials"], lowerbound=experiment_config["humandata_lowerbound"],
        upperbound=experiment_config["humandata_upperbound"], increment=experiment_config["humandata_increment"])

    baseline_config = {
        "learning_rate": 0.001,
        "n_iters": 200,
        "activation": "sigmoid",
        "hidden_units": 18,
        "loss_function": "crossentropy",
        "reinitialize_model": True,
        "optimizer": "Adam",
        "hurd_model": "Expected Utility",
        "batch_size": None
    }

    EV_config = {
        "learning_rate": 0.001,
        "n_iters": 200,
        "batch_size": None
    }

    pretrain_config = {
        "learning_rate": 0.001,
        "n_iters": 200,
        "activation": "sigmoid",
        "hidden_units": 18,
        "loss_function": "crossentropy",
        "reinitialize_model": False,
        "optimizer": "Adam",
        "hurd_model": "Expected Utility",
        "batch_size": None
    }

    run_experiments("BASELINE", training_data_arr, validation_data_arr, human_val_data, uuid, experiment_config, baseline_config)
    run_experiments("PRETRAIN", training_data_arr, validation_data_arr, human_val_data, uuid, experiment_config, pretrain_config, EV_config=EV_config)
