# due to issues with ASWS stat tests, experimenting with new statistical tests in this file

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy
from analysis import read_file, get_standard_stopping_point_of_curve
from scipy.stats import shapiro, anderson, ttest_1samp, kurtosis, skew

def normalityTestF(resid, k_error, t_error):
    k = kurtosis(resid)
    t = skew(resid)
    if k > -1*k_error and k < k_error and t > -1*t_error and t < t_error:
        return True
    return False

def moving_normality_test(array, k_error, t_error, num_data=20):
    bool_vals = np.zeros(len(array)-num_data)
    for j in range(len(array)-num_data):
        bool_vals[j] = normalityTestF(array[j : min(len(array), j+num_data)], k_error, t_error)
    return bool_vals

# given a test acc curve (and hyperparams), determine whether the training should stop
# returns True/False
###
# Overview of Algorithm:
# TODO: INSERT ALGORITHM DESCRIPTION HERE
###
def normality_stopping(acc_curve, k_error, t_error, num_data):
    norm_list = moving_normality_test(np.gradient(acc_curve), k_error, t_error, num_data)
    true_count = 0
    if len(norm_list) > 20:
        for i in range(len(norm_list)-20, len(norm_list)):
            if norm_list[i]:
                true_count += 1
        if true_count > 19:
            return True
    return False

def get_normality_stopping_point_of_model(test_acc, k_error, t_error, num_data):
    test_epoch = num_data
    stop_epoch = 399
    stop_acc = test_acc[399]
    while test_epoch < len(test_acc):
        test_acc_curve = test_acc[:test_epoch]
        should_stop = normality_stopping(test_acc_curve, k_error=k_error, t_error=t_error, num_data=num_data)
        if should_stop:
            stop_epoch = test_epoch
            stop_acc = np.amax(test_acc[:stop_epoch])
            test_epoch = len(test_acc)+1
        else:
            test_epoch += 1
    return stop_epoch, stop_acc

def normality_stopping_of_dataset(model, k_error, t_error, num_data):
    avg_standard_epochs = 0
    avg_new_epochs = 0
    avg_standard_acc = 0.0
    avg_new_acc = 0.0
    for i in [0, 1, 2, 3, 4, 75, 76, 77, 78, 79]:
        filename = "losses/" + model + "/" + model + "_" + str(i) + ".txt"
        standard_epochs, new_epochs, standard_acc, new_acc = normality_stopping_analysis(filename, k_error=k_error, t_error=t_error, num_data=num_data)
        avg_standard_epochs += standard_epochs
    avg_new_epochs += new_epochs
    avg_standard_acc += standard_acc
    avg_new_acc += new_acc
    avg_standard_epochs = avg_standard_epochs/10
    avg_new_epochs = avg_new_epochs/10
    avg_standard_acc = avg_standard_acc/10
    avg_new_acc = avg_new_acc/10
    return avg_standard_epochs, avg_new_epochs, avg_standard_acc, avg_new_acc

def normality_stopping_analysis(filename, k_error, t_error, num_data):
    train_loss, train_acc, test_loss, test_acc = read_file(filename)
    return get_stopping_points(test_acc, k_error=k_error, t_error=t_error, num_data=num_data) 

def get_stopping_points(test_acc, k_error, t_error, num_data):
    standard_epoch, standard_acc = get_standard_stopping_point_of_curve(test_acc)
    norm_epoch, norm_acc = get_normality_stopping_point_of_model(test_acc, k_error=k_error, t_error=t_error, num_data=num_data)
    return standard_epoch, norm_epoch, standard_acc, norm_acc
