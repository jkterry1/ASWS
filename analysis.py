import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import shapiro, anderson, ttest_1samp

def smoothen2(arr, gamma, count):
    smooth = np.zeros(len(arr))
    smooth[0] = arr[0]
    for i in range(1, count):
        smooth[i] = gamma * smooth[i-1] + arr[i]

    weight = 1
    for i in range(1, count):
        weight = weight * gamma + 1
        smooth[i] /= weight

    for i in range(count, len(arr)):
        tmp = 0
        for j in range(i-count+1, i+1):
            tmp = tmp * gamma + arr[j]
        smooth[i] = tmp / weight
    return smooth

def moving_shapiro(array, num_data = 20):
    p_values = np.zeros(len(array)-num_data)
    for j in range(len(array)-num_data):
        p_values[j] = shapiro(array[j : min(len(array),j + num_data)])[1]
    return p_values

def moving_ttest(array, num_data = 20):
    avgs = np.zeros(len(array)-num_data)
    for j in range(len(array)-num_data):
        avgs[j] = ttest_1samp(array[j : min(len(array),j + num_data)], 0.0)[1]
    return avgs

def read_file(filename):
    train_loss, train_acc, test_loss, test_acc = np.zeros(400),np.zeros(400),np.zeros(400),np.zeros(400)
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            epoch_data = np.array(row).astype(np.float)
            train_loss[i], train_acc[i], test_loss[i], test_acc[i] = epoch_data[1], epoch_data[2], epoch_data[3], epoch_data[4]
            test_acc[i] = test_acc[i] / 100
            i += 1
    return train_loss, train_acc, test_loss, test_acc

def get_grads(arr, count = 2):
    grads = np.zeros((count+1, 400))
    grads[0,:] = np.array([arr])
    for j in range(1, count + 1):
        grads[j,:] = np.gradient(grads[j-1,:])
    return grads

# given a model's complete test acc history, get stopping points
# returns standard epoch, aswt epoch, standard acc, best acc
def get_stopping_points(test_acc, num_data, gamma, count, local_maxima=0, slack_prop=0.05):
    standard_epoch, standard_acc = get_standard_stopping_point_of_curve(test_acc)
    aswt_epoch, aswt_acc = get_aswt_stopping_point_of_model(test_acc, gamma=gamma, num_data=num_data, count=count, local_maxima=local_maxima, slack_prop=slack_prop)
    return standard_epoch, aswt_epoch, standard_acc, aswt_acc

def parse_args():
    parser = argparse.ArgumentParser(description="analyze losses")
    parser.add_argument("-f","--fileheader", type=str)
    parser.add_argument("-g","--gamma", type=float, default=0.0)
    parser.add_argument("-c","--local_count", type=int, default=10)
    parser.add_argument("-n","--num_data", type=int, default=20)
    args = parser.parse_args()
    return args.gamma, args.fileheader, args.num_data, args.local_count


# for a given gamma, filename, num_data and count
# return standard epochs taken, new epochs taken, best epochs taken, standard accuracy, and new accuracy and best accuracy
def early_stopping_analysis(gamma, filename, num_data, count, local_maxima=0, slack_prop=0.05):
    train_loss, train_acc, test_loss, test_acc = read_file(filename)
    return get_stopping_points(test_acc, num_data, gamma, count, local_maxima=local_maxima, slack_prop=slack_prop)

###
# Inputs:
#     test_acc_set: list of full test acc curves for single model type
#     gamma, num_data, count: hyperparams
# Outputs:
#     average(standard_epoch[i]-aswt_epoch[i]) for all i -> avg_standard_epoch_diff
#     average(standard_acc[i]-aswt_acc[i]) for all i -> avg_standard_acc_diff
#     average(max_epoch[i]-aswt_epoch[i]) for all i -> avg_max_epoch_diff
#     average(max_acc[i]-aswt_acc[i]) for all i -> avg_max_acc_diff
###
def fast_early_stopping_of_dataset(test_acc_set, gamma, num_data, count, local_maxima=0, slack_prop=0.05):
    avg_standard_epoch_diff = 0.0
    avg_standard_acc_diff = 0.0
    avg_max_epoch_diff = 0.0
    avg_max_acc_diff = 0.0

    iterations = len(test_acc_set)
    for i in range(iterations):
        standard_epochs, new_epochs, standard_acc, new_acc = get_stopping_points(test_acc_set[i], num_data, gamma, count, local_maxima=local_maxima, slack_prop=slack_prop)
        best_epochs, best_acc = get_max_stopping_point_of_curve(test_acc_set[i])

        standard_epoch_diff = standard_epochs - new_epochs
        standard_acc_diff = standard_acc - new_acc
        max_epoch_diff = best_epochs - new_epochs
        max_acc_diff = best_acc - new_acc

        avg_standard_epoch_diff += standard_epoch_diff
        avg_standard_acc_diff += standard_acc_diff
        avg_max_epoch_diff += max_epoch_diff
        avg_max_acc_diff += max_acc_diff

    avg_standard_epoch_diff = avg_standard_epoch_diff/iterations
    avg_standard_acc_diff = avg_standard_acc_diff/iterations
    avg_max_epoch_diff = avg_max_epoch_diff/iterations
    avg_max_acc_diff = avg_max_acc_diff/iterations
    return avg_standard_epoch_diff, avg_standard_acc_diff, avg_max_epoch_diff, avg_max_acc_diff

def denoising_fast_early_stopping_of_dataset(test_acc_set, gamma, model, num_data, count, local_maxima_range, dataset):
    avg_standard_epochs = 0
    avg_new_epochs = 0
    avg_standard_acc = 0.0
    avg_new_acc = 0.0
    iterations = len(test_acc_set)
    for i in range(iterations):
        filename = "losses/" + model + "/" + model + "_" + str(i) + ".txt"
        input_acc = denoise_test_acc(test_acc_set[i], local_maxima_range=local_maxima_range)
        standard_epochs, standard_acc = get_standard_stopping_point_of_curve(test_acc_set[i])
        new_epochs, new_acc = get_aswt_stopping_point_of_model(input_acc, gamma=gamma, count=count, num_data=num_data)
        avg_standard_epochs += standard_epochs
        avg_new_epochs += new_epochs
        avg_standard_acc += standard_acc
        avg_new_acc += new_acc
    avg_standard_epochs = avg_standard_epochs/iterations
    avg_new_epochs = avg_new_epochs/iterations
    avg_standard_acc = avg_standard_acc/iterations
    avg_new_acc = avg_new_acc/iterations
    return avg_standard_epochs, avg_new_epochs, avg_standard_acc, avg_new_acc
    
# return average of standard epochs, average of new epochs, avg of standard accuracy, avg of new accuracy
def early_stopping_of_dataset(gamma, model, num_data, count, local_maxima, slack_prop, dataset):
    avg_standard_epochs = 0
    avg_new_epochs = 0
    avg_standard_acc = 0.0
    avg_new_acc = 0.0
    for i in [0, 1, 2, 3, 4, 75, 76, 77, 78, 79]:
        filename = "losses/" + model + "/" + model + "_" + str(i) + ".txt"
        standard_epochs, new_epochs, standard_acc, new_acc = early_stopping_analysis(gamma, filename, num_data, count, local_maxima=local_maxima, slack_prop=slack_prop)
        avg_standard_epochs += standard_epochs
        avg_new_epochs += new_epochs
        avg_standard_acc += standard_acc
        avg_new_acc += new_acc
    avg_standard_epochs = avg_standard_epochs/10
    avg_new_epochs = avg_new_epochs/10
    avg_standard_acc = avg_standard_acc/10
    avg_new_acc = avg_new_acc/10
    return avg_standard_epochs, avg_new_epochs, avg_standard_acc, avg_new_acc

#show_plots(epochs, train_loss, test_loss, smooth_train_loss, smooth_test_loss, train_acc, test_acc, smooth_train_acc, smooth_test_acc)
#show_shapiro_plots(test_acc, num_data=num_data, gamma=gamma, count=count, best_test_accuracy_epoch=best_test_accuracy_epoch, best_test_loss_epoch=best_test_loss_epoch)

##
# Standard Early Stopping Method
# Will stop when test accuracy hasn't reached new maximum in 60 epochs
# "Decrease in performance over given number of epochs"
##
def get_standard_stopping_point_of_curve(test_acc):
    standard = {}
    #standard_count = 40
    standard_count=60
    for i in range(len(test_acc)):
        if test_acc[i] == max(test_acc[i:min(i+standard_count,len(test_acc))]):
            standard["epoch_num"] = i
            standard["accuracy"] = test_acc[i]
            break
    return standard["epoch_num"], standard["accuracy"] 

##
# Patience Early Stopping Method
# Will stop when test acc has not improved in PATIENCE epochs
# improvement is when test acc has increased by atleast min_delta
##
def get_patience_stopping_point_of_curve(test_acc, patience=1, min_delta=0):
    stop_epoch = 0
    stop_acc = 0
    neg_diff_count = 0
    for i in range(1, len(test_acc)):
        diff = test_acc[i] - test_acc[i-1]
        if diff >= min_delta:
            neg_diff_count = 0
        else:
            neg_diff_count += 1
        if neg_diff_count == patience:
            stop_epoch = i
            stop_acc = max(test_acc[:i])
            return stop_epoch, stop_acc
    # only possible if model kept improving until the end
    return -1, -1

###
# Average change based early stopping
# Will stop if average delta of test acc falls below threshold (wi)
# "Average change in metric over a given number of epochs"
###
def get_averaged_stopping_point_of_curve(test_acc, window=150, min_delta_average=0.001):
    stop_epoch = 0
    stop_acc = 0
    for i in range(window, len(test_acc)):
        diff = np.diff(test_acc[(i-window):i])
        diff_avg = np.average(diff)
        if diff_avg < min_delta_average:
            stop_epoch = i
            stop_acc = max(test_acc[:i])
            return stop_epoch, stop_acc
    # only possible if model kept improving until the end
    return -1, -1

# returns standard stop epoch and acc
def get_standard_stopping_point(model, file_suffix):
    filename = "losses/" + model + "/" + model + "_" + str(file_suffix) + ".txt"
    train_loss, train_acc, test_loss, test_acc = read_file(filename)
    return get_standard_stopping_point_of_curve(test_acc)

# returns the epoch with max test acc in curve
def get_max_stopping_point_of_curve(test_acc):
    max_epoch = np.argmax(test_acc)
    return max_epoch, test_acc[max_epoch]

def get_max_stopping_point(model, file_suffix):
    filename = "losses/" + model + "/" + model + "_" + str(file_suffix) + ".txt"
    train_loss, train_acc, test_loss, test_acc = read_file(filename)
    return get_max_stopping_point_of_curve(test_acc)

# given a test acc curve (and hyperparams), determine whether the training should stop
# returns True/False
###
# Overview of Algorithm:
# If fft analysis enabled, use fft to remove local_maxima signals from input curve
# For a test acc array, with length N, create (N-num_data) buckets of size num_data
# For each bucket: 
#   determine t-test of d/dt(curve)
#   determine shapiro test of smooth(curve)
# If "SlackProp" percentage of the last "Slack" buckets is True for both tests
#    Return True
# Else
#    Return False
###
# acc_curve: input test acc curve, must be atleast length max(count, num_data)
# gamma, count: parameters for exponential smoothing
# num_data: sample size for ttest and shapiro test
# local_maxima: local max factor for fft smoothing. If set to 0, fft analysis is disabled
###
def aswt_stopping(acc_curve, gamma, count, num_data, slack=20, local_maxima=0, slack_prop=0.05):
    if local_maxima != 0:
        acc_curve = denoise_test_acc(acc_curve, local_maxima)
    ttest_fd = moving_ttest(np.gradient(acc_curve),num_data)
    smoothed = smoothen2(np.gradient(acc_curve), gamma, count)
    shapiro = moving_shapiro(smoothed,num_data)

    bar = 0.97
    ar1 = [False] * (len(ttest_fd)+slack)
    ar2 = [False] * (len(ttest_fd)+slack)
    for i in range(len(ttest_fd)):
        if shapiro[i] > bar:
            for j in range(i,i+slack):
                ar1[j] = True
        if ttest_fd[i] > bar:
            for j in range(i,i+slack):
                ar2[j] = True
    slack_req = int(slack_prop*slack)
    slack_match = 0
    for i in range(len(ttest_fd)-slack, len(ttest_fd)):
        if ar1[i] and ar2[i]:
            slack_match += 1
    if slack_match >= slack_req:
        return True 
    return False

# this is an experimental change to our proposed ASWT stopping
# this will be used to toggle either no Shapiro bias or no T-Test bias
# that is, only one or the other will be used to determine when to stop
# if not using shapiro, then t-test must be used
def even_more_augmented_aswt_stopping(acc_curve, gamma, count, num_data, slack=20, slack_prop=0.05, use_shapiro=True):
    ttest_fd = moving_ttest(np.gradient(acc_curve),num_data)
    smoothed = smoothen2(np.gradient(acc_curve), gamma, count)
    shapiro = moving_shapiro(smoothed,num_data)

    bar = 0.97
    ar1 = [False] * (len(ttest_fd)+slack)
    ar2 = [False] * (len(ttest_fd)+slack)
    for i in range(len(ttest_fd)):
        if shapiro[i] > bar:
            for j in range(i,i+slack):
                ar1[j] = True
        if ttest_fd[i] > bar:
            for j in range(i,i+slack):
                ar2[j] = True
    slack_req = int(slack_prop*slack)
    slack_match = 0
    for i in range(len(ttest_fd)-slack, len(ttest_fd)):
        if use_shapiro:
            if ar1[i]:
                slack_match += 1
        else:
            if ar2[2]:
                slack_match += 1
    if slack_match >= slack_req:
        return True 
    return False

# Given the complete test acc history of a model, simulate when to stop using ASWT
# returns epoch, test acc
# NOTE: Test Acc is the max acc that exists in the curve, up till the stopping epoch
def get_aswt_stopping_point_of_model(test_acc, gamma, num_data, count=20, local_maxima=0, slack_prop=0.05):
    test_epoch = max(num_data, count)
    stop_epoch = 399
    stop_acc = test_acc[399]
    while test_epoch < len(test_acc):
        test_acc_curve = test_acc[:test_epoch]
        should_stop = aswt_stopping(test_acc_curve, gamma=gamma, count=count, num_data=num_data, local_maxima=local_maxima, slack_prop=slack_prop)
        if should_stop:
            stop_epoch = test_epoch
            stop_acc = np.amax(test_acc[:stop_epoch])
            test_epoch = len(test_acc)+1
        else:
            test_epoch += 1
    return stop_epoch, stop_acc

def get_augmented_aswt_stopping_point_of_model(test_acc, gamma, num_data, count = 20, slack_prop=0.05, use_shapiro=True):
    test_epoch = max(num_data, count)
    stop_epoch = 399
    stop_acc = test_acc[399]
    while test_epoch < len(test_acc):
        test_acc_curve = test_acc[:test_epoch]
        should_stop = even_more_augmented_aswt_stopping(test_acc_curve, gamma=gamma, count=count, num_data=num_data, slack_prop=slack_prop, use_shapiro=use_shapiro)
        if should_stop:
            stop_epoch = test_epoch
            stop_acc = np.amax(test_acc[:stop_epoch])
            test_epoch = len(test_acc)+1
        else:
            test_epoch += 1
    return stop_epoch, stop_acc

def get_aswt_stopping_point(model, file_suffix, gamma, count, num_data, slack_prop=0.05):
    filename = "losses/" + model + "/" + model + "_" + str(file_suffix) + ".txt"
    train_loss, train_acc, test_loss, test_acc = read_file(filename)
    return get_aswt_stopping_point_of_model(test_acc, gamma, count, num_data)

def get_patience_stopping_point(model, file_suffix, patience=50):
    filename = "losses/" + model + "/" + model + "_" + str(file_suffix) + ".txt"
    train_loss, train_acc, test_loss, test_acc = read_file(filename)
    return get_patience_stopping_point_of_curve(test_acc, patience=patience)

def get_mindelta_stopping_point(model, file_suffix, min_delta=0.001):
    filename = "losses/" + model + "/" + model + "_" + str(file_suffix) + ".txt"
    train_loss, train_acc, test_loss, test_acc = read_file(filename)
    return get_patience_stopping_point_of_curve(test_acc, patience=27, min_delta=min_delta)

def get_averagees_stopping_point(model, file_suffix):
    filename = "losses/" + model + "/" + model + "_" + str(file_suffix) + ".txt"
    train_loss, train_acc, test_loss, test_acc = read_file(filename)
    return get_averaged_stopping_point_of_curve(test_acc)

# for a given curve, first find the standard stopping points/acc
# then determine the earliest epoch where the acc is > standard_acc-acc_threshold
# returns -1, -1 if a best possible does not exist
def get_best_possible_stopping_point(test_acc, acc_threshold=0.05, use_best=False):
    standard_epochs, standard_acc = get_standard_stopping_point_of_curve(test_acc)
    if use_best:
        standard_epochs, standard_acc = get_max_stopping_point_of_curve(test_acc)
    best_epochs = -1
    best_acc = -1
    for i in range(standard_epochs):
        if test_acc[i] > (standard_acc-acc_threshold):
            best_epochs = i
            best_acc = test_acc[i]
            break
    return best_epochs, best_acc

# input is model, acc file, and stopping point relative to standard
# output is fft bin freq, fft power density, and local freq maxima
def fft_analysis(model, file_suffix, stopping_point=0, local_maxima_range=1):
    filename = "losses/" + model + "/" + model + "_" + str(file_suffix) + ".txt"
    train_loss, train_acc, test_loss, test_acc = read_file(filename)
    analyzed_test_acc = test_acc
    time_series_test_acc = np.gradient(analyzed_test_acc)

    st = max(min(stopping_point, len(time_series_test_acc)), 5)
    sub_time_series = time_series_test_acc[:st]
    fft_res = np.fft.fft(sub_time_series)
    fft_len = int(len(fft_res))
    fft_res = fft_res
    ampli_curve = np.abs(fft_res)**2
    time_step = 1
    freqs = np.fft.fftfreq(fft_len, time_step)
    idx = np.argsort(freqs)
    idx = idx[len(idx)//2:]
    local_maxima = []
    for i in range(local_maxima_range, len(ampli_curve)//2-local_maxima_range):
        low_r = i-local_maxima_range
        high_r = i+local_maxima_range
        maxima_range = ampli_curve[low_r:high_r]
        ind_max = np.argmax(maxima_range)
        if type(ind_max) is np.int64:
            if ind_max == local_maxima_range:
                local_maxima.append(i)
    return freqs[idx], ampli_curve[idx], local_maxima

# input is model, acc file, and stopping point relative to standard
# output is fft bin freq, fft power density, and local freq maxima
def denoise_fft(model, file_suffix, stopping_point=0, local_maxima_range=1):
    filename = "losses/" + model + "/" + model + "_" + str(file_suffix) + ".txt"
    train_loss, train_acc, test_loss, test_acc = read_file(filename)
    analyzed_test_acc = test_acc
    time_series_test_acc = np.gradient(analyzed_test_acc)

    st = max(min(stopping_point, len(time_series_test_acc)), 5)
    sub_time_series = time_series_test_acc[:st]
    fft_res = np.fft.fft(sub_time_series)
    fft_len = int(len(fft_res))
    
    ampli_curve = np.abs(fft_res)**2
    denoised_curve = fft_res[:]
    time_step = 1
    local_maxima = []
    for i in range(local_maxima_range, len(ampli_curve)-local_maxima_range):
        low_r = i-local_maxima_range
        high_r = i+local_maxima_range
        maxima_range = ampli_curve[low_r:high_r]
        ind_max = np.argmax(maxima_range)
        if type(ind_max) is np.int64:
            if ind_max == local_maxima_range:
                local_maxima.append(i)
    for lo in local_maxima:
        denoised_curve[lo] = 0

    denoised_time = np.fft.ifft(denoised_curve).real

    return denoised_time

def denoise_test_acc(test_acc, local_maxima_range=1):
    analyzed_test_acc = test_acc
    time_series_test_acc = np.gradient(analyzed_test_acc)

    sub_time_series = time_series_test_acc[:]
    fft_res = np.fft.fft(sub_time_series)
    fft_len = int(len(fft_res))
    
    ampli_curve = np.abs(fft_res)**2
    denoised_curve = fft_res[:]
    time_step = 1
    local_maxima = []
    for i in range(local_maxima_range, len(ampli_curve)-local_maxima_range):
        low_r = i-local_maxima_range
        high_r = i+local_maxima_range
        maxima_range = ampli_curve[low_r:high_r]
        ind_max = np.argmax(maxima_range)
        if type(ind_max) is np.int64:
            if ind_max == local_maxima_range:
                local_maxima.append(i)
    for lo in local_maxima:
        denoised_curve[lo] = 0

    denoised_time = np.fft.ifft(denoised_curve).real

    return denoised_time

###
# acc threshold of 0.5%
#Max Gamma is  0.9099999999999999  max local count is  15  max num data is  20  with a max summed epoch difference of  17.72
###

###
# acc threshold of 0.05%
#Max Gamma is  0.37  max local count is  10  max num data is  30  with a max summed epoch difference of  -50.018
###

###
# acc threshold of 0.01%

###

if __name__ == "__main__":
    model_names = ["alexnet", "fc1", "fc2", "googlenet", "lenet", "resnet34", "resnet50", "resnet101", "vgg11", "vgg16", "vgg19"]
    gamma, fileheader, num_data, count = parse_args()
    graph_file = open("optimized_hypers2.csv", "w")
    graph_file.write("Model,Standard,AWST")
    for model in model_names:
        total_epochs_difference = 0
        total_accuracy_difference = 0.0
        avg_standard_epochs = 0
        avg_new_epochs = 0
        avg_standard_acc = 0.0
        avg_new_acc = 0.0
        for i in range(0,5):
            filename="losses/" + model + "/" + model + "_" + str(i) + ".txt"
            standard_epochs, new_epochs, best_epochs, standard_acc, new_acc, best_acc = early_stopping_analysis(gamma, filename, num_data, count)
            total_epochs_difference += standard_epochs - new_epochs
            total_accuracy_difference += standard_acc - new_acc
            avg_standard_epochs += standard_epochs
            avg_new_epochs += new_epochs
            avg_standard_acc += standard_acc
            avg_new_acc += new_acc
        print("standard took ", total_epochs_difference/5, "more epochs for ", total_accuracy_difference/5, "better accuracy")
        avg_standard_epochs = avg_standard_epochs / 5
        avg_new_epochs = avg_new_epochs / 5
        s1 = model + "," + str(avg_standard_epochs) + "," + str(avg_new_epochs) + "\n"
        graph_file.write(s1)
    graph_file.close()
# CSV FILE FORMAT
# model, method, run#, epochs, accuracy, gamma, local_count, num_data
