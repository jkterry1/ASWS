from analysis import fast_early_stopping_of_dataset
from analysis import early_stopping_of_dataset
from analysis import read_file
from analysis2 import normality_stopping_of_dataset
import analysis
import argparse
import threading
import numpy as np
import queue
import tqdm

model_names = ["alexnet", "fc1", "fc2", "GoogLeNet", "resnet34", "resnet50", "resnet101", "vgg11", "vgg16", "vgg19"]

###
# FFT = decomposes function
# decomposes into sins/cosines with diff amplitue freq and phases
# list of complex vals returned: only look at real vals
# 

###

# goes through all hp_grid files
# returns dict for each model containing the hp grid data
def parse_full_hp_set():
    file_prefix = "hp_grid_"
    grid_groups = {model:[] for model in model_names}
    for mod in model_names:
        f_name = file_prefix + str(mod)
        fh = open(f_name, "r")

def search_hyperparameters(hyperparams, accuracy_difference_threshold, model_test_accs, file_name):
    iterations = 0
    total_iterations = len(hyperparams)
    output_file = open(file_name, "a")
    for num_data, k, t in hyperparams:
        num_data = int(num_data)
        iterations = iterations + 1
        print(str(iterations) + "/" + str(total_iterations) + "\n", flush=True)
        for model in model_names:
            test_accs = model_test_accs[model]
            avg_std_epoch_diff, avg_new_epoch_diff, avg_std_acc_diff, avg_new_acc_diff = normality_stopping_of_dataset(model, k, t, num_data)
            output_string = str(model) + "," + str(k) + "," + str(t) + "," + str(num_data) + "," + str(avg_std_epoch_diff) + "," + str(avg_std_acc_diff) + "," + str(avg_new_epoch_diff) + "," + str(avg_new_acc_diff) + "\n"
            output_file.write(output_string)
    output_file.close()

# go through every curve, make sure a best possible stopping point exists
def analyze_best_possibles(model_test_accs, acc_threshold=0.05, use_best=False, output_average=False):
    for model in model_test_accs:
        test_accs = model_test_accs[model]
        average_stop_epoch = 0.0
        average_stop_epoch_count = 0
        for i in range(len(test_accs)):
            test_acc = test_accs[i]
            best_epochs, best_acc = analysis.get_best_possible_stopping_point(test_acc, acc_threshold=acc_threshold, use_best=use_best)
            if best_epochs == -1:
                print(model, i, " does not have best")
            else:
                average_stop_epoch += best_epochs
                average_stop_epoch_count += 1
        if average_stop_epoch_count > 0:
            average_stop_epoch = average_stop_epoch/average_stop_epoch_count
            if output_average:
                print(model,":",average_stop_epoch)
        else:
            print(model,": NULL")
      
# compare hyperparamresults against standard stopping point
def analyze_grid_data(acc_threshold=0.01):
    fh = open("hyperparameter_grid_models_slackprop.csv", "r")
    grid_data = []
    for line in fh:
        parsed = line.split(",")
        parsed[1] = float(parsed[1])
        parsed[5] = float(parsed[5])
        parsed[6] = float(parsed[6])
        parsed[7] = float(parsed[7])
        parsed[8] = float(parsed[8])
        parsed[9] = float(parsed[9])
        grid_data.append(parsed)
    fh.close()
    grid_groups = {model: [] for model in model_names}
    for dat in grid_data:
        grid_groups[dat[0]].append(dat[1:])
    print("Model,\tParameters,\tGamma,\tCount,\tNumData,\tLocalMaxima,\tSlackProp,\tAvgStdEpoch,\tAvgASWTEpoch,\tAvgStdAcc,\tAvgASWTAcc")
    for model in grid_groups:
        epoc_max = -100000
        data_max = None
        for dat in grid_groups[model]:
            if dat[6] < acc_threshold and dat[5] > epoc_max:
                data_max = dat
                epoc_max = dat[5]
        if data_max is not None:
            avg_standard_epochs, avg_new_epochs, avg_standard_acc, avg_new_acc = early_stopping_of_dataset(gamma=float(data_max[0]), model=model, num_data=int(data_max[2]), count=int(data_max[1]), local_maxima=int(data_max[3]), slack_prop=float(data_max[4]), dataset="")
            print(model, ",", model_parameter_map[model], ",", str(data_max[0]), ",", str(data_max[1]), ",", str(data_max[2]),",", str(data_max[3]),",", str(data_max[4]),",", str(avg_standard_epochs), ",", str(avg_new_epochs), ",", str(avg_standard_acc), ",", str(avg_new_acc))
        else:
            print(model,",NoSolution")

# file schema is model,gamma,count,num_data, local_max, slack_prop, avg_std_epoch_diff, avg_std_acc_diff, avg_max_epoch_diff, avg_max_acc_diff
def analyze_slackprop_dist():
    possib_slackprop_vals = np.linspace(0.05, 0.95, num=12, endpoint=True).tolist()
    slackprop_lists = {} # maps slack prop val to list of epoch differences
    for sp in possib_slackprop_vals:
        slackprop_lists[np.around(sp, decimals=3)] = []
    for model in tqdm.tqdm(model_names):
        file_name = "hp_grid3_" + model
        fh = open(file_name, "r")
        grid_data = []
        for line in fh:
            parsed = line.split(",")
            parsed[1] = float(parsed[1])
            parsed[5] = float(parsed[5])
            parsed[6] = float(parsed[6])
            parsed[7] = float(parsed[7])
            parsed[8] = float(parsed[8])
            parsed[9] = float(parsed[9])
            grid_data.append(parsed[1:])
        fh.close()
        slackprop_index = 4
        for grid_line in grid_data:
            epoch_diff = grid_line[5]
            slackprop_lists[np.around(grid_line[slackprop_index], decimals=3)].append(epoch_diff)
    outputline1 = "slackprop,"
    outputline2 = "MeanEpochDifference,"
    outputline3 = "STDofEpochDifference,"
    for sp in possib_slackprop_vals:
        outputline1 += str(sp) + ","
        np_list = np.array(slackprop_lists[np.around(sp, decimals=3)])
        sp_std = np.std(np_list)
        sp_mean = np.mean(np_list)
        outputline2 += str(sp_mean) + ","
        outputline3 += str(sp_std) + ","
    outputline1 = outputline1[:-1]
    outputline2 = outputline2[:-1]
    outputline3 = outputline3[:-1]
    print(outputline1)
    print(outputline2)
    print(outputline3)
    
def analyze_gamma_dist():
    possib_gamma_vals = [0] + list(np.linspace(0.1, 1, num=8, endpoint=False))
    slackprop_lists = {} # maps slack prop val to list of epoch differences
    for ga in possib_gamma_vals:
        slackprop_lists[np.around(ga, decimals=3)] = []
    for model in tqdm.tqdm(model_names):
        file_name = "hp_grid3_" + model
        fh = open(file_name, "r")
        grid_data = []
        for line in fh:
            parsed = line.split(",")
            parsed[1] = float(parsed[1])
            parsed[5] = float(parsed[5])
            parsed[6] = float(parsed[6])
            parsed[7] = float(parsed[7])
            parsed[8] = float(parsed[8])
            parsed[9] = float(parsed[9])
            grid_data.append(parsed[1:])
        fh.close()
        gamma_index = 0
        for grid_line in grid_data:
            epoch_diff = grid_line[5]
            slackprop_lists[np.around(grid_line[gamma_index], decimals=3)].append(epoch_diff)
    outputline1 = "Gamma,"
    outputline2 = "MeanEpochDifference,"
    outputline3 = "STDofEpochDifference,"
    for sp in possib_gamma_vals:
        outputline1 += str(sp) + ","
        np_list = np.array(slackprop_lists[np.around(sp, decimals=3)])
        sp_std = np.std(np_list)
        sp_mean = np.mean(np_list)
        outputline2 += str(sp_mean) + ","
        outputline3 += str(sp_std) + ","
    outputline1 = outputline1[:-1]
    outputline2 = outputline2[:-1]
    outputline3 = outputline3[:-1]
    print(outputline1)
    print(outputline2)
    print(outputline3)

def analyze_sample_size_dist():
    possib_samplesize_vals = np.arange(start=5, stop=20, step=2).tolist()
    slackprop_lists = {} # maps slack prop val to list of epoch differences
    for ga in possib_samplesize_vals:
        slackprop_lists[ga] = []
    for model in tqdm.tqdm(model_names):
        file_name = "hp_grid3_" + model
        fh = open(file_name, "r")
        grid_data = []
        for line in fh:
            parsed = line.split(",")
            parsed[1] = float(parsed[1])
            parsed[5] = float(parsed[5])
            parsed[6] = float(parsed[6])
            parsed[7] = float(parsed[7])
            parsed[8] = float(parsed[8])
            parsed[9] = float(parsed[9])
            grid_data.append(parsed[1:])
        fh.close()
        samplesize_index = 2
        for grid_line in grid_data:
            epoch_diff = grid_line[5]
            slackprop_lists[int(grid_line[samplesize_index])].append(epoch_diff)
    outputline1 = "num_data,"
    outputline2 = "MeanEpochDifference,"
    outputline3 = "STDofEpochDifference,"
    for sp in possib_samplesize_vals:
        outputline1 += str(sp) + ","
        np_list = np.array(slackprop_lists[sp])
        sp_std = np.std(np_list)
        sp_mean = np.mean(np_list)
        outputline2 += str(sp_mean) + ","
        outputline3 += str(sp_std) + ","
    outputline1 = outputline1[:-1]
    outputline2 = outputline2[:-1]
    outputline3 = outputline3[:-1]
    print(outputline1)
    print(outputline2)
    print(outputline3)


# file schema is model,gamma,count,num_data, local_max, slack_prop, avg_std_epoch_diff, avg_std_acc_diff, avg_max_epoch_diff, avg_max_acc_diff
def analyze_hp_grid_data(model, acc_threshold=0.005):
    file_name = "hp_grid3_" + model
    fh = open(file_name, "r")
    grid_data = []
    for line in fh:
        parsed = line.split(",")
        parsed[1] = float(parsed[1])
        parsed[5] = float(parsed[5])
        parsed[6] = float(parsed[6])
        parsed[7] = float(parsed[7])
        parsed[8] = float(parsed[8])
        parsed[9] = float(parsed[9])
        grid_data.append(parsed[1:])
    fh.close()
    epoc_max = -100000
    data_max = None
    for dat in grid_data:
        if dat[6] < acc_threshold and dat[5] > epoc_max and int(dat[3]) == 0:
            data_max = dat
            epoc_max = dat[5]
    output_dict = {}
    if data_max is not None:
        avg_standard_epochs, avg_new_epochs, avg_standard_acc, avg_new_acc = early_stopping_of_dataset(gamma=float(data_max[0]), model=model, num_data=int(data_max[2]), count=int(data_max[1]), local_maxima=int(data_max[3]), slack_prop=float(data_max[4]), dataset="")
        #print(model, ",", model_parameter_map[model], ",", str(data_max[0]), ",", str(data_max[1]), ",", str(data_max[2]),",", str(data_max[3]),",", str(data_max[4]),",", str(avg_standard_epochs), ",", str(avg_new_epochs), ",", str(avg_standard_acc), ",", str(avg_new_acc))
        print("{}, {}, {:.3f}, {}, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(model, model_parameter_map[model], data_max[0], str(data_max[1]), str(data_max[2]), str(data_max[3]), data_max[4], avg_standard_epochs, avg_new_epochs, avg_standard_acc, avg_new_acc))
        output_dict["avg_std_acc"] = avg_standard_acc
        output_dict["avg_std_epoch"] = avg_standard_epochs
        output_dict["avg_new_acc"] = avg_new_acc
        output_dict["avg_new_epoch"] = avg_new_epochs
    else:
        print(model,",NoSolution")

    output_dict["gamma"] = data_max[0]
    output_dict["count"] = data_max[1]
    output_dict["num_data"] = data_max[2]
    output_dict["local_max"] = data_max[3]
    output_dict["slack_prop"] = data_max[4]
    return output_dict

#compare hyperparam results against max acc stopping point
def analyze_grid_data_on_maxed(acc_threshold=0.01):
    fh = open("hyperparameter_grid_models_slackprop.csv", "r")
    grid_data = []
    for line in fh:
        parsed = line.split(",")
        parsed[1] = float(parsed[1])
        parsed[5] = float(parsed[5])
        parsed[6] = float(parsed[6])
        parsed[7] = float(parsed[7])
        parsed[8] = float(parsed[8])
        parsed[9] = float(parsed[9])
        grid_data.append(parsed)
    fh.close()
    grid_groups = {model: [] for model in model_names}
    for dat in grid_data:
        grid_groups[dat[0]].append(dat[1:])
    print("Model,\tParameters,\tGamma,\tCount,\tNumData,\tLocalMaxima,\tSlackProp,\tAvgMaxEpoch,\tAvgASWTEpoch,\tAvgMaxAcc,\tAvgASWTAcc")
    for model in grid_groups:
        epoc_max = -100000
        data_max = None
        for dat in grid_groups[model]:
            if dat[8] < acc_threshold and dat[7] > epoc_max:
                data_max = dat
                epoc_max = dat[7]
        if data_max is not None:
            avg_standard_epochs, avg_new_epochs, avg_standard_acc, avg_new_acc = early_stopping_of_dataset(gamma=float(data_max[0]), model=model, num_data=int(data_max[2]), count=int(data_max[1]), local_maxima=int(data_max[3]), slack_prop=float(data_max[4]), dataset="")
            print(model, ",", model_parameter_map[model], ",", str(data_max[0]), ",", str(data_max[1]), ",", str(data_max[2]),",", str(data_max[3]),",", str(data_max[4]),",", str(avg_standard_epochs), ",", str(avg_new_epochs), ",", str(avg_standard_acc), ",", str(avg_new_acc))
        else:
            print(model,",NoSolution")

def parse_args():
    parser = argparse.ArgumentParser(description="analyze losses")
    parser.add_argument("-m","--model", type=str)
    args = parser.parse_args()
    return args.model

def hp_search_new_heuristics():
    patience_values = np.arange(1, 30, step=2)
    min_delta_values = [0.001, 0.005, 0.009, 0.013, 0.017, 0.021, 0.025]
    window_values = [25, 50, 75, 100, 125, 150]
    patience_max_acc = 0.0
    min_diff_max_acc = 0.0
    avg_diff_max_acc = 0.0
    patience_max_param = []
    min_diff_max_param = []
    avg_diff_max_param = []
    count = 0
    for patience in patience_values:
        for min_delta in min_delta_values:
            count += 1
            if count % 5 == 0:
                print(count)
            # do patience and min diff search here
            # maximize average of accuracy across model
            patience_total_acc = 0.0
            min_diff_total_acc = 0.0
            for model in model_names:
                for i in range(5):
                    file_loc = "losses/" + str(model) + "/" + str(model) + "_" + str(i) + ".txt"
                    _, _, _, test_acc = analysis.read_file(file_loc)
                    _, this_patience_acc = analysis.get_patience_stopping_point_of_curve(test_acc, patience=patience)
                    _, this_min_diff_acc = analysis.get_patience_stopping_point_of_curve(test_acc, patience=patience, min_delta=min_delta)
                    patience_total_acc += this_patience_acc
                    min_diff_total_acc += this_min_diff_acc
            patience_avg_acc = patience_total_acc / len(model_names)
            min_diff_avg_acc = min_diff_total_acc / len(model_names)
            if patience_avg_acc > patience_max_acc:
                patience_max_acc = patience_avg_acc
                patience_max_param = [patience]
            if min_diff_avg_acc > min_diff_max_acc:
                min_diff_max_acc = min_diff_avg_acc
                min_diff_max_param = [patience, min_delta]
    count = 0
    for window in window_values:
        for min_delta in min_delta_values:
            count += 1
            if count % 5 == 0:
                print(count)
            # maximize average of accuracy across model
            avg_diff_total_acc = 0.0
            for model in model_names:
                for i in range(5):
                    file_loc = "losses/" + str(model) + "/" + str(model) + "_" + str(i) + ".txt"
                    _, _, _, test_acc = analysis.read_file(file_loc)
                    _, this_avg_diff_acc = analysis.get_averaged_stopping_point_of_curve(test_acc, window=window, min_delta_average=min_delta)
                    avg_diff_total_acc += this_avg_diff_acc
            avg_diff_avg_acc = avg_diff_total_acc / len(model_names)
            if avg_diff_avg_acc > avg_diff_max_acc:
                avg_diff_max_acc = avg_diff_avg_acc
                avg_diff_max_param = [window, min_delta]
    return patience_max_param, min_diff_max_param, avg_diff_max_param

# patience_hp, min_diff_hp, avg_diff_hp = hp_search_new_heuristics()
# print("PatienceStop- Patience:", patience_hp[0])
# print("Min_DiffStop- Patience:", min_diff_hp[0], "Min Delta:", min_diff_hp[1])
# print("Avg_DiffStop- Window:", avg_diff_hp[0], "Min Delta Avg:", avg_diff_hp[1])
## load all files first
arg_model = parse_args()
if arg_model in model_names:
    model_names = [arg_model]
hp_filename = "hp_grid4_" +str(arg_model)
model_parameter_map = {
"alexnet": 23272266,
"fc1": 1352510,
"fc2": 2877210,
"GoogLeNet": 6166250,
"lenet": 62006,
"resnet34": 21282122,
"resnet50": 23520842,
"resnet101": 42512970,
"vgg11": 9231114,
"vgg16": 14728266,
"vgg19": 20040522
}
model_test_accs = {}
for model in model_names:
    test_accs = []
    for i in [0, 1, 2, 3, 4, 75, 76, 77, 78, 79]:
        file_name = "losses/" + model + "/" + model + "_" + str(i) + ".txt"
        train_loss, train_acc, test_loss, test_acc = read_file(file_name)
        test_accs.append(test_acc)
    model_test_accs[model] = test_accs

k_vals = list(np.linspace(0.05, 2.5, num=10, endpoint=False))
t_vals = list(np.linspace(0.05, 2.5, num=10, endpoint=False))
num_data_vals = np.arange(start=5, stop=20, step=2)
hyperparameter_set = []
for num_data in num_data_vals:
    for k in k_vals:
        for t in t_vals:
            hyperparameter_set.append((num_data, k, t))
print("Total Search Length: ", len(hyperparameter_set), flush=True)
accuracy_difference_threshold = 0.0025
# print("Model,\tParameters,\tGamma,\tCount,\tNumData,\tLocalMaxima,\tSlackProp,\tAvgStdEpoch,\tAvgASWTEpoch,\tAvgStdAcc,\tAvgASWTAcc")
search_hyperparameters(hyperparameter_set, accuracy_difference_threshold, model_test_accs, hp_filename)
# for mname in model_names:
#     analyze_hp_grid_data(mname)
#analyze_sample_size_dist()
# analyze_grid_data(acc_threshold=accuracy_difference_threshold)
# analyze_grid_data_on_maxed(acc_threshold=accuracy_difference_threshold)

#analysis.fft_analysis("resnet101", 0)
# print("")
# analyze_best_possibles(model_test_accs, acc_threshold=accuracy_difference_threshold, output_average=True)
# analyze_best_possibles(model_test_accs, acc_threshold=accuracy_difference_threshold, output_average=True, use_best=True)
