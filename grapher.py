SAVE_TO_PGF = True

import random
import numpy as np
from scipy.stats import pearsonr
import matplotlib
if SAVE_TO_PGF:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "xelatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'font.size': 11,
    })
import matplotlib.pyplot as plt
import analysis
from analysis import fft_analysis, denoise_fft, early_stopping_of_dataset
from hyperparameter_search import analyze_hp_grid_data
import os
from operator import itemgetter

model_names = ["alexnet", "fc1", "fc2", "GoogLeNet", "resnet34", "resnet50", "resnet101", "vgg11", "vgg16", "vgg19"]

color_palette = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#000000", "#880496"]

# models = ["resnet34", "resnet50"]
# fig, ax = plt.subplots(ncols=len(models), nrows=5, sharex=True)
# for i in range(5):
#     for j in range(len(models)):
#         fft_freq, fft_amp, fft_maxima = fft_analysis(models[j], i, stopping_point=100, local_maxima_range=5)
#         denoise = denoise_fft(models[j], i, stopping_point=100, local_maxima_range=5)
#         for maxima in fft_maxima:
#             ax[i][j].axvline(x=fft_freq[maxima], color="red", linestyle="--")
#         ax[i][j].plot(fft_freq, fft_amp, "b")
#         sub_denoise = denoise[len(denoise)//2:]
#         sub_denoise = np.abs(sub_denoise)**2
#         ax[i][j].plot(fft_freq, sub_denoise, "g")
#         ax[i][j].set_xscale('log')
# plt.tight_layout()
# plt.show()

def analyze_fft_smoothing_methods():
    models = ["resnet34", "resnet50"]
    fig, ax = plt.subplots(ncols=len(models), nrows=5, sharex=True)
    for i in range(5):
        for j in range(len(models)):
            filename = "losses/" + models[j] + "/" + models[j] + "_" + str(i) + ".txt"
            train_loss, train_acc, test_loss, test_acc = analysis.read_file(filename)
            denoise = denoise_fft(models[j], i , stopping_point=100, local_maxima_range=5)
            ind = list(range(100))
            raw_curve = np.gradient(test_acc)
            raw_curve = raw_curve[:100]
            ax[i][j].plot(ind, raw_curve, "b")
            ax[i][j].plot(ind, denoise, "r")
    plt.tight_layout()
    plt.show()

def calculate_hyperparam_epoch_distribution_for_model(model, hyperparam_name):
    file_name = "hp_grid_" + str(model)
    # file schema is model,gamma,count,num_data, local_max, slack_prop, avg_std_epoch_diff, avg_std_acc_diff, avg_max_epoch_diff, avg_max_acc_diff
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
        grid_data.append(parsed)
    fh.close()
    grid_groups = {model: [] for model in model_names}
    for dat in grid_data:
        grid_groups[dat[0]].append(dat[1:])
    hyperparam_index_map = {}
    hyperparam_index_map["gamma"] = 0
    hyperparam_index_map["count"] = 1
    hyperparam_index_map["num_data"] = 2
    hyperparam_index_map["local_max"] = 3
    hyperparam_index_map["slack_prop"] = 4
    selected_hp_index = hyperparam_index_map[hyperparam_name]
    hp_population = {}
    for data_list in grid_groups[model]:
        hp_key = data_list[selected_hp_index]
        if hp_key not in hp_population:
            hp_population[hp_key]  = []
        epoch_diff = data_list[7]
        hp_population[hp_key].append(epoch_diff)
    hp_vals = []
    hp_means = []
    hp_std = []
    for hp_key in sorted(hp_population):
        hp_vals.append(hp_key)
        hp_means.append(np.mean(hp_population[hp_key]))
        hp_std.append(np.std(hp_population[hp_key]))
    return hp_vals, hp_means, hp_std

def calculate_hyperparam_epoch_distribution(hyperparam_name):
    hp_population = {}
    for model in model_names:
        file_name = "hp_grid_" + str(model)
        # file schema is model,gamma,count,num_data, local_max, slack_prop, avg_std_epoch_diff, avg_std_acc_diff, avg_max_epoch_diff, avg_max_acc_diff
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
            grid_data.append(parsed)
        fh.close()
        grid_groups = {model: [] for model in model_names}
        for dat in grid_data:
            grid_groups[dat[0]].append(dat[1:])
        hyperparam_index_map = {}
        hyperparam_index_map["gamma"] = 0
        hyperparam_index_map["count"] = 1
        hyperparam_index_map["num_data"] = 2
        hyperparam_index_map["local_max"] = 3
        hyperparam_index_map["slack_prop"] = 4
        selected_hp_index = hyperparam_index_map[hyperparam_name]
        for data_list in grid_groups[model]:
            hp_key = data_list[selected_hp_index]
            if hp_key not in hp_population:
                hp_population[hp_key]  = []
            epoch_diff = data_list[7]
            hp_population[hp_key].append(epoch_diff)
    hp_vals = []
    hp_means = []
    hp_std = []
    for hp_key in sorted(hp_population):
        hp_vals.append(hp_key)
        hp_means.append(np.mean(hp_population[hp_key]))
        hp_std.append(np.std(hp_population[hp_key]))
    return hp_vals, hp_means, hp_std

def calculate_hyperparam_acc_distribution_for_model(model, hyperparam_name):
    file_name = "hp_grid_" + str(model)
    # file schema is model,gamma,count,num_data, local_max, slack_prop, avg_std_epoch_diff, avg_std_acc_diff, avg_max_epoch_diff, avg_max_acc_diff
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
        grid_data.append(parsed)
    fh.close()
    grid_groups = {model: [] for model in model_names}
    for dat in grid_data:
        grid_groups[dat[0]].append(dat[1:])
    hyperparam_index_map = {}
    hyperparam_index_map["gamma"] = 0
    hyperparam_index_map["count"] = 1
    hyperparam_index_map["num_data"] = 2
    hyperparam_index_map["local_max"] = 3
    hyperparam_index_map["slack_prop"] = 4
    selected_hp_index = hyperparam_index_map[hyperparam_name]
    hp_population = {}
    for data_list in grid_groups[model]:
        hp_key = data_list[selected_hp_index]
        if hp_key not in hp_population:
            hp_population[hp_key]  = []
        epoch_diff = data_list[8]
        hp_population[hp_key].append(epoch_diff)
    hp_vals = []
    hp_means = []
    hp_std = []
    for hp_key in sorted(hp_population):
        hp_vals.append(hp_key)
        hp_means.append(np.mean(hp_population[hp_key]))
        hp_std.append(np.std(hp_population[hp_key]))
    return hp_vals, hp_means, hp_std

# For a given model and hyperparameter
# show K graphs, where for each graph we show the distribution of epoch diff
# where K is the number of possible values for the hyperparam
def view_hyperparam_epoch_distribution_by_model(model, hyperparam_name):
    hp_vals, epoch_means, epoch_std = calculate_hyperparam_epoch_distribution_for_model(model, hyperparam_name)
    plt.errorbar(hp_vals, epoch_means, epoch_std, linestyle="None", marker="^")
    plt.tight_layout()
    plt.show()

def view_hyperparam_acc_distribution_by_model(model, hyperparam_name):
    hp_vals, acc_means, acc_std = calculate_hyperparam_acc_distribution_for_model(model, hyperparam_name)
    plt.errorbar(hp_vals, acc_means, acc_std, linestyle="None", marker="^")
    plt.tight_layout()
    plt.show()

def view_optimized_epoch_diff_of_models(acc_threshold):
    model_diffs = []
    for model in model_names:
        output_dict = analyze_hp_grid_data(model=model, acc_threshold=acc_threshold)
        epoch_diff = output_dict["avg_std_epoch"] - output_dict["avg_new_epoch"]
        model_diffs.append(epoch_diff)
    plt.bar(model_names, model_diffs)
    plt.show()

def specific_hyperparam_set(gamma=0.9, count=10, num_data=19, slack_prop=0.9, local_max=0):
    print("Model, Average Standard Epochs, Average ASWT Epochs, Average Standard Acc, Average ASWT Acc")
    for model in model_names:
        avg_standard_epochs, avg_new_epochs, avg_standard_acc, avg_new_acc = early_stopping_of_dataset(gamma=gamma, model=model, num_data=num_data, count=count, local_maxima=local_max, slack_prop=slack_prop, dataset="")
        print(model, ",", str(avg_standard_epochs), ",", str(avg_new_epochs), ",", str(avg_standard_acc), ",", str(avg_new_acc))

def graph_time_series(xaxis, curves, labels, fname, title=""):
    line_styles = ['-', '--', '-.']
    for c in range(len(curves)):
        curve = curves[c]
        series_label = labels[c]
        color_index = c % (len(color_palette))
        plt.plot(xaxis, curve, color_palette[color_index], label=series_label, linewidth=1.25)
    plt.legend(loc="lower right")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.ylim(bottom=89, top=95)
    if SAVE_TO_PGF:
        filename = "graph_images/" + fname + ".pgf"
        plt.savefig(filename)
    else:
        plt.show()

def graph_resnet101_scheduled(min_num_epochs=0, max_num_epochs = 400):
    labels = ["Standard Trained Model 1", "Standard Trained Model 2", "Standard Trained Model 3", "Standard Trained Model 4", "Standard Trained Model 5", "ASWT Model 1", "ASWT Model 2"]
    xaxis = list(range(min_num_epochs, max_num_epochs))
    curves = []
    with open("graph_sources/resnet101_scheduled.csv", "r") as g_source:
        for i in range(7):
            curves.append([])
        r = 0
        for line_raw in g_source:
            if r != 0 and r < (max_num_epochs+1) and r > (min_num_epochs):
                line = line_raw.rstrip().split(",")
                for i in range(7):
                    curves[i].append(line[i+1])
            r += 1
        curves = np.array(curves).astype(float)
    graph_time_series(xaxis, curves, labels, "resnet101scheduled", title="ResNet101 Training on CIFAR10")

def graph_resnet101_scheduled2(min_num_epochs=0, max_num_epochs = 400):
    labels = ["ASWT Schedule 1", "ASWT Schedule 2", "StepLR Schedule", "ExponentialLR Schedule", "ReduceLR Schedule", "ADAM w/o Scheduler", "SGD w/o Scheduler"]
    xaxis = list(range(min_num_epochs, max_num_epochs))
    curves = []
    num_columns = 7
    with open("graph_sources/resnet101_scheduled2.csv", "r") as g_source:
        for i in range(num_columns):
            curves.append([])
        r = 0
        for line_raw in g_source:
            if r != 0 and r < (max_num_epochs+1) and r > (min_num_epochs):
                line = line_raw.rstrip().split(",")
                for i in range(num_columns):
                    curves[i].append(line[i+1])
            r += 1
        curves = np.array(curves).astype(float)
    graph_time_series(xaxis, curves, labels, "resnet101scheduled2", title="ResNet101 Scheduled Training on CIFAR10")

def graph_resnet101_scheduled3(min_num_epochs=0, max_num_epochs = 400):
    labels = ["ASWT Schedule 1", "ASWT Schedule 2", "StepLR Schedule", "ExponentialLR Schedule", "ReduceLR Schedule"]
    xaxis = list(range(min_num_epochs, max_num_epochs))
    curves = []
    num_columns = 5
    with open("graph_sources/resnet101_scheduled3.csv", "r") as g_source:
        for i in range(num_columns):
            curves.append([])
        r = 0
        for line_raw in g_source:
            if r != 0 and r < (max_num_epochs+1) and r > (min_num_epochs):
                line = line_raw.rstrip().split(",")
                for i in range(num_columns):
                    curves[i].append(line[i+1])
            r += 1
        curves = np.array(curves).astype(float)
    graph_time_series(xaxis, curves, labels, "resnet101scheduled3", title="ResNet101 Scheduled Training on CIFAR10")

def graph_resnet101_scheduled4(min_num_epochs=0, max_num_epochs = 400):
    labels = ["ASWS Schedule 1", "ASWS Schedule 2", "StepLR Schedule", "ReduceLR Schedule", "CyclicLR Schedule"]
    xaxis = list(range(min_num_epochs, max_num_epochs))
    curves = []
    num_columns = 5
    with open("graph_sources/resnet101_scheduled6.csv", "r") as g_source:
        for i in range(num_columns):
            curves.append([])
        r = 0
        for line_raw in g_source:
            if r != 0 and r < (max_num_epochs+1) and r > (min_num_epochs):
                line = line_raw.rstrip().split(",")
                for i in range(num_columns):
                    curves[i].append(line[i+1])
            r += 1
        curves = np.array(curves).astype(float)
    graph_time_series(xaxis, curves, labels, "resnet101scheduled6", title="ResNet101 Scheduled Training on CIFAR10")

def graph_GoogLeNet_scheduled(min_num_epochs=0, max_num_epochs = 400):
    labels = ["Standard Trained Model 1", "Standard Trained Model 2", "Standard Trained Model 3", "Standard Trained Model 4", "Standard Trained Model 5", "ASWT Model 1", "ASWT Model 2"]
    xaxis = list(range(min_num_epochs, max_num_epochs))
    curves = []
    with open("graph_sources/GoogLeNet_scheduled.csv", "r") as g_source:
        for i in range(7):
            curves.append([])
        r = 0
        for line_raw in g_source:
            if r != 0 and r < (max_num_epochs+1) and r > (min_num_epochs):
                line = line_raw.rstrip().split(",")
                for i in range(7):
                    curves[i].append(line[i+1])
            r += 1
        curves = np.array(curves).astype(float)
    graph_time_series(xaxis, curves, labels, "GoogLeNetscheduled", title="GoogLeNet Training on CIFAR10")

def graph_GoogLeNet_scheduled2(min_num_epochs=0, max_num_epochs = 400):
    labels = ["ASWT Schedule 1", "ASWT Schedule 2", "StepLR Schedule", "ExponentialLR Schedule", "ReduceLR Schedule", "ADAM w/o Scheduler", "SGD w/o Scheduler"]
    xaxis = list(range(min_num_epochs, max_num_epochs))
    curves = []
    num_columns = 7 # change as necessary
    with open("graph_sources/GoogLeNet_scheduled2.csv", "r") as g_source:
        for i in range(num_columns):
            curves.append([])
        r = 0
        for line_raw in g_source:
            if r != 0 and r < (max_num_epochs+1) and r > (min_num_epochs):
                line = line_raw.rstrip().split(",")
                for i in range(num_columns):
                    curves[i].append(line[i+1])
            r += 1
        curves = np.array(curves).astype(float)
    graph_time_series(xaxis, curves, labels, "GoogLeNetscheduled2", title="GoogLeNet Scheduled Training on CIFAR10")    

def graph_GoogLeNet_scheduled3(min_num_epochs=0, max_num_epochs = 400):
    labels = ["ASWT Schedule 1", "ASWT Schedule 2", "StepLR Schedule", "ExponentialLR Schedule", "ReduceLR Schedule"]
    xaxis = list(range(min_num_epochs, max_num_epochs))
    curves = []
    num_columns = 5 # change as necessary
    with open("graph_sources/GoogLeNet_scheduled3.csv", "r") as g_source:
        for i in range(num_columns):
            curves.append([])
        r = 0
        for line_raw in g_source:
            if r != 0 and r < (max_num_epochs+1) and r > (min_num_epochs):
                line = line_raw.rstrip().split(",")
                for i in range(num_columns):
                    curves[i].append(line[i+1])
            r += 1
        curves = np.array(curves).astype(float)
    graph_time_series(xaxis, curves, labels, "GoogLeNetscheduled3", title="GoogLeNet Scheduled Training on CIFAR10")    

def graph_GoogLeNet_scheduled4(min_num_epochs=0, max_num_epochs = 400):
    labels = ["ASWS Schedule 1", "ASWS Schedule 2", "StepLR Schedule", "ReduceLR Schedule", "CyclicLR Schedule"]
    xaxis = list(range(min_num_epochs, max_num_epochs))
    curves = []
    num_columns = 5 # change as necessary
    with open("graph_sources/GoogLeNet_scheduled6.csv", "r") as g_source:
        for i in range(num_columns):
            curves.append([])
        r = 0
        for line_raw in g_source:
            if r != 0 and r < (max_num_epochs+1) and r > (min_num_epochs):
                line = line_raw.rstrip().split(",")
                for i in range(num_columns):
                    curves[i].append(line[i+1])
            r += 1
        curves = np.array(curves).astype(float)
    graph_time_series(xaxis, curves, labels, "GoogLeNetscheduled6", title="GoogLeNet Scheduled Training on CIFAR10")   

def graph_mean_and_std(categories, means, stds, ymin=0, ymax=500, xaxis="", filename=""):
    plt.errorbar(categories, means, stds, linestyle="None", marker="^")
    plt.ylim(ymin, ymax)
    plt.xlabel(xaxis, fontsize=12)
    plt.ylabel("Mean Difference in Stopping Epoch", fontsize=12)
    fname = "graph_images/"+filename+".pgf"
    plt.savefig(fname)

def graph_hyperparamdist_file(filename, ymin=0, ymax=500, hpname="", gname=""):
    parsed = [[],[],[]]
    with open(filename, "r") as hp_dist:
        r = 0
        for line_raw in hp_dist:
            line = line_raw.rstrip().split(",")
            parsed[r] = np.array(line[1:]).astype(float)
            r += 1
    graph_mean_and_std(categories=parsed[0], means=parsed[1], stds=parsed[2], ymin=ymin, ymax=ymax, xaxis=hpname, filename=gname)

def graph_gamma_dist():
    graph_hyperparamdist_file("graph_sources/gamma_dist.txt", ymin=150, ymax=350, hpname="Gamma", gname="gammadist")

def graph_samplesize_dist():
    graph_hyperparamdist_file("graph_sources/samplesize_dist.txt", ymin=150, ymax=350, hpname="Sample Size", gname="samplesizedist")

def graph_slackprop_dist():
    graph_hyperparamdist_file("graph_sources/slackprop_dist.txt", ymin=150, ymax=350, hpname="Slack Proportion", gname="slackpropdist")

def graph_stacked_bar(categories, series_list, series_label, outputfile="", ylabel="Epochs", loc="lower right"):
    x = np.arange(len(categories))
    x = x*4
    width = 1
    fig, ax = plt.subplots()
    count = 0
    for series in series_list:
        x_off = count * width
        ax.bar(x+x_off, series, width, label=series_label[count])
        count += 1
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + 1*width)
    ax.set_xticklabels(categories)
    plt.xticks(rotation=25)
    plt.legend(loc=loc)
    if SAVE_TO_PGF:
        plt.savefig(outputfile)
    else:
        plt.show()

def graph_ASWTModelComp():
    filename = "graph_sources/ASWTModel_comp.txt"
    categories = []
    aswt_stop = []
    standard_stop = []
    with open(filename, "r") as fh:
        r = 0
        for line_raw in fh:
            line = line_raw.split(",")
            if r != 0:
                categories.append(line[0])
                aswt_stop.append(line[8])
                standard_stop.append(line[7])
            r += 1
    aswt_stop = np.array(aswt_stop).astype(float)
    standard_stop = np.array(standard_stop).astype(float)
    graph_stacked_bar(categories, standard_stop, aswt_stop)

def graph_ASWTModelComp2():
    filename = "graph_sources/ASWTModel_comp4.txt"
    categories = []
    aswt_stop = []
    standard_stop = []
    patient_stop = []
    mind_stop = []
    aveges_stop = []
    with open(filename, "r") as fh:
        r = 0
        for line_raw in fh:
            line = line_raw.split(",")
            if r != 0:
                categories.append(line[0])
                standard_stop.append(line[6])
                aswt_stop.append(line[7])
                patient_stop.append(line[8])
                mind_stop.append(line[9])
                aveges_stop.append(line[10])
            r += 1
    aswt_stop = np.array(aswt_stop).astype(float)
    standard_stop = np.array(standard_stop).astype(float)
    patient_stop = np.array(patient_stop).astype(float)
    mind_stop = np.array(mind_stop).astype(float)
    aveges_stop = np.array(aveges_stop).astype(float)
    full_series = [standard_stop, aswt_stop, patient_stop, mind_stop, aveges_stop]
    series_labels = ["Performance Stopping", "ASWS Stopping", "Patience Stopping", "Minimum Diff Stopping", "Average Diff Stopping"]
    graph_stacked_bar(categories, full_series, series_labels, "graph_images/ASWTStandardComp4.pgf", loc="upper right")

def graph_ASWTModelCompByAcc2():
    filename = "graph_sources/ASWTModel_comp4.txt"
    categories = []
    aswt_stop = []
    standard_stop = []
    patient_stop = []
    mind_stop = []
    aveges_stop = []
    with open(filename, "r") as fh:
        r = 0
        for line_raw in fh:
            line = line_raw.split(",")
            if r != 0:
                categories.append(line[0])
                standard_stop.append(line[11])
                aswt_stop.append(line[12])
                patient_stop.append(line[13])
                mind_stop.append(line[14])
                aveges_stop.append(line[15])
            r += 1
    aswt_stop = np.array(aswt_stop).astype(float)
    standard_stop = np.array(standard_stop).astype(float)
    patient_stop = np.array(patient_stop).astype(float)
    mind_stop = np.array(mind_stop).astype(float)
    aveges_stop = np.array(aveges_stop).astype(float)
    full_series = [standard_stop, aswt_stop, patient_stop, mind_stop, aveges_stop]
    series_labels = ["Performance Stopping", "ASWS Stopping", "Patience Stopping", "Minimum Diff Stopping", "Average Diff Stopping"]
    graph_stacked_bar(categories, full_series, series_labels, outputfile="graph_images/ASWTStandardCompByAcc4.pgf", ylabel="Test Accuracy")

def graph_ASWTModelCompAugmented():
    filename = "graph_sources/ASWTModel_Augmentedcomp.txt"
    categories = []
    aswt_stop = []
    not_stop = []
    noshap_stop = []
    with open(filename, "r") as fh:
        r = 0
        for line_raw in fh:
            line = line_raw.split(",")
            if r != 0:
                categories.append(line[0])
                aswt_stop.append(line[4])
                not_stop.append(line[5])
                noshap_stop.append(line[6])
            r += 1
    aswt_stop = np.array(aswt_stop).astype(float)
    not_stop = np.array(not_stop).astype(float)
    noshap_stop = np.array(noshap_stop).astype(float)
    full_series = [aswt_stop, not_stop, noshap_stop]
    series_labels = ["ASWS Stopping", "ASWS No T-Test Stopping", "ASWS No Shapiro Stopping"]
    graph_stacked_bar(categories, full_series, series_labels, "graph_images/ASWTStandardCompAugment.pgf", loc="upper right")

def graph_ASWTModelCompAugmentedByAcc():
    filename = "graph_sources/ASWTModel_Augmentedcomp.txt"
    categories = []
    aswt_stop = []
    not_stop = []
    noshap_stop = []
    with open(filename, "r") as fh:
        r = 0
        for line_raw in fh:
            line = line_raw.split(",")
            if r != 0:
                categories.append(line[0])
                aswt_stop.append(line[7])
                not_stop.append(line[8])
                noshap_stop.append(line[9])
            r += 1
    aswt_stop = np.array(aswt_stop).astype(float)
    not_stop = np.array(not_stop).astype(float)
    noshap_stop = np.array(noshap_stop).astype(float)
    full_series = [aswt_stop, not_stop, noshap_stop]
    series_labels = ["ASWS Stopping", "ASWS No T-Test Stopping", "ASWS No Shapiro Stopping"]
    graph_stacked_bar(categories, full_series, series_labels, "graph_images/ASWTStandardCompAugmentAcc.pgf", loc="lower right", ylabel="Test Accuracy")

def graph_combined_hp_dists():
    filenames = ["graph_sources/slackprop_dist.txt", "graph_sources/samplesize_dist.txt", "graph_sources/gamma_dist.txt"]
    plot_names = [r"Slack Proportion (slackProp)", r"Sample Size (n)", r"Smoothing Factor ($\gamma$)"]
    colors = ["r", "b", "g"]
    fig, ax = plt.subplots(ncols=3, nrows=1, sharey=True)
    for i in range(3):
        filename = filenames[i]
        parsed = [[],[],[]]
        with open(filename, "r") as hp_dist:
            r = 0
            for line_raw in hp_dist:
                line = line_raw.rstrip().split(",")
                parsed[r] = np.array(line[1:]).astype(float)
                r += 1
        ax[i].errorbar(parsed[0], parsed[1], parsed[2], color=colors[i], linestyle="None", marker="^")
        ax[i].set_xlabel(plot_names[i])
        if i == 0:
            ax[i].set_ylabel("Average Stopping Epoch Difference")
    plt.savefig("graph_images/CombinedEpochDists2.pgf")

def graph_diverging_accuracies(modelname, index=0):
    filename = "losses/"+modelname+"/"+modelname+"_"+str(index)+".txt"
    with open(filename, "r") as fh:
        # index 2 is train acc/100, index 4 is test acc
        train_accs = []
        test_accs = []
        for line_raw in fh:
            line = line_raw.split(",")
            train_accs.append(float(line[2])*100)
            test_accs.append(float(line[4]))
        plt.plot(train_accs, label="Train Accuracy")
        plt.plot(test_accs, label="Test Accuracy")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        filename= "graph_images/" + modelname + "_" + str(index) + "divergingacc.pgf"
        plt.show()

def graph_samplesize_vs_parameters():
    filename = "graph_sources/ASWTModel_comp.txt"
    parameters = []
    sample_size = []
    with open(filename, "r") as fh:
        r = 0
        for line_raw in fh:
            line = line_raw.split(",")
            if r != 0:
                parameters.append(line[1])
                sample_size.append(line[4])
            r += 1
    parameters = np.array(parameters).astype(float)
    sample_size = np.array(sample_size).astype(float)
    sorted_indices = np.argsort(parameters)
    param_mean = np.mean(parameters)
    param_std = np.std(parameters)
    ss_mean = np.mean(sample_size)
    ss_std = np.std(sample_size)
    param_standard = [(par-param_mean)/param_std for par in parameters]
    ss_standard = [(ss-ss_mean)/ss_std for ss in sample_size]
    r_product = [par*ss for par,ss in zip(param_standard, ss_standard)]
    r_val = sum(r_product)/float(len(parameters)-1)
    print(r_val)
    plt.plot(parameters[sorted_indices], sample_size[sorted_indices])
    text_x_index = parameters[sorted_indices][int(len(parameters)-2)]
    text_y_index = sample_size[sorted_indices][1]
    true_r_val, _ = pearsonr(parameters, sample_size)
    print(true_r_val)
    plt.text(text_x_index, text_y_index, 'R = %0.2f' % true_r_val)
    plt.ticklabel_format(style="plain")
    plt.ylabel("Optimal Sample Size")
    plt.xlabel("Model Parameters")
    filename = "graph_images/optimal_n_vs_parameters.pgf"
    plt.savefig(filename)
        
def graph_samplesize_vs_ASWT_stop():
    filename = "graph_sources/ASWTModel_comp.txt"
    parameters = []
    sample_size = []
    with open(filename, "r") as fh:
        r = 0
        for line_raw in fh:
            line = line_raw.split(",")
            if r != 0:
                parameters.append(line[8])
                sample_size.append(line[4])
            r += 1
    parameters = np.array(parameters).astype(float)
    sample_size = np.array(sample_size).astype(float)
    sorted_indices = np.argsort(parameters)
    param_mean = np.mean(parameters)
    param_std = np.std(parameters)
    ss_mean = np.mean(sample_size)
    ss_std = np.std(sample_size)
    param_standard = [(par-param_mean)/param_std for par in parameters]
    ss_standard = [(ss-ss_mean)/ss_std for ss in sample_size]
    r_product = [par*ss for par,ss in zip(param_standard, ss_standard)]
    r_val = sum(r_product)/float(len(parameters)-1)
    print(r_val)
    plt.plot(parameters[sorted_indices], sample_size[sorted_indices])
    text_x_index = parameters[sorted_indices][int(len(parameters)-2)]
    text_y_index = sample_size[sorted_indices][1]
    true_r_val, _ = pearsonr(parameters, sample_size)
    print(true_r_val)
    plt.text(text_x_index, text_y_index, 'R = %0.2f' % true_r_val)
    plt.ticklabel_format(style="plain")
    plt.ylabel("Optimal Sample Size")
    plt.xlabel("ASWT Stopping Epoch")
    filename = "graph_images/optimal_n_vs_aswt_stop.pgf"
    plt.savefig(filename)
    #plt.show()

# goes through all training data files for specified folder, and reports highest achieved test accuracy
def highest_test_accuracy_on_model(model_name):
    root_loss = "losses/"+model_name+"/"
    lr_names = ["ASWTLR1", "ASWTLR2", "CyclicLR", "ReduceLR", "StepLR"]
    for lr_name in lr_names:
        loss_folder = root_loss + lr_name + "/"
        subfiles = [fi for fi in os.listdir(loss_folder) if os.path.isfile(os.path.join(loss_folder, fi))]
        test_accuracy_map = {}
        for subfile in subfiles:
            # find highest test accuracy (last column)
            highest_acc = 0.0
            with open(loss_folder + subfile, "r") as curr_file:
                for line in curr_file:
                    if float(line.split(",")[-1]) > highest_acc:
                        highest_acc = float(line.split(",")[-1])
                test_accuracy_map[subfile] = highest_acc
        sorted_result = sorted(test_accuracy_map.items(), key=itemgetter(1))
        key, value = sorted_result[-1]
        split_file_name = key.split("_")
        index_name_split = split_file_name[1].split(".")
        index_name = index_name_split[0]
        print(lr_name, "--", index_name, ",", value)

# goes through all training data files for specified folder, and reports average achieved test accuracy
def average_test_accuracy_on_model(model_name):
    root_loss = "losses/"+model_name+"/"
    lr_names = ["ASWTLR1", "ASWTLR2", "CyclicLR", "ReduceLR", "StepLR"]
    for lr_name in lr_names:
        loss_folder = root_loss + lr_name + "/"
        subfiles = [fi for fi in os.listdir(loss_folder) if os.path.isfile(os.path.join(loss_folder, fi))]
        test_accuracy_map = {}
        for subfile in subfiles:
            # find highest test accuracy (last column)
            highest_acc = 0.0
            with open(loss_folder + subfile, "r") as curr_file:
                for line in curr_file:
                    if float(line.split(",")[-1]) > highest_acc:
                        highest_acc = float(line.split(",")[-1])
                test_accuracy_map[subfile] = highest_acc
        average_acc = 0.0
        s_size = 0
        for key in test_accuracy_map:
            #print("\t", lr_name, "--", key, ",", test_accuracy_map[key])
            average_acc += test_accuracy_map[key]
            s_size += 1
        average_acc = average_acc / s_size
        print(lr_name, "--", average_acc)

# given a list of labels, and list of series
# will generate a csv with label at top of column, and respective series as rest of column
# series must be the same length
def generate_list_csv_file(series_labels, series_list, outputname):
    with open(outputname, "w") as outputcsv:
        header_str = ""
        for lab in series_labels:
            header_str += str(lab) + ","
        header_str = header_str[:-1] + "\n"
        outputcsv.write(header_str)
        for i in range(len(series_list[0])):
            line_str = ""
            for ser in series_list:
                line_str += str(ser[i]) + ","
            line_str = line_str[:-1] + "\n"
            outputcsv.write(line_str)

###
# Runs 14-StepLR, 15-ReduceLR, 16-Cyclic
###
# series_labels = ["Epoch", "ASWS Schedule 1", "ASWS Schedule 2", "StepLR Schedule","CyclicLR Schedule", "ReduceLR Schedule"]
# epoch = list(range(1, 401))
# model_name = "GoogLeNet"
# _, _, _, aswt_1 = analysis.read_file("losses/" + model_name + "/ASWTLR1/" + model_name + "_21.txt")
# _, _, _, aswt_2 = analysis.read_file("losses/" + model_name + "/ASWTLR2/" + model_name + "_31.txt")
# _, _, _, steplr = analysis.read_file("losses/" + model_name + "/StepLR/" + model_name + "_42.txt")
# _, _, _, cycliclr = analysis.read_file("losses/" + model_name + "/CyclicLR/" + model_name + "_60.txt")
# _, _, _, reducelr = analysis.read_file("losses/" + model_name + "/ReduceLR/" + model_name + "_43.txt")

# generate_list_csv_file(series_labels, [epoch, 100*aswt_1, 100*aswt_2, 100*steplr, 100*cycliclr, 100*reducelr], outputname="graph_sources/" + model_name + "_scheduled6.csv")

#graph_GoogLeNet_scheduled4()

#highest_test_accuracy_on_model("GoogLeNet")

#graph_ASWTModelComp2()
graph_ASWTModelCompByAcc2()
#graph_combined_hp_dists()
#graph_ASWTModelCompAugmentedByAcc()

