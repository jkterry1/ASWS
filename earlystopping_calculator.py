import analysis
import os

optimal_gamma = {"alexnet":0, "fc1":0.1, "fc2":0.775, "GoogLeNet":0.2125, "resnet34":0.55, "resnet50":0.2125, "resnet101":0.1, "vgg11":0.6625, "vgg16":0.888, "vgg19":0.1}
optimal_n = {"alexnet":13, "fc1":5, "fc2":13, "GoogLeNet":19, "resnet34":17, "resnet50":19, "resnet101":15, "vgg11":19, "vgg16":17, "vgg19":17}
optimal_slackprop = {"alexnet":0.95, "fc1":0.1318, "fc2":0.05, "GoogLeNet":0.95, "resnet34":0.8682, "resnet50":0.459, "resnet101":0.95, "vgg11":0.95, "vgg16":0.705, "vgg19":0.623}
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
model_names = ["alexnet", "fc1", "fc2", "GoogLeNet", "resnet34", "resnet50", "resnet101", "vgg11", "vgg16", "vgg19"]


def print_csv(argos):
    csv_line = ""
    for argu in argos:
        csv_line += str(argu) + ","
    csv_line = csv_line[:-1]
    print(csv_line)

# for each OG training file for modelname
#   find standard stopping point and acc
# return average of these values
def standard_average_info(model):
    stop_epoch_sum = 0.0
    stop_acc_sum = 0.0
    for suffix in [0, 1, 2, 3, 4, 75, 76, 77, 78, 79]:
        stop_epoch, stop_acc = analysis.get_standard_stopping_point(model=model, file_suffix=suffix)
        stop_epoch_sum += stop_epoch
        stop_acc_sum += stop_acc
    stop_epoch_avg = float(stop_epoch_sum / 10.0)
    stop_acc_sum = float(stop_acc_sum / 10.0) 
    return stop_epoch_avg, stop_acc_sum

# for each OG training file for modelname
#   find aswt stopping point and acc
# return average of these values
def aswt_average_info(model):
    gamma = optimal_gamma[model]
    num_data = optimal_n[model]
    slack_prop = optimal_slackprop[model]
    stop_epoch_sum = 0.0
    stop_acc_sum = 0.0
    for suffix in [0, 1, 2, 3, 4, 75, 76, 77, 78, 79]:
        stop_epoch, stop_acc = analysis.get_aswt_stopping_point(model=model, file_suffix=suffix, gamma=gamma, count=20, num_data=num_data, slack_prop=slack_prop)
        stop_epoch_sum += stop_epoch
        stop_acc_sum += stop_acc
    stop_epoch_avg = float(stop_epoch_sum / 10.0)
    stop_acc_sum = float(stop_acc_sum / 10.0) 
    return stop_epoch_avg, stop_acc_sum

# for each training file for modelname
#   find patience-es stopping point and acc
# return average of these values
# hyperparams: patience=3
def patience_average_info(model):
    stop_epoch_sum = 0.0
    stop_acc_sum = 0.0
    for suffix in [0, 1, 2, 3, 4, 75, 76, 77, 78, 79]:
        stop_epoch, stop_acc = analysis.get_patience_stopping_point(model=model, file_suffix=suffix, patience=3)
        stop_epoch_sum += stop_epoch
        stop_acc_sum += stop_acc
    stop_epoch_avg = float(stop_epoch_sum / 10.0)
    stop_acc_sum = float(stop_acc_sum / 10.0) 
    return stop_epoch_avg, stop_acc_sum

# for each training file for modelname
#   find mindelta-es stopping point and acc
# return average of these values
# hyperparams: patience=27, mindelta=0.013
def mindelta_average_info(model):
    stop_epoch_sum = 0.0
    stop_acc_sum = 0.0
    for suffix in [0, 1, 2, 3, 4, 75, 76, 77, 78, 79]:
        stop_epoch, stop_acc = analysis.get_mindelta_stopping_point(model=model, file_suffix=suffix, min_delta=0.013)
        stop_epoch_sum += stop_epoch
        stop_acc_sum += stop_acc
    stop_epoch_avg = float(stop_epoch_sum / 10.0)
    stop_acc_sum = float(stop_acc_sum / 10.0) 
    return stop_epoch_avg, stop_acc_sum

# for each training file for modelname
#   find average-es stopping point and acc
# return average of these values
# hyperparams: window=150, mindeltaaverage=0.001
def averagees_average_info(model):
    stop_epoch_sum = 0.0
    stop_acc_sum = 0.0
    for suffix in [0, 1, 2, 3, 4, 75, 76, 77, 78, 79]:
        stop_epoch, stop_acc = analysis.get_averagees_stopping_point(model=model, file_suffix=suffix)
        stop_epoch_sum += stop_epoch
        stop_acc_sum += stop_acc
    stop_epoch_avg = float(stop_epoch_sum / 10.0)
    stop_acc_sum = float(stop_acc_sum / 10.0) 
    return stop_epoch_avg, stop_acc_sum

# for a specific model
# get the average aswt stopping point, aswt w/o shapiro, aswt w/o t-test stopping points for model baselines
def compare_augmented_aswt(model):
    baseline_indices = [0, 1, 2, 3, 4, 75, 76, 77, 78, 79]
    sum_aswt_epoch = 0.0
    sum_aswt_acc = 0.0
    sum_aswt_no_t_epoch = 0.0
    sum_aswt_no_t_acc = 0.0
    sum_aswt_no_shap_epoch = 0.0
    sum_aswt_no_shap_acc = 0.0
    for base in baseline_indices:
        f_name = "losses/" + model + "/" + model + "_" + str(base) + ".txt"
        _, _, _, acc_curve = analysis.read_file(f_name)
        aswt_epoch, aswt_acc = analysis.get_aswt_stopping_point_of_model(acc_curve, gamma=optimal_gamma[model], num_data=optimal_n[model], slack_prop=optimal_slackprop[model])
        aswt_no_t_epoch, aswt_no_t_acc = analysis.get_augmented_aswt_stopping_point_of_model(acc_curve, gamma=optimal_gamma[model], num_data=optimal_n[model], slack_prop=optimal_slackprop[model], use_shapiro=True)
        aswt_no_shap_epoch, aswt_no_shap_acc = analysis.get_augmented_aswt_stopping_point_of_model(acc_curve, gamma=optimal_gamma[model], num_data=optimal_n[model], slack_prop=optimal_slackprop[model], use_shapiro=False)
        sum_aswt_epoch += aswt_epoch
        sum_aswt_acc += aswt_acc
        sum_aswt_no_t_epoch += aswt_no_t_epoch
        sum_aswt_no_t_acc += aswt_no_t_acc
        sum_aswt_no_shap_epoch += aswt_no_shap_epoch
        sum_aswt_no_shap_acc += aswt_no_shap_acc

    avg_aswt_epoch = sum_aswt_epoch / len(baseline_indices)
    avg_aswt_acc = sum_aswt_acc / len(baseline_indices)
    avg_aswt_no_t_epoch = sum_aswt_no_t_epoch / len(baseline_indices)
    avg_aswt_no_t_acc = sum_aswt_no_t_acc / len(baseline_indices)
    avg_aswt_no_shap_epoch = sum_aswt_no_shap_epoch / len(baseline_indices)
    avg_aswt_no_shap_acc = sum_aswt_no_shap_acc / len(baseline_indices)
    final_output = model + "," + str(optimal_gamma[model]) + "," + str(optimal_n[model]) + "," + str(optimal_slackprop[model]) + ","
    final_output += str(avg_aswt_epoch) + "," + str(avg_aswt_no_t_epoch) + "," + str(avg_aswt_no_shap_epoch) + ","
    final_output += str(avg_aswt_acc) + "," + str(avg_aswt_no_t_acc) + "," + str(avg_aswt_no_shap_acc)
    print(final_output)

# print("Model,Parameters,Gamma,Count,NumData,SlackProp,AvgStdEpoch,AvgASWTEpoch,AvgPatEpoch,AvgMindEpoch,AvgAvgesEpoch,AvgStdAcc,AvgASWTAcc,AvgPatAcc,AvgMindAcc,AvgAvgesAcc")
# for modelname in model_names:
#     std_epoch, std_acc = standard_average_info(model=modelname)
#     aswt_epoch, aswt_acc = aswt_average_info(model=modelname)
#     pat_epoch, pat_acc = patience_average_info(model=modelname)
#     mind_epoch, mind_acc = mindelta_average_info(model=modelname)
#     avges_epoch, avges_acc = averagees_average_info(model=modelname)
#     print_csv([modelname, model_parameter_map[modelname], optimal_gamma[modelname], 20, optimal_n[modelname], optimal_slackprop[modelname], std_epoch, aswt_epoch, pat_epoch, mind_epoch, avges_epoch, std_acc, aswt_acc, pat_acc, mind_acc, avges_acc])

# print("Model,Gamma,NumData,SlackProp,AvgASWTEpoch,AvgNoTEpoch,AvgNoShapEpoch,AvgASWTAcc,AvgNoTAcc,AvgNoShapAcc")
# for modelname in model_names:
#     compare_augmented_aswt(modelname)