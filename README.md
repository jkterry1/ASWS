# This codebase is being actively maintained, please create and issue if you have issues using it

## Basics

All data files are included under losses and each folder. The main Augmented Shapiro-Wilk Stopping criterion is implemented in analysis.py, along with several helper functions and wrappers. The other comparison heuristics are also included in analysis.py, along with their wrappers. grapher.py contains all the code for generating the graphs used in the paper, and earlystopping_calculator.py includes code for generating tables and calculating some statistics from the data. hyperparameter_search.py contains all the code used to execute the grid-search on the ASWS method, along with the grid-search for the other heuristics.

## Installing

If you would like to try our code, just run `pip3 install git+https://github.com/justinkterry/ASWS`


### Example

If you wanted to try to determine the ASWS stopping point of a model, you can do so using the analysis.py file. If at anypoint during model training you wanted to perform
the stop criterion test, you can do

```
from ASWS.analysis import aswt_stopping

test_acc = [] # for storing model accuracies
for i in training_epochs:

    model.train()
    test_accuracy = model.evaluate(test_set)
    test_acc.append(test_accuracy)
    gamma = 0.5 # fill hyperparameters as desired
    num_data = 20
    slack_prop=0.1
    count = 20

    if len(test_acc) > count:
        aswt_stop_criterion = aswt_stopping(test_acc, gamma, count, num_data, slack_prop=slack_prop)

        if aswt_stop_criterion:
            print("Stop Training")

```

and if you already have finished training the model and wanted to determine the ASWS stopping point, you would need a CSV with columns Epoch, Training Loss, Training Acc, Test Loss, Test Acc. You could then use the following example

```
from ASWS.analysis import get_aswt_stopping_point_of_model, read_file

_, _, _, test_acc = read_file("modelaccuracy.csv")
gamma = 0.5 # fill hyperparameters as desired
num_data = 20
slack_prop=0.1
count = 20

stop_epoch, stop_accuracy = get_aswt_stopping_point_of_model(test_acc, gamma=gamma, num_data=num_data, count=count, slack_prop=slack_prop)

```

## pytorch-training

The pytorch-training folder contains the driver file for training each model, along with the model files which contain each network definition. The main.py file can be run out of the box for the models listed in the paper. The model to train is specified via the `--model` argument. All learning rate schedulers listed in the paper are available (via `--schedule step` etc.) and the ASWS learning rate scheduler is available via `--schedule ASWT` . The corresponding ASWS hyperparameters are passed in at the command line (for example `--gamma 0.5`). 

### Example

In order to recreate the GoogLeNet ASWT 1 scheduler from the paper, you can use the following command

`python3 main.py --model GoogLeNet --schedule ASWT --gamma 0.76 --num_data 19 --slack_prop 0.05 --lr 0.1`
