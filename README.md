# Introduction

### PSAVE

<b>PSAVE (Prioritized ShApley ValuE on feasible coalitions)</b> is a game-theoretic method for interpreting feature importance of DNNs. Traditional Shapley value suffers problems rising from unsatisfactory nature of payoff function when applying to DNNs. PSAVE tries to solve the problem by applying <b>feasible coalitions</b> and <b>feature priority</b>.

### Datasets

We use MNIST dataset and Boston House Price (BHP) dataset in our experiments.

You can download MNIST dataset from http://yann.lecun.com/exdb/mnist/, or use corresponding package provided by pytorch. The BHP dataset can be viewed in our experiment code.

### Code

We further referenced the experiment code of <b>SAGE (Shapley Additive Global importancE)</b>. And you can find SAGE's experiment code pieces in our code. SAGE's code can be downloaded from http://github.com/iancovert/sage/. In order to run our experiment, you're supposed to install the <b>sage-importance</b> package with <b>pip</b> first.

```
pip install sage-importance
```

### Usage

In fact, PSAVE is a model-agnostic method, so theoretically you can apply it to any kind of models. However, we only test its performance on DNNs. The generally way of using PSAVE is as follows.

```python
import psave
import find_fc
import w_with_fc
from d_calculator import DCalculatorPreMem
from v_calculator import VCalculator

# Load data
x, y = ...
feature_num = ...

# Load model
model = ...

# Preparations
met_path = ...
batch_size = ...

calc_v = VCalculator(x, y, model, 'mse', batch_size)
F = find_fc.find_fc(feature_num, calc_v)
calc_div = DCalculatorPreMem(x, y, model, F, 'mse', batch_size)
w = w_with_fc.get_w_with_fc(feature_num, F)

# Evalutation
res = psave.psave_whole_pre_mem(feature_num, w, F, calc_div, tp=True)
np.savetxt(met_path, res)
```

For loss function, only <b>'mse'</b> (mean square importance) and <b>'cross entropy'</b> are supported.

Unfortunately, we haven't carefully organize our code, so the robustness and expandability of our code may be very bad. In order to run our experiment code, you may need to modify our code first. For now, our code is better for reference use.

# File Structure

### core

We present two samples for using our PSAVE method on MNIST and BHP.

On MNIST, we use a heuristic method for constructing feasible coalitions on datasets with image features. And we apply 2-D Gaussian Density Function as the priority.

On BHP, we use a more general greedy algorithm to construct feasible coalitions. And we use the number of feasible coalitions one feature participates in as the priority.

##### details

We present the details of code on BHP as an example, which is more general.

- <b>boston.py</b>: The evaluation sample code on BHP.
- <b>v_calculator.py</b>: Provide callable class <b>VCalculator</b> for computing payoff function defined by SAGE.
- <b>d_calculator.py</b>: Provide callable class <b>DCalculatorPreMem</b> for computing coalition dividends with memorized searching skills.
- <b>psave.py</b>: Provide function <b>psave_whole_pre_mem</b> for computing PSAVE of all features.
- <b>find_fc.py</b>: Provide function <b>find_fc_v</b> for constructing feasible coalitions in a greedy way.
- <b>w_with_fc.py</b>: Provide function <b>get_w_with_fc</b> for compute feature priority.
- <b>imputers.py</b>:  You can reference <b>SAGE's Imputers</b>. This is for masking particular features.
- <b>model_train.py</b>: Train DNN model.
- <b>net.py</b>: Definition of DNN model.
- <b>utils</b>: Provide functions for set operations.

### feasible_condition_BHP

We validate our greedy algorithm for constructing feasible coalitions here. The core code can be viewed in <b>supera.py</b>.

### feature_selection_BHP

We do feature selection and feature importance experiments here on BHP. Also we do experiments for validating PSAVE's convergence nature. The core code can be viewed in <b>boston.py, boston_sage.py, convergence.py</b>.

### feature_selection_MNIST

We do feature selection experiments here on MNIST. The core code can be viewed in <b>feature selection.ipynb</b>. In this fold, we mainly use code from http://github.com/iancovert/sage/, so you can reference it for details.

If you try to run the code, remember to evaluate the model with PSAVE first and save the result to a local file. Then, you're supposed to modify corresponding paths in <b>feature selection.ipynb</b>.

