# Task on RLOps and LLM for automation of RL-architecture and non-fct requirements architecture


> procedure to solve the task
<div>
    <img src="./img/procedure.png" alt="BILD" style="width:100%; border:0; float:center, margin:5px" >
</div>

- ***Benefit***: 
**Automation** **of** the **analysis** process by training an LLM specifically for RL architectural features using fine tuning (here “hyperparameter tuning”). This allows the RLOps process to be significantly accelerated, since, for example, communication about the architectures used by different RL projects can be recognized and communicated more quickly.


## I. procedure

1. Collect a small dataset (around 3-5 instances) of RL source code examples that apply hyperparameter tuning. The examples should ideally use various programming languages, environments (e.g. Python scripts, Jupyter notebooks, etc.), techniques, libraries, and so on so that the examples aren't all similar. Also, find around 3-5 examples with no hyperparameter tuning.

2. Define a machine-readable schema (e.g. JSON, XML, etc.) for storing the information discovered in source code about hyperparameter tuning. At a minimum, the relevant line numbers, but additional information on, e.g. the actual hyperparameter values used, the libraries used, etc., would be interesting too.

3. Create reference (ground truth) files for each source code example, applying your defined schema.

4. Use an LLM to analyse the source code for hyperparameter tuning and output the results in your machine-readable schema. 
[Due to time constraints and a lack of current training opportunities, the LLM GPT-3.5 (= "normal" ChatGPT-3.5) was used.]

5. Perform a basic comparison, (programmatic or otherwise, e.g. (semi-)automated), to compare what the LLM found with your ground truth.


## II. repo contains

a. source code instances
b. ground truth files
c. LLM prompts with LLM output (as JSON)
d. summary/discussion


## III. sources

### source WITH HT or HPO
> source A1
> own py-file: "withHT.py" for very simple CartPole environemnt
> note: very easy and clear example


> source B1
> link: https://github.com/DLR-RM/rl-baselines3-zoo

> source C1
> Trainer.cpp
> link: https://github.com/navneet-nmk/Pytorch-RL-CPP/blob/master/Trainer.cpp

### source WITHOUT HT or HPO
> source A2
> own py-file: "withoutHT.py" for very simple CartPole environemnt
> note: very easy and clear example

> source B2

> source C2

## IV. summary/discussion
- 
Which programming language was mainly used: python, ipynt
Assumption: How do you distinguish whether a source code apply hyperparameter fine tuning or not. If fine tuning is not used, a default setting or prediefined hyperparameter is used. How can you see in the source code whether fine tuning is applied or not:

1. Heuristic for "fine tuning is used":
- **explicit specification** of hyperparameter in source code:
  - Examples/selection of hyperparameters such as: 
    - `batch_size`
    - `gamma`
    - `lr`
    - `n-trials`
    - `n-jobs`
    - `sampler`
    - `pruner`
    - `ortho_init` (True by default)
    - `policy_kwargs`
  - Examples of explicit information:
    - `[...] learning_rate = 0.001 [...]`
  - chosen RL-algorithm:
    - PPO does not necessarily require hyperparameter tuning, other algorithm does [2]
  - HPO-methods are used [1] such as:
    - `DEHB` (Distributed Evolutionary Hyperparameter Tuning for Deep Learning)
    - `RS` (Random Search)
    - `BGT` (Bayesian Graph Transformer)
    - `Grid Search`
  - HPO related libs [3]
    - `from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, BayesSearchCV`


1. Heuristic for "fine tuning is NOT used":
- use of library base
  - in python-lib Baselines3, by default "optuna" (pen source hyperparameter optimization framework) is used for optimal hyperparameter and if `optimize` is used, than Baseline3 makes use of those predefined hyperparameter, e.g.: `python -m rl_zoo3.train --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 --sampler tpe --pruner median` (NO)
    - source: "RL Baselines3 Zoo (https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html#hyperparameter-optimization)


## V. literature

[1] https://www.automl.org/hyperparameter-tuning-in-reinforcement-learning-is-easy-actually/ 

[2] https://jonathan-hui.medium.com/rl-reinforcement-learning-algorithms-comparison-76df90f180cf
- HPO provide best practice to handle multiple seeds

[3] https://scikit-learn.org/stable/modules/grid_search.html

[4] Hyperparameter Tuning: https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html
- "[...] Not all hyperparameters are tuned, and tuning enforces certain default hyperparameter settings that may be different from the official defaults. [...]
Hyperparameters not specified in rl_zoo3/hyperparams_opt.py are taken from the associated YAML file and fallback to the default values of SB3 if not present. [...]"

[5] https://rl-baselines3-zoo.readthedocs.io/en/master/index.html
- "[...] RL Baselines3 Zoo s a training framework for Reinforcement Learning (RL), [...]
It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.
In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings. [...]"

[6] RL hyperparameter tuning (RL zoo): https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tl-dr and https://github.com/DLR-RM/rl-baselines3-zoo
