# Task on RLOps and LLM for automation of RL-architecture2 and non-fct requirements architecture


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

(a) source code instances (at total 6 source files) <br>
(b) jsonResultLayout (with json layout file, based on json-schema: https://json-schema.org/draft-07/json-schema-release-notes) <br>
(c) ground truth (GT) files (6 files (3 with HT, 3 without HT) for each source code) <br>
(d) LLM prompts with LLM results (6 output-json files (3 with HT, 3 without HT), 1 md-file with LLM interaction) <br>
(e) README.md (with overview, procedure and summary (Criterias/heuristic, discussion, result)) <br>


## III. sources

### source WITH HT or HPO
> **source A1** <br>
> filename: "withHT.py" (own py-file inspired by [9], for very simple CartPole environment, different learning_rate-hyperparameters) <br>
> note: very easy and clear example

> **source B1** <br>
> filename: "sb3_agent.py" (here dehb-config as HPO-method is used) <br>
> link: https://github.com/facebookresearch/how-to-autorl/tree/main/examples

> **source C1** <br>
> filename: "hyperoptExample.py" <br>
> link: http://hyperopt.github.io/hyperopt/


### source WITHOUT HT or HPO
> **source A2** <br>
> filename: "withoutHT.py" (own py-file inspired by [9], for very simple CartPole environemnt) <br>
> note: very easy and clear example

> **source B2** <br>
> filenname: "7. Temporal Difference and Q-Learning" (Jupyter notebook file (ipynb)) <br>
> link: https://github.com/maykulkarni/Machine-Learning-Notebooks/blob/master/07.%20Reinforcement%20Learning/7.%20Temporal%20Difference%20and%20Q-Learning.ipynb

> **source C2** <br>
> filenname: "Trainer.cpp" <br>
> link: https://github.com/navneet-nmk/Pytorch-RL-CPP/blob/master/Trainer.cpp

## IV. summary

### IV-A: Criteria/heuristics

The question is when exactly one can speak of “hyperparameter tuning”. Here it is assumed that the initial definition of hyperparameters in the source code does NOT count towards hyperparameter tuning. There must be a search of a hyperparameter space (to find the combination of hyperparameters) to ultimately lead to improved performance of the RL algorithm.

Further criteria/heuristics will be listed that suggest (or not) the identification of hyperparameter tuning in this sense in a source code.
To get a better feeling for hyperparameters, here are some examples/selection of hyperparameters (in python/baselines3) for tuning a RL-Algorithm (not exhaustive, just a selection): 
  - `batch_size`
  - `buffer_size`
  - `clip_range`
  - `ent_coef`
  - `gamma`
  - `learning_starts`
  - `lr`
  - `n-jobs`
  - `n_steps`
  - `n-trials`
  - `sampler`
  - `ortho_init`
  - `policy_kwargs`
  - `pruner`
  - `target_update_interval`
  - `tau`
  - ...


1. Criteria/Heuristic for "fine tuning is applied":
   
   (a) **auto-tuning**: **HPO-methods/tools for auto-RL [1b] are used [1a] such as**:
    - `DEHB` (Distributed Evolutionary Hyperparameter Tuning for Deep Learning) [1c]
    - `PBT` [1d]
    - ...
    - **APPLY** in **source B1**

    (b) **auto/hand-tuning**:**HPO libraries or frameworks are used**:
    - This criterion can be a bit broad, since, for example, python/stable-baselines3 also uses optuna and "RL Baselines Zoo" (with already fine-tuned hyperparameters) in the background so, strictly speaking, every file with a lib import "import stable_baselines3" also implicitly uses "hyperparameter tuning". It is therefore assumed that the use of the HPO-lib would have to be combined with a concrete definition of space, as in the hyperopt example (like in case of *hyperopt*).
    - `Optuna` (python, [6], used especially used in python/stable-baselines3 [7])
    - `scikit-optimize` (python, [3][4])
      - e.g. `from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, BayesSearchCV`
    - `hyperopt` [5]
      - example for explicit space-definition: 
        ```python 
          space = hp.choice('a',
            [
                ('case 1', 1 + hp.lognormal('c1', 0, 1)),
                ('case 2', hp.uniform('c2', -10, 10))
            ])
        ```
    - **APPLY** in **source C1**

    (c) **hand-tuning**: **Loops in source code that iterate over hyperparameter settings**:
    - here a for loop would have to be combined with parameter changes
    - `Grid Search`
    - `Random Search`
    - **APPLY** in **source A1**

    (d) **auto-tuning**: **Parameterization of hyperparameters with e.g. YAML-files**
    -  `a2c.yml` [8]
    -  `ars.yml` [8]
    -  `ppo_lstm.yml` [8]
    -  `trpo.yml` [8]
    -  ...

    
<br>

1.  Criteria/Heuristic for "fine tuning is NOT applied":

  - if 1a-d does not apply
    - **APPLY** in **source A2**
    - **APPLY** in **source B2**
    - **APPLY** in **source C2**


### IV-B: discussion
> LLM:
  - LLM-related aspects and discussion can be found in "LLM_prompts_training.md" in directory: collections_LLM_prompts_results\LLM_results
  
> Heuristics enumerated fully?
  - No, only central aspects from as many different fields as possible were covered here (4 main criteria/heuristics). However, in the area of ​​"hand-tuning", for example, other patterns would have to be examined in addition to manual/iterative (for-loop) testing of suitable hyperparameters.


### IV-C: result (comparison between GT and LLM-result) 
> general result (manually evaluated)
  -  In general, just by formulating 4 central criteria for when hyperparameter tuning is used in the source code, LLM, equipped with these, was able to deliver very good results. Of a total of 6 instances of source codes, 3 with and 3 without hyperparameter tuning, a total of 5/6 (83%) could be recognized correctly and special entities like hyperparameters itself were also very good indentified.
> what does a coherent result have to look like in the result Json ("rlArchRestLayout_result_...")?
  - if one of the specific-criteria is true, the result ("hyperparameter_tuning_applied") is also true and vice versa
  - **positive result** (hyperparameter tuning was applied)
  ```
  [...]
  "hyperparameter_tuning_applied": true,
  [...]
   "specific_criteria": {
      "auto-tuning": false,
      "auto/hand-tuning": true,
      "hand-tuning": true,
      "parameterization": false
    }
  [...]
  ```
  - **negative result** (hyperparameter tuning was not applied)
  ```
  [...]
  "hyperparameter_tuning_applied": false,
  [...]
   "specific_criteria": {
      "auto-tuning": false,
      "auto/hand-tuning": false,
      "hand-tuning": false,
      "parameterization": false
    }
  [...]
  ```
> What does auto-tuning, auto/hand-tuning, hand-tuning mean?
 - Generally, I introduced them to make analyse more specific and those terms were inspired by the ICML 2023 paper of Eimer/Lindauer/Raileanu [1] (see abstract of the paper)
 - **auto-tuning**: is an approach in the area of ​​HPO (Hyperparameter Optimization) that uses automated forms of hyperparameter tuning such as HPO methods/tools (e.g. DEHB, PBT) or configuration files (YAML-files): 
   - Auto-tuned hyperparameters through configuration files are harder to detect through the LLM because they require specialized knowledge. Here, too, the only error in sourceB1 (actually with hyperparameter tuning, but not recognized) has been made by the LLM.
 - **auto/hand-tuning**: an approach of both, automation through special libraries (Optuna, scikit-optimize, hyperopt), but also (manual) specifications in the code itself (such as the exact definition of the parameter value range/space):
   - Similar to hand-tuning for an LLM, it is easier to recognize because in addition to imported libs (where some hyperparameters have been optimized), hyperparameters also have to be adjusted again (space) in the source code itself. (everything recognized correctly by the LLM)
 - **hand-tuning**: In the code itself, value ranges/space of the hyperparameters are iteratively walked through and tried out (e.g. for-loop):
   - Can be determined very well by LLM, they are very characteristic patterns used directly in the source code. (everything recognized correctly by the LLM)

> "examples" of the result JSON ("rlArchRestLayout_result_...") and detailed comparison between GT and LLM-result
- Those entities were measured in the result json file: filename, language, environment (which RL-env), technique, hyperparameter_tuning_applied, line_numbers (where are major observation points), hyperparameters, libraries, specific_criteria (which type of hyperparameter tuning was used/observed)

> entities with good result GT and LLM-result compared
  - language: always detected by LLM
  - environment: mostly good detected by LLM
  - hyperparameter_tuning_applied: very good detecion-rate (83%)
  - hyperparameters: very good result
  - libraries: very good result
  - specific_criteria: very good result since detected hyperparameter_tuning_applied was good too

> entities with medium/bad result GT and LLM-result compared
- filename (since LLM mostly does not see name): is not so important
- technique (very specific and too general result by LLM): must be improved
- line_numbers (even though the given file was the same, the detected locations were only moderate): must be improved

> conclusion/outlook

A total of 4 categories in the hand-tuned and auto-tuned areas have already delivered very good detection results but are not yet fully sufficient and still promise a lot of scope for optimization. However, more training input is needed in all areas, especially in the experience of configurations (mistake at sourceB1). However, the pattern description also needs to be expanded with regard to hand tuning. Hyperparameter tuning within the source code can be done, for example. In addition to for-loops, there are other patterns, such as through further iteration variants (while, do-while loops), recursively, through special data structures (queues, trees, stacks (peek vs. pop)) etc. can be tried out.


## V. literature

[1a] (HPO-methods/tools, HPO best practice) https://www.automl.org/hyperparameter-tuning-in-reinforcement-learning-is-easy-actually/ (full paper: https://arxiv.org/pdf/2306.01324.pdf)

[1b] https://github.com/facebookresearch/how-to-autorl/blob/main/docs/index.md

[1c] (DEHB) https://arxiv.org/pdf/2105.09821.pdf

[1d] (PBT) https://arxiv.org/pdf/1902.01894.pdf

[2] https://jonathan-hui.medium.com/rl-reinforcement-learning-algorithms-comparison-76df90f180cf
- HPO provide best practice to handle multiple seeds

[3] https://scikit-learn.org/stable/modules/grid_search.html

[4] https://scikit-optimize.github.io/stable/

[5] http://hyperopt.github.io/hyperopt/

[6] https://optuna.org/

[7] https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html#hyperparameter-optimization

[8] https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams

[9] https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

[10] RL hyperparameter tuning (RL zoo): https://github.com/DLR-RM/rl-baselines3-zoo

[11] https://rl-baselines3-zoo.readthedocs.io/en/master/index.html
- "[...] RL Baselines3 Zoo s a training framework for Reinforcement Learning (RL), [...]
It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.
In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings. [...]"

[12] general RL: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tl-dr 

[13] Hyperparameter Tuning: https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html
- "[...] Not all hyperparameters are tuned, and tuning enforces certain default hyperparameter settings that may be different from the official defaults. [...]
Hyperparameters not specified in rl_zoo3/hyperparams_opt.py are taken from the associated YAML file and fallback to the default values of SB3 if not present. [...]"

