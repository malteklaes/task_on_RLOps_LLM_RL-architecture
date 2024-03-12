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

### source WITHOUT HT or HPO
> source A2
> own py-file: "withoutHT.py" for very simple CartPole environemnt
> note: very easy and clear example

> source B2

> source C2

## IV. summary/discussion
- 
Which programming language was mainly used: python, ipynt
Assumption: How do you distinguish whether a source code uses hyperparameter fine tuning or not. If fine tuning is not used, a default setting is used. How can you see in the source code that no fine tuning is used:
- no **explicit specification** of hyperparameter:
  - Examples/selection of hyperparameters: 
    - `batch_size`
    - `gamma`
    - `lr`
    - `n-trials`
    - `n-jobs`
    - `sampler`
    - `pruner`
  - Examples of explicit information:
    - `python train.py --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 --sampler tpe --pruner median`
    - `[...] learning_rate = 0.001 [...]`

`class stable_baselines3.ppo.PPO(policy, env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, rollout_buffer_class=None, rollout_buffer_kwargs=None, target_kl=None, stats_window_size=100, tensorboard_log=None, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)`

- Use/import libraries like *optuna*:
  - `import optuna` (explicit fine tuning takes place here)



## V. literature

[1] https://www.automl.org/hyperparameter-tuning-in-reinforcement-learning-is-easy-actually/ 

[2] Hyperparameter Tuning: https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html
- "[...] Not all hyperparameters are tuned, and tuning enforces certain default hyperparameter settings that may be different from the official defaults. [...]
Hyperparameters not specified in rl_zoo3/hyperparams_opt.py are taken from the associated YAML file and fallback to the default values of SB3 if not present. [...]"

[3] RL hyperparameter tuning (RL zoo): https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tl-dr and https://github.com/DLR-RM/rl-baselines3-zoo