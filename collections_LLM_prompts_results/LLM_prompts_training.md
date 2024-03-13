# LLM collection (interaction/prompts, training, discussion)


## discussion: LLM setup and alternatives

(1) setup and reasoning about model

- GPT-3.5 as ChatGPT3.5 as LLM model
  - Due to the experimental task with limited time and focus on a schema (JSON) that recognizes and specifically demonstrates hyperparameter tuning in source code, a very advanced LLM model was used here, so that "fine tuning" via prompt engineering quickly produced very good results.

(2) alternatives (pre-defined) LLMs [1]:

- GPT-2
- Falcon
- LLaMA
- BERT 
- XLNet
- T5 
- ELECTRA




## prompts of "training-phase" (fine tuning) of the LLM via prompt engineering
- With regard to prompt engineering, the criteria/heuristics (see README.md) developed for training and fine tuning were handed over to the LLM model piece by piece.
- prompts/interaction:
  
> me
  ```
    Please consider these 4 main criteria when it comes to analyzing source code for hyperparameter fine tuning:
    1. auto-tuning: HPO-methods/tools for auto-RL [1b] are used [1a] such as DEHB, PBT
    2. auto/hand-tuning: HPO libraries or frameworks are used such as Optuna, scikit-optimize, hyperopt
    3. hand-tuning: Loops in source code that iterate over hyperparameter settings (what: here a for loop would have to be combined with parameter changes)
    4. auto-tuning: Parameterization of hyperparameters with e.g. YAML-files such as (a2c.yml ars.yml, ppo_lstm.yml, trpo.yml, ...)
```


> answer GPT-3.5
  ```
    Sure, I can break down the analysis based on the four criteria you provided:
        1. **Auto-tuning with HPO methods/tools (e.g., DEHB, PBT):**
            - Look for the integration of tools or methods that automate the hyperparameter optimization (HPO) process. This could involve algorithms like Differential Evolution for Hyperparameter Benchmarking (DEHB) or Population Based Training (PBT).
            - Check if the codebase leverages reinforcement learning (RL) techniques for automatic hyperparameter tuning, such as incorporating DEHB or PBT directly into the training loop.

        2. **Auto/hand-tuning with HPO libraries/frameworks (e.g., Optuna, scikit-optimize, hyperopt):**
            - Investigate whether the codebase utilizes popular HPO libraries or frameworks like Optuna, scikit-optimize, or hyperopt.
            - Determine if there's a mixture of automated and manual tuning approaches, where the codebase might allow for both automatic exploration (via libraries) and manual adjustment of hyperparameters.

        3. **Hand-tuning through iterative loops:**
            - Search for loops within the source code that explicitly iterate over hyperparameter settings.
            - This could involve manual adjustments of hyperparameters within a loop structure, where each iteration represents a different configuration.

        4. **Auto-tuning through parameterization with YAML files or similar:**
            - Look for YAML files or similar configuration files within the codebase that parameterize hyperparameters.
            - These files might contain predefined hyperparameter settings for different algorithms or models (e.g., a2c.yml, ars.yml, ppo_lstm.yml, trpo.yml), allowing for easy modification and experimentation.
    By assessing the presence and utilization of these criteria within the source code, you can gauge the approach taken for hyperparameter fine-tuning and understand the level of automation and flexibility in the optimization process.
```


> me
  ```
    Please use this result JSON-schema as a response template for your response:
    (see rlArchRestLayout.json)
```
> answer GPT-3.5
  ```
    (repeated prompt)
```

> me (analysis request)
  ```
  please analyze this source file with the previously defined criteria and heuristics and output your result using the previous defined result JSON-schema:
  - source file:
  <<source[A/B/C][1/2/3]>>
    
```
> answer GPT-3.5
  ```
    (see answers in directory "collections_LLM_prompts_results/LLM_results")
```


## alternative "training-phase" (fine tuning) of a LLM via prompt engineering and additional files

- alternative training of less pre-defined and well developed LLMs like GPT-3.5 via txt-files of relevant content (hyperparameter, passages of for-loops) like this:
```txt
  // hyperparameters
  batch_size
  buffer_size
  clip_range
  ent_coef
  gamma
  learning_starts
  lr
  n-jobs
  n_steps
  n-trials
  sampler
  ortho_init
  policy_kwargs
  pruner
  target_update_interval
  tau
  ...
```

## literature

[1] https://huggingface.co/docs/transformers/main/tasks/prompting
