# COSTA

This is the repository for the paper : `Cost-aware Offline Safe Meta Reinforcement Learning with Robust In-Distribution Online Task Adaptation`.

## Algorithm projects

### Installation Instructions

1. Install MuJoCo
    
    Install MuJoCo-200 accoring to [mujoco-py](https://github.com/openai/mujoco-py) and [MuJoCo website](https://www.roboti.us/license.html).
    
    Extract the downloaded `mujoco200` directory into `~/.mujoco/mujoco200`
    
    Set the env path：
    
    ```bash
    export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
    export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
    ```
    
2. Create Environment
    
    Please first create the conda environment by using:
    
    ```bash
    conda create -n costa python=3.7.9
    ```
    
    Then, install Pytorch and packages in requirements by using:
    
    ```bash
    conda install pytorch==1.2.0
    pip install -r requirements.txt
    ```
    
3. Install OpenMPI (Optional)
    
    We build our offline data collection code on OpenAI’s [Safety-Starter-Agents](https://github.com/openai/safety-starter-agents), which requires the use of OpenMPI. Therefore, if you wish to collect offline data on your own, please download and build OpenMPI-4.0.x or OpenMPI-4.1.x according to [openmpi](https://www.open-mpi.org/) and [documentation](https://www.open-mpi.org/faq/?category=building).
    
4. Install Safety-Starter-Agents (Optional)
    
    Please first activate costa env and enter the edited safety-starter-agents folder we given:
    
    ```bash
    conda activate costa
    cd safety-starter-agents
    ```
    
    Install Safety-Starter-Agents using:
    
    ```bash
    pip install -e .
    ```
    

### Offline Data Collection

We have provided the offline data used in COSTA’s experiments, you can directly download it from [offline_data](https://drive.google.com/file/d/1rq_G4Fyc7mrt_Dn6tvdVd-ZeW83by5QK/view?usp=drive_link) and use the unzipped folder `offline_data` to replace `COSTA/offline_data`.

If you wish to collect offline data on your own, please begin by installing the relevant components as outlined in sections 3 and 4 of the Installation Instructions and enter safety-starter-agents folder:

```bash
cd safety-starter-agents
```

We use the online CPO algorithm for data collection. The core code for data collection can be found in the `safety-starter-agents/scripts/mujoco_experiments.py`, while all the related environments are located in the `safety-starter-agents/rlkit/envs` directory. All the environment names we are using have "safe" appended to their end.

To collect the offline data of the first task of AntDir, please using the following instruction:

```bash
python scripts/mujoco_experiment.py --task ant-dir-safe --algo cpo --goal 0
```

In the instruction, the "--task" flag specifies the target environment, the "--algo" flag indicates the algorithm used for data collection, and the "--goal" flag represents the target task within the data collection's target environment. 

Upon the completion of the above command execution, the complete offline data will be saved in `safety-starter-agents/offline_data/ant-dir-safe_0/offline_buffer.npz`. Finally, move the npz file to `COSTA/offline_data/ant-dir-safe_0` to complete the process.

### Offline Training

First, enter the COSTA folder:

```bash
cd COSTA
```

During the Offline Training process, assuming the target environment is AntDir, you should start by running the following shell scripts separately to train the task-specific CVAEs and cost models, which are named VAE and discriminator in the code:

```bash
#!/bin/bash
for goal in {0..2}
do {
    nohup python run_vae.py --task ant-dir-safe --goal $goal
} &
done
wait
```

```bash
#!/bin/bash
for goal in {0..2}
do {
    nohup python run_discriminator.py --task ant-dir-safe --goal $goal
} &
done
wait
```

In the experiments, except for CheetahWalk, which has only two training tasks, all other environments have three training tasks. Therefore, if the environment is CheetahWalk, please change the first line of each code to `for goal in {0..1}`. The final models will be stored in 

`COSTA/run/ant-dis-safe_$goal/vae/vae.pt` and 

`COSTA/run/ant-dis-safe_$goal/discriminator/best_model.pt`.

After the training of CVAEs and cost models, just running the following command to train the context encoder:

```bash
python run_context_encoding.py --task ant-dir-safe
```

The trained models will be stored in 

`COSTA/run/ant-dir-safe/context_encoder/$training_time/model`. 

Then, you need to copy the relative path `$training_time/model/encoder100.pt` to the config file `COSTA/configs/ant-dir-safe.json`'s `[”meta_params”][”mlp_attn_path”]`.

We also provide the code for training context encoders for other baselines like FOCAL and CORRO, just using:

```bash
python run_focal_context_encoding.py --task ant-dir-safe
python run_corro_context_encoding.py --task ant-dir-safe
```

The trained models will be stored in 

`COSTA/run/ant-dir-safe/focal_context_encoder/$training_time/model` or 

`COSTA/run/ant-dir-safe/corro_context_encoder/$training_time/model`. 

Then, you need to copy the relative path `focal_context_encodere/$training_time/model/encoder100.pt` to the config file `COSTA/configs/ant-dir-safe.json`'s `[”meta_params”][”focal_path”]` or copy the relative path `corro_context_encodere/$training_time/model/encoder100.pt` to the config file `COSTA/configs/ant-dir-safe.json`'s `[”meta_params”][”corro_path”]`.

After the training of context encoder, you can run the following command to train the meta-policy:

```bash
python run_meta_cpq.py --task ant-dir-safe --seed 0
```

Also you can train the meta-policy using other algorithms:

```bash
python run_focal_cpq.py --task ant-dir-safe --seed 0
python run_corro_cpq.py --task ant-dir-safe --seed 0
python run_pearl_cpq.py --task ant-dir-safe --seed 0
python run_vanilla_cpq.py --task ant-dir-safe --seed 0
```

The trained policy will be stored in 

`COSTA/run/ant-dir-safe/meta_cpq/mlp_attn/seed_0_timestamp_$training_time/model`. 

### Online Adaptation

For running online adaptation experiments, you should first copy the relative path of trained meta-policy like 

`run/ant-dir-safe/meta_cpq/mlp_attn/seed_0_timestamp_$training_time/model/policy.pth` to the `default=` part in the code below in `COSTA/online_adaptation.py`:

```bash
parser.add_argument('--policy_path', type=str, default="")
```

Then, run the following command:

```bash
python online_adaptation.py --task ant-dir-safe --seed 0
```

The result will be stored in 

`COSTA/run/ant-dir-safe/online_adaptation/seed_0_timestamp_$adaptation_time/record`.

For environments like CheetahVel where unseen task generalization experiments need to be conducted, you can execute the following command:

```bash
python online_adaptation.py --task cheetah-vel-safe --ood 1 --seed 0
```

## Note

The implementation is based on [Safety-Starter-Agents](https://github.com/openai/safety-starter-agents), [RLKit](https://github.com/rail-berkeley/rlkit) and [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit) which are open-sourced.