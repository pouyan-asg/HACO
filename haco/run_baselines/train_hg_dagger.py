from __future__ import print_function

from haco.algo.HG_Dagger.exp_saver import Experiment
from haco.algo.HG_Dagger.model import Ensemble
from haco.algo.HG_Dagger.utils import *
from haco.utils.config import baseline_eval_config, baseline_train_config
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import datetime
import os

"""
requirement for IWR/HG-Dagger/GAIl:

create -n haco-hg-dagger python version=3.7
1. pip install loguru imageio easydict tensorboardX pyyaml  stable_baselines3 pickle5
2. conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=9.2 -c pytorch
"""

# hyperpara
BC_WARMUP_DATA_USAGE = 1000  # use human data to do warm up (maximum number of samples)
NUM_ITS = 5
STEP_PER_ITER = 1000
learning_rate = 5e-4
batch_size = 256

# need_eval = False  # we do not perform online evaluation. Instead, we evaluate by saved model
# evaluation_episode_num = 30
num_sgd_epoch = 1000  # sgd epoch on data set
device = "cuda"

# training env_config/test env config
training_config = baseline_train_config
training_config["use_render"] = True
training_config["manual_control"] = True
training_config["main_exp"] = True
eval_config = baseline_eval_config

if __name__ == "__main__":
    tm_stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    folder_dir = os.path.join(
        "hg_dagger_lr_{}_bs_{}_sgd_iter_{}_iter_batch_{}".format(
            learning_rate, batch_size, num_sgd_epoch, STEP_PER_ITER), tm_stamp)

    root_path = "/home/pouyan/phd/imitation_learning/hgdagger/HACO/logs"
    log_dir = os.path.join(root_path, folder_dir)

    exp_log = Experiment(log_dir)
    model_save_path = os.path.join(log_dir, "hg_dagger_models")
    os.mkdir(model_save_path)

    # seperate with eval env to avoid metadrive collapse
    training_env = SubprocVecEnv([lambda: HumanInTheLoopEnv(training_config)])
    eval_env = HumanInTheLoopEnv(eval_config)

    obs_shape = eval_env.observation_space.shape
    action_shape = eval_env.action_space.shape

    # fill buffer with warmup expert data
    samples = load_human_data(
        "/home/pouyan/phd/imitation_learning/hgdagger/HACO/haco/utils/human_traj_3.json", 
        data_usage=BC_WARMUP_DATA_USAGE)

    # train first epoch
    print("\033[92m\nWarm up Training ...\n\033[0m")
    agent = Ensemble(obs_shape, action_shape, device=device).to(device).float()
    X_train, y_train = samples["state"], samples["action"]
    train_model(agent, X_train, y_train,
                os.path.join(model_save_path, "model_{}.pth".format(0)),
                num_epochs=num_sgd_epoch,
                batch_size=batch_size,
                learning_rate=learning_rate,
                exp_log=exp_log)
    # if need_eval:
    #     evaluation(eval_env, agent, evaluation_episode_num=evaluation_episode_num, exp_log=exp_log)
    exp_log.end_iteration(0)

    # count
    for iteration in range(1, NUM_ITS):
        steps = 0
        episode_reward = 0
        success_num = 0
        episode_cost = 0
        done_num = 0
        state = training_env.reset()[0]
        # for user friendly :)
        print("\033[92m\nLet's Go!\033[0m")
        print(f"\033[92m\nIteration: {iteration}\033[0m")
        training_env.env_method("stop")
        print("\033[91m\nPress E to Start new iteration\033[0m")
        sample_start = time.time()

        while True:
            # main loop
            if steps % 100 == 0:
                print(f"\033[93m\nIteration: {iteration}, Steps: {steps}\033[0m")
            next_state, r, done, info = training_env.step(np.array([agent.act(torch.tensor(state, device=device))]))
            next_state = next_state[0]
            r = r[0]
            done = done[0]
            info = info[0]
            action = info["raw_action"]
            takeover = info["takeover"]

            episode_reward += r
            episode_cost += info["native_cost"]
            if takeover:
                # aggregate data only when takeover occurs
                samples["state"].append(state)
                samples["action"].append(np.array(action))
                samples["next_state"].append(next_state)
                samples["reward"].append(r)
                samples["terminal"].append(done)

            state = next_state
            steps += 1

            # train after BATCH_PER_ITER steps
            if done:
                if info["arrive_dest"]:
                    success_num += 1
                done_num += 1
                print(f"\033[93mDone Num: {done_num}\033[0m")
                if steps > STEP_PER_ITER:
                    exp_log.scalar(is_train=True, mean_episode_reward=episode_reward / done_num,
                                   mean_episode_cost=episode_cost / done_num,
                                   success_rate=success_num / done_num,
                                   mean_step_reward=episode_reward / steps,
                                   sample_time=time.time() - sample_start,
                                   buffer_size=len(samples))

                    X_train, y_train = samples["state"], samples["action"]
                    # Create new model
                    agent = Ensemble(obs_shape, action_shape, device=device).to(device).float()
                    train_model(agent, X_train, y_train,
                                os.path.join(model_save_path, "model_{}.pth".format(iteration)),
                                num_epochs=num_sgd_epoch,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                exp_log=exp_log)
                    # if need_eval:
                    #     evaluation(eval_env, agent, evaluation_episode_num=evaluation_episode_num, exp_log=exp_log)
                    break
        exp_log.end_iteration(iteration)
    training_env.close()
    eval_env.close()
