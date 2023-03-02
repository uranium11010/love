import argparse
import collections
import os
import shutil
import time
import pickle

import git
import gym_miniworld
import numpy as np
import torch
import tqdm
import wandb

import config as cfg
import dqn
import dqn_utils
from grid_world import grid
import option_wrapper
import rl
import utils
from world3d import world3d


def run_episode(env, policy, experience_observers=None, test=False,
                return_render=False, reverse_update=False):
    """Runs a single episode on the environment following the policy.

    Args:
        env (gym.Environment): environment to run on.
        policy (Policy): policy to follow.
        experience_observers (list[Callable] | None): each observer is called with
            with each experience at each timestep.

    Returns:
        episode (list[Experience]): experiences from the episode.
    """
    def maybe_render(env, instruction, action, reward, info, timestep):
        if return_render:
            render = env.render()
            render.write_text("Action: {}".format(str(action)))
            render.write_text("Instruction: {}".format(instruction))
            render.write_text("Reward: {}".format(reward))
            render.write_text("Timestep: {}".format(timestep))
            render.write_text("Info: {}".format(info))
            return render
        return None

    if experience_observers is None:
        experience_observers = []

    episode = []
    state = env.reset()
    timestep = 0
    renders = [maybe_render(env, state[1], None, 0, {}, timestep)]
    hidden_state = None
    while True:
        action, next_hidden_state = policy.act(state, hidden_state, test=test)
        next_state, reward, done, info = env.step(action)
        timestep += 1
        renders.append(maybe_render(env, next_state[1], action, reward, info, timestep))
        experience = rl.Experience(
                state, action, reward, next_state, done, info, hidden_state,
                next_hidden_state)
        episode.append(experience)
        if not reverse_update:
            for observer in experience_observers:
                observer(experience)

        if "experiences" in info:
            del info["experiences"]

        state = next_state
        hidden_state = next_hidden_state
        if done:
            break
            # return episode, renders
    if reverse_update:
        for experience in reversed(episode):
            for observer in experience_observers:
                observer(experience)
    return episode, renders


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
            '-c', '--configs', action='append', default=["configs/default.json"])
    arg_parser.add_argument(
            '-b', '--config_bindings', action='append', default=[],
            help="bindings to overwrite in the configs.")
    arg_parser.add_argument(
            "-x", "--base_dir", default="experiments",
            help="directory to log experiments")
    arg_parser.add_argument(
            "-p", "--checkpoint", default=None,
            help="path to checkpoint directory to load from or None")
    arg_parser.add_argument(
            "-f", "--force_overwrite", action="store_true",
            help="Overwrites experiment under this experiment name, if it exists.")
    arg_parser.add_argument(
            "-r", "--reverse_update", action="store_true",
            help="Traverse episode in reverse direction during updates.")
    arg_parser.add_argument(
            "-s", "--seed", default=0, help="random seed to use.", type=int)
    arg_parser.add_argument(
            "-a", "--abs_type",
            help="Type of SymSkill abstraction. If None, use HSSM.")
    arg_parser.add_argument(
            "-k", "--num_iters", type=int, default=1,
            help="Number of iterations of SymSkill.")
    arg_parser.add_argument(
            "-d", "--device", default=0, type=int,
            help="CUDA device")
    arg_parser.add_argument("exp_name", help="name of the experiment to run")
    arg_parser.add_argument("--tensorboard", action="store_true")
    arg_parser.add_argument("--wandb", action="store_true")
    arg_parser.add_argument("--num_episodes", type=int, default=150000)
    args = arg_parser.parse_args()
    config = cfg.Config.from_files_and_bindings(
            args.configs, args.config_bindings)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_dir = os.path.join(os.path.expanduser(args.base_dir), args.exp_name)
    if os.path.exists(exp_dir) and not args.force_overwrite:
        raise ValueError("Experiment already exists at: {}".format(exp_dir))
    shutil.rmtree(exp_dir, ignore_errors=True)  # remove directory if exists
    time.sleep(5)
    os.makedirs(exp_dir)

    with open(os.path.join(exp_dir, "config.json"), "w+") as f:
        config.to_file(f)
    print(config)

    with open(os.path.join(exp_dir, "metadata.txt"), "w+") as f:
        repo = git.Repo()
        f.write("Commit: {}\n\n".format(repo.head.commit))
        commit = repo.head.commit
        diff = commit.diff(None, create_patch=True)
        for patch in diff:
            f.write(str(patch))
            f.write("\n\n")

    if args.tensorboard:
        tb_writer = dqn_utils.EpisodeAndStepWriter(
                os.path.join(exp_dir, "tensorboard"))
    if args.wandb:
        wandb.init(name=f"{args.exp_name}_{args.abs_type}_{args.seed}",
                   entity="conpole2",
                   project="LOVE_CompILE")
    if args.abs_type is None:
        hssm = torch.load(config.get("checkpoint")).cpu()
        hssm._use_min_length_boundary_mask = True
        hssm.eval()

    if config.get("env") == "compile":
        env = grid.ComPILEEnv(
            1, sparse_reward=config.get("sparse_reward"),
            visit_length=config.get("visit_length"))
        train_loader = utils.compile_loader(100)[0]
        if args.abs_type is None:
            hssm.post_obs_state._output_normal = True
            hssm._output_normal = True
        else:
            import itertools
            from abstractions.compress import IAPLogN
            from abstractions.steps import Solution, AxStep
            from abstractions.abstractions import Axiom, AxiomSeq

            trajectories = np.load("compile.npy", allow_pickle=True)
            solutions = []
            for traj in trajectories:
                traj = utils.add_instruction_rewards(traj)
                states = []
                actions = []
                for i in range(1, len(traj)-1):
                    states.append(traj[i])
                    actions.append(AxStep(traj[i][1]))
                states.append(traj[-1][0])
                sol = Solution(states, actions)
                solutions.append(sol)

            abstractor = IAPLogN(solutions, 6, {"abs_type": args.abs_type})
            abstractions, new_sols = abstractor.iter_abstract(args.num_iters)
            if args.abs_type == "ax_seq_loop":
                new_abstractions = []
                for ab in abstractions:
                    for num_reps in itertools.product(*[range(2, 8)]*ab.num_loops):
                        new_abstractions.append(ab.as_abs_type("ax_seq", num_reps))
                abstractions = new_abstractions
                new_sols = AxiomSeq.get_abstracted_sols(solutions, abstractions)
            print(abstractions)
            abs2idx = {ab: env.action_space.n + idx for idx, ab in enumerate(abstractions)}
            abstracted_data = []
            shape = trajectories[0][0][0].shape
            def get_idx(a):
                if a.rule.is_abs:
                    return abs2idx[a.rule]
                return a.rule.name
            for sol in new_sols:
                # trajectory = [((np.zeros(shape, dtype=np.float32), 0), 0)]
                trajectory = []
                for s, a in zip(sol.states, sol.actions):
                    # trajectory.append((s, get_idx(a)))
                    trajectory.append(s)
                # trajectory.append((sol.states[-1], 0))
                abstracted_data.append(trajectory)

    elif config.get("env") == "3d":
        assert args.abs_type is None
        env = world3d.MultiTask3DEnv(
                seed=1, num_objects=4, visit_length=4, max_episode_steps=75,
                sparse_reward=config.get("sparse_reward"))
        env = world3d.PanoramaObservationWrapper(env)
        train_loader = utils.miniworld_loader(100)[0]
        hssm.post_obs_state._output_normal = False
        hssm._output_normal = False
    else:
        raise ValueError()

    if config.get("oracle", False):
        assert config.get("env") == "compile"
        env = option_wrapper.OracleOptionWrapper(env)
    elif args.abs_type is None:
        env = option_wrapper.OptionWrapper(
                env, hssm, train_loader, train_loader.dataset.seq_size, 1,
                threshold=config.get("threshold"),
                recurrent=config.get("recurrent"))
        hssm = hssm.cuda()
    else:
        env = option_wrapper.AbsWrapper(env, new_abstractions)

    # Use GPU if possible
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device(args.device)

    print("Device: {}".format(device))

    agent = dqn.DQNAgent.from_config(config.get("agent"), env)

    # Behavior Cloning
    if config.get("bc"):
        if config.get("env") == "compile":
          if args.abs_type is None:
              agent.behavioral_clone(
                      np.load("compile.npy", allow_pickle=True), 0,
                      num_epochs=50)
          else:
              agent.behavioral_clone(
                      abstracted_data, 0,
                      num_epochs=50)
        elif config.get("env") == "3d":
          with open("world3d.pkl", "rb") as f:
              trajectories = pickle.load(f)
          interleaved_trajectories = []
          for states, actions in zip(
                  trajectories["states"], trajectories["actions"]):
              episode = []
              for state, action, next_state in zip(states, actions, states[1:]):
                  episode.append((state[0], action, next_state[0]))
              interleaved_trajectories.append(episode)
          agent.behavioral_clone(
              np.array(interleaved_trajectories), 0, num_epochs=50)

    total_steps = 0
    train_rewards = collections.deque(maxlen=100)
    test_rewards = collections.deque(maxlen=100)
    visualize_dir = os.path.join(exp_dir, "visualize")
    os.makedirs(visualize_dir, exist_ok=False)
    for episode_num in tqdm.tqdm(range(args.num_episodes)):
        episode = run_episode(
            env, agent, experience_observers=[agent.update], reverse_update=args.reverse_update)[0]

        total_steps += sum(exp.info.get("steps", 1) for exp in episode)
        train_rewards.append(sum(exp.reward for exp in episode))

        if episode_num % 10 == 0:
            return_render = episode_num % 100 == 0
            episode, render = run_episode(
                    env, agent, test=True, return_render=return_render, reverse_update=args.reverse_update)
            test_rewards.append(sum(exp.reward for exp in episode))
            if return_render:
                frames = [frame.image() for frame in render]
                episodic_returns = sum(exp.reward for exp in episode)
                save_path = os.path.join(visualize_dir, f"{episode_num}.gif")
                frames[0].save(save_path, save_all=True, append_images=frames[1:],
                               duration=750, loop=0, optimize=True, quality=20)

        if episode_num % 50 == 0:
            if args.tensorboard:
                tb_writer.add_scalar(
                        "reward/train", np.mean(train_rewards), episode_num,
                        total_steps)

                tb_writer.add_scalar(
                        "reward/test", np.mean(test_rewards), episode_num,
                        total_steps)

                for k, v in agent.stats.items():
                    if v is not None:
                        tb_writer.add_scalar(k, v, episode_num, total_steps)

            if args.wandb:
                results = agent.stats.copy()
                results["episode_num"] = episode_num
                results["total_steps"] = total_steps
                results["reward/train"] = np.mean(train_rewards)
                results["reward/test"] = np.mean(test_rewards)
                wandb.log(results)


if __name__ == '__main__':
    main()
