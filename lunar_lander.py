import torch
import numpy as np
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

@dataclass
class Args:
    num_episodes: int = 2500
    num_steps: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 512
    minibatch_size: int = 32
    clip_coef: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    norm_adv: bool = True
    update_epochs: int = 4
    max_grad_norm: float = 0.5
    learning_rate: float = 2.5e-4
    train_agent: bool = False
    render_episodes: int = 50

def layer_init(layer, std=np.sqrt(2), bias=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space.n), std=0.01)
        )

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

args = Args()

if __name__ == "__main__":
    if args.train_agent:
        plt.ion()  # interactive mode on
        fig, ax = plt.subplots()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = gym.make("LunarLander-v3")
        agent = Agent(env).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        episode_rewards = []

        obs = torch.zeros((args.num_steps,) + env.observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps,) + env.action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps,)).to(device)
        rewards = torch.zeros((args.num_steps,)).to(device)
        dones = torch.zeros((args.num_steps,)).to(device)
        values = torch.zeros((args.num_steps,)).to(device)

        for episode in range(1, args.num_episodes+1):
            next_obs, _ = env.reset()
            next_obs = torch.tensor(next_obs).to(device)
            next_done = torch.zeros(1).to(device)
            for step in range(args.num_steps):
                obs[step] = next_obs
                dones[step] = next_done[0]
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminate, truncate, _ = env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.tensor(next_obs).to(device), torch.tensor([float(terminate or truncate)], dtype=torch.float32).to(device)
                if terminate or truncate:
                    break
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminate = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminate = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminate - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminate * lastgaelam
                returns = advantages + values
            episode_rewards.append(rewards[:step+1].sum().item())

            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[mb_inds], actions.long()[mb_inds])
                    logratio = newlogprob - logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - returns[mb_inds])**2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                print("epoch update done")
            print(f"episode {episode} finished. episode reward:{rewards[:step+1].sum().item()}")
            print(f"episode {episode}, pg_loss: {pg_loss.item():.4f}, v_loss: {v_loss.item():.4f}")
            ax.clear()
            ax.plot(episode_rewards)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Mean Reward")
            ax.set_title("LunarLander PPO Training")
            plt.pause(0.01)

        env.close()
        plt.ioff()
        plt.show()
        torch.save(agent.state_dict(), "lunar_lander_ppo.pth")
        print("trained weights saved")

    env_render = gym.make("LunarLander-v3", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env_render).to(device)
    agent.load_state_dict(torch.load("lunar_lander_ppo.pth"))
    agent.eval() 
    observation, info = env_render.reset()
    observation = torch.tensor(observation).to(device)
    episode_over = False
    solved_envs = 0
    for episode in range(args.render_episodes):
        total_reward = 0
        while not episode_over:
            action, logprob, _, value = agent.get_action_and_value(observation)  
            observation, reward, terminated, truncated, info = env_render.step(action.cpu().numpy())
            observation = torch.tensor(observation).to(device)
            total_reward += reward
            episode_over = terminated or truncated
        observation, _ = env_render.reset() 
        observation = torch.tensor(observation).to(device) 
        episode_over = False  
        if total_reward > 200:
            solved_envs += 1
        print(f"Episode finished! Total reward: {total_reward}")
    print(f"{solved_envs} out of 50 episodes solved!")
    env_render.close()
    



