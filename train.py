from env import Env
from agent import Agent
import torch
from collections import deque
from PIL import Image

import os
import numpy as np

def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # Add more extensions if needed
    image_paths = []
    
    # List all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a valid image extension
        if any(file_name.lower().endswith(ext) for ext in image_extensions):
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, file_name)
            try:
                Image.open(image_path)
                image_paths.append(image_path)
            except:
                pass
    
    return image_paths

# Example usage:
folder_path = r"C:\Users\Spher\OneDrive\Desktop\CS\AI\vision\rl_image\new\im"  # Change this to your folder path
image_paths = get_image_paths(folder_path)


num_episodes = 500

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

env = Env(image_paths, device=device)
agent = Agent().to(device)

optimizer = torch.optim.AdamW(agent.parameters())

for episode in range(num_episodes):
    states = []
    actions = []
    rewards = []
    timesteps = 0

    # sample s, a, r until episode finishes
    done = False
    obs = env.reset()
    while not done:
        action, caption = agent(obs)
        obs, reward, done = env.step(caption)

        states.append(obs)
        actions.append(action) # or action not sure yet
        rewards.append(reward)

        timesteps += 1
    
    print(f"Total rewards: {sum(rewards)}\nAverage reward: {sum(rewards)/len(rewards)}")
    log_probs = []

    # Iterate over each tensor in the list
    for tensor in actions:
        softmax_probs = torch.nn.functional.softmax(tensor, dim=-1)
        
        log_prob = torch.log(softmax_probs)
        
        log_probs.append(log_prob)

    # update policy
    batch_size = 2  # Define your desired batch size

    for i in range(0, len(rewards), batch_size):
        batch_rewards = rewards[i:i+batch_size]
        batch_log_probs = log_probs[i:i+batch_size]

        R = 0
        policy_loss = []
        returns = deque()
        for r in batch_rewards[::-1]:
            R = r + 0.98 * R
            returns.appendleft(R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())
        
        for log_prob, R in zip(batch_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        del batch_rewards[:]
        del batch_log_probs[:]

        


