import argparse
import os

import pygame
import torch

from agent import Agent
from game import SnakeGameAI


def play(model_path, episodes=5, record_gif=False, gif_path='demo.gif'):
    agent = Agent()
    agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.model.eval()
    game = SnakeGameAI()

    frames = []
    for ep in range(episodes):
        game.reset()
        done = False
        score = 0
        while not done:
            state = agent.get_state(game)
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = agent.model(state0)
            move = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move] = 1

            _, done, score = game.play_step(final_move, render=True)

            if record_gif:
                frame = pygame.surfarray.array3d(game.display)
                frames.append(frame.transpose([1, 0, 2]))

        print(f'Episode {ep + 1}: score {score}')

    if record_gif and frames:
        import imageio
        imageio.mimsave(gif_path, frames, fps=20)
        print(f'Saved {gif_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Watch a trained DQN agent play Snake.')
    parser.add_argument('--model', default=os.path.join('model', 'model.pth'), help='path to saved model weights')
    parser.add_argument('--episodes', type=int, default=5, help='number of games to play')
    parser.add_argument('--gif', action='store_true', help='record gameplay to demo.gif for your portfolio')
    args = parser.parse_args()

    play(args.model, episodes=args.episodes, record_gif=args.gif)
