import argparse

from agent import Agent
from game import SnakeGameAI
from helper import plot


def train(render=True, n_episodes=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move, render=render)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games:>5}  Score {score:>3}  Record {record:>3}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            if n_episodes and agent.n_games >= n_episodes:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DQN agent to play Snake.')
    parser.add_argument('--no-render', action='store_true', help='skip drawing the game window while training (faster)')
    parser.add_argument('--episodes', type=int, default=None, help='stop after this many games (default: run forever, Ctrl+C to stop)')
    args = parser.parse_args()

    train(render=not args.no_render, n_episodes=args.episodes)
