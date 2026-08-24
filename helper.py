import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(scores, mean_scores, filename='training_progress.png'):
    plt.figure(figsize=(8, 5))
    plt.title('Training Progress')
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), filename))
    plt.close()
