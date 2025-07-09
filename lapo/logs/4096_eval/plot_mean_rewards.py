import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

# Directory containing the result files
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Regex to extract timestamp and mean reward
FILENAME_RE = re.compile(r'results_.*_(\d{8}_\d{6})\.txt$')
MEAN_REWARD_RE = re.compile(r'Mean Return: ([\d\.\-]+)')

def parse_results():
    data = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.startswith('results_') or not fname.endswith('.txt'):
            continue
        match = re.search(r'_(\d{8}_\d{6})\.txt$', fname)
        if not match:
            continue
        timestamp_str = match.group(1)
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        with open(os.path.join(RESULTS_DIR, fname), 'r') as f:
            for line in f:
                # Updated regex to match 'Mean Return: 0.70 ± 0.90'
                m = re.search(r'Mean Return: ([\d\.\-]+)', line)
                if m:
                    mean_reward = float(m.group(1))
                    data.append((timestamp, mean_reward))
                    break
    return sorted(data)

def plot_mean_rewards(data):
    if not data:
        print('No data found!')
        return
    # X-axis: epoch numbers (200, 400, 600, ...)
    epochs = [200 * (i + 1) for i in range(len(data))]
    means = [mean for _, mean in data]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, means, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward vs Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mean_rewards.png')
    print('Plot saved as mean_rewards.png')
    # plt.show()  # Do not show interactively

if __name__ == '__main__':
    data = parse_results()
    plot_mean_rewards(data) 