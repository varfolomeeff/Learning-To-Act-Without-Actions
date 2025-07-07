#!/usr/bin/env python3
"""Quick evaluation script"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate import load_and_evaluate_policy

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python quick_eval.py <exp_name> <env_name>")
        print("Example: python quick_eval.py my_experiment coinrun")
        sys.exit(1)
    
    exp_name = sys.argv[1]
    env_name = sys.argv[2]
    
    print(f"Quick evaluation of {exp_name} on {env_name}")
    print("Running 10 episodes...")
    
    mean_return, std_return, returns = load_and_evaluate_policy(
        exp_name, env_name, num_episodes=10, device='cuda'
    )
    
    print(f"\nFinal Result: {mean_return:.2f} ± {std_return:.2f}") 