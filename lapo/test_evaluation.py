#!/usr/bin/env python3
"""Test script to verify evaluation setup works correctly"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import env_utils
from evaluate import evaluate_agent


def test_env_setup():
    """Test that environment setup works without normalization"""
    print("Testing environment setup...")
    
    try:
        env = env_utils.setup_procgen_env(
            num_envs=1,
            env_id="coinrun",
            gamma=0.99
        )
        
        # Test basic environment functionality
        obs = env.reset()
        print(f"✓ Environment created successfully")
        print(f"✓ Observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step([action])
            print(f"✓ Step {i+1}: reward = {reward[0]:.2f}, done = {done[0]}")
            if done[0]:
                break
        
        env.close()
        print("✓ Environment test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_policy_evaluation():
    """Test policy evaluation with a dummy policy"""
    print("\nTesting policy evaluation...")
    
    try:
        # Create a dummy policy
        class DummyPolicy(torch.nn.Module):
            def __init__(self, action_dim=15):
                super().__init__()
                self.action_dim = action_dim
            
            def forward(self, x):
                # Return random logits
                return torch.randn(x.shape[0], self.action_dim)
        
        policy = DummyPolicy()
        
        # Test evaluation
        mean_return, std_return, returns = evaluate_agent(
            policy, "coinrun", num_episodes=3, device='cpu'
        )
        
        print(f"✓ Policy evaluation completed")
        print(f"✓ Mean return: {mean_return:.2f}")
        print(f"✓ Number of episodes: {len(returns)}")
        print("✓ Policy evaluation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Policy evaluation test failed: {e}")
        return False


def main():
    print("Testing LAPO evaluation setup...")
    print("=" * 50)
    
    # Test environment setup
    env_ok = test_env_setup()
    
    # Test policy evaluation
    eval_ok = test_policy_evaluation()
    
    print("\n" + "=" * 50)
    if env_ok and eval_ok:
        print("✓ All tests passed! Evaluation setup is working correctly.")
        print("\nYou can now use:")
        print("  python quick_eval.py <exp_name> <env_name>")
        print("  python evaluate.py <exp_name> <env_name> [num_episodes] [device]")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 