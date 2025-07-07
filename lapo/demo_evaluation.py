#!/usr/bin/env python3
"""Demo script showing the difference between normalized and raw rewards"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import env_utils


def demo_reward_comparison():
    """Demonstrate the difference between normalized and raw rewards"""
    
    print("Демонстрация разницы между нормализованными и чистыми наградами")
    print("=" * 70)
    
    # Примеры наград для разных сред
    env_examples = {
        "coinrun": {"raw_rewards": [2.0, 5.0, 8.0, 10.0]},
        "bigfish": {"raw_rewards": [5.0, 15.0, 25.0, 35.0]},
        "fruitbot": {"raw_rewards": [-5.0, 0.0, 10.0, 25.0]},
    }
    
    for env_name, data in env_examples.items():
        print(f"\nСреда: {env_name}")
        print("-" * 40)
        
        urp = env_utils.urp_ep_return[env_name]
        expert = env_utils.expert_ep_return[env_name]
        
        print(f"URP (случайная политика): {urp:.2f}")
        print(f"Expert (экспертная политика): {expert:.2f}")
        print()
        
        print("Сырые награды -> Нормализованные награды:")
        for raw_reward in data["raw_rewards"]:
            # Старая нормализация (удаленная)
            normalized = (raw_reward - urp) / (expert - urp)
            normalized = np.clip(normalized, 0, 1)
            
            print(f"  {raw_reward:6.1f} -> {normalized:6.3f}")
        
        print(f"\nТеперь мы используем только сырые награды: {raw_reward:.1f}")
    
    print("\n" + "=" * 70)



def demo_evaluation_usage():
    """Show how to use the new evaluation scripts"""
    
    print("\n\nКак использовать новые скрипты оценки:")
    print("=" * 50)
    
    print("\n1. Быстрая оценка (10 эпизодов):")
    print("   python quick_eval.py my_experiment coinrun")
    
    print("\n2. Полная оценка (настраиваемое количество):")
    print("   python evaluate.py my_experiment coinrun 100 cuda:0")
    
    print("\n3. Тестирование установки:")
    print("   python test_evaluation.py")
    
    print("\n4. Демонстрация (этот скрипт):")
    print("   python demo_evaluation.py")


if __name__ == "__main__":
    demo_reward_comparison()
    demo_evaluation_usage()
    
    print("\n\nГотово! Теперь вы можете оценивать агентов на чистых наградах среды.") 