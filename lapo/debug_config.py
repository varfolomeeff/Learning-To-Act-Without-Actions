import torch
import sys
from omegaconf import OmegaConf
from config import Config

def debug_config():
    exp_name = "0_kanip-lipub"
    
    # Load the state dicts
    from paths import get_models_path
    models_path = get_models_path(exp_name)
    print(f"Loading from: {models_path}")
    
    state_dicts = torch.load(models_path, map_location='cpu', weights_only=False)
    print(f"Keys in state_dicts: {list(state_dicts.keys())}")
    
    # Check if policy key exists
    if "policy" in state_dicts:
        print("Policy key found in state_dicts")
    else:
        print("Policy key NOT found in state_dicts")
        print("Available keys:", list(state_dicts.keys()))
    
    # Examine the cfg structure
    cfg = state_dicts["cfg"]
    print(f"Type of cfg: {type(cfg)}")
    print(f"cfg keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else 'No keys'}")
    
    # Try to convert to dict
    if hasattr(cfg, '_content'):
        print(f"cfg._content: {cfg._content}")
    
    # Try to get the container
    if hasattr(cfg, '_get_container'):
        container = cfg._get_container()
        print(f"Container: {container}")
    
    # Try to convert to container
    try:
        cfg_container = OmegaConf.to_container(cfg)
        print(f"cfg_container: {cfg_container}")
    except Exception as e:
        print(f"Error converting to container: {e}")
    
    # Try the original config.get approach
    try:
        from config import get
        result_cfg = get(base_cfg=cfg)
        print("Successfully loaded config!")
        
        # Try to access policy from state_dicts
        if "policy" in state_dicts:
            print("Policy found, trying to create policy...")
            from utils import create_policy
            policy = create_policy(
                result_cfg.model,
                action_dim=result_cfg.model.la_dim,
                state_dict=state_dicts["policy"],
                strict_loading=True,
            )
            print("Successfully created policy!")
        else:
            print("No policy found in state_dicts")
            
    except Exception as e:
        print(f"Error in config.get: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config() 