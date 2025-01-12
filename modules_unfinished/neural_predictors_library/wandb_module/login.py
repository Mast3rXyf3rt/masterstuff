import wandb
import os

def login_wandb(api_key=None):

    try:
        from wandb_module.KEY import my_key
        api_key = my_key
    except:
        pass

    """
    Log in to wandb using the provided API key or environment variable.
    
    Args:
        api_key (str): Optional. If not provided, it will use environment variable WAND_API_KEY.
    """
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    # Ensure API key is set up
    if "WANDB_API_KEY" not in os.environ:
        raise ValueError("WandB API key is required. Please provide it as a parameter or set it in environment variables.")
    
    # Perform wandb login
    wandb.login()

