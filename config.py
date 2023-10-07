import os
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_config():
    # This file only contains model specific hyperparameters
    config = {
        # Key Hyperparameter
        "eval_weights": "C:/Users/isxzl/OneDrive/Code/multimodal-across-domains-gaze-target-detection/pretrained/best_gazefollow_gazefollow.pth",
        "device": "cuda",
        "batch_size": 4,
        "num_workers": min(8, os.cpu_count()),

        # Run metadata
        "tag": "default",
        "input_size": 224,
        "output_size": 224,

        # Training args
        "init_weights": None,
        "lr": 2.5e-4,
        "epochs": 70,
        "evaluate_every": 1,
        "save_every": 1,
        "print_every": 10,
        "no_resume": False,
        "output_dir": "output",
        "amp": None,
        "channels_last": False,
        "freeze_scene": False,
        "freeze_face": False,
        "freeze_depth": False,
        "head_da": False,
        "rgb_depth_da": False,
        "task_loss_amp_factor": 1,
        "inout_loss_amp_factor": 0,
        "rgb_depth_source_loss_amp_factor": 1,
        "rgb_depth_target_loss_amp_factor": 1,
        "adv_loss_amp_factor": 1,
        "no_wandb": False,
        "no_save": False
    }
    return Config(**config)

def update_config(config, dataset_dir, device):

    config.dataset_dir=dataset_dir
    config.device = device
    return config