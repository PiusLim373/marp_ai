import os
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from marp_ai_gym_rllib import MarpAIGym
from masked_fcnet_model import MaskedFCNet

# Register custom model
ModelCatalog.register_custom_model("masked_fcnet", MaskedFCNet)

# Register custom environment
env_name = "marp_ai_env"
register_env(env_name, lambda config: MarpAIGym(config=config, render_flag=False))

# Path to save models
checkpoint_path = os.path.join(os.getcwd(), "models")

# Define PPO config with action-masked custom model
config = (
    PPOConfig()
    .environment(env=env_name)
    .framework("torch")
    .rollouts(num_rollout_workers=8)  # Adjust based on your CPU
    .training(
        model={
            "custom_model": "masked_fcnet",
            "fcnet_hiddens": [128, 128, 128, 128],
            "fcnet_activation": "relu",
        },
        lr=5e-4,
        train_batch_size=800,
        sgd_minibatch_size=200,
        num_sgd_iter=10,
        clip_param=0.2,
        gamma=0.99,
        lambda_=0.95,
        use_gae=True,
        vf_loss_coeff=1.0,
        entropy_coeff=0.01,
    )
)
config.preprocessor_pref = None
config.model["_disable_preprocessor_api"] = True

# Run training using Ray Tune
tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        name="PPO_MARP_AI_TRAIN",
        stop={"training_iteration": 4001},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=100,
            checkpoint_at_end=True,
        ),
        storage_path=checkpoint_path,
    ),
).fit()
