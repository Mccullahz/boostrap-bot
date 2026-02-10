"""
Export Stable-Baselines3 policy to ONNX for inference in Go (onnxruntime-go).

Usage:
    python -m boostrap_bot.export_onnx path/to/model.zip path/to/output.onnx

The exported graph has a single input "obs" (shape [1, OBS_SIZE]) and outputs
"action" (shape [1, ACTION_SIZE]). Exports the deterministic (mean) action path.
"""

from pathlib import Path
import argparse
import numpy as np
import torch.nn as nn

from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.distributions import DiagGaussianDistribution

from boostrap_bot.env import RocketLeagueEnv
from boostrap_bot.env.spaces import OBS_SIZE, ACTION_SIZE


def _make_dummy_env():
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv([lambda: RocketLeagueEnv(max_episode_steps=1000)])


class DeterministicPolicyWrapper(nn.Module):
    """
    Wraps an SB3 policy so we can export obs -> deterministic action to ONNX.
    Chains: obs -> features -> latent_pi -> action_net(latent_pi) = mean.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        features = self.policy.extract_features(
            obs, self.policy.features_extractor if self.policy.share_features_extractor else self.policy.pi_features_extractor
        )
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
        mean_actions = self.policy.action_net(latent_pi)
        if isinstance(self.policy.action_dist, DiagGaussianDistribution):
            return mean_actions
        return mean_actions


def export_policy_to_onnx(
    model_path: str | Path,
    onnx_path: str | Path,
    opset_version: int = 14,
) -> None:
    """
    Load an SB3 model and export its policy (deterministic forward) to ONNX.
    """
    import torch

    model_path = Path(model_path)
    onnx_path = Path(onnx_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    for algo, loader in [(PPO, PPO.load), (SAC, SAC.load), (DDPG, DDPG.load)]:
        try:
            model = loader(str(model_path))
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not load model as PPO, SAC, or DDPG: {model_path}")

    policy = model.policy
    policy.set_training_mode(False)
    wrapper = DeterministicPolicyWrapper(policy)

    dummy_obs = np.zeros((1, OBS_SIZE), dtype=np.float32)
    obs_t = torch.from_numpy(dummy_obs)

    torch.onnx.export(
        wrapper,
        obs_t,
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset_version,
    )
    print(f"Exported policy to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export SB3 policy to ONNX")
    parser.add_argument("model", type=str, help="Path to saved model (.zip)")
    parser.add_argument("output", type=str, help="Path to output .onnx file")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()
    export_policy_to_onnx(args.model, args.output, opset_version=args.opset)


if __name__ == "__main__":
    main()
