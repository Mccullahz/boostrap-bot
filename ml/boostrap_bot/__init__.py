"""
boostrap_bot ML package.

gymn environments and training utilities for the Rocket League bot.
state/action dimensions here are the single source of truth for Go inference.
"""

from boostrap_bot.env.spaces import OBS_SIZE, ACTION_SIZE

__all__ = ["OBS_SIZE", "ACTION_SIZE"]
