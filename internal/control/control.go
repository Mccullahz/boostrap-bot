// converts the ML action vector into RLBot ControllerState.
package control

import (
	RLBot "github.com/Trey2k/RLBotGo"
)

// ACTION_SIZE is the action vector length. MUST match ml/boostrap_bot/env/spaces.py.
const ACTION_SIZE = 7

// IndexThrottle, IndexSteer, ... index into the action slice.
const (
	IndexThrottle = 0
	IndexSteer    = 1
	IndexPitch    = 2
	IndexYaw      = 3
	IndexRoll     = 4
	IndexJump     = 5
	IndexBoost    = 6
)

// JumpBoostThreshold: action value above this is treated as true for Jump/Boost.
const JumpBoostThreshold = 0.5

// ActionToControllerState fills out from the 7-float action vector.
// throttle, steer, pitch, yaw, roll are clipped to [-1, 1] no clue if this is correct.
// jump and boost are true when the corresponding value is > JumpBoostThreshold.
// if action is nil or shorter than ACTION_SIZE, out is zeroed (no input).
func ActionToControllerState(out *RLBot.ControllerState, action []float32) {
	if out == nil {
		return
	}
	*out = RLBot.ControllerState{}
	if len(action) < ACTION_SIZE {
		return
	}
	clip := func(v float32) float32 {
		if v < -1 {
			return -1
		}
		if v > 1 {
			return 1
		}
		return v
	}
	out.Throttle = clip(action[IndexThrottle])
	out.Steer = clip(action[IndexSteer])
	out.Pitch = clip(action[IndexPitch])
	out.Yaw = clip(action[IndexYaw])
	out.Roll = clip(action[IndexRoll])
	out.Jump = action[IndexJump] > JumpBoostThreshold
	out.Boost = action[IndexBoost] > JumpBoostThreshold
}
