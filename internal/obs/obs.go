// build the observation vector from RLBot GameState.
package obs

import (
	RLBot "github.com/Trey2k/RLBotGo"
)

// OBS_SIZE is the observation vector length. Must match ml/boostrap_bot/env/spaces.py.
const OBS_SIZE = 25

// layout: [0:3] ball pos, [3:6] ball vel, [6:9] car pos, [9:12] car vel,
// [12:15] car rotation (pitch,yaw,roll rad), [15:18] car ang vel, [18] boost, [19:25] pad.

// fill a slice of length OBS_SIZE from the given GameState for the given player index.
// if slice is nil or too short, FromGameState does nothing.
func FromGameState(dst []float32, gs *RLBot.GameState, playerIndex int32) {
	if gs == nil || gs.GameTick == nil || len(dst) < OBS_SIZE {
		return
	}
	tick := gs.GameTick

	// ball: position (3), velocity (3)
	ball := &tick.Ball.Physics
	dst[0] = ball.Location.X
	dst[1] = ball.Location.Y
	dst[2] = ball.Location.Z
	dst[3] = ball.Velocity.X
	dst[4] = ball.Velocity.Y
	dst[5] = ball.Velocity.Z

	// car: position (3), velocity (3), rotation (3), angular velocity (3), boost (1). [19:25] pad.
	for i := 6; i < OBS_SIZE; i++ {
		dst[i] = 0
	}
	if playerIndex >= 0 && int(playerIndex) < len(tick.Players) {
		car := &tick.Players[playerIndex]
		p := &car.Physics
		dst[6] = p.Location.X
		dst[7] = p.Location.Y
		dst[8] = p.Location.Z
		dst[9] = p.Velocity.X
		dst[10] = p.Velocity.Y
		dst[11] = p.Velocity.Z
		dst[12] = p.Rotation.Pitch
		dst[13] = p.Rotation.Yaw
		dst[14] = p.Rotation.Roll
		dst[15] = p.AngularVelocity.X
		dst[16] = p.AngularVelocity.Y
		dst[17] = p.AngularVelocity.Z
		if car.Boost >= 0 && car.Boost <= 100 {
			dst[18] = float32(car.Boost)
		}
	}
	// [19:25] padding left as zero
}
