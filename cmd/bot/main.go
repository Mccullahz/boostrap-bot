// bot connection and control for boostrap-bot.
// uses RLBotGo; supports heuristic fallback and optional ML (ONNX) inference.
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	RLBot "github.com/Trey2k/RLBotGo"

	"boostrap-bot/internal/control"
	"boostrap-bot/internal/inference"
	obspkg "boostrap-bot/internal/obs"
)

var (
	lastTouch    float32
	totalTouches int

	// mlInferencer is set at startup if model.onnx (or BOOSTRAP_MODEL_PATH) loads successfully.
	mlInferencer *inference.Inferencer
	obsBuf       []float32
)

func getInput(gameState *RLBot.GameState, rlBot *RLBot.RLBot) *RLBot.ControllerState {
	out := &RLBot.ControllerState{}

	// ML path: build obs, run ONNX, convert action → ControllerState
	if mlInferencer != nil && obsBuf != nil && gameState != nil && gameState.GameTick != nil {
		obspkg.FromGameState(obsBuf, gameState, rlBot.PlayerIndex)
		action, err := mlInferencer.Run(obsBuf)
		if err == nil {
			control.ActionToControllerState(out, action)
			return out
		}
		// on inference error, fall through to heuristic
	}

	// heuristic fallback: touch-count → jump from Trey2k example
	wasJustTouched := false
	if gameState != nil && gameState.GameTick != nil &&
		gameState.GameTick.Ball.LatestTouch.GameSeconds != 0 &&
		lastTouch != gameState.GameTick.Ball.LatestTouch.GameSeconds {
		totalTouches++
		lastTouch = gameState.GameTick.Ball.LatestTouch.GameSeconds
		wasJustTouched = true
	}
	if wasJustTouched && totalTouches <= 10 {
		rlBot.DebugMessageAdd(fmt.Sprintf("The ball was touched %d times", totalTouches))
		out.Jump = false
	} else if wasJustTouched && totalTouches > 10 {
		rlBot.DebugMessageClear()
		totalTouches = 0
		out.Jump = true
	}
	return out
}

func main() {
	log.Println("Initializing boostrap-bot..")

	// ML: try to load ONNX model
	modelPath := os.Getenv("BOOSTRAP_MODEL_PATH")
	if modelPath == "" {
		modelPath = "model.onnx"
	}
	if abs, err := filepath.Abs(modelPath); err == nil {
		modelPath = abs
	}
	if _, err := os.Stat(modelPath); err == nil {
		inf, err := inference.New(modelPath)
		if err != nil {
			log.Printf("ML model load failed (using heuristic): %v", err)
		} else {
			mlInferencer = inf
			obsBuf = make([]float32, obspkg.OBS_SIZE)
			log.Println("ML model loaded:", modelPath)
		}
	} else {
		log.Println("No model file found, using heuristic control")
	}

	host := os.Getenv("RLBT_HOST")
	portStr := os.Getenv("RLBT_PORT")
	if host == "" {
		host = "127.0.0.1"
		log.Println("RLBT_HOST not set, defaulting to localhost")
	}
	if portStr == "" {
		portStr = "23234"
		log.Println("RLBT_PORT not set, defaulting to 23234")
	}

	port, err := strconv.Atoi(portStr)
	if err != nil {
		log.Fatal(err)
	}

	rlBot, err := RLBot.Connect(port)
	if err != nil {
		log.Println("Failed to connect to RLBot, make sure the RLBot framework is running and the host and port are correct")
		log.Fatal(err)
	}

	if err := rlBot.SendReadyMessage(true, true, true); err != nil {
		log.Fatal(err)
	}

	if err := rlBot.SetGetInput(getInput); err != nil {
		log.Fatal(err)
	}
}
