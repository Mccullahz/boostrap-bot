// baseline bot connection and initialization for boostrap-bot, currently following the example from Trey2k's RLBotGo repository, will be modified to fit the needs of boostrap-bot as development continues
package main

import (
	"log"
	"fmt"
	"os"
	"strconv"

	RLBot "github.com/Trey2k/RLBotGo"
)

var lastTouch float32
var totalTouches int = 0

func getInput(gameState *RLBot.GameState, rlBot *RLBot.RLBot) *RLBot.ControllerState {
	PlayerInput := &RLBot.ControllerState{}

	// Count ball touches up to 10 and on 11 clear the messages and jump
	wasjustTouched := false
	if gameState.GameTick.Ball.LatestTouch.GameSeconds != 0 && lastTouch != gameState.GameTick.Ball.LatestTouch.GameSeconds {
		totalTouches++
		lastTouch = gameState.GameTick.Ball.LatestTouch.GameSeconds
		wasjustTouched = true
	}

	if wasjustTouched && totalTouches <= 10 {
		// DebugMessage is a helper function to let you quickly get debug text on screen. it will automatically place it so text will not overlap
		rlBot.DebugMessageAdd(fmt.Sprintf("The ball was touched %d times", totalTouches))
		PlayerInput.Jump = false
	} else if wasjustTouched && totalTouches > 10 {
		rlBot.DebugMessageClear()
		totalTouches = 0
		PlayerInput.Jump = true
	}
	return PlayerInput
}

func main() {
	log.Println("Initializing boostrap-bot..")

	// connect to RLBot -- cross platform this doesnt work, so we need to env var the connection
	host := os.Getenv("RLBT_HOST")
	portStr := os.Getenv("RLBT_PORT")

	// defaults if env vars are not set
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

	// this should use host and port, only using port for now, this is likely to break -- circle back to this
	rlBot, err := RLBot.Connect(port)
	if err != nil {
		log.Println("Failed to connect to RLBot, make sure the RLBot framework is running and the host and port are correct")
		log.Fatal(err)
	}

	// Send ready message
	err = rlBot.SendReadyMessage(true, true, true)
	if err != nil {
		panic(err)
	}

	// Set our tick handler
	err = rlBot.SetGetInput(getInput)
	fmt.Println(err.Error())
}


