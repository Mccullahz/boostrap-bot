# Building and running the bot on Windows with RLBot GUI

This guide covers building the Go bot **directly on Windows** (no Docker) and running it from **RLBot GUI**.

---

## 1. Prerequisites

- **Go 1.21+**  
  - Download: [go.dev/dl](https://go.dev/dl/)  
  - Install and ensure `go` is on your PATH (`go version` in a terminal).

- **RLBot GUI**  
  - Install from [RLBot.org](https://rlbot.org/) or the [RLBot GitHub releases](https://github.com/RLBot/RLBot/releases).  
  - Start Rocket League and leave it at the main menu (or in a match) so RLBot can attach.

- **Optional (for ML / ONNX):**  
  - A built **ONNX model** (e.g. `model.onnx` from `ml/` training + export).  
  - **onnxruntime DLL** for Windows: download the [ONNX Runtime release](https://github.com/microsoft/onnxruntime/releases) (e.g. the Windows x64 package), and use the DLL from the `lib` folder (e.g. `onnxruntime.dll`). Without this, the bot still runs using the heuristic only.

---

## 2. Build the bot

Open a terminal (PowerShell or Command Prompt) and go to the repo root:

```powershell
cd C:\path\to\boostrap-bot
```

Build the executable:

```powershell
go build -o boostrap-bot.exe ./cmd/bot
```

This produces **`boostrap-bot.exe`** in the current directory. You can use another name or path (e.g. `bin\boostrap-bot.exe`) if you prefer; just use that path when configuring RLBot.

---

## 3. Run with RLBot GUI

RLBot runs **external (non-Python) bots** by starting your executable and expecting it to connect to RLBot over TCP. The bot receives game state and sends controller input over that connection.

### Step 1: Start RLBot GUI

- Start **RLBot GUI**.
- (Optional) Start **Rocket League** and get to the main menu or a match.

### Step 2: Add the bot as an executable / script

- In RLBot GUI, add a new bot (e.g. “Add” or “Configure”).
- Set the bot type to run an **external executable** (often labeled “Script” or “Executable” depending on the GUI version).
- **Command / executable path:**  
  Use the **full path** to your built exe, e.g.  
  `C:\path\to\boostrap-bot\boostrap-bot.exe`
- **Working directory (if the GUI has it):**  
  Set to the directory that contains `boostrap-bot.exe` (and, if you use ML, `model.onnx` and the onnxruntime DLL).  
  Example: `C:\path\to\boostrap-bot`

RLBot will pass the connection port (and sometimes a player index) via **command-line flags**. The RLBotGo library uses:

- `-player-index N` (player index)
- `-rlbot-version` and `-rlbot-dll-directory` (so the process starts correctly when launched by RLBot)

So you **do not** need to run the exe manually in most setups: RLBot GUI will start `boostrap-bot.exe` with the right flags and port.

### Step 3: Start a match

- Choose your bot (e.g. boostrap-bot) and add opponents if desired.
- Start the match. RLBot will:
  1. Launch `boostrap-bot.exe` with the correct port and player index.
  2. Your bot connects to RLBot, receives game state each tick, and sends back controller state.

If the bot fails to connect, check:

- RLBot GUI is actually running and the match has started.
- No firewall blocking localhost (e.g. `127.0.0.1:23234`).
- The executable path and working directory in the GUI are correct.

---

## 4. Optional: ML model and ONNX runtime

To use the **ML (ONNX) path** instead of the heuristic:

1. **Place the ONNX model**  
   Put your exported `model.onnx` in the **same directory as `boostrap-bot.exe`** (or set `BOOSTRAP_MODEL_PATH` to its full path; see below).

2. **Provide the onnxruntime DLL**  
   The bot uses [onnxruntime_go](https://github.com/yalue/onnxruntime_go), which loads the ONNX Runtime C library at runtime:
   - Download the matching [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) Windows x64 build.
   - Put **`onnxruntime.dll`** (from the release’s `lib` folder) in the same directory as `boostrap-bot.exe`, **or** set the environment variable **`ONNXRUNTIME_SHARED_LIBRARY_PATH`** to the full path of that DLL (e.g. in RLBot’s run configuration or system env).

3. **Environment variables (optional)**  
   - **`BOOSTRAP_MODEL_PATH`**  
     Path to `model.onnx`. If unset, the bot looks for `model.onnx` in the current working directory (the “working directory” RLBot uses when it starts the exe).
   - **`ONNXRUNTIME_SHARED_LIBRARY_PATH`**  
     Path to `onnxruntime.dll` if it’s not next to the exe.

If the model or DLL is missing or invalid, the bot falls back to the **heuristic** (touch-count jump behavior) and does not require ONNX to run.

---

## 5. Summary

| Step | Action |
|------|--------|
| 1 | Install Go and RLBot GUI. |
| 2 | `go build -o boostrap-bot.exe ./cmd/bot` in repo root. |
| 3 | In RLBot GUI, add bot → set executable to `boostrap-bot.exe` and working dir to the exe’s folder. |
| 4 | Start match; RLBot launches the exe and the bot connects. |
| 5 | (Optional) Put `model.onnx` and `onnxruntime.dll` in the same folder (or set env vars) to use the ML path. |

No Docker is required for this flow; everything runs natively on Windows with RLBot GUI.
