// loads an ONNX policy and runs inference (obs -> action).
// requires onnxruntime shared library; see docs/build-windows.md.
package inference

import (
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"boostrap-bot/internal/control"
	obspkg "boostrap-bot/internal/obs"
)

const (
	inputName  = "obs"
	outputName = "action"
)

// inferencer runs ONNX inference. safe for use from a single goroutine at a time
// call Run from the same goroutine that calls getInput.
type Inferencer struct {
	mu       sync.Mutex
	input    *ort.Tensor[float32]
	output   *ort.Tensor[float32]
	session  *ort.AdvancedSession
	obsBuf   []float32
	actionBuf []float32
}

// load the ONNX model from modelPath and initializes the ONNX runtime.
// if fail, return nil, error, then use heuristic control only.
func New(modelPath string) (*Inferencer, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is empty")
	}
	if path := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); path != "" {
		ort.SetSharedLibraryPath(path)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("onnx runtime init: %w", err)
	}
	// input [1, OBS_SIZE], output [1, ACTION_SIZE]
	inputShape := ort.NewShape(1, obspkg.OBS_SIZE)
	outputShape := ort.NewShape(1, control.ACTION_SIZE)
	obsData := make([]float32, obspkg.OBS_SIZE)
	inputTensor, err := ort.NewTensor(inputShape, obsData)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	session, err := ort.NewAdvancedSession(modelPath,
		[]string{inputName}, []string{outputName},
		[]ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("create session: %w", err)
	}
	return &Inferencer{
		input:     inputTensor,
		output:    outputTensor,
		session:   session,
		obsBuf:    obsData,
		actionBuf: make([]float32, control.ACTION_SIZE),
	}, nil
}

// copies obs into the input tensor, runs the session, returns action slice.
// obs must have length at least obspkg.OBS_SIZE; only the first OBS_SIZE elements are used.
// do not retain returned slice.
func (inf *Inferencer) Run(obs []float32) (action []float32, err error) {
	if inf == nil {
		return nil, fmt.Errorf("inferencer is nil")
	}
	inf.mu.Lock()
	defer inf.mu.Unlock()
	n := obspkg.OBS_SIZE
	if len(obs) < n {
		n = len(obs)
	}
	copy(inf.obsBuf, obs[:n])
	for i := n; i < len(inf.obsBuf); i++ {
		inf.obsBuf[i] = 0
	}
	if err := inf.session.Run(); err != nil {
		return nil, fmt.Errorf("session run: %w", err)
	}
	outData := inf.output.GetData()
	copy(inf.actionBuf, outData)
	return inf.actionBuf, nil
}

func (inf *Inferencer) Close() error {
	if inf == nil {
		return nil
	}
	inf.mu.Lock()
	defer inf.mu.Unlock()
	var err error
	if inf.session != nil {
		err = inf.session.Destroy()
		inf.session = nil
	}
	if inf.input != nil {
		inf.input.Destroy()
		inf.input = nil
	}
	if inf.output != nil {
		inf.output.Destroy()
		inf.output = nil
	}
	return err
}
