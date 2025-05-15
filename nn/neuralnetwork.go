package nn

import (
	"github.com/OfflineBot/gomat/matrix2"
	"github.com/OfflineBot/gonn/forward"
)

type NeuralNetwork struct {
	// input/output shape 
	input_size, output_size int
	// hidden layer
	linear []forward.Linear
	// last layer
	out_linear forward.Linear
}


func NewNeuralNetwork(input_size, output_size int) NeuralNetwork {
	return NeuralNetwork {
		input_size: input_size,
		output_size: output_size,
		linear: []forward.Linear{},
	}
}

func (m* NeuralNetwork) AddLinear(layer_size int, activ, deriv func(matrix2.Matrix2[float32]) matrix2.Matrix2[float32]) {
	input_size := 0
	if len(m.linear) == 0 {
		input_size = m.input_size
	} else {
		input_size = m.linear[len(m.linear)-1].GetOutputSize()
	}
	m.linear = append(m.linear, forward.NewLinear(input_size, layer_size, activ, deriv))
}


