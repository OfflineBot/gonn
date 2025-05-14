package nn

import (
	"fmt"

	"github.com/OfflineBot/gomat/matrix2"
	"github.com/OfflineBot/gonn/activation"
	"github.com/OfflineBot/gonn/forward"
)

type NeuralNetwork struct {
	input_size, output_size int
	linear []forward.Linear
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

func (m* NeuralNetwork) ForwardLinear(input matrix2.Matrix2[float32]) matrix2.Matrix2[float32] {
	m.out_linear = forward.NewLinear(m.linear[len(m.linear)-1].GetOutputSize(), m.output_size, activation.ReLU, activation.DerivReLU)
	fmt.Println("0 Iteration")
	for i := range len(m.linear) {
		fmt.Println(fmt.Sprintf("%d from %d", i, len(m.linear) - 1))
		item := m.linear[i]
		item.Forward(input)
		a_value, a_e := item.GetAOutput()
		if a_e != nil {
			panic(a_e)
		}
		input = a_value
	}

	fmt.Println("Done")
	m.out_linear.Forward(input)
	z_value, z_e := m.out_linear.GetZOutput()
	if z_e != nil {
		panic(z_e)
	}
	return z_value
}

func (m* NeuralNetwork) PrintLayout() {
	fmt.Println("Network:")
	output_size := 0
	for i := range len(m.linear) {
		input_size := m.linear[i].GetInputSize()
		output_size = m.linear[i].GetOutputSize()
		fmt.Println(fmt.Sprintf("- Linear: %d %d", input_size, output_size))
	}
	fmt.Println(fmt.Sprintf("- Linear: %d %d", output_size, m.output_size))
	fmt.Println("Done")
}

