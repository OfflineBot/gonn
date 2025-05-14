
package nn

import (
	"github.com/OfflineBot/gonn/forward"
	"github.com/OfflineBot/gonn/activation"
	"fmt"
)

type NeuralNetwork struct {
	linear []forward.Linear
}


func NewNeuralNetwork() NeuralNetwork {
	return NeuralNetwork {
		linear: []forward.Linear{},
	}
}

func (m* NeuralNetwork) AddLinear(input_size, output_size int, activ activation.Activation) {
	m.linear = append(m.linear, forward.NewLinear(input_size, output_size, activ))
}

func (m* NeuralNetwork) PrintLayout() {
	fmt.Println("Network:")
	fmt.Println(fmt.Sprintf("- InputSize: x - x"))
	for i := range len(m.linear) {
		fmt.Println(fmt.Sprintf("- Linear: %d %d", m.linear[i].GetInputSize(), m.linear[i].GetOutputSize()))
	}
	fmt.Println("- OutputSize: y - y")
	fmt.Println("Done")
}

