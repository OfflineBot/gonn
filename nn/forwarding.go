
package nn

import (
	"github.com/OfflineBot/gomat/matrix2"
	"github.com/OfflineBot/gonn/activation"
)

func (m* NeuralNetwork) ForwardLinear(input *matrix2.Matrix2[float32]) matrix2.Matrix2[float32] {
	m.out_linear = forward.NewLinear(m.linear[len(m.linear)-1].GetOutputSize(), m.output_size, activation.ReLU, activation.DerivReLU)
	for i := range len(m.linear) {
		item := m.linear[i]
		item.Forward(input)
		a_value, a_e := item.GetAOutput()
		if a_e != nil {
			panic(a_e)
		}
		input = &a_value
	}

	m.out_linear.Forward(input)
	z_value, z_e := m.out_linear.GetZOutput()
	if z_e != nil {
		panic(z_e)
	}
	return z_value
}

