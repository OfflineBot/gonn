package nn

import (
	"github.com/OfflineBot/gomat/matrix2"
)

func (m* NeuralNetwork) BackwardLinear(output matrix2.Matrix2[float32], truth matrix2.Matrix2[float32]) {
	output.SubMatrix2(&truth)
	m.out_linear.SetDelta(output)
	last_delta := output
	last_weight := m.out_linear.GetWeight()

	for i := len(m.linear) - 1; i >= 0; i-- {
		item := m.linear[i]
		item.Backward(last_delta, last_weight)
		
		last_delta = item.GetDelta()
		last_weight = item.GetWeight()
	}
}

