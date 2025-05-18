package forward

import (
	"github.com/OfflineBot/gomat/matrix2"
)


func (m* Linear) Backward(error matrix2.Matrix2[float32], other_weight matrix2.Matrix2[float32]) {
	other_weight = other_weight.T()
	weight_dot := error.Dot(&other_weight)
	z, e := m.GetZOutput()
	if e != nil {
		panic("Matrix not initialized yet!")
	}
	deriv := m.derivative(z)
	weight_dot.MulMatrix2(&deriv)
	m.delta = weight_dot
}

