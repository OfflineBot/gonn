package forward

import (
	"fmt"

	"github.com/OfflineBot/gomat/matrix2"
)


func (m* Linear) Backward(error matrix2.Matrix2[float32], other_weight matrix2.Matrix2[float32]) {
	fmt.Println("Shapes:")
	fmt.Println(error.Shape())
	error = error.T()
	fmt.Println(error.Shape())
	other_weight.Shape()
	weight_dot := error.Dot(&other_weight)
	fmt.Println("Done")
	z, e := m.GetZOutput()
	if e != nil {
		panic("Matrix not initialized yet!")
	}
	deriv := m.derivative(z)
	weight_dot.MulMatrix2(&deriv)
	m.delta = weight_dot
}

