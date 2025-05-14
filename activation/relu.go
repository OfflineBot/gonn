
package activation

import(
	"github.com/OfflineBot/gomat/matrix2"
)

type ReLU struct{}

func (m *ReLU) Activ(x matrix2.Matrix2[float32]) matrix2.Matrix2[float32] {
	for i := range x.Shape()[0] {
		for j := range x.Shape()[1] {
			val := x.GetValue(i, j)
			if val <= 0 { 
				x.SetValue(0.0, i, j)
			}
		}
	}
	return x
}


func (m* ReLU) Deriv(x matrix2.Matrix2[float32]) matrix2.Matrix2[float32] {
	for i := range x.Shape()[0] {
		for j := range x.Shape()[1] {
			val := x.GetValue(i, j)
			if val <= 0 { 
				x.SetValue(0.0, i, j)
			} else {
				x.SetValue(1.0, i, j)
			}
		}
	}
	return x
}

