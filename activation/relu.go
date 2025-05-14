
package activation

import(
	"github.com/OfflineBot/gomat/matrix2"
)


func ReLU(x matrix2.Matrix2[float32]) matrix2.Matrix2[float32] {
	out := matrix2.EmptyMatrix2[float32](x.Shape()[0], x.Shape()[1])
	for i := range x.Shape()[0] {
		for j := range x.Shape()[1] {
			val := x.GetValue(i, j)
			if val <= 0 { 
				out.SetValue(0.0, i, j)
			} else {
				out.SetValue(val, i, j)
			}
		}
	}
	return *out
}


func DerivReLU(x matrix2.Matrix2[float32]) matrix2.Matrix2[float32] {
	out := matrix2.EmptyMatrix2[float32](x.Shape()[0], x.Shape()[1])
	for i := range x.Shape()[0] {
		for j := range x.Shape()[1] {
			val := x.GetValue(i, j)
			if val <= 0 { 
				out.SetValue(0.0, i, j)
			} else {
				out.SetValue(1.0, i, j)
			}
		}
	}
	return *out
}

