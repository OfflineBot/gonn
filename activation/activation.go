package activation

import "github.com/OfflineBot/gomat/matrix2"


type Activation interface {
	Activ(matrix2.Matrix2[float32]) matrix2.Matrix2[float32]
	Deriv(matrix2.Matrix2[float32]) matrix2.Matrix2[float32]
}


