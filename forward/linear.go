package forward

import (
	"github.com/OfflineBot/gomat/matrix1"
	"github.com/OfflineBot/gomat/matrix2"
	"github.com/OfflineBot/gonn/activation"
)

type Linear struct {
	activation activation.Activation
	input, output int
	weight matrix2.Matrix2[float32]
	bias matrix1.Matrix1[float32]
}


func NewLinear(input_size, output_size int, activ activation.Activation) Linear {
	return Linear{
		activation: activ,
		input: input_size,
		output: output_size,
		weight: *matrix2.RandomMatrix2[float32](input_size, output_size, -1.0, 1.0),
		bias: *matrix1.RandomMatrix1[float32](output_size, -1.0, 1.0),
	}
}

func (m* Linear) GetInputSize() int {
	return m.input
}

func (m* Linear) GetOutputSize() int {
	return m.output
}
