package forward

import (
	"fmt"
	"github.com/OfflineBot/gomat/matrix1"
	"github.com/OfflineBot/gomat/matrix2"
)

type Linear struct {
	activation, derivative func(matrix2.Matrix2[float32]) matrix2.Matrix2[float32]
	input, output int
	weight matrix2.Matrix2[float32]
	bias matrix1.Matrix1[float32]

	z *matrix2.Matrix2[float32]
	a *matrix2.Matrix2[float32]
}


func NewLinear(input_size, output_size int, activ, deriv func(matrix2.Matrix2[float32]) matrix2.Matrix2[float32]) Linear {
	return Linear{
		activation: activ,
		derivative: deriv,
		input: input_size,
		output: output_size,
		weight: *matrix2.RandomMatrix2[float32](input_size, output_size, -1.0, 1.0),
		bias: *matrix1.RandomMatrix1[float32](output_size, -1.0, 1.0),
		z: nil,
		a: nil,
	}
}

func (m* Linear) Forward(x matrix2.Matrix2[float32]) {
	dot_product := x.Dot(&m.weight)
	dot_product.AddMatrix1(&m.bias)
	m.z = dot_product
	a := m.activation(*dot_product)
	m.a = &a
}

func (m* Linear) GetInputSize() int {
	return m.input
}

func (m* Linear) GetOutputSize() int {
	return m.output
}

func (m* Linear) GetZOutput() (matrix2.Matrix2[float32], error) {
	if m.z == nil {
		empty_matrix := matrix2.EmptyMatrix2[float32](0, 0)
		return *empty_matrix, fmt.Errorf("Matrix not created yet!")
	} else {
		return *m.z, nil
	}
}

func (m* Linear) GetAOutput() (matrix2.Matrix2[float32], error) {
	if m.a == nil {
		empty_matrix := matrix2.EmptyMatrix2[float32](0, 0)
		return *empty_matrix, fmt.Errorf("Matrix not created yet!")
	} else {
		return *m.a, nil
	}
}

func (m* Linear) GetWeight() matrix2.Matrix2[float32] {
	return m.weight
}

func (m* Linear) GetBias() matrix1.Matrix1[float32] {
	return m.bias
}


