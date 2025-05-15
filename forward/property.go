
package forward

import (
	"fmt"
	"github.com/OfflineBot/gomat/matrix2"
	"github.com/OfflineBot/gomat/matrix1"
)

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


