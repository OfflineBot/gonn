
package nn

import "fmt"

func (m* NeuralNetwork) PrintLayout() {
	fmt.Println("Network:")
	output_size := 0
	for i := range len(m.linear) {
		input_size := m.linear[i].GetInputSize()
		output_size = m.linear[i].GetOutputSize()
		fmt.Println(fmt.Sprintf("- Linear: %d %d", input_size, output_size))
	}
	fmt.Println(fmt.Sprintf("- Linear: %d %d", output_size, m.output_size))
	fmt.Println("Done")
}

