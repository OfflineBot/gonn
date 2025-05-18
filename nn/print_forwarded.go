package nn

import "fmt"


func (m* NeuralNetwork) PrintFowarded() {
	for i := range len(m.linear) {
		fmt.Print("Iteration: ")
		fmt.Println(i)
		item := m.linear[i]
		z, _ := item.GetZOutput()
		z.Println()
	}
}

