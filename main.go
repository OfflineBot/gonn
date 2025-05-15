package main

import (
	"github.com/OfflineBot/gomat/matrix2"
	"github.com/OfflineBot/gonn/activation"
	"github.com/OfflineBot/gonn/nn"
)


// Testing Area
func main() {
	input := matrix2.NewMatrix2([][]float32 {
		{1.0, 2.0},
		{4.0, 3.0},
	});

	nn := nn.NewNeuralNetwork(2, 1)

	nn.AddLinear(2, activation.ReLU, activation.DerivReLU)

	out := nn.ForwardLinear(input)
	nn.BackwardLinear()
	nn.UpdateParameter()

	out.Println()
}

