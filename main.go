package main

import (
	"github.com/OfflineBot/gomat/matrix2"
	"github.com/OfflineBot/gonn/activation"
	"github.com/OfflineBot/gonn/nn"
)


func main() {
	a := matrix2.NewMatrix2([][]float32 {
		[]float32 {1.0, 1.0},
		[]float32 {1.0, 1.0},
	});

	x := nn.NewNeuralNetwork(2, 1)
	x.AddLinear(4, activation.ReLU, activation.DerivReLU)
	x.AddLinear(8, activation.ReLU, activation.DerivReLU)
	x.AddLinear(16, activation.ReLU, activation.DerivReLU)
	x.PrintLayout()
	out := x.ForwardLinear(*a)
	out.Println()
}

