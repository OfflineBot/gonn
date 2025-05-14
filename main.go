package main

import (
	"github.com/OfflineBot/gonn/nn"
	"github.com/OfflineBot/gonn/activation"
)

func X(x float32) float32 {
	return x
}

func main() {
	x := nn.NewNeuralNetwork()
	x.AddLinear(3, 4, &activation.ReLU{})
	x.PrintLayout()
}

