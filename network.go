package mtcnn

import (
	"math"
)

type Data2D struct {
	Width, Height int
	Channels      int
	Values        []float32
}

func NewData2D(width, height, channels int) Data2D {
	return Data2D{
		Width: width, Height: height,
		Channels: channels,
		Values:   make([]float32, width*height*channels),
	}
}

func (d Data2D) TotalSize() int {
	return d.Width * d.Height * d.Channels
}

func (d Data2D) GetIndex(x, y, channel int) int {
	return x*(d.Height*d.Channels) + y*d.Channels + channel
}

type Layer2D interface {
	Apply(Data2D) Data2D
}

type Conv2D struct {
	kerWidth, kerHeight int
	channels            int
	outputs             int

	// weights is of size kerWidth*kerHeight*channels*outputs, and is indexed in this order.
	weights []float32

	// biases is of size outputs.
	biases []float32
}

func NewConv2D(kerWidth, kerHeight, channels, outputs int, weights []float32, biases []float32) *Conv2D {
	if kerWidth*kerHeight*channels*outputs != len(weights) {
		panic("number of weights does not match the specified kernel")
	}
	if outputs != len(biases) {
		panic("number of biases does not match the specified outputs")
	}

	return &Conv2D{
		kerWidth:  kerWidth,
		kerHeight: kerHeight,
		channels:  channels,
		outputs:   outputs,
		weights:   weights,
		biases:    biases,
	}
}

func (c *Conv2D) indexWeight(x, y, channel, output int) int {
	return (x * c.kerHeight * c.channels * c.outputs) +
		(y * c.channels * c.outputs) +
		(channel * c.outputs) + output
}

func (c *Conv2D) apply33(input Data2D) (result Data2D) {
	if input.Width < 3 || input.Height < 3 {
		panic("input's dimensions are too small")
	}
	if input.Channels != c.channels {
		panic("input has a different number of channels")
	}
	result = NewData2D(input.Width-2, input.Height-2, c.outputs)

	for x := 0; x < result.Width; x++ {
		for y := 0; y < result.Height; y++ {
			resIndex := result.GetIndex(x, y, 0)
			kerIndex := input.GetIndex(x, y, 0)

			for output := 0; output < c.outputs; output++ {
				result.Values[resIndex+output] = c.biases[output]
			}

			for channel := 0; channel < c.channels; channel++ {
				xycWeightIndex00 := c.indexWeight(0, 0, channel, 0)
				xycWeightIndex01 := c.indexWeight(0, 1, channel, 0)
				xycWeightIndex02 := c.indexWeight(0, 2, channel, 0)
				xycWeightIndex10 := c.indexWeight(1, 0, channel, 0)
				xycWeightIndex11 := c.indexWeight(1, 1, channel, 0)
				xycWeightIndex12 := c.indexWeight(1, 2, channel, 0)
				xycWeightIndex20 := c.indexWeight(2, 0, channel, 0)
				xycWeightIndex21 := c.indexWeight(2, 1, channel, 0)
				xycWeightIndex22 := c.indexWeight(2, 2, channel, 0)

				inputValue00 := input.Values[kerIndex+input.GetIndex(0, 0, channel)]
				inputValue01 := input.Values[kerIndex+input.GetIndex(0, 1, channel)]
				inputValue02 := input.Values[kerIndex+input.GetIndex(0, 2, channel)]
				inputValue10 := input.Values[kerIndex+input.GetIndex(1, 0, channel)]
				inputValue11 := input.Values[kerIndex+input.GetIndex(1, 1, channel)]
				inputValue12 := input.Values[kerIndex+input.GetIndex(1, 2, channel)]
				inputValue20 := input.Values[kerIndex+input.GetIndex(2, 0, channel)]
				inputValue21 := input.Values[kerIndex+input.GetIndex(2, 1, channel)]
				inputValue22 := input.Values[kerIndex+input.GetIndex(2, 2, channel)]

				for output := 0; output < c.outputs; output++ {
					result.Values[resIndex+output] +=
						inputValue00*c.weights[xycWeightIndex00+output] +
							inputValue01*c.weights[xycWeightIndex01+output] +
							inputValue02*c.weights[xycWeightIndex02+output] +
							inputValue10*c.weights[xycWeightIndex10+output] +
							inputValue11*c.weights[xycWeightIndex11+output] +
							inputValue12*c.weights[xycWeightIndex12+output] +
							inputValue20*c.weights[xycWeightIndex20+output] +
							inputValue21*c.weights[xycWeightIndex21+output] +
							inputValue22*c.weights[xycWeightIndex22+output]
				}
			}
		}
	}

	return
}

func (c *Conv2D) Apply(input Data2D) (result Data2D) {
	// unrolled function
	if c.kerWidth == 3 && c.kerHeight == 3 {
		return c.apply33(input)
	}

	if input.Width < c.kerWidth || input.Height < c.kerHeight {
		panic("input's dimensions are too small")
	}
	if input.Channels != c.channels {
		panic("input has a different number of channels")
	}
	result = NewData2D(input.Width-c.kerWidth+1, input.Height-c.kerHeight+1, c.outputs)

	for x := 0; x < result.Width; x++ {
		xResIndex := result.GetIndex(x, 0, 0)
		for y := 0; y < result.Height; y++ {
			xyResIndex := xResIndex + result.GetIndex(0, y, 0)

			for kx := 0; kx < c.kerWidth; kx++ {
				xKerIndex := input.GetIndex(x+kx, 0, 0)
				xWeightIndex := c.indexWeight(kx, 0, 0, 0)
				for ky := 0; ky < c.kerHeight; ky++ {
					xyKerIndex := xKerIndex + input.GetIndex(0, y+ky, 0)
					xyWeightIndex := xWeightIndex + c.indexWeight(0, ky, 0, 0)

					for channel := 0; channel < c.channels; channel++ {
						xycWeightIndex := xyWeightIndex + c.indexWeight(0, 0, channel, 0)
						inputValue := input.Values[xyKerIndex+channel]

						for output := 0; output < c.outputs; output++ {
							result.Values[xyResIndex+output] +=
								inputValue * c.weights[xycWeightIndex+output]
						}
					}
				}
			}
		}
	}

	for output := 0; output < c.outputs; output++ {
		for x := 0; x < result.Width; x++ {
			for y := 0; y < result.Height; y++ {
				result.Values[result.GetIndex(x, y, output)] += c.biases[output]
			}
		}
	}

	return
}

type PReLU struct {
	// parameters is the size of channels.
	parameters []float32

	axis int
}

func NewPReLU(parameters []float32, axis int) *PReLU {
	return &PReLU{parameters: parameters, axis: axis}
}

func (pl *PReLU) Apply(input Data2D) (result Data2D) {
	var diff1, diff2, diff3 int
	var max1, max2, max3 int
	switch pl.axis {
	case 0:
		diff1, diff2, diff3 = input.Channels, 1, input.Height*input.Channels
		max1, max2, max3 = input.Height, input.Channels, input.Width
		if input.Width != len(pl.parameters) {
			panic("input has a different number of rows")
		}
	case 1:
		diff1, diff2, diff3 = input.Height*input.Channels, 1, input.Channels
		max1, max2, max3 = input.Width, input.Channels, input.Height
		if input.Height != len(pl.parameters) {
			panic("input has a different number of columns")
		}
	case 2:
		diff1, diff2, diff3 = input.Height*input.Channels, input.Channels, 1
		max1, max2, max3 = input.Width, input.Height, input.Channels
		if input.Channels != len(pl.parameters) {
			panic("input has a different number of channels")
		}
	default:
		panic("axis must be one of 0, 1, or 2")
	}

	result = NewData2D(input.Width, input.Height, input.Channels)
	for i3 := 0; i3 < max3; i3++ {
		p := pl.parameters[i3]
		for i1 := 0; i1 < max1; i1++ {
			for i2 := 0; i2 < max2; i2++ {
				idx := i1*diff1 + i2*diff2 + i3*diff3
				if value := input.Values[idx]; 0 < value {
					result.Values[idx] = value
				} else {
					result.Values[idx] = value * p
				}
			}
		}
	}

	return
}

func NewSoftmax(axis int) *Softmax {
	if !(axis == 0 || axis == 1 || axis == 2) {
		panic("axis must be one of 1, 2, or 3")
	}
	return &Softmax{axis: axis}
}

type Softmax struct {
	axis int
}

func (s *Softmax) Apply(input Data2D) (result Data2D) {
	var diff1, diff2, diff3 int
	var max1, max2, max3 int
	switch s.axis {
	case 0:
		diff1, diff2, diff3 = input.Channels, 1, input.Height*input.Channels
		max1, max2, max3 = input.Height, input.Channels, input.Width
	case 1:
		diff1, diff2, diff3 = input.Height*input.Channels, 1, input.Channels
		max1, max2, max3 = input.Width, input.Channels, input.Height
	case 2:
		diff1, diff2, diff3 = input.Height*input.Channels, input.Channels, 1
		max1, max2, max3 = input.Width, input.Height, input.Channels
	default:
		panic("axis must be one of 1, 2, or 3")
	}

	result = NewData2D(input.Width, input.Height, input.Channels)

	for i1 := 0; i1 < max1; i1++ {
		for i2 := 0; i2 < max2; i2++ {
			baseIdx := i1*diff1 + i2*diff2
			var expSum float32
			for i3 := 0; i3 < max3; i3++ {
				idx := baseIdx + i3*diff3
				e := float32(math.Exp(float64(input.Values[idx])))
				result.Values[idx] = e
				expSum += e
			}
			for i3 := 0; i3 < max3; i3++ {
				idx := baseIdx + i3*diff3
				result.Values[idx] /= expSum
			}
		}
	}

	return
}

type MaxPooling2D struct {
	poolWidth, poolHeight int
	strideX, strideY      int
}

func NewMaxPooling2D(poolWidth, poolHeight, strideX, strideY int) *MaxPooling2D {
	return &MaxPooling2D{
		poolWidth:  poolWidth,
		poolHeight: poolHeight,
		strideX:    strideX,
		strideY:    strideY,
	}
}

func (mp *MaxPooling2D) Apply(input Data2D) (result Data2D) {
	result = NewData2D((input.Width-mp.poolWidth+mp.strideX-1)/mp.strideX+1, (input.Height-mp.poolHeight+mp.strideY-1)/mp.strideY+1, input.Channels)

	for channel := 0; channel < input.Channels; channel++ {
		for x := 0; x < result.Width; x++ {
			for y := 0; y < result.Height; y++ {
				outputIndex := result.GetIndex(x, y, channel)
				result.Values[outputIndex] = float32(math.Inf(-1))

				for px := 0; px < mp.poolWidth && x*mp.strideX+px < input.Width; px++ {
					for py := 0; py < mp.poolHeight && y*mp.strideY+py < input.Height; py++ {
						current := input.Values[input.GetIndex(x*mp.strideX+px, y*mp.strideY+py, channel)]
						if result.Values[outputIndex] < current {
							result.Values[outputIndex] = current
						}
					}
				}
			}
		}
	}

	return result
}

type Flatten struct{}

func (f *Flatten) Apply(input Data2D) (result Data2D) {
	result = NewData2D(input.TotalSize(), 1, 1)
	copy(result.Values, input.Values)
	return
}

type Dense struct {
	inputSize, outputSize int

	weights []float32
	bias    []float32
}

func NewDense(inputSize, outputSize int, weights, bias []float32) *Dense {
	return &Dense{
		inputSize: inputSize, outputSize: outputSize,
		weights: weights, bias: bias,
	}
}

func (d *Dense) Apply(input Data2D) (result Data2D) {
	if input.Width != d.inputSize || input.Height != 1 || input.Channels != 1 {
		panic("incorrect input dimensions")
	}
	result = NewData2D(d.outputSize, 1, 1)

	for i := 0; i < d.inputSize; i++ {
		for j := 0; j < d.outputSize; j++ {
			result.Values[j] += d.weights[j+i*d.outputSize] * input.Values[i]
		}
	}

	for i := 0; i < d.outputSize; i++ {
		result.Values[i] += d.bias[i]
	}

	return result
}

type LayerData struct {
	Layer2D
	InputIndex int
}

type Model struct {
	Layers  []LayerData
	Outputs []int
}

func (m *Model) Compute(input Data2D) []Data2D {
	results := make([]Data2D, len(m.Layers))
	for i, layer := range m.Layers {
		if i == 0 {
			results[i] = layer.Apply(input)
		} else {
			results[i] = layer.Apply(results[layer.InputIndex])
		}
	}

	outputs := make([]Data2D, len(m.Outputs))
	for i, layer := range m.Outputs {
		outputs[i] = results[layer]
	}

	return outputs
}
