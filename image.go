package mtcnn

import (
	"image"
	"math"

	"golang.org/x/image/draw"
)

// downScaleImage scales the given image by the given scale. It assumes scale is in the range [0, 1].
// TODO: change implementation to area instead of nearest neighbor.
func downScaleImage(img image.Image, scale float64) image.Image {
	newWidth := int(math.Ceil(float64(img.Bounds().Dx()) * scale))
	newHeight := int(math.Ceil(float64(img.Bounds().Dy()) * scale))

	resImg := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
	draw.NearestNeighbor.Scale(resImg, resImg.Bounds(), img, img.Bounds(), draw.Over, nil)
	return resImg
}

func ImageToData2D(img image.Image) Data2D {
	data := NewData2D(img.Bounds().Dx(), img.Bounds().Dy(), 3)
	for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
		for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
			r, g, b, _ := img.At(x, y).RGBA()
			copy(data.Values[data.GetIndex(x-img.Bounds().Min.X, y-img.Bounds().Min.Y, 0):],
				[]float32{float32(r>>8) / 255, float32(g>>8) / 255, float32(b>>8) / 255})
		}
	}
	return data
}

func ImageToData2DTanspose(img image.Image) Data2D {
	data := NewData2D(img.Bounds().Dx(), img.Bounds().Dy(), 3)
	for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
		for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
			r, g, b, _ := img.At(x, y).RGBA()
			copy(data.Values[data.GetIndex(y-img.Bounds().Min.Y, x-img.Bounds().Min.X, 0):],
				[]float32{float32(r>>8) / 255, float32(g>>8) / 255, float32(b>>8) / 255})
		}
	}
	return data
}

func scaleImageData2D(input Data2D, targetWidth, targetHeight int) Data2D {
	if input.Width >= targetWidth && input.Height >= targetHeight {
		return downscaleData2D(input, targetWidth, targetHeight)
	}
	if input.Width <= targetWidth && input.Height <= targetHeight {
		return upscaleData2D(input, targetWidth, targetHeight)
	}
	panic("cannot upscale and downscale image data in different axis")
}

func downscaleData2D(input Data2D, targetWidth, targetHeight int) Data2D {
	result := NewData2D(targetWidth, targetHeight, input.Channels)
	widthScale, heightScale := float32(input.Width)/float32(targetWidth), float32(input.Height)/float32(targetHeight)

	for x := 0; x < result.Width; x++ {
		for y := 0; y < result.Height; y++ {
			minX, maxX := widthScale*float32(x), widthScale*float32(x+1)
			minY, maxY := heightScale*float32(y), heightScale*float32(y+1)
			if maxX > float32(input.Width) {
				maxX = float32(input.Width)
			}
			if maxY > float32(input.Height) {
				maxY = float32(input.Height)
			}
			minFullX, maxFullX := int(math.Ceil(float64(minX))), int(math.Floor(float64(maxX)))
			minFullY, maxFullY := int(math.Ceil(float64(minY))), int(math.Floor(float64(maxY)))
			totalArea := widthScale * heightScale

			for c := 0; c < result.Channels; c++ {
				var sum float32

				if diffX := float32(minFullX) - minX; diffX > 0 {
					if diffY := float32(minFullY) - minY; diffY > 0 {
						v := input.Values[input.GetIndex(minFullX-1, minFullY-1, c)]
						sum += v * diffX * diffY
					}
					for sy := minFullY; sy < maxFullY; sy++ {
						v := input.Values[input.GetIndex(minFullX-1, sy, c)]
						sum += v * diffX
					}
					if diffY := maxY - float32(maxFullY); diffY > 0 {
						v := input.Values[input.GetIndex(minFullX-1, maxFullY, c)]
						sum += v * diffX * diffY
					}
				}

				for sx := minFullX; sx < maxFullX; sx++ {
					if diffY := float32(minFullY) - minY; diffY > 0 {
						v := input.Values[input.GetIndex(sx, minFullY-1, c)]
						sum += v * diffY
					}
					for sy := minFullY; sy < maxFullY; sy++ {
						v := input.Values[input.GetIndex(sx, sy, c)]
						sum += v
					}
					if diffY := maxY - float32(maxFullY); diffY > 0 {
						v := input.Values[input.GetIndex(sx, maxFullY, c)]
						sum += v * diffY
					}
				}

				if diffX := maxX - float32(maxFullX); diffX > 0 {
					if diffY := float32(minFullY) - minY; diffY > 0 {
						v := input.Values[input.GetIndex(maxFullX, minFullY-1, c)]
						sum += v * diffX * diffY
					}
					for sy := minFullY; sy < maxFullY; sy++ {
						v := input.Values[input.GetIndex(maxFullX, sy, c)]
						sum += v * diffX
					}
					if diffY := maxY - float32(maxFullY); diffY > 0 {
						v := input.Values[input.GetIndex(maxFullX, maxFullY, c)]
						sum += v * diffX * diffY
					}
				}
				result.Values[result.GetIndex(x, y, c)] = sum / totalArea
			}
		}
	}

	return result
}

func upscaleData2D(input Data2D, targetWidth, targetHeight int) Data2D {
	result := NewData2D(targetWidth, targetHeight, input.Channels)
	widthScale, heightScale := float32(targetWidth)/float32(input.Width), float32(targetHeight)/float32(input.Height)

	for x := 0; x < input.Width; x++ {
		for y := 0; y < input.Height; y++ {
			minX, maxX := widthScale*float32(x), widthScale*float32(x+1)
			minY, maxY := heightScale*float32(y), heightScale*float32(y+1)
			if maxX > float32(targetWidth) {
				maxX = float32(targetWidth)
			}
			if maxY > float32(targetHeight) {
				maxY = float32(targetHeight)
			}
			minFullX, maxFullX := int(math.Ceil(float64(minX))), int(math.Floor(float64(maxX)))
			minFullY, maxFullY := int(math.Ceil(float64(minY))), int(math.Floor(float64(maxY)))

			for c := 0; c < result.Channels; c++ {
				v := input.Values[input.GetIndex(x, y, c)]

				if diffX := float32(minFullX) - minX; diffX > 0 {
					if diffY := float32(minFullY) - minY; diffY > 0 {
						result.Values[result.GetIndex(minFullX-1, minFullY-1, c)] += v * diffX * diffY
					}
					for sy := minFullY; sy < maxFullY; sy++ {
						result.Values[result.GetIndex(minFullX-1, sy, c)] += v * diffX
					}
					if diffY := maxY - float32(maxFullY); diffY > 0 {
						result.Values[result.GetIndex(minFullX-1, maxFullY, c)] += v * diffX * diffY
					}
				}
				for sx := minFullX; sx < maxFullX; sx++ {
					if diffY := float32(minFullY) - minY; diffY > 0 {
						result.Values[result.GetIndex(sx, minFullY-1, c)] += v * diffY
					}
					for sy := minFullY; sy < maxFullY; sy++ {
						result.Values[result.GetIndex(sx, sy, c)] = v
					}
					if diffY := maxY - float32(maxFullY); diffY > 0 {
						result.Values[result.GetIndex(sx, maxFullY, c)] += v * diffY
					}
				}
				if diffX := maxX - float32(maxFullX); diffX > 0 {
					if diffY := float32(minFullY) - minY; diffY > 0 {
						result.Values[result.GetIndex(maxFullX, minFullY-1, c)] += v * diffX * diffY
					}
					for sy := minFullY; sy < maxFullY; sy++ {
						result.Values[result.GetIndex(maxFullX, sy, c)] += v * diffX
					}
					if diffY := maxY - float32(maxFullY); diffY > 0 {
						result.Values[result.GetIndex(maxFullX, maxFullY, c)] += v * diffX * diffY
					}
				}
			}
		}
	}

	return result
}

func transposeImageData(input Data2D) Data2D {
	result := NewData2D(input.Height, input.Width, input.Channels)
	copy(result.Values, input.Values)

	for x := 0; x < input.Width; x++ {
		for y := 0; y < input.Height; y++ {
			for c := 0; c < input.Channels; c++ {
				result.Values[result.GetIndex(y, x, c)] = input.Values[input.GetIndex(x, y, c)]
			}
		}
	}

	return result
}
