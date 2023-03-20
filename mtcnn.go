package mtcnn

import (
	"encoding/binary"
	"image"
	"io"
	"math"
	"sort"
)

func roundToZero(x float32) float32 {
	if x > 0 {
		return float32(math.Floor(float64(x)))
	}
	return float32(math.Ceil(float64(x)))
}

type boundingBox struct {
	q1x, q1y, q2x, q2y float32
	score              float32
}

type region struct {
	dx1, dy1, dx2, dy2 float32
}

// generateBoundingBoxes takes the scores and reg returned by the pnet and returns the bounding boxes and their regions.
// it expects imap and reg to be to be transposed.
func generateBoundingBoxes(imap, reg Data2D, scale, t float32) ([]boundingBox, []region) {
	if imap.Channels != 1 {
		panic("imap must have one channel")
	}

	const stride, cellsize = 2, 12

	var boundingBoxes []boundingBox
	var regs []region

	for y := 0; y < imap.Height; y++ {
		for x := 0; x < imap.Width; x++ {
			score := imap.Values[imap.GetIndex(x, y, 0)]
			if score < t {
				continue
			}
			dx1 := reg.Values[reg.GetIndex(x, y, 0)]
			dy1 := reg.Values[reg.GetIndex(x, y, 1)]
			dx2 := reg.Values[reg.GetIndex(x, y, 2)]
			dy2 := reg.Values[reg.GetIndex(x, y, 3)]

			boundingBoxes = append(boundingBoxes, boundingBox{
				q1x:   roundToZero(float32(stride*x+1) / scale),
				q1y:   roundToZero(float32(stride*y+1) / scale),
				q2x:   roundToZero(float32(stride*x+cellsize) / scale),
				q2y:   roundToZero(float32(stride*y+cellsize) / scale),
				score: score,
			})
			regs = append(regs, region{
				dx1: dx1,
				dy1: dy1,
				dx2: dx2,
				dy2: dy2,
			})
		}
	}

	return boundingBoxes, regs
}

func nonMaximumSuppresion(boxes []boundingBox, threshold float32, isMethodMin bool) []int {
	sortedIndices := make([]int, len(boxes))
	for i := 0; i < len(sortedIndices); i++ {
		sortedIndices[i] = i
	}
	sort.Slice(sortedIndices, func(i, j int) bool {
		return boxes[sortedIndices[i]].score < boxes[sortedIndices[j]].score
	})
	sortedIndicesSwitch := make([]int, 0, len(boxes))

	var pickedIndices []int
	for len(sortedIndices) > 0 {
		i := sortedIndices[len(sortedIndices)-1]
		pickedIndices = append(pickedIndices, i)

		x11 := boxes[i].q1x
		y11 := boxes[i].q1y
		x21 := boxes[i].q2x
		y21 := boxes[i].q2y
		area1 := (x21 - x11 + 1) * (y21 - y11 + 1)

		for _, boxIdx := range sortedIndices[:len(sortedIndices)-1] {
			x12 := boxes[boxIdx].q1x
			y12 := boxes[boxIdx].q1y
			x22 := boxes[boxIdx].q2x
			y22 := boxes[boxIdx].q2y
			area2 := (x22 - x12 + 1) * (y22 - y12 + 1)

			xx1 := math.Max(float64(x11), float64(x12))
			yy1 := math.Max(float64(y11), float64(y12))
			xx2 := math.Min(float64(x21), float64(x22))
			yy2 := math.Min(float64(y21), float64(y22))
			w, h := math.Max(0, xx2-xx1+1), math.Max(0, yy2-yy1+1)
			intersection := float32(w * h)

			var o float32
			if isMethodMin {
				o = intersection / float32(math.Min(float64(area1), float64(area2)))
			} else {
				o = intersection / (area1 + area2 - intersection)
			}
			if o <= threshold {
				sortedIndicesSwitch = append(sortedIndicesSwitch, boxIdx)
			}
		}
		sortedIndices, sortedIndicesSwitch = sortedIndicesSwitch, sortedIndices[:0]
	}

	return pickedIndices
}

func rerec(bb boundingBox) boundingBox {
	w := bb.q2x - bb.q1x
	h := bb.q2y - bb.q1y
	maxSide := w
	if w < h {
		maxSide = h
	}
	bb.q1x = bb.q1x + 0.5*(w-maxSide)
	bb.q1y = bb.q1y + 0.5*(h-maxSide)
	bb.q2x = bb.q1x + maxSide
	bb.q2y = bb.q1y + maxSide
	return bb
}

type padding struct {
	dy, edy    int32
	dx, edx    int32
	y, ey      int32
	x, ex      int32
	tmpw, tmph int32
}

func pad(bb boundingBox, width, height int32) padding {
	p := padding{
		tmpw: int32(bb.q2x - bb.q1x + 1),
		tmph: int32(bb.q2y - bb.q1y + 1),
		dx:   1,
		dy:   1,
		x:    int32(bb.q1x),
		y:    int32(bb.q1y),
		ex:   int32(bb.q2x),
		ey:   int32(bb.q2y),
	}

	p.edx = p.tmpw
	p.edy = p.tmph

	if p.ex > width {
		p.edx = -p.ex + width + p.tmpw
		p.ex = width
	}
	if p.ey > height {
		p.edy = -p.ey + height + p.tmph
		p.ey = height
	}
	if p.x < 1 {
		p.dx = 2 - p.x
		p.x = 1
	}
	if p.y < 1 {
		p.dy = 2 - p.y
		p.y = 1
	}
	return p
}

func bbreg(bb boundingBox, reg region) boundingBox {
	w := bb.q2x - bb.q1x + 1
	h := bb.q2y - bb.q1y + 1
	return boundingBox{
		q1x:   bb.q1x + reg.dx1*w,
		q1y:   bb.q1y + reg.dy1*h,
		q2x:   bb.q2x + reg.dx2*w,
		q2y:   bb.q2y + reg.dy2*h,
		score: bb.score,
	}
}

func ReadData2DFromBin(r io.Reader) ([]Data2D, error) {
	buf := make([]byte, 0, 1024*32)
	bo := binary.LittleEndian

	buf = buf[:4]
	n, err := r.Read(buf)
	if err != nil {
		if err == io.EOF && n == 0 {
			return nil, io.EOF
		}
		return nil, err
	}
	datas := make([]Data2D, bo.Uint32(buf))

	for i := 0; i < len(datas); i++ {
		buf = buf[:12]
		n, err = r.Read(buf)
		if err != nil {
			if err == io.EOF && n == 0 {
				return nil, io.EOF
			}
			return nil, err
		}
		datas[i] = NewData2D(int(bo.Uint32(buf[0:4])), int(bo.Uint32(buf[4:8])), int(bo.Uint32(buf[8:12])))

		toRead := datas[i].TotalSize()
		var read int = 0
		for read < toRead {
			leftToRead := toRead - read
			if 4*leftToRead < cap(buf) {
				buf = buf[:4*leftToRead]
			} else {
				buf = buf[:cap(buf)]
			}
			n, err = r.Read(buf)
			if err != nil {
				if err == io.EOF && n == 0 {
					return nil, io.EOF
				}
				return nil, err
			}
			for j := 0; j < len(buf)/4; j++ {
				datas[i].Values[read+j] = math.Float32frombits(bo.Uint32(buf[4*j : 4*(j+1)]))
			}
			read += n / 4
		}
	}

	return datas, nil
}

func getPNet(weights [][]float32) *Model {
	return &Model{
		Layers: []LayerData{
			{NewConv2D(3, 3, 3, 10, weights[0], weights[1]), 0},
			{NewPReLU(weights[2], 2), 0},
			{NewMaxPooling2D(2, 2, 2, 2), 1},

			{NewConv2D(3, 3, 10, 16, weights[3], weights[4]), 2},
			{NewPReLU(weights[5], 2), 3},

			{NewConv2D(3, 3, 16, 32, weights[6], weights[7]), 4},
			{NewPReLU(weights[8], 2), 5},

			{NewConv2D(1, 1, 32, 2, weights[9], weights[10]), 6},
			{NewSoftmax(2), 7},

			{NewConv2D(1, 1, 32, 4, weights[11], weights[12]), 6},
		},
		Outputs: []int{9, 8},
	}
}

func getRNet(weights [][]float32) *Model {
	return &Model{
		Layers: []LayerData{
			{NewConv2D(3, 3, 3, 28, weights[0], weights[1]), 0},
			{NewPReLU(weights[2], 2), 0},
			{NewMaxPooling2D(3, 3, 2, 2), 1},

			{NewConv2D(3, 3, 28, 48, weights[3], weights[4]), 2},
			{NewPReLU(weights[5], 2), 3},
			{NewMaxPooling2D(3, 3, 2, 2), 4},

			{NewConv2D(2, 2, 48, 64, weights[6], weights[7]), 5},
			{NewPReLU(weights[8], 2), 6},
			{&Flatten{}, 7},
			{NewDense(3*3*64, 128, weights[9], weights[10]), 8},
			{NewPReLU(weights[11], 0), 9},

			{NewDense(128, 2, weights[12], weights[13]), 10},
			{NewSoftmax(0), 11},

			{NewDense(128, 4, weights[14], weights[15]), 10},
		},
		Outputs: []int{13, 12},
	}
}

func getONet(weights [][]float32) *Model {
	return &Model{
		Layers: []LayerData{
			{NewConv2D(3, 3, 3, 32, weights[0], weights[1]), 0},
			{NewPReLU(weights[2], 2), 0},
			{NewMaxPooling2D(3, 3, 2, 2), 1},

			{NewConv2D(3, 3, 32, 64, weights[3], weights[4]), 2},
			{NewPReLU(weights[5], 2), 3},
			{NewMaxPooling2D(3, 3, 2, 2), 4},

			{NewConv2D(3, 3, 64, 64, weights[6], weights[7]), 5},
			{NewPReLU(weights[8], 2), 6},
			{NewMaxPooling2D(2, 2, 2, 2), 7},

			{NewConv2D(2, 2, 64, 128, weights[9], weights[10]), 8},
			{NewPReLU(weights[11], 2), 9},

			{&Flatten{}, 10},
			{NewDense(3*3*128, 256, weights[12], weights[13]), 11},
			{NewPReLU(weights[14], 0), 12},

			{NewDense(256, 2, weights[15], weights[16]), 13},
			{NewSoftmax(0), 14},

			{NewDense(256, 4, weights[17], weights[18]), 13},
			{NewDense(256, 10, weights[19], weights[20]), 13},
		},
		Outputs: []int{16, 17, 15},
	}
}

func stage1(pnet *Model, img image.Image, scales []float32, threshold float32) []boundingBox {
	var allBoxes []boundingBox
	var allRegs []region

	originalImgData := ImageToData2D(img)
	for _, scale := range scales {
		scaledImgData := scaleImageData2D(originalImgData,
			int(math.Ceil(float64(scale)*float64(img.Bounds().Dx()))),
			int(math.Ceil(float64(scale)*float64(img.Bounds().Dy()))))
		for i := 0; i < len(scaledImgData.Values); i++ {
			scaledImgData.Values[i] = (scaledImgData.Values[i] - 0.5) * 255 / 128
		}
		out := pnet.Compute(scaledImgData)

		imap := NewData2D(out[1].Width, out[1].Height, 1)
		for i := 0; i < imap.TotalSize(); i++ {
			imap.Values[i] = out[1].Values[i*2+1]
		}
		boxes, regs := generateBoundingBoxes(imap, out[0], scale, threshold)
		picks := nonMaximumSuppresion(boxes, 0.5, false)

		for _, pick := range picks {
			allBoxes = append(allBoxes, boxes[pick])
			allRegs = append(allRegs, regs[pick])
		}
	}

	var boundingBoxes []boundingBox

	if len(allBoxes) > 0 {
		picks := nonMaximumSuppresion(allBoxes, 0.7, false)
		for _, pick := range picks {
			regw := allBoxes[pick].q2x - allBoxes[pick].q1x
			regh := allBoxes[pick].q2y - allBoxes[pick].q1y

			boundingBoxes = append(boundingBoxes, boundingBox{
				q1x:   roundToZero(allBoxes[pick].q1x + allRegs[pick].dy1*regw),
				q1y:   roundToZero(allBoxes[pick].q1y + allRegs[pick].dx1*regh),
				q2x:   roundToZero(allBoxes[pick].q2x + allRegs[pick].dy2*regw),
				q2y:   roundToZero(allBoxes[pick].q2y + allRegs[pick].dx2*regh),
				score: allBoxes[pick].score,
			})
		}
		for i := 0; i < len(boundingBoxes); i++ {
			boundingBoxes[i] = rerec(boundingBoxes[i])
		}
	}

	return boundingBoxes
}

func stage2(rnet *Model, img image.Image, boundingBoxes []boundingBox, threshold float32) []boundingBox {
	imgData := ImageToData2D(img)

	rnetResult := make([]boundingBox, 0, len(boundingBoxes))
	mvs := make([]region, 0, len(boundingBoxes))
	for k := 0; k < len(boundingBoxes); k++ {
		p := pad(boundingBoxes[k], int32(img.Bounds().Dx()), int32(img.Bounds().Dy()))

		tmp := NewData2D(int(p.tmpw), int(p.tmph), 3)
		for x := p.dx - 1; x < p.edx; x++ {
			sx := x - (p.dx - 1) + (p.x - 1)
			for y := p.dy - 1; y < p.edy; y++ {
				sy := y - (p.dy - 1) + (p.y - 1)

				for c := 0; c < 3; c++ {
					tmp.Values[tmp.GetIndex(int(x), int(y), c)] =
						imgData.Values[imgData.GetIndex(int(sx), int(sy), c)]
				}
			}
		}

		tmp = scaleImageData2D(tmp, 24, 24)
		for i := 0; i < len(tmp.Values); i++ {
			tmp.Values[i] = (tmp.Values[i] - 0.5) * 255 / 128
		}
		out := rnet.Compute(tmp)

		score := out[1].Values[1]
		if score > threshold {
			rnetResult = append(rnetResult, boundingBox{
				q1x:   boundingBoxes[k].q1x,
				q1y:   boundingBoxes[k].q1y,
				q2x:   boundingBoxes[k].q2x,
				q2y:   boundingBoxes[k].q2y,
				score: score,
			})
			mvs = append(mvs, region{
				dx1: out[0].Values[0],
				dy1: out[0].Values[1],
				dx2: out[0].Values[2],
				dy2: out[0].Values[3],
			})
		}
	}

	picks := nonMaximumSuppresion(rnetResult, 0.7, false)
	result := make([]boundingBox, len(picks))
	for i, pick := range picks {
		result[i] = bbreg(rnetResult[pick], mvs[pick])
		result[i] = rerec(result[i])
	}

	return result
}

func stage3(onet *Model, img image.Image, boundingBoxes []boundingBox, threshold float32) []FaceData {
	imgData := ImageToData2D(img)

	faceBoxes := make([]boundingBox, 0, len(boundingBoxes))
	onetResult := make([]boundingBox, 0, len(boundingBoxes))
	allPoints := make([][10]float32, 0, len(boundingBoxes))

	for k := 0; k < len(boundingBoxes); k++ {
		p := pad(boundingBoxes[k], int32(img.Bounds().Dx()), int32(img.Bounds().Dy()))

		tmp := NewData2D(int(p.tmpw), int(p.tmph), 3)
		for x := p.dx - 1; x < p.edx; x++ {
			sx := x - (p.dx - 1) + (p.x - 1)
			for y := p.dy - 1; y < p.edy; y++ {
				sy := y - (p.dy - 1) + (p.y - 1)

				for c := 0; c < 3; c++ {
					tmp.Values[tmp.GetIndex(int(x), int(y), c)] =
						imgData.Values[imgData.GetIndex(int(sx), int(sy), c)]
				}
			}
		}

		tmp = scaleImageData2D(tmp, 48, 48)

		for i := 0; i < len(tmp.Values); i++ {
			tmp.Values[i] = (tmp.Values[i] - 0.5) * 255 / 128
		}
		out := onet.Compute(tmp)

		score := out[2].Values[1]
		if score > threshold {
			points := out[1]
			mv := out[0]

			faceBoxes = append(faceBoxes, bbreg(boundingBoxes[k], region{
				dx1: mv.Values[0],
				dy1: mv.Values[1],
				dx2: mv.Values[2],
				dy2: mv.Values[3],
			}))
			onetResult = append(onetResult, boundingBoxes[k])
			allPoints = append(allPoints, [10]float32{})
			copy(allPoints[len(allPoints)-1][:], points.Values)
		}
	}

	picks := nonMaximumSuppresion(faceBoxes, 0.7, true)
	fd := make([]FaceData, 0, len(picks))
	for _, pick := range picks {
		w := onetResult[pick].q2x - onetResult[pick].q1x + 1
		h := onetResult[pick].q2y - onetResult[pick].q1y + 1

		fd = append(fd, FaceData{
			Box:        image.Rect(int(faceBoxes[pick].q1x), int(faceBoxes[pick].q1y), int(faceBoxes[pick].q2x), int(faceBoxes[pick].q2y)),
			Confidence: faceBoxes[pick].score,

			LeftEye: image.Point{
				X: int(w*allPoints[pick][0] + onetResult[pick].q1x - 1),
				Y: int(h*allPoints[pick][5] + onetResult[pick].q1y - 1),
			},
			RightEye: image.Point{
				X: int(w*allPoints[pick][1] + onetResult[pick].q1x - 1),
				Y: int(h*allPoints[pick][6] + onetResult[pick].q1y - 1),
			},
			Nose: image.Point{
				X: int(w*allPoints[pick][2] + onetResult[pick].q1x - 1),
				Y: int(h*allPoints[pick][7] + onetResult[pick].q1y - 1),
			},
			MouthLeft: image.Point{
				X: int(w*allPoints[pick][3] + onetResult[pick].q1x - 1),
				Y: int(h*allPoints[pick][8] + onetResult[pick].q1y - 1),
			},
			MouthRight: image.Point{
				X: int(w*allPoints[pick][4] + onetResult[pick].q1x - 1),
				Y: int(h*allPoints[pick][9] + onetResult[pick].q1y - 1),
			},
		})
	}

	return fd
}
