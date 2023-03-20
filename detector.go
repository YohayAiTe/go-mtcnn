package mtcnn

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"image"
	"io"
	"math"
)

type FaceData struct {
	Box        image.Rectangle
	Confidence float32

	LeftEye, RightEye, Nose, MouthLeft, MouthRight image.Point
}

type DetectorOptions struct {
	MinFaceSize     int
	StageThresholds [3]float32
	ScaleFactor     float32
}

var defaultDetectorOptions = DetectorOptions{
	MinFaceSize:     20,
	StageThresholds: [3]float32{0.6, 0.6, 0.7},
	ScaleFactor:     0.709,
}

type Detector struct {
	pnet, rnet, onet *Model

	options *DetectorOptions
}

func NewDetector(pnetWeights, rnetWeights, onetWeights [][]float32, options *DetectorOptions) *Detector {
	if options == nil {
		options = &defaultDetectorOptions
	}

	return &Detector{
		pnet:    getPNet(pnetWeights),
		rnet:    getRNet(rnetWeights),
		onet:    getONet(onetWeights),
		options: options,
	}
}

func getScalePyramid(m, minLayer, scaleFactor float32) []float32 {
	var scales []float32

	var currentScale float32 = 1
	for minLayer >= 12 {
		scales = append(scales, m*currentScale)
		currentScale *= scaleFactor
		minLayer *= scaleFactor
	}

	return scales
}

func (d *Detector) DetectFaces(img image.Image) []FaceData {
	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	m := 12 / float32(d.options.MinFaceSize)
	minLayer := float32(width) * m
	if height < width {
		minLayer = float32(height) * m
	}
	scales := getScalePyramid(m, minLayer, d.options.ScaleFactor)
	_ = scales

	boundingBoxes := stage1(d.pnet, img, scales, d.options.StageThresholds[0])
	boundingBoxes = stage2(d.rnet, img, boundingBoxes, d.options.StageThresholds[1])
	return stage3(d.onet, img, boundingBoxes, d.options.StageThresholds[2])
}

func ReadWeightsFromBin(r io.Reader) (weights [][]float32, err error) {
	buf := make([]byte, 0, 1024*32)
	bo := binary.LittleEndian

	buf = buf[:4]
	_, err = r.Read(buf)
	if err != nil {
		return
	}

	weights = make([][]float32, bo.Uint32(buf))
	if len(weights) == 0 {
		return
	}
	for i := 0; i < len(weights); i++ {
		buf = buf[:4]
		_, err = r.Read(buf)
		if err != nil {
			return
		}
		weights[i] = make([]float32, bo.Uint32(buf))

		toRead, read := len(weights[i]), 0
		for read < toRead {
			leftToRead := toRead - read
			if 4*leftToRead < cap(buf) {
				buf = buf[:4*leftToRead]
			} else {
				buf = buf[:cap(buf)]
			}
			_, err = r.Read(buf)
			if err != nil {
				return
			}
			for j := 0; j < len(buf)/4; j++ {
				weights[i][read+j] = math.Float32frombits(bo.Uint32(buf[4*j : 4*(j+1)]))
			}
			read += len(buf) / 4
		}
	}

	return
}

var (
	//go:embed weights/pnet_weights.bin
	pnetWeightsFile []byte

	//go:embed weights/rnet_weights.bin
	rnetWeightsFile []byte

	//go:embed weights/onet_weights.bin
	onetWeightsFile []byte
)

var defaultDetector *Detector

var DefaultPnetWeights, DefaultRnetWeights, DefaultOnetWeights [][]float32

func init() {
	DefaultPnetWeights, _ = ReadWeightsFromBin(bytes.NewReader(pnetWeightsFile))
	DefaultRnetWeights, _ = ReadWeightsFromBin(bytes.NewReader(rnetWeightsFile))
	DefaultOnetWeights, _ = ReadWeightsFromBin(bytes.NewReader(onetWeightsFile))
	defaultDetector = NewDetector(DefaultPnetWeights, DefaultRnetWeights, DefaultOnetWeights, nil)
}

func DetectFaces(img image.Image) []FaceData {
	return defaultDetector.DetectFaces(img)
}
