package main

import (
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"os"
	"sync"

	"github.com/yohayaite/go-mtcnn"
)

func drawSquare(img draw.Image, x, y int, r int, c color.Color) {
	for dx := -r; dx <= r; dx++ {
		if x+dx < img.Bounds().Min.X || x+dx >= img.Bounds().Max.X {
			continue
		}
		for dy := -r; dy <= r; dy++ {
			if y+dy < img.Bounds().Min.Y || y+dy >= img.Bounds().Max.Y {
				continue
			}

			img.Set(x+dx, y+dy, c)
		}
	}
}

func detectFacesAndSave(inputFilename, outputFilename string, wg *sync.WaitGroup) {
	log.Printf("starting to read and detect faces from %s\n", inputFilename)
	inputFile, err := os.Open(inputFilename)
	if err != nil {
		log.Fatal(err)
	}
	defer inputFile.Close()
	img, _, err := image.Decode(inputFile)
	if err != nil {
		log.Fatal(err)
	}

	facedata := mtcnn.DetectFaces(img)

	log.Printf("finished detecting faces, starting to save %s\n", outputFilename)
	imgCopy := image.NewRGBA(img.Bounds())
	draw.Draw(imgCopy, img.Bounds(), img, img.Bounds().Min, draw.Over)
	for _, facedatum := range facedata {
		for x := facedatum.Box.Min.X; x < facedatum.Box.Max.X; x++ {
			imgCopy.Set(x, facedatum.Box.Min.Y, color.White)
			imgCopy.Set(x, facedatum.Box.Max.Y, color.White)
		}
		for y := facedatum.Box.Min.Y; y < facedatum.Box.Max.Y; y++ {
			imgCopy.Set(facedatum.Box.Min.X, y, color.White)
			imgCopy.Set(facedatum.Box.Max.X, y, color.White)
		}

		r := 1
		drawSquare(imgCopy, facedatum.LeftEye.X, facedatum.LeftEye.Y, r, color.RGBA{0, 0, 0, 255})
		drawSquare(imgCopy, facedatum.RightEye.X, facedatum.RightEye.Y, r, color.RGBA{255, 0, 0, 255})
		drawSquare(imgCopy, facedatum.Nose.X, facedatum.Nose.Y, r, color.RGBA{0, 255, 0, 255})
		drawSquare(imgCopy, facedatum.MouthLeft.X, facedatum.MouthLeft.Y, r, color.RGBA{0, 0, 255, 255})
		drawSquare(imgCopy, facedatum.MouthRight.X, facedatum.MouthRight.Y, r, color.RGBA{255, 0, 255, 255})
	}

	outputFile, err := os.Create(outputFilename)
	if err != nil {
		log.Fatal(err)
	}
	defer outputFile.Close()
	err = jpeg.Encode(outputFile, imgCopy, nil)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("finished saving %s\n", outputFilename)
	wg.Done()
}

func main() {
	imageFiles := []struct {
		inputFilename, outputFilename string
	}{
		// source: https://commons.wikimedia.org/wiki/File:Neil_deGrasse_Tyson_-_NASA_Advisory_Council.jpg
		{"images/Neil_deGrasse_Tyson.jpg", "images/Neil_deGrasse_Tyson_out.jpg"},
		// source: https://commons.wikimedia.org/wiki/File:Ohio_farmer_David_Brandt.jpg
		{"images/Ohio_farmer_David_Brandt.jpg", "images/Ohio_farmer_David_Brandt_out.jpg"},
		// source: https://commons.wikimedia.org/wiki/File:ZIPFY_family-walking.jpg
		{"images/ZIPFY_family-walking.jpg", "images/ZIPFY_family-walking_out.jpg"},
	}
	var wg sync.WaitGroup
	for _, files := range imageFiles {
		wg.Add(1)
		go detectFacesAndSave(files.inputFilename, files.outputFilename, &wg)
	}
	wg.Wait()
}
