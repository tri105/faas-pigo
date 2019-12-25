package function

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"io/ioutil"
	"math"
	"net/http"
	"os"
	"time"

	pigo "github.com/esimov/pigo/core"
	"github.com/fogleman/gg"
)

var dc *gg.Context

// FaceDetector struct contains Pigo face detector general settings.
type FaceDetector struct {
	cascadeFile  string
	minSize      int
	maxSize      int
	shiftFactor  float64
	scaleFactor  float64
	iouThreshold float64
}

// DetectionResult contains the coordinates of the detected faces and the base64 converted image.
type DetectionResult struct {
	ImageName  string
	TotalFaces int
	Faces      []image.Rectangle
	//ImageBase64 string
	Time string
}

// Result contains final json return
type Result struct {
	Status      string
	TotalTime   string
	TotalImages int
	Data        []DetectionResult
}

// Handle is main function to received request (http.Request) and processing to return the response (via http.ResponseWriter)
func Handle(w http.ResponseWriter, r *http.Request) {
	// Start time
	//start2 := time.Now()
	parseErr := r.ParseMultipartForm(128 << 20)
	// Check multipart data
	if parseErr != nil {
		http.Error(w, parseErr.Error(), http.StatusBadRequest)
		return
	}
	if r.MultipartForm == nil || r.MultipartForm.File == nil {
		http.Error(w, "Expecting multipart form file", http.StatusBadRequest)
		return
	}
	if err := verifyRequest(r); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	//elapsed2 := time.Since(start2)
	// Declare variable
	var (
		resp  DetectionResult
		rects []image.Rectangle
		//image   []byte
		outcome   []DetectionResult
		result    Result
		numImages int
	)
	numImages = 0
	fd := NewFaceDetector("./data/facefinder", 20, 2000, 0.1, 1.1, 0.18)
	// For loop to process all image in "image" field
	// Start time
	begin := time.Now()
	for _, h := range r.MultipartForm.File["image"] {
		// Start time
		start := time.Now()
		file, err := h.Open()
		if err != nil {
			http.Error(w, "Failed to get media form file", http.StatusBadRequest)
			return
		}
		defer file.Close()

		// Create temporary file in /tmp with prefix image
		tmpfile, err := ioutil.TempFile("/tmp", "image")
		if err != nil {
			http.Error(w, "Unable to create temp file", http.StatusInternalServerError)
			return
		}
		if _, err = io.Copy(tmpfile, file); err != nil {
			http.Error(w, "Unable to copy file to tmp folder", http.StatusInternalServerError)
			return
		}
		defer os.Remove(tmpfile.Name()) // Remove tmpfile after processing
		// Face detection
		faces, err := fd.DetectFaces(tmpfile.Name())
		if err != nil {
			http.Error(w, "Error on face detection", http.StatusInternalServerError)
			return
		}
		// Return result (rect location and image encode)
		var errs error
		rects, _, errs = fd.DrawFaces(faces, false)
		if errs != nil {
			http.Error(w, "Error creating image output", http.StatusInternalServerError)
			return
		}
		elapsed := time.Since(start)
		resp = DetectionResult{
			ImageName:  h.Filename,
			TotalFaces: len(rects),
			Faces:      rects,
			//ImageBase64: base64.StdEncoding.EncodeToString(image),
			Time: elapsed.String(),
		}
		numImages++
		// Append image result to overall
		outcome = append(outcome, resp)
	}
	end := time.Since(begin)
	result = Result{
		Status:      "success",
		TotalTime:   end.String(),
		TotalImages: numImages,
		Data:        outcome,
	}
	joutcome, err := json.Marshal(result)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error encoding output: %s", err), http.StatusInternalServerError)
		return
	}
	// Return result
	w.WriteHeader(http.StatusOK)
	w.Header().Set("Content-Type", "application/json")
	w.Write(joutcome)
}

// verifyRequest will check and verify the multipart/form-data input include image file or not
func verifyRequest(r *http.Request) error {
	if _, ok := r.MultipartForm.File["image"]; !ok {
		return fmt.Errorf("No image uploaded. Please try again")
	}
	return nil
}

// NewFaceDetector initialises the constructor function.
func NewFaceDetector(cf string, minSize, maxSize int, shf, scf, iou float64) *FaceDetector {
	return &FaceDetector{
		cascadeFile:  cf,
		minSize:      minSize,
		maxSize:      maxSize,
		shiftFactor:  shf,
		scaleFactor:  scf,
		iouThreshold: iou,
	}
}

// DetectFaces run the detection algorithm over the provided source image.
func (fd *FaceDetector) DetectFaces(source string) ([]pigo.Detection, error) {
	src, err := pigo.GetImage(source)
	if err != nil {
		return nil, err
	}

	pixels := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	dc = gg.NewContext(cols, rows)
	dc.DrawImage(src, 0, 0)

	cParams := pigo.CascadeParams{
		MinSize:     fd.minSize,
		MaxSize:     fd.maxSize,
		ShiftFactor: fd.shiftFactor,
		ScaleFactor: fd.scaleFactor,
		ImageParams: pigo.ImageParams{
			Pixels: pixels,
			Rows:   rows,
			Cols:   cols,
			Dim:    cols,
		},
	}

	cascadeFile, err := ioutil.ReadFile(fd.cascadeFile)
	if err != nil {
		return nil, err
	}

	pigo := pigo.NewPigo()
	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier, err := pigo.Unpack(cascadeFile)
	if err != nil {
		return nil, err
	}

	// Run the classifier over the obtained leaf nodes and return the detection results.
	// The result contains quadruplets representing the row, column, scale and detection score.
	faces := classifier.RunCascade(cParams, 0)

	// Calculate the intersection over union (IoU) of two clusters.
	faces = classifier.ClusterDetections(faces, fd.iouThreshold)

	return faces, nil
}

// DrawFaces marks the detected faces with a circle in case isCircle is true, otherwise marks with a rectangle.
func (fd *FaceDetector) DrawFaces(faces []pigo.Detection, isCircle bool) ([]image.Rectangle, []byte, error) {
	var (
		qThresh float32 = 5.0
		rects   []image.Rectangle
	)

	for _, face := range faces {
		if face.Q > qThresh {
			if isCircle {
				dc.DrawArc(
					float64(face.Col),
					float64(face.Row),
					float64(face.Scale/2),
					0,
					2*math.Pi,
				)
			} else {
				dc.DrawRectangle(
					float64(face.Col-face.Scale/2),
					float64(face.Row-face.Scale/2),
					float64(face.Scale),
					float64(face.Scale),
				)
			}
			rects = append(rects, image.Rect(
				face.Col-face.Scale/2,
				face.Row-face.Scale/2,
				face.Scale,
				face.Scale,
			))
			dc.SetLineWidth(2.0)
			dc.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 255, G: 255, B: 0, A: 255}))
			dc.Stroke()
		}
	}

	img := dc.Image()

	filename := fmt.Sprintf("/tmp/%d.jpg", time.Now().UnixNano())

	output, err := os.OpenFile(filename, os.O_CREATE|os.O_RDWR, 0755)
	if err != nil {
		return nil, nil, err
	}
	defer os.Remove(filename)

	jpeg.Encode(output, img, &jpeg.Options{Quality: 100})

	rf, err := ioutil.ReadFile(filename)
	return rects, rf, err
}
