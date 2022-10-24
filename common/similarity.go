package common

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"golang.org/x/exp/constraints"
)

type featurizable interface {
	constraints.Integer | constraints.Float
}

// Dot function returns the dot or scalar product of two equal-length vectors
func Dot[T featurizable](v1 []T, v2 []T) (dot T, err error) {
	if len(v1) != len(v2) {
		err = errors.New("unequal length vectors provided")
		return
	}
	if len(v1) == 0 {
		err = errors.New("zero length vectors provided")
		return
	}
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
	}
	return
}

// Cosine similarity function returns the cosine similarity of two equal-length vectors
// the cosine similarity is calculated as Cos(x, y) = x . y / ||x|| * ||y||
func Cosine[T featurizable](v1 []T, v2 []T) (cosine float64, err error) {
	if len(v1) != len(v2) {
		err = errors.New("unequal length vectors provided")
		return
	}
	if len(v1) == 0 {
		err = errors.New("zero length vectors provided")
		return
	}
	var dot T
	var s1, s2 float64
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
		s1 += math.Pow(float64(v1[i]), 2)
		s2 += math.Pow(float64(v2[i]), 2)
	}
	if s1 == 0 || s2 == 0 {
		err = errors.New("vectors should not be null (all zeros)")
		return
	}
	cosine = float64(dot) / (math.Sqrt(s1) * math.Sqrt(s2))
	return
}

func EuclideanSimilarity[T featurizable](v1, v2 []T) (similarity float64, err error) {
	euclideanDistance, err := EuclideanDistance(v1, v2)
	if err == nil {
		similarity = 1.0 - euclideanDistance
	}
	return
}

func EuclideanDistance[T featurizable](v1, v2 []T) (distance float64, err error) {
	if len(v1) != len(v2) {
		err = errors.New("unequal length vectors provided")
		return
	}
	distance = 0
	for i := 0; i < len(v1); i++ {
		distance += math.Pow(float64(v1[i]-v2[i]), 2)
	}
	distance = math.Sqrt(distance)
	return
}

func CosineDistance[T featurizable](v1, v2 []T) (distance float64, err error) {
	cos, err := Cosine(v1, v2)
	if err == nil {
		distance = 1.0 - cos
	}
	return
}

func ArgMax[T featurizable](data []T) int {
	var maxVal T
	var maxIdx int
	for i, v := range data {
		if v >= maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

func areAllEqual[T featurizable](a, b [][]T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !areEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}

func areEqual[T featurizable](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func Mean[T featurizable](data [][]T) []T {
	mean := make([]T, len(data[0]))
	for _, v := range data {
		for j := 0; j < len(mean); j++ {
			mean[j] = mean[j] + v[j]
		}
	}
	for j := 0; j < len(mean); j++ {
		mean[j] = mean[j] / T(len(data))
	}
	return mean
}

func KMeans[T featurizable](seed int64, data [][]T, k int, maxIterations int, similarityMeasure func([]T, []T) (float64, error)) ([][]T, error) {

	if k > len(data) {
		return data, fmt.Errorf("the size of the data set must at least equal k")
	}

	// Randomly select initial centroids on the dataset
	rand.Seed(seed)
	permutation := rand.Perm(len(data))

	previousCentroidsData := make([][]T, k)
	centroidsData := make([][]T, k)
	for i := 0; i < k; i++ {
		centroidsData[i] = data[permutation[i]]
	}

	similarities := make([]float64, k)
	var closestCentroidIndex int

	for i := 0; i < maxIterations && !areAllEqual(centroidsData, previousCentroidsData); i++ {
		clusters := map[int][][]T{}
		for _, vector := range data {
			// compute distance of point to every centroid
			for centroidIndex, centroidDataVector := range centroidsData {
				similarities[centroidIndex], _ = similarityMeasure(vector, centroidDataVector)
			}

			// find closest centroid to assign the point to
			closestCentroidIndex = ArgMax(similarities)
			clusters[closestCentroidIndex] = append(clusters[closestCentroidIndex], vector)
		}

		// set current centroids to previous
		previousCentroidsData = centroidsData

		// set new centroids to mean of points belonging to them
		for centroidIndex, points := range clusters {
			centroidsData[centroidIndex] = Mean(points)
		}
	}
	return centroidsData, nil
}

type DocumentRelevance struct {
	DocumentVector []float64
	Similarity     float64
}
