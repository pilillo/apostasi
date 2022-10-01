package common

import (
	"errors"
	"math"

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

type DocumentRelevance struct {
	DocumentVector []float64
	Similarity     float64
}
