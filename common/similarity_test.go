package common

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMain(m *testing.M) {

	exitVal := m.Run()
	os.Exit(exitVal)
}

func TestKMeans(t *testing.T) {
	points := [][]float64{
		{1.0, 1.0},
		{2.0, 1.0},
		{4.0, 3.0},
		{5.0, 4.0},
	}
	seed := int64(1234)
	_, err := KMeans(seed, points, 10, 1, EuclideanSimilarity[float64])
	assert.ErrorContains(t, err, "the size of the data set must at least equal k")
	centroids, err := KMeans(seed, points, 2, 200, EuclideanSimilarity[float64])
	assert.NoError(t, err)
	assert.Equal(t, [][]float64{
		{4.5, 3.5},
		{1.5, 1},
	}, centroids, "wrong clusters found")
}

func TestMean(t *testing.T) {
	points := [][]float64{
		{1, 1, 1},
		{2, 3, 4},
		{0, 2, 7},
	}
	mean := Mean(points)
	expectedMean := []float64{
		1.0, 2.0, 4.0,
	}
	assert.Equal(t, expectedMean, mean, "wrong mean calculated")
}

func TestArgMax(t *testing.T) {
	assert.Equal(t, 1, ArgMax([]int{1, 50, 2, 20}), "wrong argmax index")
}
