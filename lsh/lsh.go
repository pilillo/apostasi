package lsh

import (
	"fmt"
	"math/rand"
	"sort"
	"strconv"

	"github.com/pilillo/apostasi/common"
	"gonum.org/v1/gonum/stat/combin"
)

type lshUtil struct {
	seed          int64
	numBits       int
	randomVectors [][]float64
	table         map[int64][]any
}

func NewLshUtil(seed int64, numBits int) *lshUtil {
	rand.Seed(seed)
	return &lshUtil{seed: seed, numBits: numBits, table: map[int64][]any{}}
}

func (lsh *lshUtil) generateRandFloatVectors(min float64, max float64, numFeatures int, numSplits int) [][]float64 {

	res := make([][]float64, numSplits)
	for i := range res {
		res[i] = lsh.generateRandFloatVector(min, max, numFeatures)
	}
	return res
}

func (lsh *lshUtil) generateRandFloatVector(min float64, max float64, numFeatures int) []float64 {
	res := make([]float64, numFeatures)
	for i := range res {
		res[i] = min + rand.Float64()*(max-min)
	}
	return res
}

func (lsh *lshUtil) Init(min float64, max float64, numFeatures int, numSplits int) {
	lsh.randomVectors = lsh.generateRandFloatVectors(min, max, numFeatures, numSplits)
}

func (lsh *lshUtil) dot(v1 []float64, v2 []float64) (dot float64) {
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
	}
	return
}

func (lsh *lshUtil) encodeVector(point []float64) (int64, error) {
	sig := ""
	for i := 0; i < len(point); i++ {
		if lsh.dot(point, lsh.randomVectors[i]) >= 0.0 {
			sig += "1"
		} else {
			sig += "0"
		}
	}

	return strconv.ParseInt(sig, 2, 64)
}

func (lsh *lshUtil) InsertOne(index any, v []float64) error {
	bucketIndex, err := lsh.encodeVector(v)
	if err != nil {
		return err
	}
	bucketContent, exists := lsh.table[bucketIndex]
	if !exists {
		bucketContent = []any{}
		lsh.table[bucketIndex] = bucketContent
	}
	lsh.table[bucketIndex] = append(bucketContent, index)
	return nil
}

func (lsh *lshUtil) Insert(data map[any][]float64) error {
	for k, v := range data {
		if err := lsh.InsertOne(k, v); err != nil {
			return err
		}
	}
	return nil
}

func (lsh *lshUtil) flip(queryBucket int64, flipBits []int) int64 {
	for _, b := range flipBits {
		queryBucket ^= 1 << b
	}
	return queryBucket
}

func (lsh *lshUtil) getBucketsInRadius(queryBucket int64, radius int) []int64 {
	//var err error
	res := []int64{}
	for r := 0; r <= radius; r++ {
		combs := combin.Combinations(lsh.numBits, r)
		for _, c := range combs {
			candidateBucket := lsh.flip(queryBucket, c)
			if _, ok := lsh.table[candidateBucket]; ok {
				res = append(res, candidateBucket)
			}
		}
	}
	return res
}

func (lsh *lshUtil) Query(point []float64, searchRadius int) ([]any, error) {
	// retrieve query bucket
	queryBucket, err := lsh.encodeVector(point)
	if err != nil {
		return nil, err
	}
	_, exists := lsh.table[queryBucket]
	if !exists {
		return nil, fmt.Errorf("missing target bucket %v", queryBucket)
	}

	// retrieve neighboring buckets and collect their documents
	candidates := []any{}
	for _, bucket := range lsh.getBucketsInRadius(queryBucket, searchRadius) {
		// concatenate slices
		candidates = append(candidates, lsh.table[bucket]...)
	}
	return candidates, nil
}

func (lsh *lshUtil) SortByDescendingSimilarity(queryPoint []float64, candidates [][]float64) []common.DocumentRelevance {
	// sort documents in descending similarity from query bucket
	similarities := make([]common.DocumentRelevance, len(candidates))

	for i, candidate := range candidates {
		cos, _ := common.Cosine(queryPoint, candidate)
		similarities[i] = common.DocumentRelevance{
			DocumentVector: candidate,
			Similarity:     cos,
		}
	}

	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].Similarity > similarities[j].Similarity
	})

	return similarities
}
