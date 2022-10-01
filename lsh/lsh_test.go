package lsh

import (
	"os"
	"strconv"
	"testing"

	"github.com/pilillo/apostasi/common"
	"github.com/stretchr/testify/assert"
)

var lshUtilTestInstance *lshUtil

func TestMain(m *testing.M) {
	lshUtilTestInstance = NewLshUtil(1234, 16)
	lshUtilTestInstance.Init(0.0, 1.0, 7, 10)
	exitVal := m.Run()
	os.Exit(exitVal)
}

func TestInit(t *testing.T) {
	assert.Equal(t, len(lshUtilTestInstance.randomVectors), 10, "wrong number of random vectors initialized")
}

func TestDot(t *testing.T) {
	dot := lshUtilTestInstance.dot([]float64{1, 2, 3, 4, 5, 6}, []float64{7, 8, 9, 10, 11, 12})
	assert.Equal(t, dot, float64(217))
}

func TestEncodeVector(t *testing.T) {
	p := []float64{1, 2, 3, 4, 5, 6, 7}
	v, err := lshUtilTestInstance.encodeVector(p)
	assert.Nil(t, err)
	assert.Equal(t, int64(127), v, "wrong encoding for input vector")
}

func TestInsertOne(t *testing.T) {
	assert.Equal(t, 0, len(lshUtilTestInstance.table))
	lshUtilTestInstance.InsertOne(0, []float64{1, 2, 3, 4, 5})
	assert.Equal(t, 1, len(lshUtilTestInstance.table))
}

func TestInsert(t *testing.T) {
	lshUtilTestInstance.table = map[int64][]any{}
	assert.Equal(t, 0, len(lshUtilTestInstance.table))
	data := map[any][]float64{}
	lshUtilTestInstance.Insert(data)
	assert.Equal(t, 0, len(lshUtilTestInstance.table))
	data = map[any][]float64{
		1: {1, 2, 3, 4, 5},
	}
	lshUtilTestInstance.Insert(data)
	assert.Equal(t, 1, len(lshUtilTestInstance.table))
	assert.Equal(t, map[int64][]interface{}{31: {1}}, lshUtilTestInstance.table)
}

func TestFlip(t *testing.T) {
	i, _ := strconv.ParseInt("0001", 2, 64)
	assert.Equal(t, int64(0), lshUtilTestInstance.flip(i, []int{0}))
	i, _ = strconv.ParseInt("0010", 2, 64)
	assert.Equal(t, int64(0), lshUtilTestInstance.flip(i, []int{1}))
	i, _ = strconv.ParseInt("0010", 2, 64)
	assert.Equal(t, int64(4), lshUtilTestInstance.flip(i, []int{1, 2}))
}

func TestBucketsInRadius(t *testing.T) {
	lshUtilTestInstance.table = map[int64][]any{
		0: []any{}, 1: []any{}, 2: []any{}, 4: []any{}, 8: []any{}, 16: []any{},
		32: []any{}, 64: []any{}, 128: []any{}, 256: []any{}, 1024: []any{},
		2048: []any{}, 4096: []any{}, 8192: []any{}, 16384: []any{}, 32768: []any{},
	}

	bucketsInRadius := lshUtilTestInstance.getBucketsInRadius(0, 0)
	assert.Equal(t, []int64{0}, bucketsInRadius)
	bucketsInRadius = lshUtilTestInstance.getBucketsInRadius(0, 1)
	assert.Equal(t, []int64{0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384, 32768}, bucketsInRadius)
	lshUtilTestInstance.table = map[int64][]any{}
}

func TestQuery(t *testing.T) {
	lshUtilTestInstance.table = map[int64][]any{
		// bucketId : { docId ...}
		127: {},
	}
	documents, err := lshUtilTestInstance.Query([]float64{0, 0, 0, 1, 1, 1, 1}, 0)
	assert.NoError(t, err)
	assert.Empty(t, documents)

	lshUtilTestInstance.table = map[int64][]any{
		// bucketId : { docId ...}
		127: {1, 2, 3, 4},
	}
	documents, err = lshUtilTestInstance.Query([]float64{0, 0, 0, 1, 1, 1, 1}, 0)
	assert.NoError(t, err)
	assert.NotEmpty(t, documents)
	assert.Equal(t, []any{1, 2, 3, 4}, documents)
}

func TestSortByDescendingDistance(t *testing.T) {
	lshUtilTestInstance.table = map[int64][]any{
		// bucketId : { docId ...}
		127: {1, 2, 3, 4},
	}
	queryDocumentVector := []float64{0, 0, 0, 1, 1, 1, 1}
	relevance := lshUtilTestInstance.SortByDescendingSimilarity(
		queryDocumentVector,
		[][]float64{
			{0, 0, 0, 1, 0, 0, 0},
			{0, 0, 0, 1, 0, 0, 1},
			{0, 0, 0, 1, 1, 1, 0},
		},
	)
	var s float64
	previousSimilarity := 1.0
	for _, r := range relevance {
		s, _ = common.Cosine(r.DocumentVector, queryDocumentVector)
		assert.Equal(t, s, r.Similarity)
		// checking for increasing distance (decreasing similarity)
		assert.GreaterOrEqual(t, previousSimilarity, r.Similarity)
		previousSimilarity = r.Similarity
	}
	assert.Equal(t,
		[]common.DocumentRelevance{
			{DocumentVector: []float64{0, 0, 0, 1, 1, 1, 0}, Similarity: 0.8660254037844387},
			{DocumentVector: []float64{0, 0, 0, 1, 0, 0, 1}, Similarity: 0.7071067811865475},
			{DocumentVector: []float64{0, 0, 0, 1, 0, 0, 0}, Similarity: 0.5},
		},
		relevance,
	)
}
