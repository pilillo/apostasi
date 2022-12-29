package annoy

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/pilillo/apostasi/common"
)

type Index interface {
	FindSimilarById(id int64, k int, bucketScale float64) (neighbours []int64, err error)
	FindSimilarByVector(v []float64, k int, bucketScale float64) (neighbours []int64, err error)
	SortCandidates(idToDistance map[int64]float64) ([]int64, error)
}

type index struct {
	// k ... num items in a leaf node
	k    int
	size int
	// trees ... trees indices
	trees []*node
	// nodes ... maps node ids to actual nodes that can be traversed
	nodes map[nodeId]*node
	// items ... maps item ids to actual items (i.e., index+vector pairs)
	items map[dataItemId]*dataItem
}

func (i *index) FindSimilarById(id int64, k int, bucketScale float64) ([]int64, error) {
	it, ok := i.items[dataItemId(id)]
	if !ok {
		return nil, errors.New(fmt.Sprintf("No item found for id: %d", id))
	}
	return i.FindSimilarByVector(it.vector, k, bucketScale)
}

func (i *index) FindSimilarByVector(v []float64, k int, bucketScale float64) (neighbours []int64, err error) {
	// 1. init priority queue and insert the root nodes of all trees
	pq := priorityQueue{}
	for i, r := range i.trees {
		n := &queueItem{
			value:    r.id,
			index:    i,
			priority: math.Inf(-1),
		}
		pq = append(pq, n)
	}

	bucketSize := int(float64(k) * bucketScale)
	annMap := make(map[dataItemId]struct{}, bucketSize)

	// 2. search for candidates in all trees
	heap.Init(&pq)
	for pq.Len() > 0 && len(annMap) < bucketSize {
		q, ok := heap.Pop(&pq).(*queueItem)
		d := q.priority
		n, ok := i.nodes[q.value]
		if !ok {
			return nil, errors.New("invalid index")
		}

		if len(n.leafItems) > 0 {
			for _, id := range n.leafItems {
				annMap[id] = struct{}{}
			}
			continue
		}

		dp := calculateDirection(n.split, v)
		heap.Push(&pq, &queueItem{
			value:    n.leftChild.id,
			priority: math.Max(d, dp),
		})
		heap.Push(&pq, &queueItem{
			value:    n.rightChild.id,
			priority: math.Max(d, -dp),
		})
	}

	// 3. calculate cross-similarity between query vector v and all candidates
	idToDist := make(map[int64]float64, len(annMap))
	//candidates := make([]int64, 0, len(annMap))
	for id := range annMap {
		iid := int64(id)
		//candidates = append(candidates, iid)
		if idToDist[iid], err = common.CosineDistance(i.items[id].vector, v); err != nil {
			return nil, err
		}
	}

	// 3. sort candidates by distance asc
	candidates, err := i.SortCandidates(idToDist)
	if err != nil {
		return nil, err
	}

	// 4. return top k
	if len(candidates) > k {
		candidates = candidates[:k]
	}
	return candidates, nil
}

func (i *index) SortCandidates(idToDistance map[int64]float64) ([]int64, error) {
	candidates := make([]int64, 0, len(idToDistance))

	for id := range idToDistance {
		iid := int64(id)
		candidates = append(candidates, iid)
	}

	// sort candidates by descending similarity / ascending distance
	sort.SliceStable(candidates, func(i, j int) bool {
		return idToDistance[candidates[i]] < idToDistance[candidates[j]]
	})

	return candidates, nil
}

type dataItemId int64
type dataItem struct {
	id     dataItemId
	vector []float64
}

func rawDataFromDataItems(dataItems []*dataItem) [][]float64 {
	result := make([][]float64, len(dataItems))
	for i, v := range dataItems {
		result[i] = v.vector
	}
	return result
}

func dataItemsFromRawData(rawData [][]float64) ([]*dataItem, map[dataItemId]*dataItem) {
	dataItems := make([]*dataItem, len(rawData))
	indexedDataItems := make(map[dataItemId]*dataItem, len(rawData))
	for i, v := range rawData {
		dataItems[i] = &dataItem{id: dataItemId(i), vector: v}
		indexedDataItems[dataItems[i].id] = dataItems[i]
	}
	return dataItems, indexedDataItems
}

func NewIndex(rawData [][]float64, size int, numberOfTrees int, k int) (Index, error) {

	// convert the input matrix to indexed data items so that they can be moved around properly
	dataItems, indexedDataItems := dataItemsFromRawData(rawData)

	index := &index{
		k:     k,
		size:  size,
		trees: make([]*node, numberOfTrees),
		nodes: map[nodeId]*node{}, // map nodeId to node
		items: indexedDataItems,   // map dataItemId to dataItem
	}

	// init trees
	for t := 0; t < numberOfTrees; t++ {
		root := NewNode(dataItems)

		index.trees[t] = root
		index.nodes[root.id] = root
	}

	// build multiple trees in parallel
	var wg sync.WaitGroup
	wg.Add(numberOfTrees)
	for _, treeRoot := range index.trees {
		go func(tr *node, k int) {
			defer wg.Done()
			tr.build(dataItems, k)
		}(treeRoot, k)
	}
	wg.Wait()
	return index, nil
}

func getSplit(dataItems []*dataItem) []float64 {
	seed := time.Now().UnixNano()
	centroids, _ := common.KMeans(seed, rawDataFromDataItems(dataItems), 2, 200, common.EuclideanSimilarity[float64])

	split := make([]float64, len(centroids[0]))
	for d := 0; d < len(centroids[0]); d++ {
		v := centroids[0][d] - centroids[1][d]
		split[d] += v
	}
	return split
}
