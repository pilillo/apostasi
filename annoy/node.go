package annoy

import "github.com/google/uuid"

type nodeId string

type node struct {
	id nodeId

	split      []float64
	leftChild  *node
	rightChild *node

	leafItems []dataItemId
}

func NewNode(dataItems []*dataItem) *node {
	return &node{
		id:    nodeId(uuid.New().String()),
		split: getSplit(dataItems),

		leftChild:  nil,
		rightChild: nil,

		leafItems: []dataItemId{},
	}
}

func (n *node) build(dataItems []*dataItem, k int) {
	// base case, the node is a leaf has items are less than k
	if len(dataItems) <= k {
		n.leafItems = make([]dataItemId, len(dataItems))
		for i, dataItem := range dataItems {
			n.leafItems[i] = dataItem.id
		}
	} else {
		// inductive case, the node must be split into its children
		leftItems := []*dataItem{}
		rightItems := []*dataItem{}
		for _, dataItem := range dataItems {
			if calculateDirection(dataItem.vector, n.split) > 0 {
				rightItems = append(rightItems, dataItem)
			} else {
				leftItems = append(leftItems, dataItem)
			}
		}
		// avoid splitting too much on a side only unless the split is greater than the min k size
		if len(leftItems) <= k || len(rightItems) <= k {
			n.leafItems = make([]dataItemId, len(dataItems))
			for i, dataItem := range dataItems {
				n.leafItems[i] = dataItem.id
			}
		} else {
			// build left child
			n.leftChild = NewNode(leftItems)
			n.leftChild.build(leftItems, k)
			// build right child
			n.rightChild = NewNode(rightItems)
			n.rightChild.build(rightItems, k)
		}
	}
}

func calculateDirection(point, target []float64) float64 {
	direction := 0.0
	for i := range point {
		direction += point[i] * target[i]
	}
	return direction
}
