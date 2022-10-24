package annoy

type queueItem struct {
	index    int
	value    nodeId
	priority float64
}

type priorityQueue []*queueItem

func (q priorityQueue) Len() int {
	return len(q)
}

func (q priorityQueue) Swap(i, j int) {
	q[i], q[j] = q[j], q[i]
	q[i].index = i
	q[j].index = j
}

func (q priorityQueue) Less(i, j int) bool {
	return q[i].priority < q[j].priority
}

func (q *priorityQueue) Push(x any) {
	l := len(*q)
	queueItem := x.(*queueItem)
	queueItem.index = l
	*q = append(*q, queueItem)
}

func (q *priorityQueue) Pop() any {
	old := *q
	n := len(old)
	queueItem := old[n-1]
	queueItem.index = -1
	*q = old[0 : n-1]
	return queueItem
}
