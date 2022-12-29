package annoy

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMain(m *testing.M) {

	exitVal := m.Run()
	os.Exit(exitVal)
}

type capital struct {
	country   string
	city      string
	latitude  float64
	longitude float64
}

type world struct {
	capitals []capital
}

func NewWorld() world {
	return world{
		capitals: []capital{
			{"Aland Islands", "Mariehamn", 60.116667, 19.900000},
			{"Afghanistan", "Kabul", 34.516666666666666, 69.183333},
			{"Albania", "Tirana", 41.31666666666667, 19.816667},
			{"Algeria", "Algiers", 36.75, 3.050000},
			{"Andorra", "Andorra la Vella", 42.5, 1.516667},
			{"Antigua and Barbuda", "Saint John's", 17.116666666666667, -61.850000},
			{"Argentina", "Buenos Aires", -34.583333333333336, -58.666667},
			{"Armenia", "Yerevan", 40.166666666666664, 44.500000},
			{"Austria", "Vienna", 48.2, 16.366667},
			{"Azerbaijan", "Baku", 40.38333333333333, 49.866667},
			{"Belarus", "Minsk", 53.9, 27.566667},
			{"Belgium", "Brussels", 50.833333333333336, 4.333333},
			{"Belize", "Belmopan", 17.25, -88.766667},
			{"Bosnia and Herzegovina", "Sarajevo", 43.86666666666667, 18.416667},
			{"Bulgaria", "Sofia", 42.68333333333333, 23.316667},
			{"Croatia", "Zagreb", 45.8, 16.000000},
			{"Cyprus", "Nicosia", 35.166666666666664, 33.366667},
			{"Czech Republic", "Prague", 50.083333333333336, 14.466667},
			{"Denmark", "Copenhagen", 55.666666666666664, 12.583333},
			{"Estonia", "Tallinn", 59.43333333333333, 24.716667},
			{"Faroe Islands", "Torshavn", 62, -6.766667},
			{"Finland", "Helsinki", 60.166666666666664, 24.933333},
			{"France", "Paris", 48.86666666666667, 2.333333},
			{"Georgia", "Tbilisi", 41.68333333333333, 44.833333},
			{"Germany", "Berlin", 52.516666666666666, 13.400000},
			{"Gibraltar", "Gibraltar", 36.13333333333333, -5.350000},
			{"Greece", "Athens", 37.983333333333334, 23.733333},
			{"Greenland", "Nuuk", 64.18333333333334, -51.750000},
			{"Vatican City", "Vatican City", 41.9, 12.450000},
			{"Hungary", "Budapest", 47.5, 19.083333},
			{"Iceland", "Reykjavik", 64.15, -21.950000},
			{"Ireland", "Dublin", 53.31666666666667, -6.233333},
			{"Isle of Man", "Douglas", 54.15, -4.483333},
			{"Israel", "Jerusalem", 31.766666666666666, 35.233333},
			{"Italy", "Rome", 41.9, 12.483333},
			{"Jamaica", "Kingston", 18, -76.800000},
			{"Japan", "Tokyo", 35.68333333333333, 139.750000},
			{"Jordan", "Amman", 31.95, 35.933333},
			{"Kosovo", "Pristina", 42.666666666666664, 21.166667},
			{"Latvia", "Riga", 56.95, 24.100000},
			{"Liechtenstein", "Vaduz", 47.13333333333333, 9.516667},
			{"Lithuania", "Vilnius", 54.68333333333333, 25.316667},
			{"Luxembourg", "Luxembourg", 49.6, 6.116667},
			{"Macedonia", "Skopje", 42, 21.433333},
			{"Malta", "Valletta", 35.88333333333333, 14.500000},
			{"Moldova", "Chisinau", 47, 28.850000},
			{"Monaco", "Monaco", 43.733333333333334, 7.416667},
			{"Montenegro", "Podgorica", 42.43333333333333, 19.266667},
			{"Netherlands", "Amsterdam", 52.35, 4.916667},
			{"Norway", "Oslo", 59.916666666666664, 10.750000},
			{"Poland", "Warsaw", 52.25, 21.000000},
			{"Portugal", "Lisbon", 38.71666666666667, -9.133333},
			{"Romania", "Bucharest", 44.43333333333333, 26.100000},
			{"Russia", "Moscow", 55.75, 37.600000},
			{"San Marino", "San Marino", 43.93333333333333, 12.416667},
			{"Serbia", "Belgrade", 44.833333333333336, 20.500000},
			{"Slovakia", "Bratislava", 48.15, 17.116667},
			{"Slovenia", "Ljubljana", 46.05, 14.516667},
			{"Spain", "Madrid", 40.4, -3.683333},
			{"Svalbard", "Longyearbyen", 78.21666666666667, 15.633333},
			{"Sweden", "Stockholm", 59.333333333333336, 18.050000},
			{"Switzerland", "Bern", 46.916666666666664, 7.466667},
			{"Turkey", "Ankara", 39.93333333333333, 32.866667},
		},
	}
}

func (w world) toDataset() [][]float64 {
	dataset := make([][]float64, len(w.capitals))
	for index, capital := range w.capitals {
		dataset[index] = []float64{capital.latitude, capital.longitude}
	}
	return dataset
}

func (w world) getCityFromId(id int) (string, string, float64, float64) {
	return w.capitals[id].city, w.capitals[id].country, w.capitals[id].latitude, w.capitals[id].longitude
}

func TestAnnoy(t *testing.T) {

	w := NewWorld()

	k := 5
	treeNum := 10

	index, err := NewIndex(w.toDataset(), 2, treeNum, k)
	assert.NoError(t, err)
	assert.NotNil(t, index)

	city, country, lat, long := w.getCityFromId(34)
	assert.Equal(t, "Rome", city)
	assert.Equal(t, "Italy", country)
	assert.Equal(t, 41.9, lat)
	assert.Equal(t, 12.483333, long)

	bucketScale := float64(5)

	candidateIdToDistances := map[int64]float64{int64(0): float64(10.0), int64(1): float64(5.0), int64(2): float64(1.0), int64(3): float64(0.0)}
	sortedCandidates, err := index.SortCandidates(candidateIdToDistances)
	assert.NoError(t, err)
	assert.NotNil(t, sortedCandidates)
	assert.Equal(t, []int64{3, 2, 1, 0}, sortedCandidates)

	// distance from Rome, Italy - 41.9, 12.483333
	n, err := index.FindSimilarByVector([]float64{lat, long}, k, bucketScale)
	assert.NoError(t, err)
	assert.NotNil(t, n)
	assert.Equal(t, []int64{34, 28, 60, 17, 54}, n)

	c, _, _, _ := w.getCityFromId(34)
	assert.Equal(t, "Rome", c)

	c, _, _, _ = w.getCityFromId(28)
	assert.Equal(t, "Vatican City", c)

	c, _, _, _ = w.getCityFromId(60)
	assert.Equal(t, "Stockholm", c)

	c, _, _, _ = w.getCityFromId(17)
	assert.Equal(t, "Prague", c)

	c, _, _, _ = w.getCityFromId(54)
	assert.Equal(t, "San Marino", c)

	n, err = index.FindSimilarById(34, k, bucketScale)
	assert.NoError(t, err)
	assert.NotNil(t, n)
	assert.Equal(t, []int64{34, 28, 60, 17, 54}, n)
}
