package faiss

import (
	"fmt"
	"testing"
)

func TestX(t *testing.T) {
	params, _, err := NewSearchParams(SearchParams{
		SearchType: SearchTypeIVF,
		Nprobe:     1,
	})
	if err != nil {
		t.Error(err)
		return
	}
	fmt.Println(params)
}
