package faiss

/*
#include <stdlib.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/impl/AuxIndexStructures_c.h>
#include <faiss/c_api/index_factory_c.h>
#include <faiss/c_api/IndexIVF_c.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

// Index is a Faiss index.
//
// Note that some index implementations do not support all methods.
// Check the Faiss wiki to see what operations an index supports.
type Index interface {
	// D returns the dimension of the indexed vectors.
	D() int

	// IsTrained returns true if the index has been trained or does not require
	// training.
	IsTrained() bool

	// Ntotal returns the number of indexed vectors.
	Ntotal() int64

	// MetricType returns the metric type of the index.
	MetricType() int

	// Train trains the index on a representative set of vectors.
	Train(x []float32) error

	// Add adds vectors to the index.
	Add(x []float32) error

	// AddWithIDs is like Add, but stores xids instead of sequential IDs.
	AddWithIDs(x []float32, xids []int64) error

	// Search queries the index with the vectors in x.
	// Returns the IDs of the k nearest neighbors for each query vector and the
	// corresponding distances.
	Search(x []float32, k int64) (distances []float32, labels []int64, err error)

	SearchWithParameters(x []float32, k int64, params *SearchParams) (distances []float32, labels []int64, err error)

	// RangeSearch queries the index with the vectors in x.
	// Returns all vectors with distance < radius.
	RangeSearch(x []float32, radius float32) (*RangeSearchResult, error)

	// Reset removes all vectors from the index.
	Reset() error

	// RemoveIDs removes the vectors specified by sel from the index.
	// Returns the number of elements removed and error.
	RemoveIDs(sel *IDSelector) (int, error)

	// Delete frees the memory used by the index.
	Delete()

	cPtr() *C.FaissIndex
}

type faissIndex struct {
	idx *C.FaissIndex
}

func (idx *faissIndex) cPtr() *C.FaissIndex {
	return idx.idx
}

func (idx *faissIndex) D() int {
	return int(C.faiss_Index_d(idx.idx))
}

func (idx *faissIndex) IsTrained() bool {
	return C.faiss_Index_is_trained(idx.idx) != 0
}

func (idx *faissIndex) Ntotal() int64 {
	return int64(C.faiss_Index_ntotal(idx.idx))
}

func (idx *faissIndex) MetricType() int {
	return int(C.faiss_Index_metric_type(idx.idx))
}

func (idx *faissIndex) Train(x []float32) error {
	n := len(x) / idx.D()
	if c := C.faiss_Index_train(idx.idx, C.idx_t(n), (*C.float)(&x[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *faissIndex) Add(x []float32) error {
	n := len(x) / idx.D()
	if c := C.faiss_Index_add(idx.idx, C.idx_t(n), (*C.float)(&x[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *faissIndex) AddWithIDs(x []float32, xids []int64) error {
	n := len(x) / idx.D()
	if c := C.faiss_Index_add_with_ids(
		idx.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		(*C.idx_t)(&xids[0]),
	); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *faissIndex) Search(x []float32, k int64) (
	distances []float32, labels []int64, err error,
) {
	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)
	if c := C.faiss_Index_search(
		idx.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		err = getLastError()
	}
	return
}

func (idx *faissIndex) SearchWithParameters(x []float32, k int64, params *SearchParams) (
	distances []float32, labels []int64, err error,
) {
	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)
	searchParams, f, err := NewSearchParams(params)
	defer f()
	if err != nil {
		return nil, nil, err
	}

	if c := C.faiss_Index_search_with_params(
		idx.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		searchParams,
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		err = getLastError()
	}
	return
}

func (idx *faissIndex) RangeSearch(x []float32, radius float32) (
	*RangeSearchResult, error,
) {
	n := len(x) / idx.D()
	var rsr *C.FaissRangeSearchResult
	if c := C.faiss_RangeSearchResult_new(&rsr, C.idx_t(n)); c != 0 {
		return nil, getLastError()
	}
	if c := C.faiss_Index_range_search(
		idx.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.float(radius),
		rsr,
	); c != 0 {
		return nil, getLastError()
	}
	return &RangeSearchResult{rsr}, nil
}

func (idx *faissIndex) Reset() error {
	if c := C.faiss_Index_reset(idx.idx); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *faissIndex) RemoveIDs(sel *IDSelector) (int, error) {
	var nRemoved C.size_t
	if c := C.faiss_Index_remove_ids(idx.idx, sel.sel, &nRemoved); c != 0 {
		return 0, getLastError()
	}
	return int(nRemoved), nil
}

func (idx *faissIndex) Delete() {
	C.faiss_Index_free(idx.idx)
}

// RangeSearchResult is the result of a range search.
type RangeSearchResult struct {
	rsr *C.FaissRangeSearchResult
}

// Nq returns the number of queries.
func (r *RangeSearchResult) Nq() int {
	return int(C.faiss_RangeSearchResult_nq(r.rsr))
}

// Lims returns a slice containing start and end indices for queries in the
// distances and labels slices returned by Labels.
func (r *RangeSearchResult) Lims() []int {
	var lims *C.size_t
	C.faiss_RangeSearchResult_lims(r.rsr, &lims)
	length := r.Nq() + 1
	return (*[1 << 30]int)(unsafe.Pointer(lims))[:length:length]
}

// Labels returns the unsorted IDs and respective distances for each query.
// The result for query i is labels[lims[i]:lims[i+1]].
func (r *RangeSearchResult) Labels() (labels []int64, distances []float32) {
	lims := r.Lims()
	length := lims[len(lims)-1]
	var clabels *C.idx_t
	var cdist *C.float
	C.faiss_RangeSearchResult_labels(r.rsr, &clabels, &cdist)
	labels = (*[1 << 30]int64)(unsafe.Pointer(clabels))[:length:length]
	distances = (*[1 << 30]float32)(unsafe.Pointer(cdist))[:length:length]
	return
}

// Delete frees the memory associated with r.
func (r *RangeSearchResult) Delete() {
	C.faiss_RangeSearchResult_free(r.rsr)
}

// IndexImpl is an abstract structure for an index.
type IndexImpl struct {
	Index
}

// IndexFactory builds a composite index.
// description is a comma-separated list of components.
func IndexFactory(d int, description string, metric int) (*IndexImpl, error) {
	cdesc := C.CString(description)
	defer C.free(unsafe.Pointer(cdesc))
	var idx faissIndex
	c := C.faiss_index_factory(&idx.idx, C.int(d), cdesc, C.FaissMetricType(metric))
	if c != 0 {
		return nil, getLastError()
	}
	return &IndexImpl{&idx}, nil
}

type SearchType int

const (
	SearchTypeFlat SearchType = iota
	SearchTypeIVF
	SearchTypeIVFPQ
)

type SearchParams struct {
	SearchType SearchType
	Nprobe     int     // 用于IVF类型
	MaxCodes   int64   // 最大检查的编码数
	Polarity   float32 // 用于某些特定搜索
}

// NewSearchParams 创建搜索参数
func NewSearchParams(params *SearchParams) (*C.FaissSearchParameters, func(), error) {
	var searchParams *C.FaissSearchParameters
	var cleanupFunc func()
	switch params.SearchType {
	case SearchTypeFlat:
		// Flat索引使用默认参数
		searchParams = nil
		cleanupFunc = func() {}

	case SearchTypeIVF, SearchTypeIVFPQ:
		var ivfSearchParams *C.FaissSearchParametersIVF
		// 创建IVF搜索参数
		ivfParams := C.faiss_SearchParametersIVF_new_with(
			&ivfSearchParams,
			nil,
			C.size_t(params.Nprobe),
			C.size_t(params.MaxCodes),
		)
		if ivfParams == nil {
			return nil, nil, errors.New("failed to create IVF search parameters")
		}
		// 转换为通用参数
		searchParams = (*C.FaissSearchParameters)(unsafe.Pointer(ivfParams))
		cleanupFunc = func() {
			C.free(ivfParams)
		}

	default:
		return nil, nil, errors.New("unsupported search type")
	}

	return searchParams, cleanupFunc, nil
}
