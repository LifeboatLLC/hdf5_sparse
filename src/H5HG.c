/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the LICENSE file, which can be found at the root of the source code       *
 * distribution tree, or in https://www.hdfgroup.org/licenses.               *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Purpose:	Operations on the global heap.  The global heap is the set of
 *		all collections and each collection contains one or more
 *		global heap objects.  An object belongs to exactly one
 *		collection.  A collection is treated as an atomic entity for
 *		the purposes of I/O and caching.
 *
 *		Each file has a small cache of global heap collections called
 *		the CWFS list and recently accessed collections with free
 *		space appear on this list.  As collections are accessed the
 *		collection is moved toward the front of the list.  New
 *		collections are added to the front of the list while old
 *		collections are added to the end of the list.
 *
 *		The collection model reduces the overhead which would be
 *		incurred if the global heap were a single object, and the
 *		CWFS list allows the library to cheaply choose a collection
 *		for a new object based on object size, amount of free space
 *		in the collection, and temporal locality.
 */

/****************/
/* Module Setup */
/****************/

#include "H5HGmodule.h" /* This source code file is part of the H5HG module */

/***********/
/* Headers */
/***********/
#include "H5private.h"   /* Generic Functions			*/
#include "H5Eprivate.h"  /* Error handling		  	*/
#include "H5Fprivate.h"  /* File access				*/
#include "H5FLprivate.h" /* Free Lists                               */
#include "H5HGpkg.h"     /* Global heaps				*/
#include "H5MFprivate.h" /* File memory management		*/
#include "H5MMprivate.h" /* Memory management			*/

/****************/
/* Local Macros */
/****************/

/*
 * The maximum number of links allowed to a global heap object.
 */
#define H5HG_MAXLINK 65535

/*
 * The maximum number of indices allowed in a global heap object.
 */
#define H5HG_MAXIDX 65535

/******************/
/* Local Typedefs */
/******************/

/********************/
/* Package Typedefs */
/********************/

/********************/
/* Local Prototypes */
/********************/

static haddr_t H5HG__create(H5F_t *f, size_t size);
static size_t  H5HG__alloc(H5F_t *f, H5HG_heap_t *heap, size_t size, unsigned *heap_flags_ptr);

static H5HG_local_heapset_t *H5HG__alloc_local_heapset(size_t nalloc);
static herr_t                H5HG__grow_local_heapset(H5HG_local_heapset_t **heapset_ptr, size_t min_nalloc);
static void                  H5HG__trim_local_heapset(H5HG_local_heapset_t *heapset);

static herr_t H5HG__reserve_local(H5F_t *f, H5HG_local_heapset_t *heapset, H5HG_heap_t *heap, size_t need);

static herr_t H5HG__local_heap_alloc_size(const H5HG_heap_t *heap, size_t *size_out);

/*********************/
/* Package Variables */
/*********************/

/* Package initialization variable */
bool H5_PKG_INIT_VAR = false;

/* Declare a free list to manage the H5HG_heap_t struct */
H5FL_DEFINE(H5HG_heap_t);

/* Declare a free list to manage sequences of H5HG_obj_t's */
H5FL_SEQ_DEFINE(H5HG_obj_t);

/* Declare a PQ free list to manage heap chunks */
H5FL_BLK_DEFINE(gheap_chunk);

/*****************************/
/* Library Private Variables */
/*****************************/

/*******************/
/* Local Variables */
/*******************/

/*-------------------------------------------------------------------------
 * Function:	H5HG__create
 *
 * Purpose:	Creates a global heap collection of the specified size.  If
 *		SIZE is less than some minimum it will be readjusted.  The
 *		new collection is allocated in the file and added to the
 *		beginning of the CWFS list.
 *
 * Return:	Success:	Ptr to a cached heap.  The pointer is valid
 *				only until some other hdf5 library function
 *				is called.
 *
 *		Failure:	NULL
 *
 *-------------------------------------------------------------------------
 */
static haddr_t
H5HG__create(H5F_t *f, size_t size)
{
    H5HG_heap_t *heap = NULL;
    uint8_t     *p    = NULL;
    haddr_t      addr = HADDR_UNDEF;
    size_t       n;
    haddr_t      ret_value = HADDR_UNDEF; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    if (size < H5HG_MINSIZE)
        size = H5HG_MINSIZE;
    size = H5HG_ALIGN(size);

    /* Create it */
    H5_CHECK_OVERFLOW(size, size_t, hsize_t);
    if (HADDR_UNDEF == (addr = H5MF_alloc(f, H5FD_MEM_GHEAP, (hsize_t)size)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, HADDR_UNDEF, "unable to allocate file space for global heap");
    if (NULL == (heap = H5FL_CALLOC(H5HG_heap_t)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, HADDR_UNDEF, "memory allocation failed");
    heap->addr   = addr;
    heap->size   = size;
    heap->shared = H5F_SHARED(f);

    if (NULL == (heap->chunk = H5FL_BLK_MALLOC(gheap_chunk, size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, HADDR_UNDEF, "memory allocation failed");
    memset(heap->chunk, 0, size);
    heap->nalloc = H5HG_NOBJS(f, size);
    heap->nused  = 1; /* account for index 0, which is used for the free object */
    if (NULL == (heap->obj = H5FL_SEQ_MALLOC(H5HG_obj_t, heap->nalloc)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, HADDR_UNDEF, "memory allocation failed");

    /* Initialize the header */
    H5MM_memcpy(heap->chunk, H5HG_MAGIC, (size_t)H5_SIZEOF_MAGIC);
    p    = heap->chunk + H5_SIZEOF_MAGIC;
    *p++ = H5HG_VERSION;
    *p++ = 0; /*reserved*/
    *p++ = 0; /*reserved*/
    *p++ = 0; /*reserved*/
    H5F_ENCODE_LENGTH(f, p, size);

    /*
     * Padding so free space object is aligned. If malloc returned memory
     * which was always at least H5HG_ALIGNMENT aligned then we could just
     * align the pointer, but this might not be the case.
     */
    n = (size_t)H5HG_ALIGN(p - heap->chunk) - (size_t)(p - heap->chunk);
    p += n;

    /* The freespace object */
    heap->obj[0].size = size - H5HG_SIZEOF_HDR(f);
    assert(H5HG_ISALIGNED(heap->obj[0].size));
    heap->obj[0].nrefs = 0;
    heap->obj[0].begin = p;
    UINT16ENCODE(p, 0); /*object ID*/
    UINT16ENCODE(p, 0); /*reference count*/
    UINT32ENCODE(p, 0); /*reserved*/
    H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);

    /* Add this heap to the beginning of the CWFS list */
    if (H5F_cwfs_add(f, heap) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, HADDR_UNDEF,
                    "unable to add global heap collection to file's CWFS");

    /* Add the heap to the cache */
    if (H5AC_insert_entry(f, H5AC_GHEAP, addr, heap, H5AC__NO_FLAGS_SET) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, HADDR_UNDEF, "unable to cache global heap collection");

    ret_value = addr;

done:
    /* Cleanup on error */
    if (!H5_addr_defined(ret_value)) {
        if (H5_addr_defined(addr)) {
            /* Release the space on disk */
            if (H5MF_xfree(f, H5FD_MEM_GHEAP, addr, (hsize_t)size) < 0)
                HDONE_ERROR(H5E_BTREE, H5E_CANTFREE, HADDR_UNDEF, "unable to free global heap");

            /* Check if the heap object was allocated */
            if (heap)
                /* Destroy the heap object */
                if (H5HG__free(heap) < 0)
                    HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, HADDR_UNDEF,
                                "unable to destroy global heap collection");
        } /* end if */
    }     /* end if */

    FUNC_LEAVE_NOAPI(ret_value)
} /* H5HG__create() */

/*-------------------------------------------------------------------------
 * Function:	H5HG__protect
 *
 * Purpose:	Convenience wrapper around H5AC_protect on an indirect block
 *
 * Return:	Pointer to indirect block on success, NULL on failure
 *
 *-------------------------------------------------------------------------
 */
H5HG_heap_t *
H5HG__protect(H5F_t *f, haddr_t addr, unsigned flags)
{
    H5HG_heap_t *heap;             /* Global heap */
    H5HG_heap_t *ret_value = NULL; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check arguments */
    assert(f);
    assert(H5_addr_defined(addr));

    /* only H5AC__READ_ONLY_FLAG may appear in flags */
    assert((flags & (unsigned)(~H5AC__READ_ONLY_FLAG)) == 0);

    /* Lock the heap into memory */
    if (NULL == (heap = (H5HG_heap_t *)H5AC_protect(f, H5AC_GHEAP, addr, f, flags)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, NULL, "unable to protect global heap");

    /* Set the heap's address */
    heap->addr = addr;

    /* Set the return value */
    ret_value = heap;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5HG__protect() */

/*-------------------------------------------------------------------------
 * Function:	H5HG__alloc
 *
 * Purpose:	Given a heap with enough free space, this function will split
 *		the free space to make a new empty heap object and initialize
 *		the header.  SIZE is the exact size of the object data to be
 *		stored. It will be increased to make room for the object
 *		header and then rounded up for alignment.
 *
 * Return:	Success:	The heap object ID of the new object.
 *
 *		Failure:	0
 *
 *-------------------------------------------------------------------------
 */
static size_t
H5HG__alloc(H5F_t *f, H5HG_heap_t *heap, size_t size, unsigned *heap_flags_ptr)
{
    size_t   idx;
    uint8_t *p;
    size_t   need      = H5HG_SIZEOF_OBJHDR(f) + H5HG_ALIGN(size);
    size_t   ret_value = 0; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(heap);
    assert(heap->obj[0].size >= need);
    assert(heap_flags_ptr);

    /*
     * Find an ID for the new object. ID zero is reserved for the free space
     * object.
     */
    if (heap->nused <= H5HG_MAXIDX)
        idx = heap->nused++;
    else {
        for (idx = 1; idx < heap->nused; idx++)
            if (NULL == heap->obj[idx].begin)
                break;
    } /* end else */

    assert(idx < heap->nused);

    /* Check if we need more room to store heap objects */
    if (idx >= heap->nalloc) {
        size_t      new_alloc; /* New allocation number */
        H5HG_obj_t *new_obj;   /* New array of object descriptions */

        /* Determine the new number of objects to index */
        /* nalloc is *not* guaranteed to be a power of 2! - NAF 10/26/09 */
        new_alloc = MIN(MAX(heap->nalloc * 2, (idx + 1)), (H5HG_MAXIDX + 1));
        assert(idx < new_alloc);

        /* Reallocate array of objects */
        if (NULL == (new_obj = H5FL_SEQ_REALLOC(H5HG_obj_t, heap->obj, new_alloc)))
            HGOTO_ERROR(H5E_HEAP, H5E_CANTALLOC, 0, "memory allocation failed");

        /* Clear newly allocated space */
        memset(&new_obj[heap->nalloc], 0, (new_alloc - heap->nalloc) * sizeof(heap->obj[0]));

        /* Update heap information */
        heap->nalloc = new_alloc;
        heap->obj    = new_obj;
        assert(heap->nalloc > heap->nused);
    } /* end if */

    /* Initialize the new object */
    heap->obj[idx].nrefs = 0;
    heap->obj[idx].size  = size;
    heap->obj[idx].begin = heap->obj[0].begin;
    p                    = heap->obj[idx].begin;
    UINT16ENCODE(p, idx);
    UINT16ENCODE(p, 0); /*nrefs*/
    UINT32ENCODE(p, 0); /*reserved*/
    H5F_ENCODE_LENGTH(f, p, size);

    /* Fix the free space object */
    if (need == heap->obj[0].size) {
        /*
         * All free space has been exhausted from this collection.
         */
        heap->obj[0].size  = 0;
        heap->obj[0].begin = NULL;
    } /* end if */
    else if (heap->obj[0].size - need >= H5HG_SIZEOF_OBJHDR(f)) {
        /*
         * Some free space remains and it's larger than a heap object header,
         * so write the new free heap object header to the heap.
         */
        heap->obj[0].size -= need;
        heap->obj[0].begin += need;
        p = heap->obj[0].begin;
        UINT16ENCODE(p, 0); /*id*/
        UINT16ENCODE(p, 0); /*nrefs*/
        UINT32ENCODE(p, 0); /*reserved*/
        H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);
        assert(H5HG_ISALIGNED(heap->obj[0].size));
    } /* end else-if */
    else {
        /*
         * Some free space remains but it's smaller than a heap object header,
         * so we don't write the header.
         */
        heap->obj[0].size -= need;
        heap->obj[0].begin += need;
        assert(H5HG_ISALIGNED(heap->obj[0].size));
    }

    /* Mark the heap as dirty */
    *heap_flags_ptr |= H5AC__DIRTIED_FLAG;

    /* Set the return value */
    ret_value = idx;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__alloc() */

/*-------------------------------------------------------------------------
 * Function:	H5HG_extend
 *
 * Purpose:	Extend a heap to hold an object of SIZE bytes.
 *		SIZE is the exact size of the object data to be
 *		stored. It will be increased to make room for the object
 *		header and then rounded up for alignment.
 *
 * Return:	Success:	Non-negative
 *
 *		Failure:	Negative
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG_extend(H5F_t *f, haddr_t addr, size_t need)
{
    H5HG_heap_t *heap       = NULL;               /* Pointer to heap to extend */
    unsigned     heap_flags = H5AC__NO_FLAGS_SET; /* Flags to unprotecting heap */
    size_t       old_size;                        /* Previous size of the heap's chunk */
    uint8_t     *new_chunk;                       /* Pointer to new chunk information */
    uint8_t     *p;                               /* Pointer to raw heap info */
    unsigned     u;                               /* Local index variable */
    herr_t       ret_value = SUCCEED;             /* Return value */

    FUNC_ENTER_NOAPI_NOINIT

    /* Check args */
    assert(f);
    assert(H5_addr_defined(addr));

    /* Protect the heap */
    if (NULL == (heap = H5HG__protect(f, addr, H5AC__NO_FLAGS_SET)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, FAIL, "unable to protect global heap");

    /* Re-allocate the heap information in memory */
    if (NULL == (new_chunk = H5FL_BLK_REALLOC(gheap_chunk, heap->chunk, (heap->size + need))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "new heap allocation failed");
    memset(new_chunk + heap->size, 0, need);

    /* Adjust the size of the heap */
    old_size = heap->size;
    heap->size += need;

    /* Encode the new size of the heap */
    p = new_chunk + H5_SIZEOF_MAGIC + 1 /* version */ + 3 /* reserved */;
    H5F_ENCODE_LENGTH(f, p, heap->size);

    /* Move the pointers to the existing objects to their new locations */
    for (u = 0; u < heap->nused; u++)
        if (heap->obj[u].begin)
            heap->obj[u].begin = new_chunk + (heap->obj[u].begin - heap->chunk);

    /* Update the heap chunk pointer now */
    heap->chunk = new_chunk;

    /* Update the free space information for the heap  */
    heap->obj[0].size += need;
    if (heap->obj[0].begin == NULL)
        heap->obj[0].begin = heap->chunk + old_size;
    p = heap->obj[0].begin;
    UINT16ENCODE(p, 0); /*id*/
    UINT16ENCODE(p, 0); /*nrefs*/
    UINT32ENCODE(p, 0); /*reserved*/
    H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);
    assert(H5HG_ISALIGNED(heap->obj[0].size));

    /* Resize the heap in the cache */
    if (H5AC_resize_entry(heap, heap->size) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTRESIZE, FAIL, "unable to resize global heap in cache");

    /* Mark the heap as dirty */
    heap_flags |= H5AC__DIRTIED_FLAG;

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, heap->addr, heap, heap_flags) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, FAIL, "unable to unprotect heap");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG_extend() */

/*-------------------------------------------------------------------------
 * Function:	H5HG_insert
 *
 * Purpose:	A new object is inserted into the global heap.  It will be
 *		placed in the first collection on the CWFS list which has
 *		enough free space and that collection will be advanced one
 *		position in the list.  If no collection on the CWFS list has
 *		enough space then  a new collection will be created.
 *
 *		It is legal to push a zero-byte object onto the heap to get
 *		the reference count features of heap objects.
 *
 * Return:	Success:	Non-negative, and a heap object handle returned
 *				through the HOBJ pointer.
 *
 *		Failure:	Negative
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG_insert(H5F_t *f, size_t size, const void *obj, H5HG_t *hobj /*out*/)
{
    size_t       need; /*total space needed for object		*/
    size_t       idx;
    haddr_t      addr; /* Address of heap to add object within */
    H5HG_heap_t *heap       = NULL;
    unsigned     heap_flags = H5AC__NO_FLAGS_SET;
    herr_t       ret_value  = SUCCEED; /* Return value */

    FUNC_ENTER_NOAPI_TAG(H5AC__GLOBALHEAP_TAG, FAIL)

    /* Check args */
    assert(f);
    assert(0 == size || obj);
    assert(hobj);

    if (0 == (H5F_INTENT(f) & H5F_ACC_RDWR))
        HGOTO_ERROR(H5E_HEAP, H5E_WRITEERROR, FAIL, "no write intent on file");

    /* Find a large enough collection on the CWFS list */
    need = H5HG_SIZEOF_OBJHDR(f) + H5HG_ALIGN(size);

    /* Look for a heap in the file's CWFS that has enough space for the object */
    addr = HADDR_UNDEF;
    if (H5F_cwfs_find_free_heap(f, need, &addr) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_NOTFOUND, FAIL, "error trying to locate heap");

    /*
     * If we didn't find any collection with enough free space then allocate a
     * new collection large enough for the message plus the collection header.
     */
    if (!H5_addr_defined(addr)) {
        addr = H5HG__create(f, need + H5HG_SIZEOF_HDR(f));

        if (!H5_addr_defined(addr))
            HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, FAIL, "unable to allocate a global heap collection");
    } /* end if */
    assert(H5_addr_defined(addr));

    if (NULL == (heap = H5HG__protect(f, addr, H5AC__NO_FLAGS_SET)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, FAIL, "unable to protect global heap");

    /* Split the free space to make room for the new object */
    if (0 == (idx = H5HG__alloc(f, heap, size, &heap_flags)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTALLOC, FAIL, "unable to allocate global heap object");

    /* Copy data into the heap */
    if (size > 0)
        H5MM_memcpy(heap->obj[idx].begin + H5HG_SIZEOF_OBJHDR(f), obj, size);
    heap_flags |= H5AC__DIRTIED_FLAG;

    /* Return value */
    hobj->addr = heap->addr;
    hobj->idx  = idx;

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, heap->addr, heap, heap_flags) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, FAIL, "unable to unprotect heap.");

    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* H5HG_insert() */

/*-------------------------------------------------------------------------
 * Function:	H5HG_read
 *
 * Purpose:	Reads the specified global heap object into the buffer OBJECT
 *		supplied by the caller.  If the caller doesn't supply a
 *		buffer then one will be allocated.  The buffer should be
 *		large enough to hold the result.
 *
 * Return:	Success:	The buffer containing the result.
 *
 *		Failure:	NULL
 *
 *-------------------------------------------------------------------------
 */
void *
H5HG_read(H5F_t *f, H5HG_t *hobj, void *object /*out*/, size_t *buf_size)
{
    H5HG_heap_t *heap = NULL;          /* Pointer to global heap object */
    size_t       size;                 /* Size of the heap object */
    uint8_t     *p;                    /* Pointer to object in heap buffer */
    void        *orig_object = object; /* Keep a copy of the original object pointer */
    void        *ret_value   = NULL;   /* Return value */

    FUNC_ENTER_NOAPI_TAG(H5AC__GLOBALHEAP_TAG, NULL)

    /* Check args */
    assert(f);
    assert(hobj);

    /* Heap object idx 0 is the free space in the heap and should never be given out */
    if (0 == hobj->idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* Load the heap */
    if (NULL == (heap = H5HG__protect(f, hobj->addr, H5AC__READ_ONLY_FLAG)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, NULL, "unable to protect global heap");
    if (hobj->idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);
    if (NULL == heap->obj[hobj->idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad heap pointer, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    size = heap->obj[hobj->idx].size;
    p    = heap->obj[hobj->idx].begin + H5HG_SIZEOF_OBJHDR(f);

    /* Allocate a buffer for the object read in, if the user didn't give one */
    if (!object && NULL == (object = H5MM_malloc(size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "memory allocation failed");
    H5MM_memcpy(object, p, size);

    /*
     * Advance the heap in the CWFS list. We might have done this already
     * with the H5AC_protect(), but it won't hurt to do it twice.
     */
    if (heap->obj[0].begin) {
        if (H5F_cwfs_advance_heap(f, heap, false) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTMODIFY, NULL, "can't adjust file's CWFS");
    } /* end if */

    /* If the caller would like to know the heap object's size, set that */
    if (buf_size)
        *buf_size = size;

    /* Set return value */
    ret_value = object;

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, hobj->addr, heap, H5AC__NO_FLAGS_SET) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, NULL, "unable to release object header");

    if (NULL == ret_value && NULL == orig_object && object)
        H5MM_free(object);

    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* end H5HG_read() */

/*-------------------------------------------------------------------------
 * Function:	H5HG_link
 *
 * Purpose:	Adjusts the link count for a global heap object by adding
 *		ADJUST to the current value.  This function will fail if the
 *		new link count would overflow.  Nothing special happens when
 *		the link count reaches zero; in order for a heap object to be
 *		removed one must call H5HG_remove().
 *
 * Return:	Success:	Number of links present after the adjustment.
 *
 *		Failure:	Negative
 *
 *-------------------------------------------------------------------------
 */
int
H5HG_link(H5F_t *f, const H5HG_t *hobj, int adjust)
{
    H5HG_heap_t *heap       = NULL;
    unsigned     heap_flags = H5AC__NO_FLAGS_SET;
    int          ret_value  = -1; /* Return value */

    FUNC_ENTER_NOAPI_TAG(H5AC__GLOBALHEAP_TAG, FAIL)

    /* Check args */
    assert(f);
    assert(hobj);
    if (0 == (H5F_INTENT(f) & H5F_ACC_RDWR))
        HGOTO_ERROR(H5E_HEAP, H5E_WRITEERROR, FAIL, "no write intent on file");

    /* Heap object idx 0 is the free space in the heap and should never be given out */
    if (0 == hobj->idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* Load the heap */
    if (NULL == (heap = H5HG__protect(f, hobj->addr, H5AC__NO_FLAGS_SET)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, FAIL, "unable to protect global heap");

    if (adjust != 0) {
        if (hobj->idx >= heap->nused)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                        hobj->addr, hobj->idx);
        if (NULL == heap->obj[hobj->idx].begin)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap pointer, heap object = {%" PRIxHADDR ", %zu}",
                        hobj->addr, hobj->idx);
        if ((heap->obj[hobj->idx].nrefs + adjust) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "new link count would be out of range");
        if ((heap->obj[hobj->idx].nrefs + adjust) > H5HG_MAXLINK)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "new link count would be out of range");
        heap->obj[hobj->idx].nrefs += adjust;
        heap_flags |= H5AC__DIRTIED_FLAG;
    } /* end if */

    /* Set return value */
    ret_value = heap->obj[hobj->idx].nrefs;

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, hobj->addr, heap, heap_flags) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, FAIL, "unable to release object header");

    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* end H5HG_link() */

/*-------------------------------------------------------------------------
 * Function:    H5HG_get_obj_size
 *
 * Purpose:     Returns the size of a global heap object.
 * Return:      Success:        Non-negative
 *
 *              Failure:        Negative
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG_get_obj_size(H5F_t *f, H5HG_t *hobj, size_t *obj_size)
{
    H5HG_heap_t *heap      = NULL;    /* Pointer to global heap object */
    herr_t       ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_NOAPI_TAG(H5AC__GLOBALHEAP_TAG, FAIL)

    /* Check args */
    assert(f);
    assert(hobj);
    assert(obj_size);

    /* Heap object idx 0 is the free space in the heap and should never be given out */
    if (0 == hobj->idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* Load the heap */
    if (NULL == (heap = H5HG__protect(f, hobj->addr, H5AC__READ_ONLY_FLAG)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, FAIL, "unable to protect global heap");

    /* Sanity check the heap object */
    if (hobj->idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);
    if (NULL == heap->obj[hobj->idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap pointer, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* Set object size */
    *obj_size = heap->obj[hobj->idx].size;

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, hobj->addr, heap, H5AC__NO_FLAGS_SET) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, FAIL, "unable to release object header");

    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* end H5HG_get_obj_size() */

/*-------------------------------------------------------------------------
 * Function:	H5HG_remove
 *
 * Purpose:	Removes the specified object from the global heap.
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG_remove(H5F_t *f, H5HG_t *hobj)
{
    H5HG_heap_t *heap = NULL;
    uint8_t     *p = NULL, *obj_start = NULL;
    size_t       need;
    unsigned     u;
    unsigned     flags     = H5AC__NO_FLAGS_SET; /* Whether the heap gets deleted */
    herr_t       ret_value = SUCCEED;            /* Return value */

    FUNC_ENTER_NOAPI_TAG(H5AC__GLOBALHEAP_TAG, FAIL)

    /* Check args */
    assert(f);
    assert(hobj);
    if (0 == (H5F_INTENT(f) & H5F_ACC_RDWR))
        HGOTO_ERROR(H5E_HEAP, H5E_WRITEERROR, FAIL, "no write intent on file");

    /* Heap object idx 0 is the free space in the heap and should never be given out */
    if (0 == hobj->idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* Load the heap */
    if (NULL == (heap = H5HG__protect(f, hobj->addr, H5AC__NO_FLAGS_SET)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTPROTECT, FAIL, "unable to protect global heap");

    /* Sanity check the heap object (split around bugfix below) */
    if (hobj->idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap index, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    /* When the application selects the same location to rewrite the VL element by using H5Sselect_elements,
     * it can happen that the entry has been removed by first rewrite.  Here we simply skip the removal of
     * the entry and let the second rewrite happen (see HDFFV-10635).  In the future, it'd be nice to handle
     * this situation in H5T_conv_vlen in H5Tconv.c instead of this level (HDFFV-10648). */
    if (heap->obj[hobj->idx].nrefs == 0 && heap->obj[hobj->idx].size == 0 && !heap->obj[hobj->idx].begin)
        HGOTO_DONE(SUCCEED);

    /* Finish sanity checking the heap object */
    if (NULL == heap->obj[hobj->idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad heap pointer, heap object = {%" PRIxHADDR ", %zu}",
                    hobj->addr, hobj->idx);

    obj_start = heap->obj[hobj->idx].begin;
    /* Include object header size */
    need = H5HG_ALIGN(heap->obj[hobj->idx].size) + H5HG_SIZEOF_OBJHDR(f);

    /* Move the new free space to the end of the heap */
    for (u = 0; u < heap->nused; u++)
        if (heap->obj[u].begin > heap->obj[hobj->idx].begin)
            heap->obj[u].begin -= need;
    if (NULL == heap->obj[0].begin) {
        heap->obj[0].begin = heap->chunk + (heap->size - need);
        heap->obj[0].size  = need;
        heap->obj[0].nrefs = 0;
    } /* end if */
    else
        heap->obj[0].size += need;
    memmove(obj_start, obj_start + need, heap->size - (size_t)((obj_start + need) - heap->chunk));
    if (heap->obj[0].size >= H5HG_SIZEOF_OBJHDR(f)) {
        p = heap->obj[0].begin;
        UINT16ENCODE(p, 0); /*id*/
        UINT16ENCODE(p, 0); /*nrefs*/
        UINT32ENCODE(p, 0); /*reserved*/
        H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);
    } /* end if */
    memset(heap->obj + hobj->idx, 0, sizeof(H5HG_obj_t));
    flags |= H5AC__DIRTIED_FLAG;

    if ((heap->obj[0].size + H5HG_SIZEOF_HDR(f)) == heap->size) {
        /*
         * The collection is empty. Remove it from the CWFS list and return it
         * to the file free list.
         */
        flags |=
            H5AC__DELETED_FLAG |
            H5AC__FREE_FILE_SPACE_FLAG; /* Indicate that the object was deleted, for the unprotect call */
    }                                   /* end if */
    else {
        /*
         * If the heap is in the CWFS list then advance it one position.  The
         * H5AC_protect() might have done that too, but that's okay.  If the
         * heap isn't on the CWFS list then add it to the end.
         */
        if (H5F_cwfs_advance_heap(f, heap, true) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTMODIFY, FAIL, "can't adjust file's CWFS");
    } /* end else */

done:
    if (heap && H5AC_unprotect(f, H5AC_GHEAP, hobj->addr, heap, flags) < 0)
        HDONE_ERROR(H5E_HEAP, H5E_CANTUNPROTECT, FAIL, "unable to release object header");

    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* end H5HG_remove() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__free
 *
 * Purpose:     Destroys a global heap collection in memory
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__free(H5HG_heap_t *heap)
{
    herr_t ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check arguments */
    assert(heap);

    /* Remove the heap from the CWFS list */
    if (H5F_cwfs_remove_heap(heap->shared, heap) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTREMOVE, FAIL, "can't remove heap from file's CWFS");

    if (heap->chunk)
        heap->chunk = H5FL_BLK_FREE(gheap_chunk, heap->chunk);
    if (heap->obj)
        heap->obj = H5FL_SEQ_FREE(H5HG_obj_t, heap->obj);
    heap = H5FL_FREE(H5HG_heap_t, heap);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5HG__free() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__free_local
 *
 * Purpose:     Frees a chunk-local heap object.
 *
 *              This differs from H5HG__free() because the chunk-local heap is
 *              not on the CWFS list and is not a standalone metadata-cache
 *              object. It is owned by the decoded structured chunk.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                          --AZO    07/12/26
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__free_local(H5HG_heap_t *heap)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(heap);

    /* Release the serialized H5HG collection image owned by this local heap. */
    if (heap->chunk) {

        heap->chunk = H5FL_BLK_FREE(gheap_chunk, heap->chunk);
    }

    /*
     * Release the in-memory object table used to map local indices to records
     * within the serialized heap image
     */
    if (heap->obj) {

        heap->obj = H5FL_SEQ_FREE(H5HG_obj_t, heap->obj);
    }

    /*
     * Release the heap descriptor itself. No CWFS removal is required because
     * local heaps are never registered with the file-wide global heap machinery.
     */
    heap = H5FL_FREE(H5HG_heap_t, heap);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__free_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__create_local
 *
 * Purpose:     Creates an in-memory H5HG-style heap for use as a
 *              chunk-local VL heap.
 *
 *              Unlike H5HG__create(), this routine does not allocate
 *              a standalone global-heap collection in the file, assign
 *              a file address, add the heap to the CWFS list, or insert
 *              it into the metadata cache. The returned heap is owned
 *              by the decoded structured chunk and will later be
 *              serialized into H5_SECTION_VL.
 *
 *              The heap image uses the existing H5HG collection layout.
 *              Object-table index zero is initialized as the free-space
 *              record, and indices beginning at one are available for
 *              payload objects.
 *
 * Return:      Success: Pointer to the created local heap
 *              Failure: NULL
 *
 *                                                  -- AZO  6/26/26
 *-------------------------------------------------------------------------
 */
H5HG_heap_t *
H5HG__create_local(H5F_t *f, size_t init_size)
{
    H5HG_heap_t *heap         = NULL;      /* New local heap being constructed */
    uint8_t     *p            = NULL;      /* Current position in encoded heap image */
    size_t       size         = init_size; /* Requested/adjusted heap image size */
    size_t       aligned_size = 0;         /* Result of safely aligning SIZE */
    size_t       pad_size     = 0;         /* Padding before the first heap object */
    H5HG_heap_t *ret_value    = NULL;      /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);

    /*
     * Use the normal min H5HG collection size so the local heap has
     * enough room for its header, free-space record, and initial objects.
     */
    if (size < H5HG_MINSIZE) {

        size = H5HG_MINSIZE;
    }

    /*
     * Align the complete heap image into the H5HG allocation boundary.
     * A wrapped alignment result would be smaller than the requested size.
     */
    aligned_size = H5HG_ALIGN(size);

    if (aligned_size < size)
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "chunk-local heap initial size overflow");

    size = aligned_size;

    /* The heap must be large enough to contain its collection header. */
    if (size < H5HG_SIZEOF_HDR(f))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "chunk-local heap size is smaller than its header");

    /* Allocate and initialize the in-memory heap descriptor. */
    if (NULL == (heap = H5FL_CALLOC(H5HG_heap_t)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "memory allocation failed");

    heap->addr = HADDR_UNDEF;
    heap->size = size;

    /*
     * The local heap may use file-width information while the decoded
     * structured chunk is alive, but it is never placed on the file CWFS
     * list or in the metadata cache.
     */
    heap->shared = H5F_SHARED(f);

    /* Allocate and clear the serialized H5HG collection image. */
    if (NULL == (heap->chunk = H5FL_BLK_MALLOC(gheap_chunk, size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "memory allocation failed");

    memset(heap->chunk, 0, size);

    /*
     * Object indices are 16-bit, so there is no reason to allocate more
     * object-table entries than the format can address. This is especially
     * important for dedicated oversized heaps.
     */
    heap->nalloc = MIN(H5HG_NOBJS(f, size), ((size_t)H5HG_MAXIDX + 1));
    heap->nused  = 1; /* Index zero is the free-space object */
    heap->nlive  = 0; /* No live payload objects have been inserted */

    if (0 == heap->nalloc)
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "invalid local heap object-table size");

    if (heap->nalloc > (SIZE_MAX / sizeof(heap->obj[0])))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "local heap object-table size overflow");

    if (NULL == (heap->obj = H5FL_SEQ_MALLOC(H5HG_obj_t, heap->nalloc)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "memory allocation failed");

    memset(heap->obj, 0, heap->nalloc * sizeof(heap->obj[0]));

    /*
     * Initialize the collection header:
     * (signature, version, three reserved bytes, encoded collection size)
     */
    H5MM_memcpy(heap->chunk, H5HG_MAGIC, (size_t)H5_SIZEOF_MAGIC);

    p    = heap->chunk + H5_SIZEOF_MAGIC;
    *p++ = H5HG_VERSION;
    *p++ = 0; /* Reserved */
    *p++ = 0; /* Reserved */
    *p++ = 0; /* Reserved */

    H5F_ENCODE_LENGTH(f, p, size);

    /*
     * Align the beginning of object zero in the same way as a normal H5HG
     * collection.
     */
    pad_size = (size_t)H5HG_ALIGN(p - heap->chunk) - (size_t)(p - heap->chunk);
    p += pad_size;

    /* Initialize object zero as the free-space object */
    heap->obj[0].size  = size - H5HG_SIZEOF_HDR(f);
    heap->obj[0].nrefs = 0;
    heap->obj[0].begin = p;

    assert(H5HG_ISALIGNED(heap->obj[0].size));

    /* Write the serialized free-space object header into the heap image. */
    UINT16ENCODE(p, 0); /* Object ID */
    UINT16ENCODE(p, 0); /* Reference count */
    UINT32ENCODE(p, 0); /* Reserved */
    H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);

    ret_value = heap;

done:
    /* Destroy any partially initialized heap if creation fails.  */
    if ((NULL == ret_value && heap)) {

        if ((H5HG__free_local(heap) < 0))
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, NULL, "unable to destroy local heap");
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__create_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__insert_local
 *
 * Purpose:     Inserts an object into a chunk-local heap and returns the
 *              local object index.
 *
 *              This routine is a per-heap insertion primitive. The caller
 *              must supply an existing local heap with sufficient free
 *              space and available object-index capacity. This routine
 *              does not create a local heap, extend the supplied heap,
 *              select a different heap, or otherwise make heap-set
 *              allocation-policy decisions.
 *
 *              Selection and creation of member heaps is performed by
 *              the outer chunk-local heap-set manager. If the supplied
 *              heap cannot hold the object, this routine fails and
 *              allows the heap-set manager to select or create another
 *              heap.
 *
 *              SIZE can be zero. OBJ must be non-NULL when SIZE is
 *              greater than zero. A zero-length object is still a
 *              real payload object, receives a nonzero object index,
 *              and is included in the heap's live-object count.
 *
 *              On success, IDX_OUT receives the object-table index
 *              assigned within HEAP. Object index zero remains reserved
 *              for the free-space record.
 *
 * Return:      SUCCEED/FAIL
 *
 *
 *                                                  -- AZO   07/12/26
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__insert_local(H5F_t *f, H5HG_heap_t *heap, size_t size, const void *obj, size_t *idx_out)
{
    size_t   aligned_size = 0;                  /* Aligned payload size */
    size_t   need         = 0;                  /* Total space needed for object */
    size_t   idx          = 0;                  /* Allocated object index */
    unsigned heap_flags   = H5AC__NO_FLAGS_SET; /* Ignored for local heaps */
    herr_t   ret_value    = SUCCEED;            /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(heap);
    assert(0 == size || obj);
    assert(idx_out);

    /*
     * Ensure that callers never receive a stale index when insertion fails.
     */
    *idx_out = 0;

    /* Compute the aligned payload size */
    aligned_size = H5HG_ALIGN(size);

    if (aligned_size < size)
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap object size overflow");

    if ((size > 0) && (NULL == obj))
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "non-empty chunk-local object has no source buffer");

    if (aligned_size > (SIZE_MAX - H5HG_SIZEOF_OBJHDR(f)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap object allocation size overflow");

    need = H5HG_SIZEOF_OBJHDR(f) + aligned_size;

    /*
     * NLIVE above the representable object-index range indicates broken
     * internal bookkeeping. NLIVE equal to the limit simply means that
     * this member heap cannot accept another object.
     */
    if (heap->nlive > H5HG_MAXIDX)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap live-object count is inconsistent");

    assert(heap->nlive <= H5HG_MAXIDX);

    if (heap->nlive == H5HG_MAXIDX)
        HGOTO_ERROR(H5E_HEAP, H5E_NOSPACE, FAIL, "chunk-local heap object index space is exhausted");
    /*
     * This is now a per-heap insertion primitive. It does not extend the heap.
     * The outer heap-set manager decides whether another heap should be
     * selected or created.
     */
    if (heap->obj[0].size < need)
        HGOTO_ERROR(H5E_HEAP, H5E_NOSPACE, FAIL, "chunk-local heap does not have enough free space");

    if (0 == (idx = H5HG__alloc(f, heap, size, &heap_flags)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTALLOC, FAIL, "unable to allocate chunk-local heap object");

    /* Copy the logical payload. Zero-length objects require no copy. */
    if (size > 0) {
        H5MM_memcpy(heap->obj[idx].begin + H5HG_SIZEOF_OBJHDR(f), obj, size);
    }

    /* Count valid zero-length objects too. They are real payload records. */
    heap->nlive++;

    /* An individual H5HG heap must never exceed its available object-index space. */
    assert(heap->nlive <= H5HG_MAXIDX);

    *idx_out = idx;

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end of H5HG__insert_local */

/*-------------------------------------------------------------------------
 * Function:    H5HG__read_local
 *
 * Purpose:     Reads an object from a chunk-local H5HG-style heap.
 *
 *              Unlike H5HG_read(), this routine operates on an already
 *              decoded heap supplied by the caller. The heap is owned
 *              by the current structured chunk and is not located or
 *              protected through the metadata cache.
 *
 *              IDX identifies the object within the supplied heap. Object
 *              index zero is reserved for the heap's free-space record
 *              and is not valid payload object.
 *
 *              If OBJECT is NULL, no data is copied and *BUF_SIZE is set
 *              to the logical payload size. This provides a size-query
 *              operation without allocating a destination buffer.
 *
 *              If OBJECT is non-NULL, *BUF_SIZE is the capacity of the
 *              caller-owned destination buffer on entry. If that buffer
 *              is too small, *BUF_SIZE is updated with the required size
 *              and the routine fails without copying data.
 *
 *              On a successful read, *BUF_SIZE is set to the logical
 *              payload size. This routine never allocates the
 *              destination.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                                      -- AZO 06/28/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__read_local(H5F_t *f, const H5HG_heap_t *heap, size_t idx, void *object, size_t *buf_size)
{
    size_t   size      = 0;       /* Size of the heap object */
    uint8_t *p         = NULL;    /* Pointer to the object in the heap */
    herr_t   ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(heap);
    assert(heap->chunk);
    assert(heap->obj);
    assert(buf_size);

    /*
     * BUF_SIZE is required for both size queries and normal reads.
     * Keep the runtime check since assertions are not present in release
     * builds.
     */
    if (NULL == buf_size)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "chunk-local heap read requires buffer size");

    /* Heap object idx 0 is free space in the heap */
    if (0 == idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad local heap index");

    /* Verify that the supplied index identifies an entry in the local heap's object table */
    if (idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "local heap index out of range");

    /*
     * Removed objects have cleared table entries. A NULL begin pointer therefore means
     * that the requested local object no longer exists.
     */
    if (NULL == heap->obj[idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad local heap object pointer");

    /*
     * Object table stores the unaligned payload size. The serialized payload begins
     * immediately after the object's H5HG header.
     */
    size = heap->obj[idx].size;
    p    = heap->obj[idx].begin + H5HG_SIZEOF_OBJHDR(f);

    /*
     * OBJECT == NULL is a size query. Do not allocate or copy anything.
     * This also handles a valid zero-length object naturally.
     */
    if (NULL == object) {
        *buf_size = size;
        HGOTO_DONE(SUCCEED);
    }

    /*
     * With a destination buffer, *BUF_SIZE is its capacity on entry.
     * Never copy if doing so would overrun the caller's buffer.
     *
     * Report the required size even on this failure so the caller knows how
     * much space is needed for a retry.
     */
    if (*buf_size < size) {
        *buf_size = size;

        HGOTO_ERROR(H5E_HEAP, H5E_NOSPACE, FAIL, "chunk-local heap read buffer is too small.");
    }
    /*
     * Copy only the logical payload bytes. Any alignment padding stored in
     * the heap image is not part of the returned object.
     */
    if (size > 0) {

        H5MM_memcpy(object, p, size);
    }
    /*
     * Return the logical payload size.
     */
    *buf_size = size;

done:

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__read_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__is_empty_local
 *
 * Purpose:     Checks whether a chunk-local heap contains any allocated
 *              payload objects.
 *
 *              The heap maintains a count of live payload objects as
 *              objects are inserted, removed, and reconstructed during
 *              decode. Object zero, which represents the free-space
 *              extent is not included in this count.
 *
 * Return:      TRUE if no live payload objects remain
 *              FALSE if at least one live payload object remains.
 *
 *                                                  -- AZO   7/09/26
 *
 *-------------------------------------------------------------------------
 */
htri_t
H5HG__is_empty_local(const H5HG_heap_t *heap)
{
    htri_t ret_value;

    FUNC_ENTER_PACKAGE_NOERR

    /* Check argument */
    assert(heap);
    assert(heap->nlive <= H5HG_MAXIDX);

    /* Object zero is not included in the live payload count */
    ret_value = (0 == heap->nlive);

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__is_empty_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__get_obj_size_local
 *
 * Purpose:     Returns the payload size of an object stored in a
 *              chunk-local H5HG-style heap.
 *
 * Return:      Success:            Non-negative
 *
 *              Failure:            Negative
 *
 *                                                  -- AZO 7/11/26
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__get_obj_size_local(H5F_t *f, const H5HG_heap_t *heap, size_t idx, size_t *obj_size)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(heap);
    assert(obj_size);

    /* Ensure validation failures do not leave an old caller value behind */
    *obj_size = 0;

    /* Heap object idx 0 is the free space in the heap and should not be given out */
    if (0 == idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad chunk-local heap index");

    /* Sanity check the heap object */
    if (idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap index is out of range");

    if (NULL == heap->obj[idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap object has been removed or is invalid");

    /* Set object size */
    *obj_size = heap->obj[idx].size;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__get_obj_size_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__remove_local
 *
 * Purpose:     Removes an object from a chunk-local heap. The heap itself is
 *              not freed here; the caller owns heap lifetime.
 *
 *              Unlike H5HG_remove(), this routine operates on an in-memory
 *              heap owned by the current structured chunk rather than a
 *              standalone global heap collection stored in the file.
 *              The caller supplies the heap directly so no metadata-cache
 *              protection or file-space management is performed.
 *
 *              If heap_empty is non-null. it is set to indicate whether
 *              any allocated objects remain agter the removal completes.
 *
 * Return:      Non-negative on success/Negative on failure
 *
 *                                                   -- AZO   7/11/26
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__remove_local(H5F_t *f, H5HG_heap_t *heap, size_t idx, hbool_t *heap_empty)
{
    uint8_t *p         = NULL;    /* Pointer into encoded heap image */
    uint8_t *obj_start = NULL;    /* Beginning of object being removed */
    size_t   aligned_size;        /* Aligned payload size */
    size_t   need      = 0;       /* Complete serialized object extent */
    size_t   move_size = 0;       /* Bytes shifted during compaction */
    unsigned u;                   /* Object-table index */
    herr_t   ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(heap);

    /*
     * Initialize the optional heap-empty result so callers do not observe a
     * stale value if validation fails before the heap is modified.
     */
    if (heap_empty) {

        *heap_empty = false;
    }

    /* Heap object idx 0 is the free space in the heap and should not be shared */
    if (0 == idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "bad local heap index");

    /* Sanity check the heap object */
    if (idx >= heap->nused)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "local heap index out of range");

    /*
     * An already-cleared entry is an error. The outer heap-set manager relies on
     * each successful call removing exactly one live object.
     */
    if (NULL == heap->obj[idx].begin)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL,
                    "chunk-local heap object has already been removed or is invalid.");

    /* A live object-table entry must correspond to at least one object in the maintained
     * live-payload count.
     */
    if (0 == heap->nlive)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap live-count is inconsistent");

    assert(heap->nlive > 0);

    obj_start = heap->obj[idx].begin;

    /* Compute the aligned payload size. */
    aligned_size = H5HG_ALIGN(heap->obj[idx].size);

    /* Ensure size is in valid range */
    if (aligned_size < heap->obj[idx].size)
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap object size overflow");

    if (aligned_size > (SIZE_MAX - H5HG_SIZEOF_OBJHDR(f)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap removal size overflow");

    /*
     * Compute the serialized extent occupied by this object. The extent
     * includes both the object header and the aligned payload because
     * the entire record will be removed from the heap image.
     */
    need = aligned_size + H5HG_SIZEOF_OBJHDR(f);

    /*
     * Verify that the object extent lies entirely inside the local
     * heap image. This becomes particularly important after local
     * heaps are decoded from on-disk section data.
     */
    if ((obj_start < heap->chunk) || (obj_start >= (heap->chunk + heap->size)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "local heap object pointer is outside heap image");

    if (need > (heap->size - (size_t)(obj_start - heap->chunk)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "local heap object extends beyond heap image");

    /*
     * Validate the new free-space size before modifying object pointers,
     * object zero, or the serialized heap image.
     */
    if (need > (SIZE_MAX - heap->obj[0].size))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap free-space size overflow");

    /* The previous bounds checks guarantee that this subtraction cannot underflow */
    move_size = (heap->size - (size_t)((obj_start + need) - heap->chunk));

    /*
     * Compact the heap by shifting every object that follows the removed
     * record toward the beginning of the heap image. Update each object's
     * cached pointer before moving the serialized bytes.
     */
    for (u = 0; u < heap->nused; u++) {

        if ((heap->obj[u].begin) && (heap->obj[u].begin > obj_start)) {

            heap->obj[u].begin -= need;
        }
    }

    /*
     * Add the reclaimed extent to object zero. If no serialized free-space
     * record currently exists, create one at the new end of the used region.
     */
    if (NULL == heap->obj[0].begin) {
        heap->obj[0].begin = heap->chunk + (heap->size - need);
        heap->obj[0].size  = need;
        heap->obj[0].nrefs = 0;
    }
    else {

        heap->obj[0].size += need;
    }

    /*
     * Remove the serialized object record from the heap image by sliding
     * the remaining bytes towards the beginning of the heap.
     */
    memmove(obj_start, obj_start + need, move_size);

    /*
     * Rewrite the free-space object's serialized header so the heap
     * image remains internally consistent after compaction.
     */
    if ((heap->obj[0].size >= H5HG_SIZEOF_OBJHDR(f))) {
        p = heap->obj[0].begin;

        UINT16ENCODE(p, 0); /* Object ID */
        UINT16ENCODE(p, 0); /* Reference count */
        UINT32ENCODE(p, 0); /* Reserved */
        H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);
    }

    /* Clear the in-memory object-table entry so its index may be reused. */
    memset(heap->obj + idx, 0, sizeof(H5HG_obj_t));

    /* The removal is complete. Update the maintained live-payload count. */
    heap->nlive--;

    assert(heap->nlive <= H5HG_MAXIDX);

    if (heap_empty) {

        *heap_empty = (0 == heap->nlive);
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__remove_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__encode_local
 *
 * Purpose:     Creates a serialized copy of a chunk-local H5HG-style heap.
 *
 *              The H5HG heap-management routines maintain HEAP->CHUNK
 *              directly in the existing serialized global-heap collection
 *              format. Consequently, this routine does not rebuild the heap
 *              record by record. It allocates an output buffer, copies the
 *              current encoded image, and returns the image and its length
 *              to the caller.
 *
 *              On success, *IMAGE_OUT points to a buffer allocated with
 *              H5MM_malloc(), and *IMAGE_LEN_OUT contains its size. The
 *              caller is responsible for releasing the buffer with
 *              H5MM_free().
 *
 *              This routine exports only one embedded H5HG collection
 *              image. H5HG__encode_local_heapset() owns the outer
 *              H5_SECTION_VL heap-set header and directory. SCC remains
 *              responsible for section framing, filtering, checksums, and
 *              persistence.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   7/14/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__encode_local(H5F_t *f, const H5HG_heap_t *heap, uint8_t **image_out, size_t *image_len_out)
{
    uint8_t *image        = NULL;    /* Newly allocated serialized heap image */
    uint8_t *p            = NULL;    /* Collection length field */
    size_t   encoded_size = 0;       /* Header and live object records */
    herr_t   ret_value    = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(heap);
    assert(image_out);
    assert(image_len_out);

    /*
     * Initialize both outputs so a failed encode cannot leave stale values
     * in the caller.
     */
    *image_out     = NULL;
    *image_len_out = 0;

    /*
     * A chunk-local heap is embedded in its owning structured chunk and
     * therefore must not have a standalone global-heap file address.
     */
    if (H5_addr_defined(heap->addr))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap unexpectedly has a file address");

    if (NULL == heap->chunk || NULL == heap->obj)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap has no serialized image");

    if (heap->size < H5HG_SIZEOF_HDR(f))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL,
                    "chunk-local heap image is smaller than the minimum heap size");

    if (heap->obj[0].size > heap->size - H5HG_SIZEOF_HDR(f))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "invalid chunk-local heap free-space size");

    /*
     * Local insertion and removal keep all live records before a single
     * trailing free-space extent. Exclude that extent from serialization.
     * Required padding within live records remains part of the image.
     */
    encoded_size = heap->size - heap->obj[0].size;

    if (!H5HG_ISALIGNED(encoded_size))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local encoded heap size is not aligned");

    if (heap->obj[0].size > 0 && heap->obj[0].begin != heap->chunk + encoded_size)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap free space is not a trailing extent");

    /*
     * Allocate an independent output image. SCC may retain or transform this
     * buffer after the decoded in-memory heap has been released.
     */
    if (NULL == (image = H5MM_malloc(encoded_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate encoded chunk-local heap image");

    /*
     * The serialized collection size describes this compact image.
     * Leave the resident image, capacity, and allocation accounting intact.
     */
    H5MM_memcpy(image, heap->chunk, encoded_size);

    p = image + H5_SIZEOF_MAGIC + 4;
    H5F_ENCODE_LENGTH(f, p, encoded_size);

    *image_out     = image;
    *image_len_out = encoded_size;
    image          = NULL;

done:
    if (image) {

        H5MM_free(image);
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__encode_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__decode_local
 *
 * Purpose:     Reconstructs an in-memory chunk-local H5HG-style heap from
 *              a serialized H5HG collection image.
 *
 *              This routine performs the same image parsing and object-table
 *              reconstruction as the ordinary global-heap metadata-cache
 *              deserializer, but the resulting heap is not added to the
 *              file's CWFS list and is not registered as an independent
 *              metadata-cache object.
 *
 *              IMAGE is expected to contain exactly one embedded H5HG
 *              collection image. For structured-chunk VL data,
 *              H5HG__decode_local_heapset() validates the outer heap-set
 *              header and directory and passes each active member image to
 *              this routine. SCC handles section framing, filtering, and
 *              checksum verification before heap-set decoding begins.
 *
 *              Compact images contain the collection header and live object records
 *              without unused capacity. Older images containing a trailing free-space
 *              extent are also accepted. Resident image allocation follows the encoded
 *              length; later insertions reserve additional capacity as necessary.
 *
 *              The returned heap is owned by the decoded structured chunk
 *              and must eventually be released with H5HG__free_local().
 *
 * Return:      Success: Pointer to a newly decoded chunk-local heap
 *              Failure: NULL
 *
 *                                              -- AZO   7/14/26
 *
 *-------------------------------------------------------------------------
 */
H5HG_heap_t *
H5HG__decode_local(H5F_t *f, const void *image, size_t len)
{
    H5HG_heap_t   *heap      = NULL; /* Local heap being reconstructed */
    uint8_t       *p         = NULL; /* Current record in copied heap image */
    const uint8_t *p_end     = NULL; /* Last valid byte in copied heap image */
    size_t         max_idx   = 0;    /* Largest nonzero object index observed */
    size_t         nalloc    = 0;    /* Initial object-table allocation size */
    size_t         need      = 0;    /* Complete serialized record size */
    unsigned       idx       = 0;    /* Decoded heap object index */
    uint8_t       *begin     = NULL; /* Beginning of serialized object record */
    size_t         new_alloc = 0;    /* Enlarged table size */
    H5HG_obj_t    *new_obj   = NULL; /* Reallocated object table */
    size_t         aligned_size;     /* Aligned logical payload length */
    H5HG_heap_t   *ret_value = NULL; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(f);
    assert(image);

    /*
     * Validate the length before calculating P_END. This prevents an
     * invalid or zero length from underflowing the end-pointer expression.
     */
    if (len < H5HG_SIZEOF_HDR(f))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "chunk-local H5HG image is smaller than its header");

    /* Allocate the in-memory heap descriptor. */
    if ((NULL == (heap = H5FL_CALLOC(H5HG_heap_t))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "unable to allocate chunk-local heap descriptor");

    /*
     * A local heap has no standalone file address. Retain the file-shared
     * state because decoding length fields depends on the file's configured
     * size width.
     */
    heap->addr   = HADDR_UNDEF;
    heap->shared = H5F_SHARED(f);

    /* Allocate and retain a private copy of the serialized collection. */
    if ((NULL == (heap->chunk = H5FL_BLK_MALLOC(gheap_chunk, len))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "unable to allocate chunk-local heap image");

    H5MM_memcpy(heap->chunk, image, len);

    /*
     * Parse the copied image so all reconstructed object pointers refer to
     * HEAP->CHUNK rather than to the caller's input buffer.
     */
    p_end = heap->chunk + len - 1;

    /* Decode and validate the global-heap collection header. */
    if ((H5_IS_BUFFER_OVERFLOW(heap->chunk, H5HG_SIZEOF_HDR(f), p_end)))
        HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                    "ran off end of chunk-local heap image while decoding header");

    /*
     * Decode the embedded H5HG collection header locally. The equivalent
     * metadata-cache helper is static to H5HGcache.c, so the chunk-local path
     * performs the small amount of header parsing it needs here without
     * changing the legacy cache implementation.
     */
    {
        const uint8_t *hdr = heap->chunk;

        /* Collection signature */
        if (H5_IS_BUFFER_OVERFLOW(hdr, H5_SIZEOF_MAGIC, p_end))
            HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                        "ran off end of chunk-local heap image while decoding signature");

        if (memcmp(hdr, H5HG_MAGIC, (size_t)H5_SIZEOF_MAGIC) != 0)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad chunk-local H5HG collection signature");

        hdr += H5_SIZEOF_MAGIC;

        /* Collection format version */
        if (H5_IS_BUFFER_OVERFLOW(hdr, 1, p_end))
            HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                        "ran off end of chunk-local heap image while decoding version");

        if (H5HG_VERSION != *hdr++)
            HGOTO_ERROR(H5E_HEAP, H5E_VERSION, NULL, "unsupported chunk-local H5HG collection version");

        /*
         * Skip the three reserved bytes in the existing H5HG collection
         * header. This preserves the legacy global-heap image format.
         */
        if (H5_IS_BUFFER_OVERFLOW(hdr, 3, p_end))
            HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                        "ran off end of chunk-local heap image while decoding reserved header bytes");

        hdr += 3;

        /* Complete serialized collection size */
        if (H5_IS_BUFFER_OVERFLOW(hdr, H5F_sizeof_size(f), p_end))
            HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                        "ran off end of chunk-local heap image while decoding collection size");

        H5F_DECODE_LENGTH(f, hdr, heap->size);

        if (heap->size < H5HG_SIZEOF_HDR(f))
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "chunk-local H5HG collection size is too small");
    }

    /*
     * The supplied buffer must contain exactly one complete H5HG collection.
     * A mismatch indicates truncation or trailing bytes in the section data.
     */
    if (heap->size != len)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL,
                    "encoded heap size does not match chunk-local image length");

    /* Serialized object records begin after the aligned collection header. */
    p = heap->chunk + H5HG_SIZEOF_HDR(f);

    /* Always retain entry zero, even for a header-only collection. */
    nalloc = MAX((size_t)1, MIN(H5HG_NOBJS(f, heap->size), ((size_t)H5HG_MAXIDX + 1)));

    if (0 == nalloc)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "invalid chunk-local heap object-table size");

    if ((NULL == (heap->obj = H5FL_SEQ_CALLOC(H5HG_obj_t, nalloc))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "unable to allocate chunk-local heap object table");

    heap->nalloc = nalloc;
    /* Reconstruct the number of live payload objects while walking the serialized records. */
    heap->nlive = 0;

    /*
     * Walk the serialized collection and reconstruct the in-memory object
     * table. Index zero represents free space; nonzero indices represent
     * stored payload objects.
     */
    while (p < (heap->chunk + heap->size)) {
        /*
         * A trailing extent smaller than an object header cannot contain a
         * payload record and is represented as headerless free space.
         */
        if ((size_t)((heap->chunk + heap->size) - p) < H5HG_SIZEOF_OBJHDR(f)) {

            if (NULL != heap->obj[0].begin)
                HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "chunk-local free-space object is already defined");

            heap->obj[0].size  = (size_t)((heap->chunk + heap->size) - p);
            heap->obj[0].nrefs = 0;
            heap->obj[0].begin = p;

            /*
             * Move directly to the end of the collection. This extent was
             * calculated from the validated heap bounds.
             */
            p += heap->obj[0].size;
        }
        else {
            need  = 0; /* Complete serialized record size */
            idx   = 0; /* Decoded heap object index */
            begin = p; /* Beginning of serialized object record */

            /* Decode the two-byte object index. */
            if (H5_IS_BUFFER_OVERFLOW(p, 2, p_end))
                HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                            "ran off end of chunk-local heap while decoding object index");

            UINT16DECODE(p, idx);

            /*
             * Enlarge the object table if the encoded object index exceeds
             * the initial size-based estimate.
             */
            if (idx >= heap->nalloc) {
                /*
                 * Grow enough to represent IDX, but never beyond the 16-bit
                 * object-index space.
                 */
                new_alloc = MIN(MAX(heap->nalloc * 2, (size_t)idx + 1), ((size_t)H5HG_MAXIDX + 1));

                assert((size_t)idx < new_alloc);

                if (NULL == (new_obj = H5FL_SEQ_REALLOC(H5HG_obj_t, heap->obj, new_alloc)))
                    HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL,
                                "unable to enlarge chunk-local heap object table");

                memset(&new_obj[heap->nalloc], 0, (new_alloc - heap->nalloc) * sizeof(new_obj[0]));

                heap->obj    = new_obj;
                heap->nalloc = new_alloc;
            }

            /*
             * Reject duplicate serialized object indices. Each index must
             * identify at most one record in the collection.
             */
            if ((heap->obj[idx].begin))
                HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "duplicate object index in chunk-local heap image");

            /* Decode the object's reference count. */
            if ((H5_IS_BUFFER_OVERFLOW(p, 2, p_end)))
                HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                            "ran off end of chunk-local heap while decoding reference count");

            UINT16DECODE(p, heap->obj[idx].nrefs);

            /*
             * Skip the four reserved bytes in the existing H5HG object
             * header. Their format remains unchanged from normal global
             * heaps.
             */
            if ((H5_IS_BUFFER_OVERFLOW(p, 4, p_end)))
                HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                            "ran off end of chunk-local heap while decoding reserved field");

            p += 4;

            /* Decode the logical, unaligned payload length. */
            if ((H5_IS_BUFFER_OVERFLOW(p, H5F_sizeof_size(f), p_end)))
                HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                            "ran off end of chunk-local heap while decoding object size");

            H5F_DECODE_LENGTH(f, p, heap->obj[idx].size);

            /*
             * Object-table entries point to the beginning of the complete
             * serialized record, including the object header.
             */
            heap->obj[idx].begin = begin;

            if (idx > 0) {
                aligned_size = H5HG_ALIGN(heap->obj[idx].size);

                if (aligned_size < heap->obj[idx].size)
                    HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "chunk-local heap object size overflow");

                if (aligned_size > SIZE_MAX - H5HG_SIZEOF_OBJHDR(f))
                    HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "chunk-local heap object-record size overflow");

                /*
                 * A normal object record contains the object header followed
                 * by its aligned payload.
                 */
                need = H5HG_SIZEOF_OBJHDR(f) + aligned_size;

                if ((size_t)idx > max_idx) {

                    max_idx = (size_t)idx;
                }
            }
            else {
                /*
                 * A serialized object-zero record represents a free-space extent that
                 * includes its own object header. Smaller trailing free extents are
                 * represented without a header and are handled separately above.
                 */
                if (heap->obj[0].size < H5HG_SIZEOF_OBJHDR(f))
                    HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "invalid chunk-local free-space record size");

                need = heap->obj[0].size;
            }

            /* Verify that the complete record lies within the heap image. */
            if (H5_IS_BUFFER_OVERFLOW(begin, need, p_end))
                HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL,
                            "chunk-local heap object extends beyond heap image");

            /*
             * Each valid nonzero record represents one live payload object.
             * Valid zero-length payload objects are included.
             */
            if (idx > 0) {
                heap->nlive++;
            }

            p = begin + need;

        } /* end else */

    } /* end while */

    /* The parser must consume the complete encoded collection. */
    if (p != (heap->chunk + heap->size))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "chunk-local heap image was only partially decoded");

    if (!H5HG_ISALIGNED(heap->obj[0].size))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "decoded chunk-local free-space extent is not aligned");

    /*
     * Set the next never-issued object index. Cleared entries below this
     * value may later be reused after the 16-bit index range wraps.
     */
    heap->nused = (max_idx > 0) ? max_idx + 1 : 1;

    ret_value = heap;
    heap      = NULL;

done:
    /*
     * Local heaps have not been added to CWFS or H5AC, so partial decode
     * cleanup must use the chunk-local free routine.
     */
    if (heap) {

        if (H5HG__free_local(heap) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, NULL, "unable to destroy partially decoded chunk-local heap");
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__decode_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__local_heap_alloc_size
 *
 * Purpose:     Computes the resident allocation owned by a single
 *              chunk-local H5HG heap.
 *
 *              The returned size includes:
 *
 *                  - the H5HG_heap_t structure
 *                  - the serialized heap image in HEAP->CHUNK
 *                  - the allocated H5HG_obj_t object table
 *
 *              This routine reports allocated memory, not the number of
 *              bytes currently occupied by live payload objects. In
 *              particular, unused object-table capacity and unused space
 *              within the heap image are still part of the resident
 *              allocation and must be included.
 *
 *              The heap-set layer uses this value when maintaining its
 *              cached resident allocation size for SCC memory accounting.
 *              The calculation is constant-time and does not scan heap
 *              objects or payload data.
 *
 *              SIZE_OUT is set to zero before any fallible calculation.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   09/12/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5HG__local_heap_alloc_size(const H5HG_heap_t *heap, size_t *size_out)
{
    size_t total;               /* Total resident allocation */
    size_t obj_size;            /* Allocated object-table size */
    herr_t ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Package-private caller requirements */
    assert(heap);
    assert(size_out);

    /* Do not return a stale size if a later calculation fails */
    *size_out = 0;

    /* Compute the allocated object-table size without overflowing size_t */
    if (heap->nalloc > SIZE_MAX / sizeof(heap->obj[0]))
        HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, FAIL, "local heap object-table allocation size overflow");

    obj_size = heap->nalloc * sizeof(heap->obj[0]);

    /* Add the heap structure and serialized heap image safely */
    if (heap->size > SIZE_MAX - sizeof(*heap))
        HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, FAIL, "local heap allocation size overflow");

    total = sizeof(*heap) + heap->size;

    /* Add the allocated object table safely */
    if (obj_size > SIZE_MAX - total)
        HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, FAIL, "local heap allocation size overflow");

    total += obj_size;

    /* Return the complete allocation owned by this member heap */
    *size_out = total;

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__local_heap_alloc_size() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__get_local_heapset_alloc_size
 *
 * Purpose:     Returns the complete resident allocation currently owned by
 *              a chunk-local H5HG heap set.
 *
 *              The heap-set allocation size is maintained incrementally as
 *              the heap set changes. It includes:
 *
 *                  - the H5HG_local_heapset_t structure
 *                  - the trailing member-heap pointer array
 *                  - all active member H5HG_heap_t structures
 *                  - all active member heap images
 *                  - all active member object tables
 *
 *              This routine does not walk the heap slots or inspect
 *              individual heap objects. It simply returns the cached
 *              HEAPSET->ALLOC_SIZE value, making the query constant-time.
 *
 *              A NULL HEAPSET represents a chunk with no chunk-local VL
 *              storage and therefore has an allocation size of zero.
 *
 *              This interface allows the structured chunk/SCC code to
 *              account for VL resident memory without depending on the
 *              internal H5HG heap-set representation.
 *
 *              SIZE_OUT is always set on success.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   09/12/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__get_local_heapset_alloc_size(const H5HG_local_heapset_t *heapset, size_t *size_out)
{
    FUNC_ENTER_PACKAGE_NOERR

    /* Package-private caller requirement */
    assert(size_out);

    /*
     * A NULL heap set is a normal state for a chunk that has no resident
     * chunk-local VL payload storage.
     */
    if (NULL == heapset) {

        *size_out = 0;
    }
    else {

        *size_out = heapset->alloc_size;
    }

    FUNC_LEAVE_NOAPI(SUCCEED)

} /* end H5HG__get_local_heapset_alloc_size() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__alloc_local_heapset
 *
 * Purpose:     Allocates an empty chunk-local heap-set structure with
 *              storage for NALLOC stable heap-pointer slots.
 *
 *              The heap-pointer slots are stored as trailing storage
 *              in the same allocation as H5HG_local_heapset_t. All slots
 *              are initialized to NULL. NSLOTS and NLIVE are initially
 *              zero, while NALLOC records the number of available ptr slots.
 *
 *              If NALLOC is zero, return error.
 *
 *              This routine allocates only the outer-heap-set manager.
 *              It does not allocate any member of H5HG_heap_t objects.
 *
 * Return:      Success: Pointer to a new H5HG_local_heapset_t on success.
 *              Failure: NULL
 *
 *                                              -- AZO   8/24/26
 *
 *-------------------------------------------------------------------------
 */
static H5HG_local_heapset_t *
H5HG__alloc_local_heapset(size_t nalloc)
{
    H5HG_local_heapset_t *heapset    = NULL;
    size_t                alloc_size = 0;
    H5HG_local_heapset_t *ret_value  = NULL;

    FUNC_ENTER_PACKAGE

    /*
     * Return error if nalloc is zero.
     */
    if (0 == nalloc)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, NULL, "zero-capacity chunk-local heap set requested.");

    /* Heap slots are descriptor-visible 16-bit values. */
    if (nalloc > H5HG_LOCAL_MAX_HEAP_SLOTS)
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL,
                    "chunk-local heap-set slot capacity exceeds format limits.");

    /* Check the flexible array allocation calculation before performing it. */
    if (nalloc > ((SIZE_MAX - sizeof(*heapset)) / sizeof(heapset->heaps[0])))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "chunk-local heap-set allocation size overflow");

    alloc_size = sizeof(*heapset) + nalloc * sizeof(heapset->heaps[0]);

    /* Allocate the outer manager and its trailing heap-slot array as one contiguous block. */
    if (NULL == (heapset = H5MM_malloc(alloc_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, NULL, "unable to allocate chunk-local heap set");

    /* Also initializes every trailing heap slot to NULL. */
    memset(heapset, 0, alloc_size);

    heapset->nalloc = nalloc;

    heapset->alloc_size = alloc_size;

    /* These conditions describe the required initial state of every newly allocated
     * heap set. If one fails, it indicates a programming error.
     */
    assert(0 == heapset->nslots);
    assert(0 == heapset->nlive);
    assert(heapset->nalloc == nalloc);

    ret_value = heapset;
    heapset   = NULL;

done:

    /* Only free HEAPSET here if ownership was not transferred to RET_VALUE */
    if (heapset) {
        H5MM_free(heapset);
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__alloc_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__grow_local_heapset
 *
 * Purpose:     Enlarges the trailing heap-pointer storage of existing
 *              chunk-local heap set so that it can hold at least
 *              MIN_NALLOC heap slots.
 *
 *              Existing stable heap-slot entries are preserved and newly
 *              allocated entries are initialized to NULL. The allocation
 *              may move as a result of H5MM_realloc(), so the caller
 *              passes the owning heap-set pointer by address and this
 *              routine updates *HEAPSET_PTR when necessary.
 *
 *              This routine changes only the in-memory slot capacity
 *              recorded by NALLOC. It does not change NSLOTS, create any
 *              member H5HG_heap_t objects, or alter descriptor-visible
 *              heap slot numbers.
 *
 *              MIN_NALLOC must be greater than zero. If the current allocation
 *              already provides at least MIN_NALLOC slots, the request
 *              is already satisfied and this routine succeeds without
 *              modifying the heapset.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   8/24/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5HG__grow_local_heapset(H5HG_local_heapset_t **heapset_ptr, size_t min_nalloc)
{
    H5HG_local_heapset_t *heapset     = NULL;
    H5HG_local_heapset_t *new_heapset = NULL;
    size_t                old_nalloc  = 0;
    size_t                new_nalloc  = 0;
    size_t                alloc_size  = 0;
    size_t                growth_size = 0;
    herr_t                ret_value   = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(heapset_ptr);
    assert(*heapset_ptr);

    if (NULL == heapset_ptr || NULL == *heapset_ptr)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "invalid chunk-local heap-set pointer");

    heapset = *heapset_ptr;

    if (0 == heapset->nalloc)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap set has zero allocated capacity");

    if (heapset->nalloc > H5HG_LOCAL_MAX_HEAP_SLOTS)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap-set capacity is inconsistent");

    if (heapset->nslots > heapset->nalloc)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL,
                    "chunk-local heap-set slot count exceeds allocated capacity");

    if (0 == min_nalloc)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "zero chunk-local heap-set capacity requested");

    if (min_nalloc > H5HG_LOCAL_MAX_HEAP_SLOTS)
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap-set slot limit exceeded");

    if (min_nalloc <= heapset->nalloc)
        HGOTO_DONE(SUCCEED);

    old_nalloc = heapset->nalloc;
    new_nalloc = old_nalloc;

    /* Keep stable slot numbers; only the trailing pointer capacity grows.
     * Check before doubling, both to avoid overflow and to respect the
     * descriptor-visible slot limit. Capacities decoded from disk need
     * not be powers of two.
     */
    while (new_nalloc < min_nalloc) {
        if (new_nalloc > H5HG_LOCAL_MAX_HEAP_SLOTS / 2) {
            new_nalloc = H5HG_LOCAL_MAX_HEAP_SLOTS;
        }
        else {
            new_nalloc *= 2;
        }
    }

    /*
     * Validate all arithmetic before realloc().
     *
     * Once realloc succeeds, the original HEAPSET pointer may no longer be
     * valid, so there must be no fallible validation between realloc and
     * publishing the new pointer.
     */
    if ((new_nalloc - old_nalloc) > SIZE_MAX / sizeof(heapset->heaps[0]))
        HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, FAIL, "heap-set allocation growth overflow");

    growth_size = (new_nalloc - old_nalloc) * sizeof(heapset->heaps[0]);

    if (growth_size > SIZE_MAX - heapset->alloc_size)
        HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, FAIL, "heap-set resident allocation overflow");

    if (new_nalloc > ((SIZE_MAX - sizeof(*heapset)) / sizeof(heapset->heaps[0])))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap-set allocation size overflow");

    alloc_size = sizeof(*heapset) + new_nalloc * sizeof(heapset->heaps[0]);

    /*
     * From this point until *HEAPSET_PTR is updated, do not perform any
     * operation that can fail.
     */
    if (NULL == (new_heapset = H5MM_realloc(heapset, alloc_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to enlarge chunk-local heap set");

    memset(&new_heapset->heaps[old_nalloc], 0, (new_nalloc - old_nalloc) * sizeof(new_heapset->heaps[0]));

    new_heapset->alloc_size += growth_size;
    new_heapset->nalloc = new_nalloc;

    /*
     * realloc() may have moved the flexible-array object.
     */
    *heapset_ptr = new_heapset;

    assert((*heapset_ptr)->nslots <= (*heapset_ptr)->nalloc);
    assert((*heapset_ptr)->nalloc >= min_nalloc);
    assert((*heapset_ptr)->nalloc <= H5HG_LOCAL_MAX_HEAP_SLOTS);

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__grow_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__trim_local_heapset
 *
 * Purpose:     Removes trailing unused heap slots from the descriptor-
 *              visible high-water slot count of a chunk-local heap set.
 *
 *              While the last slot below NSLOTS contains NULL, NSLOTS is
 *              decremented. Interior unused slots are not removed or
 *              renumbered because heap-slot values are stored in persistent
 *              VL descriptors and must remain stable.
 *
 *              This routine does not shrink the allocated trailing pointer
 *              storage recorded by NALLOC and does not free or move any
 *              member H5HG_heap_t objects. It only updates NSLOTS.
 *
 *              This is used after an empty trailing member heap has been
 *              released or after rollback of an insertion that appended
 *              a new slot.
 *
 * Return:      void
 *
 *                                              -- AZO   8/24/26
 *
 *-------------------------------------------------------------------------
 */
static void
H5HG__trim_local_heapset(H5HG_local_heapset_t *heapset)
{
    FUNC_ENTER_PACKAGE_NOERR

    assert(heapset);

    /*
     * These conditions indicate an internal bookkeeping bug. They should
     * already have been prevented by the heap-set manager.
     */
    assert(heapset->nalloc > 0);
    assert(heapset->nalloc <= H5HG_LOCAL_MAX_HEAP_SLOTS);
    assert(heapset->nslots <= heapset->nalloc);

    /*
     * Remove only trailing holes. Interior NULL slots must remain in place
     * because their numeric positions are persistent descriptor-visible IDs.
     *
     * Example:
     *          slot 0 -> heap
     *          slot 1 -> NULL
     *          slot 2 -> heap
     *          slot 3 -> NULL
     *
     * NSLOTS can drop from 4 to 3 because slot 3 is unused.
     *
     * Slot 1 must remain because slot 2 is still live and existing VL descriptors
     * may contain heap-slot value 2. Renumbering slot 2 would invalidate those refs.
     */
    while ((heapset->nslots > 0) && (NULL == heapset->heaps[heapset->nslots - 1])) {
        heapset->nslots--;
    }

    /* This function does not change NALLOC. We keep the already-allocated pointer
     * capacity available for later reuse instead of reallocating memory every time
     * a trailing member heap disapears.
     */
    assert(heapset->nslots <= heapset->nalloc);

    FUNC_LEAVE_NOAPI_VOID
} /* end H5HG__trim_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__create_local_heapset
 *
 * Purpose:     Creates an empty chunk-local heap-set manager.
 *
 *              The new heap set initially has capacity for one stable
 *              heap slot, but no descriptor-visible slots and no member
 *              heaps. Individual H5HG_heap_t objects are created later,
 *              as needed, by H5HG__insert_local_heapset().
 *
 * Return:      Pointer to a new H5HG_local_heapset_t on success, NULL
 *              on failure.
 *
 *                                              -- AZO   8/24/26
 *
 *-------------------------------------------------------------------------
 */
H5HG_local_heapset_t *
H5HG__create_local_heapset(void)
{
    H5HG_local_heapset_t *ret_value = NULL;

    FUNC_ENTER_PACKAGE

    /*
     * Allocate the manager with room for its first heap pointer.
     * This does NOT create an H5HG heap.
     */
    if (NULL == (ret_value = H5HG__alloc_local_heapset(1)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, NULL, "unable to create chunk-local heap set.");

    /* A newly created manager contains no actual heaps yet. */
    assert(0 == ret_value->nslots);
    assert(0 == ret_value->nlive);
    assert(1 == ret_value->nalloc);

done:
    FUNC_LEAVE_NOAPI(ret_value);
} /* end H5HG__create_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__free_local_heapset
 *
 * Purpose:     Frees a chunk-local heap set and all member heaps currently
 *              owned by it.
 *
 *              HEAPSET may be NULL because chunk-local heap state is
 *              optional and created lazily. In that case there is nothing
 *              to release and the routine succeeds.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   8/24/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__free_local_heapset(H5HG_local_heapset_t *heapset)
{
    size_t u;
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* An absent optional heap set requires no cleanup. */
    if (NULL == heapset)
        HGOTO_DONE(SUCCEED);

    assert(heapset->nalloc > 0);
    assert(heapset->nalloc <= H5HG_LOCAL_MAX_HEAP_SLOTS);
    assert(heapset->nslots <= heapset->nalloc);

    /* Only descriptor-visible slots can own member heaps. */
    for (u = 0; u < heapset->nslots; u++) {

        if (heapset->heaps[u]) {
            if (H5HG__free_local(heapset->heaps[u]) < 0)
                HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free member chunk-local heap");
        }
    }

    H5MM_free(heapset);

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__free_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__reserve_local
 *
 * Purpose:     Reserve image and object-table capacity for one local insertion.
 *              Grow a normal member geometrically up to the normal bound.
 *              Allocate all replacement storage before changing the member;
 *              allocation failure leaves its image, table, and accounting intact.
 *              Object IDs and stable heap slots do not change during relocation.
 *
 *
 *
 *                                                  -- AZO  09/24/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5HG__reserve_local(H5F_t *f, H5HG_local_heapset_t *heapset, H5HG_heap_t *heap, size_t need)
{
    uint8_t    *new_chunk = NULL; /* Replacement image, privately owned until commit */
    H5HG_obj_t *new_obj   = NULL; /* Replacement object table */
    uint8_t    *p;
    size_t      new_size;
    size_t      new_nalloc;
    size_t      required;
    size_t      image_delta;
    size_t      table_delta;
    size_t      free_offset;
    size_t      u;
    herr_t      ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(f);
    assert(heapset);
    assert(heap);
    assert(heap->nalloc > 0);
    assert(heap->nused <= heap->nalloc);

    new_size   = heap->size;
    new_nalloc = heap->nalloc;

    if (heap->obj[0].size < need) {

        if (heap->size > H5HG_LOCAL_NORMAL_HEAP_SIZE ||
            need - heap->obj[0].size > H5HG_LOCAL_NORMAL_HEAP_SIZE - heap->size)
            HGOTO_ERROR(H5E_HEAP, H5E_NOSPACE, FAIL, "local heap cannot grow within normal bound");

        required = heap->size + (need - heap->obj[0].size);

        while (new_size < required) {
            new_size =
                new_size > H5HG_LOCAL_NORMAL_HEAP_SIZE / 2 ? H5HG_LOCAL_NORMAL_HEAP_SIZE : new_size * 2;
        }
    }

    /* Reserve the next index before H5HG__alloc() can increment NUSED.
     * Once all indices have been issued, insertion reuses a cleared entry.
     */
    if (heap->nused <= H5HG_MAXIDX && heap->nused >= new_nalloc)
        new_nalloc = MIN(MAX(new_nalloc * 2, heap->nused + 1), ((size_t)H5HG_MAXIDX + 1));

    image_delta = new_size - heap->size;

    if (new_nalloc > SIZE_MAX / sizeof(heap->obj[0]))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "local heap object-table size overflow");

    table_delta = (new_nalloc - heap->nalloc) * sizeof(heap->obj[0]);

    if (image_delta > SIZE_MAX - heapset->alloc_size ||
        table_delta > SIZE_MAX - heapset->alloc_size - image_delta)
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "local heap allocation accounting overflow");

    if (image_delta) {
        if (NULL == (new_chunk = H5FL_BLK_MALLOC(gheap_chunk, new_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to grow local heap image");

        H5MM_memcpy(new_chunk, heap->chunk, heap->size);
        memset(new_chunk + heap->size, 0, image_delta);
    }
    if (table_delta) {
        if (NULL == (new_obj = H5FL_SEQ_MALLOC(H5HG_obj_t, new_nalloc)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to grow local heap object table");

        H5MM_memcpy(new_obj, heap->obj, heap->nalloc * sizeof(heap->obj[0]));
        memset(new_obj + heap->nalloc, 0, table_delta);
    }

    /* No fallible operations remain. Relocate while the old image is alive,
     * so pointer subtraction never uses freed storage.
     */
    if (new_obj) {

        heap->obj    = H5FL_SEQ_FREE(H5HG_obj_t, heap->obj);
        heap->obj    = new_obj;
        new_obj      = NULL;
        heap->nalloc = new_nalloc;
    }
    if (new_chunk) {

        free_offset = heap->size - heap->obj[0].size;

        for (u = 1; u < heap->nused; u++) {

            if (heap->obj[u].begin) {
                heap->obj[u].begin = new_chunk + (size_t)(heap->obj[u].begin - heap->chunk);
            }
        }

        /* Handles both an exhausted image and a headerless free tail. */
        heap->obj[0].begin = new_chunk + free_offset;
        heap->obj[0].size += image_delta;
        heap->obj[0].nrefs = 0;

        p = new_chunk + H5_SIZEOF_MAGIC + 4;

        H5F_ENCODE_LENGTH(f, p, new_size);

        if (heap->obj[0].size >= H5HG_SIZEOF_OBJHDR(f)) {
            p = heap->obj[0].begin;
            UINT16ENCODE(p, 0);
            UINT16ENCODE(p, 0);
            UINT32ENCODE(p, 0);
            H5F_ENCODE_LENGTH(f, p, heap->obj[0].size);
        }

        heap->chunk = H5FL_BLK_FREE(gheap_chunk, heap->chunk);
        heap->chunk = new_chunk;
        new_chunk   = NULL;
        heap->size  = new_size;
    }

    heapset->alloc_size += image_delta + table_delta;

done:
    if (new_chunk) {
        new_chunk = H5FL_BLK_FREE(gheap_chunk, new_chunk);
    }
    if (new_obj) {
        new_obj = H5FL_SEQ_FREE(H5HG_obj_t, new_obj);
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5HG__reserve_local() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__insert_local_heapset
 *
 * Purpose:     Inserts an object into a chunk-local heap set and returns
 *              the stable heap-slot index and per-heap object index that
 *              identify the new payload.
 *
 *              Normal member heaps are searched in increasing stable-slot
 *              order using first fit. If no existing normal heap can
 *              accept the object, an unused stable slot is reused or a
 *              new slot is appended and a new bounded H5HG heap is created.
 *
 *              An object whose complete serialized extent is too large for a
 *              normal heap is placed in a dedicated larger member heap.
 *
 *              SIZE may be zero. A zero-length VL payload is still a real
 *              object and receives a nonzero object index.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   8/25/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__insert_local_heapset(H5F_t *f, H5HG_local_heapset_t **heapset_ptr, size_t size, const void *obj,
                           uint16_t *heap_slot_out, uint16_t *obj_idx_out)
{
    H5HG_local_heapset_t *heapset           = NULL;     /* Current heap-set manager */
    H5HG_heap_t          *heap              = NULL;     /* Member heap being examined */
    H5HG_heap_t          *selected_heap     = NULL;     /* Existing heap selected for insertion */
    H5HG_heap_t          *grow_heap         = NULL;     /* First member able to grow */
    H5HG_heap_t          *new_heap          = NULL;     /* Newly created member heap */
    size_t                aligned_size      = 0;        /* Aligned payload size */
    size_t                need              = 0;        /* Complete encoded object extent */
    size_t                min_heap_size     = 0;        /* Minimum heap size for new member */
    size_t                heap_size         = 0;        /* Size selected for new member */
    size_t                free_slot         = 0;        /* First reusable NULL stable slot */
    size_t                slot              = SIZE_MAX; /* Selected stable heap-slot index */
    size_t                selected_slot     = SIZE_MAX; /* Slot of selected existing heap */
    size_t                grow_slot         = SIZE_MAX; /* Slot of first growable member */
    size_t                obj_idx           = 0;        /* Per-heap object index */
    size_t                member_alloc_size = 0;        /* Resident allocation of new member heap */
    hbool_t               created_heapset   = false;    /* This call created outer manager */
    hbool_t               oversized         = false;    /* Object requires dedicated large heap */
    hbool_t               found_free_slot   = false;    /* A reusable stable slot was found */
    herr_t                ret_value         = SUCCEED;  /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(f);
    assert(heapset_ptr);
    assert(heap_slot_out);
    assert(obj_idx_out);

    /*
     * A zero-length payload is valid. OBJ is required only when there are
     * actual payload bytes to copy.
     */
    if ((size > 0) && (NULL == obj))
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "non-empty chunk-local payload has no source buffer");

    /* Do not expose stale output values after a failed insertion. */
    *heap_slot_out = 0;
    *obj_idx_out   = 0;

    /* Determine the complete serialized extent required by the object. */
    aligned_size = H5HG_ALIGN(size);

    if (aligned_size < size)
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local object size overflow");

    if (aligned_size > (SIZE_MAX - H5HG_SIZEOF_OBJHDR(f)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local object extent overflow");

    need = H5HG_SIZEOF_OBJHDR(f) + aligned_size;

    /* A newly created member heap also requires its collection header. */
    if (need > (SIZE_MAX - H5HG_SIZEOF_HDR(f)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap size overflow");

    min_heap_size = H5HG_SIZEOF_HDR(f) + need;

    /*
     * Objects whose complete minimum heap image exceeds the normal bound
     * receive a dedicated oversized member heap.
     */
    oversized = (min_heap_size > H5HG_LOCAL_NORMAL_HEAP_SIZE);

    /*
     * Lazily create only the outer manager. A member heap is not created
     * until an existing member has been ruled out.
     */
    if (NULL == *heapset_ptr) {
        if (NULL == (*heapset_ptr = H5HG__create_local_heapset()))
            HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, FAIL, "unable to create chunk-local heap set");

        created_heapset = true;
    }

    heapset = *heapset_ptr;

    /* Validate manager state before indexing its trailing pointer array. */
    if ((0 == heapset->nalloc) || (heapset->nslots > heapset->nalloc) ||
        (heapset->nalloc > H5HG_LOCAL_MAX_HEAP_SLOTS))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "invalid chunk-local heap-set state");

    assert(heapset->nslots <= heapset->nalloc);

    /*
     * Search the stable-slot range.
     *
     * Normal objects first use existing free space, then a growable member.
     * Oversized
     * objects always receive a dedicated heap, but we still remember the
     * first unused stable slot so it can be reused.
     */
    for (slot = 0; slot < heapset->nslots; slot++) {
        heap = heapset->heaps[slot];

        if (NULL == heap) {

            if (!found_free_slot) {
                free_slot       = slot;
                found_free_slot = true;
            }

            continue;
        }

        assert(heap->nlive <= H5HG_MAXIDX);

        if ((!oversized) && (heap->size <= H5HG_LOCAL_NORMAL_HEAP_SIZE) && (heap->nlive < H5HG_MAXIDX) &&
            (heap->obj[0].size >= need)) {

            selected_heap = heap;
            selected_slot = slot;
            break;
        }

        /* Prefer any existing free extent before growing the first candidate. */
        if (!oversized && !grow_heap && heap->size < H5HG_LOCAL_NORMAL_HEAP_SIZE &&
            heap->nlive < H5HG_MAXIDX && heap->obj[0].size < need &&
            need - heap->obj[0].size <= H5HG_LOCAL_NORMAL_HEAP_SIZE - heap->size) {
            grow_heap = heap;
            grow_slot = slot;
        }
    }

    if (!selected_heap && grow_heap) {
        selected_heap = grow_heap;
        selected_slot = grow_slot;
    }

    /*
     * If an existing member heap can accept the object, use it directly.
     * Check any possible object-table allocation growth before insertion so
     * there is no accounting failure after the payload has been committed.
     */
    if (selected_heap) {
        /* Reserve both allocations and account for them before insertion.
         * H5HG__alloc() will not need to allocate on this path.
         */
        if (H5HG__reserve_local(f, heapset, selected_heap, need) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTALLOC, FAIL, "unable to reserve local heap capacity");

        if (H5HG__insert_local(f, selected_heap, size, obj, &obj_idx) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTINSERT, FAIL, "unable to insert object into chunk-local heap");

        assert(obj_idx > 0);
        assert(obj_idx <= UINT16_MAX);
        assert(selected_slot <= UINT16_MAX);
        heapset->nlive++;

        *heap_slot_out = (uint16_t)selected_slot;
        *obj_idx_out   = (uint16_t)obj_idx;

    } /* end if */
    else {
        /*
         * No existing member heap can accept the object. Reuse the first unused
         * stable slot if possible; otherwise append a new descriptor-visible slot.
         */
        if (found_free_slot) {
            slot = free_slot;
        }
        else {
            if (heapset->nslots >= H5HG_LOCAL_MAX_HEAP_SLOTS)
                HGOTO_ERROR(H5E_HEAP, H5E_NOSPACE, FAIL, "chunk-local heap-slot space is exhausted");

            slot = heapset->nslots;

            if (slot >= heapset->nalloc) {
                if (H5HG__grow_local_heapset(heapset_ptr, slot + 1) < 0)
                    HGOTO_ERROR(H5E_HEAP, H5E_CANTALLOC, FAIL, "unable to grow chunk-local heap set");

                /* realloc() may have moved the outer manager. */
                heapset = *heapset_ptr;
            }
        }

        /*
         * Normal members start at the minimum needed for their first object,
         * rather than allocating the normal limit up front. Oversized objects receive
         * a dedicated member large enough for their complete serialized extent.
         */
        heap_size = MAX(min_heap_size, (size_t)H5HG_MINSIZE);

        if (NULL == (new_heap = H5HG__create_local(f, heap_size)))
            HGOTO_ERROR(H5E_HEAP, H5E_CANTINIT, FAIL, "unable to create chunk-local member heap");

        /*
         * Insert before publishing the new heap into the stable slot. Until
         * publication, NEW_HEAP remains locally owned and can be freed directly
         * if anything fails.
         */
        if (H5HG__insert_local(f, new_heap, size, obj, &obj_idx) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTINSERT, FAIL, "unable to insert object into new chunk-local heap");

        assert(obj_idx > 0);
        assert(obj_idx <= UINT16_MAX);
        assert(slot <= UINT16_MAX);

        /*
         * Determine the complete resident allocation owned by the new member
         * before publishing it into the heap set.
         */
        if (H5HG__local_heap_alloc_size(new_heap, &member_alloc_size) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTGET, FAIL, "unable to calculate member heap allocation");

        if ((member_alloc_size) > (SIZE_MAX - heapset->alloc_size))
            HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap-set resident allocation overflow");

        /*
         * Publish the fully constructed member. After this point the heap set
         * owns it.
         */
        heapset->heaps[slot] = new_heap;
        new_heap             = NULL;

        heapset->alloc_size += member_alloc_size;

        /*
         * Appending makes a new slot descriptor-visible. Reusing an interior
         * hole leaves the stable-slot high-water mark unchanged.
         */
        if (slot == heapset->nslots) {
            heapset->nslots++;
        }

        heapset->nlive++;

        *heap_slot_out = (uint16_t)slot;
        *obj_idx_out   = (uint16_t)obj_idx;

    } /* end else */

done:
    /*
     * NEW_HEAP has not been published into the manager, so it remains safe
     * to destroy directly on failure.
     */
    if (new_heap) {

        if (H5HG__free_local(new_heap) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free uncommitted chunk-local heap");
    }

    /*
     * Restore the original NULL state when this operation created an outer
     * manager but failed before publishing a live payload.
     */
    if ((ret_value < 0) && (created_heapset) && (*heapset_ptr) && (0 == (*heapset_ptr)->nlive)) {

        if (H5HG__free_local_heapset(*heapset_ptr) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free unused chunk-local heap set");

        *heapset_ptr = NULL;
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__insert_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__read_local_heapset()
 *
 * Purpose:     Resolves a stable heap-slot/object-index reference and
 *              reads the referenced payload from the selected member
 *              H5HG heap.
 *
 *              Heap-slot and object-index validation is performed
 *              before delegating to H5HG__read_local().
 *
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   8/25/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__read_local_heapset(H5F_t *f, const H5HG_local_heapset_t *heapset, uint16_t heap_slot, uint16_t obj_idx,
                         void *object, size_t *buf_size)
{
    const H5HG_heap_t *heap      = NULL;    /* Member heap selected by HEAP_SLOT */
    herr_t             ret_value = SUCCEED; /* read result / return value */

    FUNC_ENTER_PACKAGE

    assert(f);
    assert(buf_size);

    /*
     * Report an error on missing heap set.
     */
    if (NULL == heapset)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap set does not exist");

    assert(heapset->nalloc > 0);
    assert(heapset->nslots <= heapset->nalloc);

    /* Object index zero is reserved for the per-heap free-space record. */
    if (0 == obj_idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local object index zero is reserved");

    if ((size_t)heap_slot >= heapset->nslots)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap slot is out of range.");

    if (NULL == (heap = heapset->heaps[heap_slot]))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap slot is unused.");

    /*
     * H5HG__read_local() handles OBJECT == NULL & OBJECT != NULL
     */
    if (H5HG__read_local(f, heap, (size_t)obj_idx, object, buf_size) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTGET, FAIL, "unable to read chunk-local heap object");

done:
    FUNC_LEAVE_NOAPI(ret_value);

} /* end H5HG__read_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__remove_local_heapset
 *
 * Purpose:     Removes an object identified by a stable heap-slot and
 *              per-heap object index.
 *
 *              The actual object removal and compaction are performed
 *              by H5HG__remove_local(). After a successful removal, the
 *              heap-set live-object count is decremented.
 *
 *              If the selected member heap becomes empty, that H5HG_heap_t
 *              is freed and its stable slot becomes NULL and available for
 *              later reuse. Interior slots are never renumbered. Trailing
 *              unused slots may be removed from NSLOTS.
 *
 *              This routine never frees the outer heap-set object.
 *
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   8/25/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__remove_local_heapset(H5F_t *f, H5HG_local_heapset_t *heapset, uint16_t heap_slot, uint16_t obj_idx)
{
    H5HG_heap_t *heap              = NULL;    /* Member heap containing referenced object */
    hbool_t      heap_empty        = false;   /* Whether member heap became empty after removal */
    size_t       member_alloc_size = 0;       /* Member allocation size */
    herr_t       ret_value         = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* sanity check */
    assert(f);

    /*
     * These reference-related errors can result from bad descriptor/file
     * data, so they are ordinary FAIL conditions rather than assertions.
     */
    if (NULL == heapset)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap set does not exist");

    if (0 == obj_idx)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local object index zero is reserved");

    assert(heapset->nalloc > 0);
    assert(heapset->nslots <= heapset->nalloc);

    if ((size_t)heap_slot >= heapset->nslots)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap slot is out of range");

    if (NULL == (heap = heapset->heaps[heap_slot]))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap slot is unused");

    /*
     * A reference to a supposedly live object while the maintained total is
     * zero is inconsistent. Treat it as an error rather than decrementing
     * through zero.
     */
    if (0 == heapset->nlive)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap-set live-object count is inconsistent");

    /*
     * Determine and validate this member's current resident allocation before
     * modifying either the member heap or the heap-set bookkeeping.
     *
     * If the removal empties this member, exactly this amount will be
     * subtracted from HEAPSET->ALLOC_SIZE before the member is released.
     */
    if (H5HG__local_heap_alloc_size(heap, &member_alloc_size) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTGET, FAIL, "unable to calculate member heap allocation");

    if (member_alloc_size > heapset->alloc_size)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "heap-set allocation accounting is inconsistent");

    /*
     * H5HG__remove_local() now succeeds only when one real live object is
     * actually removed. Therefore one successful call corresponds to
     * exactly one heap-set NLIVE decrement.
     */
    if (H5HG__remove_local(f, heap, (size_t)obj_idx, &heap_empty) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTREMOVE, FAIL, "unable to remove chunk-local heap object");

    heapset->nlive--;

    /*
     * If the member became empty, release the complete member allocation.
     *
     * H5HG__free_local() does not alter the outer heap-set object. The
     * numeric slot remains stable and is simply marked unused.
     */
    if (heap_empty) {

        assert(0 == heap->nlive);

        if (H5HG__free_local(heap) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free empty chunk-local heap");

        heapset->alloc_size -= member_alloc_size;
        heapset->heaps[heap_slot] = NULL;

        /*
         * Interior NULL slots remain so descriptor-visible slot numbers stay
         * stable. Only unused slots at the end may be removed from NSLOTS.
         */
        H5HG__trim_local_heapset(heapset);
    }

    assert(heapset->nslots <= heapset->nalloc);

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__remove_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__is_empty_local_heapset
 *
 * Purpose:     Determines whether a chunk-local heap set contains any live
 *              payload objects.
 *
 *              A NULL heap-set pointer represents a chunk that has no
 *              chunk-local VL storage and is therefore empty.
 *
 *
 * Return:      TRUE if empty, FALSE otherwise.
 *
 *                                              -- AZO   8/25/26
 *
 *-------------------------------------------------------------------------
 */
htri_t
H5HG__is_empty_local_heapset(const H5HG_local_heapset_t *heapset)
{
    htri_t ret_value; /* TRUE if heap set contains no live payload objects */

    FUNC_ENTER_PACKAGE_NOERR

    /*
     * NULL means the chunk never needed a heap set. This is not an error but rather
     * a normal empty state.
     */
    if (NULL == heapset) {
        ret_value = true;
    }
    else {
        assert(heapset->nslots <= heapset->nalloc);

        /*
         * NLIVE is maintained during insertion/removal specifically so emptiness
         * is O(1).
         */
        ret_value = (0 == heapset->nlive);
    }

    FUNC_LEAVE_NOAPI(ret_value);

} /* end H5HG__is_empty_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__encode_local_heapset
 *
 * Purpose:     Serializes a chunk-local heap set into one logical
 *              H5_SECTION_VL image.
 *
 *              The image contains a small heap-set header followed by one
 *              {offset, length} directory entry for each stable heap slot.
 *              A NULL slot is encoded as {0, 0}. Each active slot points to
 *              one ordinary H5HG image produced by H5HG__encode_local().
 *
 *              A NULL or empty heap set produces no image.
 *
 *              On success, *IMAGE_OUT is allocated with H5MM_malloc() and
 *              must be released by the caller with H5MM_free().
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   8/25/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__encode_local_heapset(H5F_t *f, const H5HG_local_heapset_t *heapset, uint8_t **image_out,
                           size_t *image_len_out)
{
    uint8_t *image         = NULL;  /* Complete encoded heap-set image */
    uint8_t *dir           = NULL;  /* Current directory entry */
    uint8_t *data          = NULL;  /* Current member-heap image position */
    uint8_t *heap_image    = NULL;  /* Temporary encoded member heap */
    size_t   heap_len      = 0;     /* Encoded member-heap length */
    size_t   image_size    = 0;     /* Total heap-set image size */
    size_t   dir_size      = 0;     /* Total directory size */
    size_t   offset        = 0;     /* Member image offset */
    size_t   counted_nlive = 0;     /* Live objects counted across active member heaps */
    size_t   u;                     /* Heap slot index */
    size_t   member_image_size = 0; /* Compact encoded member length */
    herr_t   ret_value         = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* Package-private caller requirements. */
    assert(f);
    assert(image_out);
    assert(image_len_out);

    *image_out     = NULL;
    *image_len_out = 0;

    /*
     * No heap set, or a heap set with no live objects, means there is no
     * H5_SECTION_VL heap image to encode.
     */
    if (NULL == heapset)
        HGOTO_DONE(SUCCEED);

    if (0 == heapset->nlive) {

        if (0 != heapset->nslots)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "empty chunk-local heap set has active slots");

        assert(0 == heapset->nslots);

        HGOTO_DONE(SUCCEED);
    }

    /*
     * Do not generate persistent bytes from an inconsistent in-memory manager.
     * Keep the assertions as debug checks, but validate the state in release
     * builds as well.
     */
    if ((0 == heapset->nalloc) || (0 == heapset->nslots) || (heapset->nslots > heapset->nalloc) ||
        (heapset->nalloc > H5HG_LOCAL_MAX_HEAP_SLOTS))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "invalid chunk-local heap-set state");

    assert(heapset->nslots <= heapset->nalloc);
    assert(heapset->nalloc <= H5HG_LOCAL_MAX_HEAP_SLOTS);

    /*
     * Calculate the fixed header and stable-slot directory size.
     */
    if (heapset->nslots > ((SIZE_MAX - H5HG_LOCAL_HEAPSET_SIZEOF_HDR) / H5HG_LOCAL_HEAPSET_SIZEOF_DIRENT(f)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap-set directory size overflow");

    dir_size   = heapset->nslots * H5HG_LOCAL_HEAPSET_SIZEOF_DIRENT(f);
    image_size = H5HG_LOCAL_HEAPSET_SIZEOF_HDR + dir_size;

    /*
     * Sum compact member-image lengths, excluding unused resident capacity.
     */
    for (u = 0; u < heapset->nslots; u++) {
        if (heapset->heaps[u]) {

            const H5HG_heap_t *member = heapset->heaps[u];

            if (NULL == member->obj || member->size < H5HG_SIZEOF_HDR(f) ||
                member->obj[0].size > member->size - H5HG_SIZEOF_HDR(f))
                HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "invalid chunk-local member heap state");

            member_image_size = member->size - member->obj[0].size;

            if (member_image_size > SIZE_MAX - image_size)
                HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap-set image size overflow");

            if (member->nlive > SIZE_MAX - counted_nlive)
                HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, FAIL, "chunk-local heap-set live-object count overflow");

            image_size += member_image_size;
            counted_nlive += member->nlive;
        }
    }

    if (counted_nlive != heapset->nlive)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "chunk-local heap-set live-object count is inconsistent");

    if (NULL == (image = H5MM_malloc(image_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate chunk-local heap-set image");

    /*
     * Encode the small outer heap-set header.
     */
    dir = image;

    H5MM_memcpy(dir, H5HG_LOCAL_HEAPSET_MAGIC, (size_t)H5_SIZEOF_MAGIC);

    dir += H5_SIZEOF_MAGIC;

    *dir++ = H5HG_LOCAL_HEAPSET_VERSION;

    /* Reserved V1 header bytes must be zero. */
    memset(dir, 0, H5HG_LOCAL_HEAPSET_NRESERVED);
    dir += H5HG_LOCAL_HEAPSET_NRESERVED;

    UINT32ENCODE(dir, (uint32_t)heapset->nslots);

    /*
     * Member H5HG images begin immediately after the complete directory.
     */
    data = image + H5HG_LOCAL_HEAPSET_SIZEOF_HDR + dir_size;

    for (u = 0; u < heapset->nslots; u++) {

        if (NULL == heapset->heaps[u]) {
            /*
             * Preserve an unused stable slot in the directory.
             */
            H5F_ENCODE_LENGTH(f, dir, (size_t)0);
            H5F_ENCODE_LENGTH(f, dir, (size_t)0);
        }
        else {
            /*
             * Let the existing single-heap helper produce the H5HG image.
             */
            if (H5HG__encode_local(f, heapset->heaps[u], &heap_image, &heap_len) < 0)
                HGOTO_ERROR(H5E_HEAP, H5E_CANTENCODE, FAIL, "unable to encode chunk-local member heap");

            /*
             * Encoded length excludes the resident member's trailing free capacity.
             */
            if (heap_len != heapset->heaps[u]->size - heapset->heaps[u]->obj[0].size)
                HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "encoded chunk-local heap size is inconsistent");

            offset = (size_t)(data - image);

            H5F_ENCODE_LENGTH(f, dir, offset);
            H5F_ENCODE_LENGTH(f, dir, heap_len);

            H5MM_memcpy(data, heap_image, heap_len);
            data += heap_len;

            H5MM_free(heap_image);
            heap_image = NULL;
        }
    }

    assert((size_t)(data - image) == image_size);

    *image_out     = image;
    *image_len_out = image_size;
    image          = NULL;

done:
    if (heap_image)
        H5MM_free(heap_image);

    if (image)
        H5MM_free(image);

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__encode_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__decode_local_heapset
 *
 * Purpose:     Reconstructs a chunk-local heap set from one logical
 *              H5_SECTION_VL image.
 *
 *              The heap-set header gives the number of stable slots.
 *              Each directory entry contains the offset and encoded length
 *              of one member H5HG image. An entry of {0, 0} represents an
 *              unused stable slot.
 *
 *              Active member images are decoded with H5HG__decode_local()
 *              and restored to their original stable slots. The heap-set
 *              NLIVE count is reconstructed from the decoded member heaps.
 *
 * Return:      Success: Pointer to decoded H5HG_local_heapset_t
 *              Failure: NULL
 *
 *                                              -- AZO   8/25/26
 *
 *-------------------------------------------------------------------------
 */
H5HG_local_heapset_t *
H5HG__decode_local_heapset(H5F_t *f, const void *image, size_t len)
{
    const uint8_t        *p                 = (const uint8_t *)image;
    H5HG_local_heapset_t *heapset           = NULL;
    H5HG_heap_t          *heap              = NULL;
    uint32_t              encoded_nslots    = 0;
    size_t                nslots            = 0;
    size_t                dir_size          = 0;
    size_t                last_range_end    = 0;
    size_t                heap_offset       = 0;
    size_t                heap_len          = 0;
    size_t                member_alloc_size = 0;
    size_t                u;
    H5HG_local_heapset_t *ret_value = NULL;

    FUNC_ENTER_PACKAGE

    assert(f);
    assert(image);

    /*
     * A zero-length H5_SECTION_VL is represented by a NULL heap-set pointer
     * and should not be passed to this decoder.
     */
    if (len < H5HG_LOCAL_HEAPSET_SIZEOF_HDR)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "chunk-local heap-set image is too small");

    /* Validate the outer heap-set signature. */
    if (memcmp(p, H5HG_LOCAL_HEAPSET_MAGIC, (size_t)H5_SIZEOF_MAGIC) != 0)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "bad chunk-local heap-set signature");

    p += H5_SIZEOF_MAGIC;

    /* Validate the V1 heap-set format version. */
    if (*p++ != H5HG_LOCAL_HEAPSET_VERSION)
        HGOTO_ERROR(H5E_HEAP, H5E_VERSION, NULL, "unsupported chunk-local heap-set version");

    /*
     * All reserved header bytes must be zero in V1.
     */
    for (u = 0; u < H5HG_LOCAL_HEAPSET_NRESERVED; u++) {

        if (p[u] != 0)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "invalid chunk-local heap-set reserved bytes");
    }

    p += H5HG_LOCAL_HEAPSET_NRESERVED;

    /* Decode the number of descriptor-visible stable slots. */
    UINT32DECODE(p, encoded_nslots);
    nslots = (size_t)encoded_nslots;

    if ((0 == nslots) || (nslots > H5HG_LOCAL_MAX_HEAP_SLOTS))
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "invalid chunk-local heap-set slot count");

    /*
     * Validate the complete directory size before reading any entries.
     */
    if (nslots > ((SIZE_MAX - H5HG_LOCAL_HEAPSET_SIZEOF_HDR) / H5HG_LOCAL_HEAPSET_SIZEOF_DIRENT(f)))
        HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "chunk-local heap-set directory size overflow");

    dir_size = nslots * H5HG_LOCAL_HEAPSET_SIZEOF_DIRENT(f);

    if (H5HG_LOCAL_HEAPSET_SIZEOF_HDR + dir_size > len)
        HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL, "chunk-local heap-set directory extends beyond image");

    /*
     * Allocate exactly enough pointer capacity for the persistent stable-slot
     * directory. H5HG__alloc_local_heapset() also initializes ALLOC_SIZE for
     * the outer manager and its trailing pointer array.
     */
    if (NULL == (heapset = H5HG__alloc_local_heapset(nslots)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTALLOC, NULL, "unable to allocate chunk-local heap set");

    heapset->nslots = nslots;

    /*
     * Active images are written in increasing stable-slot order. Keep track
     * of the previous range end to reject overlap and out-of-order ranges.
     */
    last_range_end = H5HG_LOCAL_HEAPSET_SIZEOF_HDR + dir_size;

    for (u = 0; u < nslots; u++) {

        H5F_DECODE_LENGTH(f, p, heap_offset);
        H5F_DECODE_LENGTH(f, p, heap_len);

        /* {0, 0} represents an unused stable slot. */
        if ((0 == heap_offset) && (0 == heap_len)) {
            continue;
        }

        /* Exactly one zero field is never a valid active directory entry. */
        if ((0 == heap_offset) || (0 == heap_len))
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "invalid chunk-local heap-set directory entry");

        if (heap_offset < last_range_end)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL,
                        "chunk-local member heap ranges overlap or are out of order");

        if ((heap_offset > len) || (heap_len > (len - heap_offset)))
            HGOTO_ERROR(H5E_HEAP, H5E_OVERFLOW, NULL, "chunk-local member heap extends beyond image");

        /*
         * Reconstruct this complete embedded H5HG collection.
         */
        if (NULL == (heap = H5HG__decode_local(f, (const uint8_t *)image + heap_offset, heap_len)))
            HGOTO_ERROR(H5E_HEAP, H5E_CANTDECODE, NULL, "unable to decode chunk-local member heap");

        /*
         * Active stable slots must contain at least one live payload.
         * Empty members should have been removed before serialization and
         * represented by a {0,0} directory entry instead.
         */
        if (0 == heap->nlive)
            HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL,
                        "active chunk-local member heap contains no live objects");

        /*
         * Determine the resident allocation owned by this member before
         * transferring ownership to the heap set.
         */
        if (H5HG__local_heap_alloc_size(heap, &member_alloc_size) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTGET, NULL, "unable to calculate decoded member allocation");

        if ((member_alloc_size) > (SIZE_MAX - heapset->alloc_size))
            HGOTO_ERROR(H5E_HEAP, H5E_BADRANGE, NULL, "decoded heap-set allocation size overflow");

        /*
         * Restore the member to its original stable slot and update the
         * aggregate live-object and resident-allocation bookkeeping.
         */
        heapset->heaps[u] = heap;
        heapset->nlive += heap->nlive;
        heapset->alloc_size += member_alloc_size;

        heap = NULL;

        last_range_end = heap_offset + heap_len;

    } /* end for */

    /*
     * A non-empty serialized heap-set image must represent at least one live
     * payload. A chunk with no live VL payloads uses a zero-length VL section
     * instead of an encoded empty heap set.
     */
    if (0 == heapset->nlive)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL,
                    "non-empty heap-set image contains no live payload objects");

    /*
     * No trailing bytes are permitted after the final active member image.
     */
    if (last_range_end != len)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, NULL, "unexpected data in chunk-local heap-set image");

    assert(heapset->nslots <= heapset->nalloc);

    ret_value = heapset;
    heapset   = NULL;

done:
    /*
     * HEAP has not yet been attached if it remains non-NULL here.
     */
    if (heap) {

        if (H5HG__free_local(heap) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, NULL, "unable to free partially decoded member heap");
    }

    /*
     * This also frees any member heaps whose ownership was already
     * transferred during a partial decode.
     */
    if (heapset) {

        if (H5HG__free_local_heapset(heapset) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, NULL,
                        "unable to free partially decoded chunk-local heap set");
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__decode_local_heapset() */

/*-------------------------------------------------------------------------
 * Function:    H5HG__copy_local_heapset
 *
 * Purpose:     Creates an independent copy of a chunk-local heap set.
 *
 *              The copy preserves descriptor-visible heap-slot and object
 *              indices. This allows structured-chunk operations to modify
 *              the copy and commit it only after the complete operation
 *              succeeds.
 *
 *              A NULL or logically empty source produces a NULL copy.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                              -- AZO   09/12/26
 *-------------------------------------------------------------------------
 */
herr_t
H5HG__copy_local_heapset(H5F_t *f, const H5HG_local_heapset_t *heapset, H5HG_local_heapset_t **copy_out)
{
    H5HG_local_heapset_t *copy       = NULL;
    uint8_t              *image      = NULL;
    size_t                image_size = 0;
    herr_t                ret_value  = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(f);
    assert(copy_out);

    *copy_out = NULL;

    /*
     * Encoding preserves stable heap-slot and object indices. A NULL or
     * empty source intentionally produces a zero-length image.
     */
    if (H5HG__encode_local_heapset(f, heapset, &image, &image_size) < 0)
        HGOTO_ERROR(H5E_HEAP, H5E_CANTENCODE, FAIL, "unable to encode chunk-local heap-set copy");

    if (0 == image_size) {
        assert(NULL == image);
        HGOTO_DONE(SUCCEED);
    }

    if (NULL == image)
        HGOTO_ERROR(H5E_HEAP, H5E_BADVALUE, FAIL, "encoded chunk-local heap-set image is missing");

    if (NULL == (copy = H5HG__decode_local_heapset(f, image, image_size)))
        HGOTO_ERROR(H5E_HEAP, H5E_CANTDECODE, FAIL, "unable to decode chunk-local heap-set copy");

    *copy_out = copy;
    copy      = NULL;

done:
    if (image)
        H5MM_free(image);

    if (copy)
        if (H5HG__free_local_heapset(copy) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free incomplete chunk-local heap-set copy");

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5HG__copy_local_heapset() */