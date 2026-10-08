/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://www.hdfgroup.org/licenses.               *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* Purpose: Abstract indexed (chunked) I/O functions.  The logical
 *          multi-dimensional dataspace is regularly partitioned into
 *          same-sized "chunks", the first of which is aligned with the
 *          logical origin.  The chunks are indexed by different methods,
 *          that map a chunk index to disk address.  Each chunk can be
 *          compressed independently and the chunks may move around in the
 *          file as their storage requirements change.
 *
 * Cache:   Disk I/O is performed in units of chunks and H5MF_alloc()
 *          contains code to optionally align chunks on disk block
 *          boundaries for performance.
 *
 *          The chunk cache is an extendible hash indexed by a function
 *          of storage B-tree address and chunk N-dimensional offset
 *          within the dataset.  Collisions are not resolved -- one of
 *          the two chunks competing for the hash slot must be preempted
 *          from the cache.  All entries in the hash also participate in
 *          a doubly-linked list and entries are penalized by moving them
 *          toward the front of the list.  When a new chunk is about to
 *          be added to the cache the heap is pruned by preempting
 *          entries near the front of the list to make room for the new
 *          entry which is added to the end of the list.
 */

/****************/
/* Module Setup */
/****************/
#define H5D_FRIEND
#include "H5Dmodule.h" /* This source code file is part of the H5D module */
#define H5SC_FRIEND    /*suppress error about including H5SCpkg    */

/***********/
/* Headers */
/***********/
#include "H5private.h" /* Generic Functions            */
#ifdef H5_HAVE_PARALLEL
#include "H5ACprivate.h" /* Metadata cache            */
#endif                   /* H5_HAVE_PARALLEL */
#include "H5CXprivate.h" /* API Contexts                         */
#include "H5Dpkg.h"      /* Dataset functions            */
#include "H5Eprivate.h"  /* Error handling              */
#include "H5FLprivate.h" /* Free Lists                           */
#include "H5Fprivate.h"  /* File functions            */
#include "H5Iprivate.h"  /* IDs                      */
#include "H5MFprivate.h" /* File memory management               */
#include "H5MMprivate.h" /* Memory management            */
#include "H5PBprivate.h" /* Page Buffer	                         */
#include "H5SCpkg.h"     /* Shared chunk cache functions            */
#include "H5SLprivate.h" /* Skip Lists                               */
#include "H5VMprivate.h" /* Vector and array functions        */

#include "H5HGprivate.h" /* Heapset functions */

/****************/
/* Local Macros */
/****************/

/* Length of sequence lists requested from dataspace selections */
#define SEQ_LIST_LEN 128

/* Length of allocated arrays for building vector I/O operations */
#define VECTOR_LEN 8

/* Sanity check on chunk index types */
#define H5D_STRUCT_CHUNK_STORAGE_INDEX_CHK(storage)                                                          \
    do {                                                                                                     \
        assert((H5D_CHUNK_IDX_EARRAY == (storage)->idx_type &&                                               \
                H5D_COPS_STRUCT_CHUNK_EARRAY == (storage)->ops) ||                                           \
               (H5D_CHUNK_IDX_FARRAY == (storage)->idx_type &&                                               \
                H5D_COPS_STRUCT_CHUNK_FARRAY == (storage)->ops) ||                                           \
               (H5D_CHUNK_IDX_BT2 == (storage)->idx_type && H5D_COPS_STRUCT_CHUNK_BT2 == (storage)->ops) ||  \
               (H5D_CHUNK_IDX_SINGLE == (storage)->idx_type &&                                               \
                H5D_COPS_STRUCT_CHUNK_SINGLE == (storage)->ops));                                            \
    } while (0)

/******************/
/* Local Typedefs */
/******************/

/* Intermediate struct for the chunk cache memory format */
typedef struct H5D_chunk_cache_mem_t {
    void                 *data_buf;   /* Buffer pointer to the data values */
    void                 *sel_buf;    /* Buffer pointer to the encoded selection */
    H5S_t                *sel_space;  /* Dataspace for encoded selection */
    H5HG_local_heapset_t *vl_heapset; /* Variable-length heapset */
    /* size tracking */
    size_t sel_nbytes;      /* nbytes for selection */
    size_t sel_alloc_size;  /* alloc_size for selection */
    size_t data_nbytes;     /* nbytes for data values */
    size_t data_alloc_size; /* alloc_size for data values */
} H5D_chunk_cache_mem_t;

/********************/
/* Local Prototypes */
/********************/

/*
 * Layout I/O callbacks for structured chunk
 */
static herr_t H5D__struct_chunk_construct(H5F_t H5_ATTR_UNUSED *f, H5D_t *dset);
static herr_t H5D__struct_chunk_init(H5F_t *f, const H5D_t *const dset, hid_t dapl_id);
static herr_t H5D__struct_chunk_io_init(H5D_io_info_t *io_info, H5D_dset_io_info_t *dinfo);
static herr_t H5D__struct_chunk_mdio_init(H5D_io_info_t *io_info, H5D_dset_io_info_t *dinfo);
static herr_t H5D__struct_chunk_io_term(H5D_io_info_t H5_ATTR_UNUSED *io_info, H5D_dset_io_info_t *di);
static herr_t H5D__struct_chunk_dest(H5D_t *dset);

/* Helper routines for above layout callbacks */
static herr_t H5D__struct_chunk_may_use_select_io(H5D_io_info_t            *io_info,
                                                  const H5D_dset_io_info_t *dset_info);
static herr_t H5D__struct_chunk_io_init_selections(H5D_io_info_t *io_info, H5D_dset_io_info_t *dinfo);
static herr_t H5D__struct_chunk_set_info_real(H5O_layout_struct_chunk_t *layout, unsigned ndims,
                                              const hsize_t *curr_dims, const hsize_t *max_dims);

/*
 *  Shared chunk cache layout callbacks for structured chunk
 *
 */
static herr_t H5D__struct_chunk_lookup(H5D_t *dset, size_t count, const hsize_t *scaled[] /*in*/,
                                       haddr_t *addr[] /*out*/, hsize_t *size[] /*out*/,
                                       hsize_t *defined_values_size[] /*out*/, size_t *size_hint[] /*out*/,
                                       size_t *defined_values_size_hint[] /*out*/, void **udata /*out*/);

static herr_t H5D__struct_chunk_decode(H5D_t *dset, size_t *nbytes /*in,out*/, size_t *alloc_size /*in,out*/,
                                       bool partial_bound, void **chunk /*in,out*/, void *udata);

static herr_t H5D__struct_chunk_decode_defined_values(H5D_t *dset, size_t *nbytes /*in,out*/,
                                                      size_t *alloc_size /*in,out*/, bool partial_bound,
                                                      void **chunk /*in,out*/, void *udata);

static herr_t H5D__struct_chunk_new_chunk(H5D_t *dset, bool fill, size_t *nbytes /*out*/,
                                          size_t *buf_size /*out*/, void **chunk /*chunk*/,
                                          void **udata /*out*/);

static herr_t H5D__struct_chunk_condense(H5D_t *dset, size_t *nbytes /*in, out*/, void **chunk /*in, out*/,
                                         void *udata);

static herr_t H5D__struct_chunk_encode(H5D_t *dset, hsize_t *write_size /*out*/,
                                       hsize_t *write_buf_alloc /*out*/, bool partial_bound,
                                       const void *chunk, void *udata, void **write_buf /*out*/);

static herr_t H5D__struct_chunk_encode_in_place(H5D_t *dset, size_t *write_size /*out*/, bool partial_bound,
                                                void **chunk /*in,out*/, void *udata);

static herr_t H5D__struct_chunk_evict(H5D_t *dset, void *chunk, void *udata);

static herr_t H5D__struct_chunk_insert(H5D_t *dset, size_t count, const hsize_t *scaled[] /*in*/,
                                       haddr_t *addr[] /*in,out*/, hsize_t old_disk_size[],
                                       hsize_t new_disk_size[], void *chunk[] /*in*/, void *udata[]);

static herr_t H5D__struct_chunk_vector_read(H5D_t *dset, haddr_t addr, const H5S_t *file_space_in,
                                            bool partial_bound, void *chunk /*in*/, size_t *vec_count /*out*/,
                                            haddr_t **offsets /*out*/, size_t **sizes /*out*/,
                                            bool *vector_possible /*out*/, bool *require_values /*out*/,
                                            void *udata);

static herr_t H5D__struct_chunk_vector_write(H5D_t *dset, haddr_t addr, const H5S_t *file_space_in,
                                             bool partial_bound, void *chunk /*in*/,
                                             size_t *vec_count /*out*/, haddr_t **offsets /*out*/,
                                             size_t **sizes /*out*/, bool *vector_possible /*out*/,
                                             bool *require_values /*out*/, void *udata);

static herr_t H5D__struct_chunk_scatter_mem(H5D_dset_io_info_t *dset_info, H5D_io_type_info_t *io_type_info,
                                            const H5S_t *mem_space, const H5S_t *file_space,
                                            const void *chunk, void *udata);

static herr_t H5D__struct_chunk_gather_mem(H5D_dset_io_info_t *dset_info, H5D_io_type_info_t *io_type_info,
                                           const H5S_t *mem_space, const H5S_t *file_space,
                                           size_t *nbytes /*in,out*/, size_t *alloc_size /*in,out*/,
                                           size_t *alloc_size_total /*in,out*/, void *chunk, void *udata);

static herr_t H5D__struct_chunk_fill(H5D_dset_io_info_t *dset_info, H5D_io_type_info_t *io_type_info,
                                     H5S_t *space, size_t *nbytes /*in,out*/, size_t *alloc_size /*in,out*/,
                                     size_t *alloc_size_total /*in,out*/, void *chunk, void *udata);

static herr_t H5D__struct_chunk_defined_values(H5D_t *dset, const H5S_t *selection, void *chunk,
                                               H5S_t **defined_values /*out*/, void *udata);

static herr_t H5D__struct_chunk_erase_values(H5D_t *dset, const H5S_t *selection, size_t *nbytes /*in,out*/,
                                             size_t *alloc_size /*in,out*/, void *chunk,
                                             bool *delete_chunk /*out*/, void *udata);

static herr_t H5D__struct_chunk_evict_values(H5D_t *dset, size_t *nbytes /*in,out*/,
                                             size_t *alloc_size /*in,out*/, void *chunk, void *udata);

static herr_t H5D__struct_chunk_layout_query(H5D_t *dset, hsize_t *chunk_dims, bool *encode_decode_necessary,
                                             bool *partial_bound_chunks_different_encoding);

static herr_t H5D__struct_chunk_delete_chunk(H5D_t *dset, const hsize_t *scaled /*in*/, haddr_t addr,
                                             hsize_t disk_size);

/*
 *  Shared chunk cache layout callbacks for structured global heap/heapset
 *
 */
static herr_t H5D__struct_chunk_get_alloc_size(const H5D_chunk_cache_mem_t *chk, size_t *alloc_size_out);

static herr_t H5D__struct_chunk_get_vlen_ref_size(H5D_t *dset, size_t *ref_nbytes);

static herr_t H5D__struct_chunk_prepare_vlen_type(H5D_t *dset, const H5T_t *file_type,
                                                  const H5T_t *other_type, bool file_is_src,
                                                  H5T_t **chunk_file_type, H5T_path_t **chunk_tpath,
                                                  size_t *ref_nbytes);

static herr_t H5D__struct_chunk_vlen_convert(const H5T_vlen_chunk_ctx_t *ctx, H5T_path_t *tpath,
                                             const H5T_t *src_type, const H5T_t *dst_type, size_t nelmts,
                                             void *buf, void *bkg);

/*********************/
/* Package Variables */
/*********************/
/* Layout I/O callbacks for structured chunk */
const H5D_layout_ops_t H5D_LOPS_STRUCT_CHUNK[1] = {{
    H5D__struct_chunk_construct,      /* construct */
    H5D__struct_chunk_init,           /* init */
    H5D__struct_chunk_is_space_alloc, /* is_space_alloc */
    H5D__struct_chunk_is_data_cached, /* is_data_cached */
    H5D__struct_chunk_io_init,        /* io_init */
    H5D__struct_chunk_mdio_init,      /* mdio_init */
    NULL,                             /* ser_read */
    NULL,                             /* ser_write */
    NULL,                             /* readvv */
    NULL,                             /* writevv */
    NULL,                             /* flush */
    H5D__struct_chunk_io_term,        /* io_term */
    H5D__struct_chunk_dest            /* dest */
}};

/* Shared Chunk Cache layout callbacks for structured chunked */
const H5SC_layout_ops_t H5SC_LOPS_STRUCT_CHUNK[1] = {{
    H5D__struct_chunk_lookup,                /* lookup */
    H5D__struct_chunk_decode,                /* decode */
    H5D__struct_chunk_decode_defined_values, /* decode_defined_values */
    H5D__struct_chunk_new_chunk,             /* new_chunk */
    H5D__struct_chunk_condense,              /* condense */
    H5D__struct_chunk_encode,                /* encode */
    H5D__struct_chunk_encode_in_place,       /* encode_in_place */
    H5D__struct_chunk_evict,                 /* evict */
    H5D__struct_chunk_insert,                /* insert */
    NULL,                                    /* selection_read */
    H5D__struct_chunk_vector_read,           /* vector_read */
    NULL,                                    /* selection_write */
    H5D__struct_chunk_vector_write,          /* vector_write */
    H5D__struct_chunk_scatter_mem,           /* scatter_mem */
    H5D__struct_chunk_gather_mem,            /* gather_mem */
    H5D__struct_chunk_fill,                  /* fill */
    H5D__struct_chunk_defined_values,        /* defined_values */
    H5D__struct_chunk_erase_values,          /* erase_values */
    H5D__struct_chunk_evict_values,          /* evict_values */
    H5D__struct_chunk_layout_query,          /* layout_query */
    H5D__struct_chunk_delete_chunk           /* delete_chunk */
}};

/*******************/
/* Local Variables */
/*******************/

/* Declare extern free list to manage the H5S_sel_iter_t struct */
H5FL_EXTERN(H5S_sel_iter_t);

/* Declare extern free list to manage sequences of size_t */
H5FL_SEQ_EXTERN(size_t);

/* Declare extern free list to manage sequences of hsize_t */
H5FL_SEQ_EXTERN(hsize_t);

/* Declare extern free list to manage the H5D_piece_info_t struct */
H5FL_EXTERN(H5D_piece_info_t);

/* Declare extern free list to manage the H5D_chunk_info_t struct */
H5FL_DEFINE(H5D_chunk_map_t);

/* Declare a free list to manage blocks of scat_buf data */
H5FL_BLK_DEFINE(scat_buf);

/*
 * Helper routines for layout callbacks
 */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_create
 *
 * Purpose:    Creates a new chunked storage index and initializes the
 *        layout information with information about the storage.  The
 *        layout info should be immediately written to the object header.
 *
 * Return:    Non-negative on success (with the layout information initialized
 *        and ready to write to an object header). Negative on failure.
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5D__struct_chunk_create(const H5D_t *dset /*in,out*/)
{
    H5D_chk_idx_info_t          idx_info; /* Chunked index info */
    H5O_storage_struct_chunk_t *store     = &(dset->shared->layout.storage.u.struct_chunk);
    herr_t                      ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(dset);
    assert(H5D_STRUCT_CHUNK == dset->shared->layout.type);
    assert(dset->shared->layout.u.struct_chunk.ndims > 0 &&
           dset->shared->layout.u.struct_chunk.ndims <= H5O_LAYOUT_NDIMS);
    H5D_STRUCT_CHUNK_STORAGE_INDEX_CHK(store);

#ifndef NDEBUG
    {
        unsigned u; /* Local index variable */

        for (u = 0; u < dset->shared->layout.u.struct_chunk.ndims; u++)
            assert(dset->shared->layout.u.struct_chunk.dim[u] > 0);
    }
#endif

    /* Compose chunked index info struct */
    idx_info.f           = dset->oloc.file;
    idx_info.stc_pline   = &dset->shared->dcpl_cache.stc_pline;
    idx_info.stc_layout  = &dset->shared->layout.u.struct_chunk;
    idx_info.stc_storage = store;

    /* Create the index for the chunks */
    if ((store->ops->create)(&idx_info) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "can't create chunk index");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__chunk_create() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_may_use_select_io
 *
 * Purpose:    A small internal function to if it may be possible to use
 *             selection I/O.
 *
 * Return:    true or false
 *
 *
 * Updated:     Selection I/O is disabled for structured chunks whose datatype
 *              contains variable-length data. Their file-side descriptors
 *              reference payloads owned by the decoded chunk's local H5HG heap
 *              set and must be processed through the SCC representation and
 *              chunk-local H5T conversion callbacks.
 *
 *              Allowing selection I/O would bypass that decoding, ownership,
 *              and conversion path. When VL data is detected, selection I/O
 *              is therefore disabled and H5D_SEL_IO_CHUNK_CACHE is recorded
 *              as the reason before returning successfully.
 *
 *                                              -- AZO   9/14/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_may_use_select_io(H5D_io_info_t *io_info, const H5D_dset_io_info_t *dset_info)
{
    const H5D_t *dataset       = NULL; /* Local pointer to dataset info */
    htri_t       has_vlen_type = false;
    herr_t       ret_value     = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity check */
    assert(io_info);
    assert(dset_info);

    dataset = dset_info->dset;
    assert(dataset);

    /*
     * VL structured chunks must pass through the decoded SCC representation
     * so H5T can resolve file-side descriptors through the chunk-local H5HG
     * heap set. Selection I/O would bypass that ownership and conversion path.
     */
    if ((has_vlen_type = H5T_detect_class(dataset->shared->type, H5T_VLEN, false)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to determine whether datatype contains VL");

    if (has_vlen_type) {
        io_info->use_select_io = H5D_SELECTION_IO_MODE_OFF;
        io_info->no_selection_io_cause |= H5D_SEL_IO_CHUNK_CACHE;
        HGOTO_DONE(SUCCEED);
    }

    /* Don't use selection I/O if there are filters on the dataset (for now) */
    if (dataset->shared->dcpl_cache.stc_pline.tot_filt_nsects > 0) {
        io_info->use_select_io = H5D_SELECTION_IO_MODE_OFF;
        io_info->no_selection_io_cause |= H5D_SEL_IO_DATASET_FILTER;
    }
    else {
        bool page_buf_enabled;

        /* Check if the page buffer is enabled */
        if (H5PB_enabled(io_info->f_sh, H5FD_MEM_DRAW, &page_buf_enabled) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't check if page buffer is enabled");
        if (page_buf_enabled) {
            /* Note that page buffer is disabled in parallel */
            io_info->use_select_io = H5D_SELECTION_IO_MODE_OFF;
            io_info->no_selection_io_cause |= H5D_SEL_IO_PAGE_BUFFER;
        }
    }

    /* Remove coding for checking if chunks in this dataset may be cached */

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_may_use_select_io() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_io_init_selections
 *
 * Purpose:        Initialize the chunk mappings
 *
 * Return:        Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_io_init_selections(H5D_io_info_t *io_info, H5D_dset_io_info_t *dinfo)
{
    H5D_chunk_map_t   *fm;                 /* Convenience pointer to chunk map */
    const H5D_t       *dataset;            /* Local pointer to dataset info */
    const H5T_t       *mem_type;           /* Local pointer to memory datatype */
    H5S_t             *tmp_mspace = NULL;  /* Temporary memory dataspace */
    bool               iter_init  = false; /* Selection iteration info has been initialized */
    char               bogus;              /* "bogus" buffer to pass to selection iterator */
    H5D_io_info_wrap_t io_info_wrap;
    herr_t             ret_value = SUCCEED; /* Return value        */

    FUNC_ENTER_PACKAGE

    assert(io_info);
    assert(dinfo);

    /* Set convenience pointers */
    fm = dinfo->layout_io_info.chunk_map;
    assert(fm);
    dataset  = dinfo->dset;
    mem_type = dinfo->type_info.mem_type;

    /* Special case for only one element in selection */
    /* (usually appending a record) */
    if (dinfo->nelmts == 1
#ifdef H5_HAVE_PARALLEL
        && !(io_info->using_mpi_vfd)
#endif /* H5_HAVE_PARALLEL */
        && H5S_SEL_ALL != H5S_GET_SELECT_TYPE(dinfo->file_space)) {
        /* Initialize skip list for chunk selections */
        fm->use_single = true;

        /* Remove coding to use/setup the chunk cache's copy of single_piece_info */

        /* Initialize single chunk dataspace */
        if (NULL == dataset->shared->struct_chunk.single_space) {
            /* Make a copy of the dataspace for the dataset */
            if ((dataset->shared->struct_chunk.single_space = H5S_copy(dinfo->file_space, true, false)) ==
                NULL)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy file space");

            /* Resize chunk's dataspace dimensions to size of chunk */
            if (H5S_set_extent_real(dataset->shared->struct_chunk.single_space, fm->chunk_dim) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSET, FAIL, "can't adjust chunk dimensions");

            /* Set the single chunk dataspace to 'all' selection */
            if (H5S_select_all(dataset->shared->struct_chunk.single_space, true) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL, "unable to set all selection");
        } /* end if */
        fm->single_space = dataset->shared->struct_chunk.single_space;
        assert(fm->single_space);

        /* Allocate the single chunk information */
        if (NULL == dataset->shared->struct_chunk.single_piece_info)
            if (NULL == (dataset->shared->struct_chunk.single_piece_info = H5FL_MALLOC(H5D_piece_info_t)))
                HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "can't allocate chunk info");
        fm->single_piece_info = dataset->shared->struct_chunk.single_piece_info;
        assert(fm->single_piece_info);

        /* Reset chunk template information */
        fm->mchunk_tmpl = NULL;

        /* Set up chunk mapping for single element */
        if (H5D__create_piece_map_single(dinfo, io_info) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL,
                        "unable to create chunk selections for single element");
    } /* end if */
    else {
        bool sel_hyper_flag; /* Whether file selection is a hyperslab */

        /* Initialize skip list for chunk selections */
        if (NULL == dataset->shared->struct_chunk.sel_chunks)
            if (NULL == (dataset->shared->struct_chunk.sel_chunks = H5SL_create(H5SL_TYPE_HSIZE, NULL)))
                HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, FAIL, "can't create skip list for chunk selections");
        fm->dset_sel_pieces = dataset->shared->struct_chunk.sel_chunks;
        assert(fm->dset_sel_pieces);

        /* We are not using single element mode */
        fm->use_single = false;

        /* Get type of selection on disk & in memory */
        if ((fm->fsel_type = H5S_GET_SELECT_TYPE(dinfo->file_space)) < H5S_SEL_NONE)
            HGOTO_ERROR(H5E_DATASET, H5E_BADSELECT, FAIL, "unable to get type of selection");
        if ((fm->msel_type = H5S_GET_SELECT_TYPE(dinfo->mem_space)) < H5S_SEL_NONE)
            HGOTO_ERROR(H5E_DATASET, H5E_BADSELECT, FAIL, "unable to get type of selection");

        /* If the selection is NONE or POINTS, set the flag to false */
        if (fm->fsel_type == H5S_SEL_POINTS || fm->fsel_type == H5S_SEL_NONE)
            sel_hyper_flag = false;
        else
            sel_hyper_flag = true;

        /* Check if file selection is a not a hyperslab selection */
        if (sel_hyper_flag) {
            /* Build the file selection for each chunk */
            if (H5S_SEL_ALL == fm->fsel_type) {
                if (H5D__create_piece_file_map_all(dinfo, io_info) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to create file chunk selections");
            } /* end if */
            else {
                /* Sanity check */
                assert(fm->fsel_type == H5S_SEL_HYPERSLABS);

                if (H5D__create_piece_file_map_hyper(dinfo, io_info) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to create file chunk selections");
            } /* end else */
        }     /* end if */
        else {
            H5S_sel_iter_op_t iter_op; /* Operator for iteration */

            /* set opdata for H5D__piece_mem_cb */
            io_info_wrap.io_info = io_info;
            io_info_wrap.dinfo   = dinfo;
            iter_op.op_type      = H5S_SEL_ITER_OP_LIB;
            iter_op.u.lib_op     = H5D__piece_file_cb;

            /* Spaces might not be the same shape, iterate over the file selection directly */
            if (H5S_select_iterate(&bogus, dataset->shared->type, dinfo->file_space, &iter_op,
                                   &io_info_wrap) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to create file chunk selections");

            /* Reset "last piece" info */
            fm->last_index      = (hsize_t)-1;
            fm->last_piece_info = NULL;
        } /* end else */

        /* Build the memory selection for each chunk */
        if (sel_hyper_flag && H5S_SELECT_SHAPE_SAME(dinfo->file_space, dinfo->mem_space) == true) {
            /* Reset chunk template information */
            fm->mchunk_tmpl = NULL;

            /* If the selections are the same shape, use the file chunk
             * information to generate the memory chunk information quickly.
             */
            if (H5D__create_piece_mem_map_hyper(dinfo) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to create memory chunk selections");
        } /* end if */
        else if (sel_hyper_flag && fm->f_ndims == 1 && fm->m_ndims == 1 &&
                 H5S_SELECT_IS_REGULAR(dinfo->mem_space) && H5S_SELECT_IS_SINGLE(dinfo->mem_space)) {
            if (H5D__create_piece_mem_map_1d(dinfo) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to create file chunk selections");
        } /* end else-if */
        else {
            H5S_sel_iter_op_t iter_op;   /* Operator for iteration */
            size_t            elmt_size; /* Memory datatype size */

            /* Make a copy of equivalent memory space */
            if ((tmp_mspace = H5S_copy(dinfo->mem_space, true, false)) == NULL)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy memory space");

            /* De-select the mem space copy */
            if (H5S_select_none(tmp_mspace) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTINIT, FAIL, "unable to de-select memory space");

            /* Save chunk template information */
            fm->mchunk_tmpl = tmp_mspace;

            /* Create selection iterator for memory selection */
            if (0 == (elmt_size = H5T_get_size(mem_type)))
                HGOTO_ERROR(H5E_DATATYPE, H5E_BADSIZE, FAIL, "datatype size invalid");
            if (H5S_select_iter_init(&(fm->mem_iter), dinfo->mem_space, elmt_size, 0) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTINIT, FAIL, "unable to initialize selection iterator");
            iter_init = true; /* Selection iteration info has been initialized */

            /* set opdata for H5D__piece_mem_cb */
            io_info_wrap.io_info = io_info;
            io_info_wrap.dinfo   = dinfo;
            iter_op.op_type      = H5S_SEL_ITER_OP_LIB;
            iter_op.u.lib_op     = H5D__piece_mem_cb;

            /* Spaces aren't the same shape, iterate over the memory selection directly */
            if (H5S_select_iterate(&bogus, dataset->shared->type, dinfo->file_space, &iter_op,
                                   &io_info_wrap) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to create memory chunk selections");
        } /* end else */
    }     /* end else */

done:
    /* Release the [potentially partially built] chunk mapping information if an error occurs */
    if (ret_value < 0) {
        if (tmp_mspace && !fm->mchunk_tmpl)
            if (H5S_close(tmp_mspace) < 0)
                HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                            "can't release memory chunk dataspace template");
        if (H5D__struct_chunk_io_term(io_info, dinfo) < 0)
            HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to release chunk mapping");
    } /* end if */

    if (iter_init && H5S_SELECT_ITER_RELEASE(&(fm->mem_iter)) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to release selection iterator");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_io_init_selections() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_set_info_real
 *
 * Purpose:     Internal routine to set the information about chunks for a dataset
 *
 * Return:      SUCCEED/FAIL
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_set_info_real(H5O_layout_struct_chunk_t *layout, unsigned ndims, const hsize_t *curr_dims,
                                const hsize_t *max_dims)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(layout);
    assert(curr_dims);

    /* Can happen when corrupt files are parsed */
    if (ndims == 0)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "number of dimensions cannot be zero");

    /* Compute the # of chunks in dataset dimensions */
    layout->nchunks     = 1;
    layout->max_nchunks = 1;
    for (unsigned u = 0; u < ndims; u++) {
        /* Round up to the next integer # of chunks, to accommodate partial chunks */
        layout->chunks[u] = ((curr_dims[u] + layout->dim[u]) - 1) / layout->dim[u];
        if (H5S_UNLIMITED == max_dims[u])
            layout->max_chunks[u] = H5S_UNLIMITED;
        else {
            /* Sanity check */
            if (layout->dim[u] == 0)
                HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "dimension size must be > 0, dim = %u ", u);

            layout->max_chunks[u] = ((max_dims[u] + layout->dim[u]) - 1) / layout->dim[u];
        }

        /* Accumulate the # of chunks */
        layout->nchunks *= layout->chunks[u];
        layout->max_nchunks *= layout->max_chunks[u];
    }

    /* Get the "down" sizes for each dimension */
    H5VM_array_down(ndims, layout->chunks, layout->down_chunks);
    H5VM_array_down(ndims, layout->max_chunks, layout->max_down_chunks);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_set_info_real() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_set_info
 *
 * Purpose:    Sets the information about chunks for a dataset
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5D__struct_chunk_set_info(const H5D_t *dset)
{
    herr_t ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);

    /* Set the base layout information */
    if (H5D__struct_chunk_set_info_real(&dset->shared->layout.u.struct_chunk, dset->shared->ndims,
                                        dset->shared->curr_dims, dset->shared->max_dims) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTSET, FAIL, "can't set layout's chunk info");

    /* Call the index's "resize" callback */
    if (dset->shared->layout.storage.u.struct_chunk.ops->resize &&
        (dset->shared->layout.storage.u.struct_chunk.ops->resize)(&dset->shared->layout.u.struct_chunk) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTSET, FAIL, "unable to resize chunk index information");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_set_info() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_set_sizes
 *
 * Purpose:     Sets chunk and type sizes.
 *
 * Return:      SUCCEED/FAIL
 *
 * Updated:
 *              Added support for H5_SECTION_VL
 *
 *                                      --AZO   09/15/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5D__struct_chunk_set_sizes(H5D_t *dset)
{
    uint64_t chunk_size;            /* Size of chunk in bytes */
    unsigned max_enc_bytes_per_dim; /* Max. number of bytes required to encode this dimension */
    unsigned u;                     /* Iterator */
    htri_t   has_vlen_type;
    herr_t   ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);

    /* Increment # of chunk dimensions, to account for datatype size as last element */
    dset->shared->layout.u.struct_chunk.ndims++;

    /* Set the last dimension of the chunk size to the size of the datatype */
    dset->shared->layout.u.struct_chunk.dim[dset->shared->layout.u.struct_chunk.ndims - 1] =
        (uint32_t)H5T_GET_SIZE(dset->shared->type);

    /* Compute number of bytes to use for encoding chunk dimensions */
    max_enc_bytes_per_dim = 0;
    for (u = 0; u < (unsigned)dset->shared->layout.u.struct_chunk.ndims; u++) {
        unsigned enc_bytes_per_dim; /* Number of bytes required to encode this dimension */

        /* Get encoded size of dim, in bytes */
        enc_bytes_per_dim = (H5VM_log2_gen(dset->shared->layout.u.struct_chunk.dim[u]) + 8) / 8;

        /* Check if this is the largest value so far */
        if (enc_bytes_per_dim > max_enc_bytes_per_dim)
            max_enc_bytes_per_dim = enc_bytes_per_dim;
    } /* end for */
    assert(max_enc_bytes_per_dim > 0 && max_enc_bytes_per_dim <= 8);
    dset->shared->layout.u.struct_chunk.enc_bytes_per_dim = max_enc_bytes_per_dim;

    /* Compute and store the total size of a chunk */
    /* (Use 64-bit value to ensure that we can detect >4GB chunks) */
    for (u = 1, chunk_size = (uint64_t)dset->shared->layout.u.struct_chunk.dim[0];
         u < dset->shared->layout.u.struct_chunk.ndims; u++)
        chunk_size *= (uint64_t)dset->shared->layout.u.struct_chunk.dim[u];

    dset->shared->layout.u.struct_chunk.size = chunk_size;

    /* Remove the following check: */
    /* Check for chunk larger than can be represented in 32-bits */
    /* (Chunk size is encoded in 32-bit value in v1 B-tree records) */
    /* if (chunk_size > (uint64_t)0xffffffff)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "chunk size must be < 4GB");
    H5_CHECKED_ASSIGN(dset->shared->layout.u.chunk.size, uint32_t, chunk_size, uint64_t); */

    /* Detect whether the datatype has a VL component */
    if ((has_vlen_type = H5T_detect_class(dset->shared->type, H5T_VLEN, false)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "unable to detect vlen datatypes?");

    /* Set up info for structured chunk composition */
    if (has_vlen_type) {
        /*
         * Sparse VL structured chunks contain three logical sections:
         *
         *      H5_SECTION_SELECTION
         *      H5_SECTION_FIXED
         *      H5_SECTION_VL
         *
         * The fixed section contains the file-side VL descriptors.
         * The Vl section contains the versioned chunk-local H5HG
         * heap-set image referenced by those descriptors.
         *
         */
        dset->shared->layout.storage.u.struct_chunk.nsects    = H5_SECTION_NUM; /* 3 */
        dset->shared->layout.storage.u.struct_chunk.nsects_md = H5_SECTION_NUM; /* 3 */

        dset->shared->layout.storage.u.struct_chunk.seq_sects_md[0] = H5_SECTION_SELECTION; /* 0 */
        dset->shared->layout.storage.u.struct_chunk.seq_sects_md[1] = H5_SECTION_FIXED;     /* 1 */
        dset->shared->layout.storage.u.struct_chunk.seq_sects_md[2] = H5_SECTION_VL;        /* 2 */
    }
    else { /* Fixed-size data */
        dset->shared->layout.storage.u.struct_chunk.nsects          = 2;
        dset->shared->layout.storage.u.struct_chunk.nsects_md       = 1;
        dset->shared->layout.storage.u.struct_chunk.seq_sects_md[0] = 0;
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_set_sizes */

/*-------------------------------------------------------------------------
 * Function:    H5D_struct_chunk_idx_reset
 *
 * Purpose:    Reset index information
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5D_struct_chunk_idx_reset(H5O_storage_struct_chunk_t *storage, bool reset_addr)
{
    herr_t ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_NOAPI(FAIL)

    /* Sanity checks */
    assert(storage);
    assert(storage->ops);
    H5D_STRUCT_CHUNK_STORAGE_INDEX_CHK(storage);

    /* Reset index structures */
    if ((storage->ops->reset)(storage, reset_addr) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "unable to reset chunk index info");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D_chunk_idx_reset() */

/*
 * Layout I/O callbacks for structured chunk
 */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_construct
 *
 * Purpose:    Constructs new chunked layout information for dataset
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_construct(H5F_t H5_ATTR_UNUSED *f, H5D_t *dset)
{
    unsigned u;                   /* Local index variable */
    herr_t   ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(f);
    assert(dset);

    /* Check for invalid chunk dimension rank */
    if (0 == dset->shared->layout.u.struct_chunk.ndims)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "no chunk information set?");
    if (dset->shared->layout.u.struct_chunk.ndims != dset->shared->ndims)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "dimensionality of chunks doesn't match the dataspace");

    /* Set chunk sizes */
    H5D__struct_chunk_set_sizes(dset);
    assert((unsigned)(dset->shared->layout.u.struct_chunk.ndims) <=
           NELMTS(dset->shared->layout.u.struct_chunk.dim));

    /* Chunked storage is not compatible with external storage (currently) */
    if (dset->shared->dcpl_cache.efl.nused > 0)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "external storage not supported with chunked layout");

    /* Sanity check dimensions */
    for (u = 0; u < dset->shared->layout.u.struct_chunk.ndims - 1; u++) {
        /* Don't allow zero-sized chunk dimensions */
        if (0 == dset->shared->layout.u.struct_chunk.dim[u])
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "chunk size must be > 0, dim = %u ", u);

        /*
         * The chunk size of a dimension with a fixed size cannot exceed
         * the maximum dimension size. If any dimension size is zero, there
         * will be no such restriction.
         */
        if (dset->shared->curr_dims[u] && dset->shared->max_dims[u] != H5S_UNLIMITED &&
            dset->shared->max_dims[u] < dset->shared->layout.u.struct_chunk.dim[u])
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL,
                        "chunk size must be <= maximum dimension size for fixed-sized dimensions");
    } /* end for */

    /* Reset address and pointer of the array struct for the chunked storage index */
    if (H5D_struct_chunk_idx_reset(&dset->shared->layout.storage.u.struct_chunk, true) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to reset chunked storage index");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_construct() */

/*-------------------------------------------------------------------------
 *
 * Purpose:    Called when the dataset is initialized.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_init(H5F_t *f, const H5D_t *const dset, hid_t H5_ATTR_UNUSED dapl_id)
{
    H5D_chk_idx_info_t          idx_info; /* Chunked index info */
    H5O_storage_struct_chunk_t *storage   = &(dset->shared->layout.storage.u.struct_chunk);
    bool                        idx_init  = false;
    herr_t                      ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity check */
    assert(f);
    assert(dset);
    H5D_STRUCT_CHUNK_STORAGE_INDEX_CHK(storage);

    /* Coding for raw data chunk cache for a dataset is removed */

    /* Compose chunked index info struct */
    idx_info.f           = f;
    idx_info.stc_pline   = &dset->shared->dcpl_cache.stc_pline;
    idx_info.stc_layout  = &dset->shared->layout.u.struct_chunk;
    idx_info.stc_storage = storage;

    /* Allocate any indexing structures */
    if (storage->ops->init && (storage->ops->init)(&idx_info, dset->shared->space, dset->oloc.addr) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "can't initialize indexing information");
    idx_init = true;

    /* Set the number of chunks in dataset, etc. */
    if (H5D__struct_chunk_set_info(dset) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to set # of chunks for dataset");

done:
    if (FAIL == ret_value) {

        if (idx_init && storage->ops->dest && (storage->ops->dest)(&idx_info) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "unable to release chunk index info");
    }
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_init() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_is_space_alloc
 *
 * Purpose:    Query if space is allocated for layout
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
bool
H5D__struct_chunk_is_space_alloc(const H5O_storage_t *store)
{
    const H5O_storage_struct_chunk_t *storage   = &(store->u.struct_chunk);
    bool                              ret_value = false; /* Return value */

    FUNC_ENTER_PACKAGE_NOERR

    /* Sanity checks */
    assert(store);
    H5D_STRUCT_CHUNK_STORAGE_INDEX_CHK(storage);

    /* Query index layer */
    ret_value = (storage->ops->is_space_alloc)(storage);

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_is_space_alloc() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_is_data_cached
 *
 * Purpose:     Indicate that structured-chunk reads must proceed through
 *              the shared chunk cache even when the persistent chunk index
 *              has not yet been allocated.
 *
 *              With write-back caching, SCC may contain authoritative dirty
 *              resident chunks that are not represented by the on-disk
 *              structured-chunk index. SCC also handles true sparse misses
 *              and supplies the appropriate fill-value representation.
 *
 * Return:      true
 *-------------------------------------------------------------------------
 */
bool
H5D__struct_chunk_is_data_cached(const H5D_shared_t H5_ATTR_UNUSED *shared)
{
    FUNC_ENTER_PACKAGE_NOERR

    assert(shared);
    assert(shared->layout.type == H5D_STRUCT_CHUNK);
    assert(shared->layout.sc_ops);

    /*
     * This callback is used by H5D__read() as a routing decision, not as a
     * per-selection cache-hit query. Structured-chunk reads must reach SCC
     * because resident write-back state may exist without persistent index
     * allocation.
     */
    FUNC_LEAVE_NOAPI(true)
}

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_allocated_cb
 *
 * Purpose:    Simply counts the number of chunks for a dataset.
 *
 * Return:    Success:    Non-negative
 *        Failure:    Negative
 *
 *-------------------------------------------------------------------------
 */
static int
H5D__struct_chunk_allocated_cb(const void *rec, void *_udata)
{
    const H5D_struct_chunk_rec_t *chunk_rec = (const H5D_struct_chunk_rec_t *)rec;
    hsize_t                      *nbytes    = (hsize_t *)_udata;

    FUNC_ENTER_PACKAGE_NOERR

    *(hsize_t *)nbytes += chunk_rec->nbytes;

    FUNC_LEAVE_NOAPI(H5_ITER_CONT)
} /* H5D__chunk_allocated_cb() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_allocated
 *
 * Purpose:    Return the number of bytes allocated in the file for storage
 *        of raw data in the structured chunk dataset
 *
 * Return:    Success:    Number of bytes stored in all chunks.
 *        Failure:    0
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5D__struct_chunk_allocated(const H5D_t *dset, hsize_t *nbytes)
{
    H5D_chk_idx_info_t          idx_info;        /* Chunked index info */
    hsize_t                     chunk_bytes = 0; /* Number of bytes allocated for chunks */
    H5O_storage_struct_chunk_t *sc          = &(dset->shared->layout.storage.u.struct_chunk);
    herr_t                      ret_value   = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(dset->shared);
    H5D_STRUCT_CHUNK_STORAGE_INDEX_CHK(sc);

    /* Probably need to do this when shared chunk cache is ready */
    /* Search for cached chunks that haven't been written out */

    /* Compose chunked index info struct */
    idx_info.f           = dset->oloc.file;
    idx_info.stc_pline   = &dset->shared->dcpl_cache.stc_pline;
    idx_info.stc_layout  = &dset->shared->layout.u.struct_chunk;
    idx_info.stc_storage = sc;

    /* Iterate over the chunks */
    if ((sc->ops->iterate)(&idx_info, H5D__struct_chunk_allocated_cb, &chunk_bytes) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL,
                    "unable to retrieve allocated chunk information from index");

    /* Set number of bytes for caller */
    *nbytes = chunk_bytes;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__chunk_allocated() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_io_init
 *
 * Purpose:    Performs initialization before any sort of I/O on the raw data
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_io_init(H5D_io_info_t *io_info, H5D_dset_io_info_t *dinfo)
{
    const H5D_t     *dataset = dinfo->dset;         /* Local pointer to dataset info */
    H5D_chunk_map_t *fm;                            /* Convenience pointer to chunk map */
    hssize_t         old_offset[H5O_LAYOUT_NDIMS];  /* Old selection offset */
    htri_t           file_space_normalized = false; /* File dataspace was normalized */
    unsigned         f_ndims;                       /* The number of dimensions of the file's dataspace */
    int              sm_ndims; /* The number of dimensions of the memory buffer's dataspace (signed) */
    unsigned         u;        /* Local index variable */
    herr_t           ret_value = SUCCEED; /* Return value        */

    FUNC_ENTER_PACKAGE

    /* Allocate chunk map */
    if (NULL == (dinfo->layout_io_info.chunk_map = H5FL_MALLOC(H5D_chunk_map_t)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "unable to allocate chunk map");
    fm = dinfo->layout_io_info.chunk_map;

    /* Get layout for dataset */
    dinfo->layout = &(dataset->shared->layout);

    /* Initialize "last chunk" information */
    fm->last_index      = (hsize_t)-1;
    fm->last_piece_info = NULL;

    /* Clear other fields */
    fm->mchunk_tmpl       = NULL;
    fm->dset_sel_pieces   = NULL;
    fm->single_space      = NULL;
    fm->single_piece_info = NULL;

    /* Initialize selection type in memory and file */
    fm->msel_type = H5S_SEL_ERROR;
    fm->fsel_type = H5S_SEL_ERROR;

    /* Check if the memory space is scalar & make equivalent memory space */
    if ((sm_ndims = H5S_GET_EXTENT_NDIMS(dinfo->mem_space)) < 0)
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "unable to get dimension number");
    /* Set the number of dimensions for the memory dataspace */
    H5_CHECKED_ASSIGN(fm->m_ndims, unsigned, sm_ndims, int);

    /* Get rank for file dataspace */
    fm->f_ndims = f_ndims = dataset->shared->layout.u.struct_chunk.ndims - 1;

    /* Normalize hyperslab selections by adjusting them by the offset */
    /* (It might be worthwhile to normalize both the file and memory dataspaces
     * before any (contiguous, chunked, etc) file I/O operation, in order to
     * speed up hyperslab calculations by removing the extra checks and/or
     * additions involving the offset and the hyperslab selection -QAK)
     */
    if ((file_space_normalized = H5S_hyper_normalize_offset(dinfo->file_space, old_offset)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTSET, FAIL, "unable to normalize selection");

    /* Decide the number of chunks in each dimension */
    for (u = 0; u < f_ndims; u++)
        /* Keep the size of the chunk dimensions as hsize_t for various routines */
        fm->chunk_dim[u] = dinfo->layout->u.struct_chunk.dim[u];

    if (H5D__struct_chunk_io_init_selections(io_info, dinfo) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to create file and memory chunk selections");

    /* Check if we're performing selection I/O and save the result if it hasn't
     * been disabled already */
    if (io_info->use_select_io != H5D_SELECTION_IO_MODE_OFF)
        if (H5D__struct_chunk_may_use_select_io(io_info, dinfo) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't check if selection I/O is possible");

    /* Calculate type conversion buffer size if necessary.  Currently only implemented for selection I/O. */
    if (io_info->use_select_io != H5D_SELECTION_IO_MODE_OFF &&
        !(dinfo->type_info.is_xform_noop && dinfo->type_info.is_conv_noop)) {
        H5SL_node_t *chunk_node; /* Current node in chunk skip list */

        /* Iterate through nodes in chunk skip list */
        chunk_node = H5D_CHUNK_GET_FIRST_NODE(dinfo);
        while (chunk_node) {
            H5D_piece_info_t *piece_info; /* Chunk information */

            /* Get the actual chunk information from the skip list node */
            piece_info = H5D_CHUNK_GET_NODE_INFO(dinfo, chunk_node);

            /* Handle type conversion buffer */
            H5D_INIT_PIECE_TCONV(io_info, dinfo, piece_info)

            /* Advance to next chunk in list */
            chunk_node = H5D_CHUNK_GET_NEXT_NODE(dinfo, chunk_node);
        }
    }

#ifdef H5_HAVE_PARALLEL
    /*
     * If collective metadata reads are enabled, ensure all ranks
     * have the dataset's chunk index open (if it was created) to
     * prevent possible metadata inconsistency issues or unintentional
     * independent metadata reads later on.
     */
    if (H5F_SHARED_HAS_FEATURE(io_info->f_sh, H5FD_FEAT_HAS_MPI) &&
        H5F_shared_get_coll_metadata_reads(io_info->f_sh) &&
        H5D__chunk_is_space_alloc(&dataset->shared->layout.storage)) {
        H5O_storage_chunk_t *sc = &(dataset->shared->layout.storage.u.chunk);
        H5D_chk_idx_info_t   idx_info;
        bool                 index_is_open;

        idx_info.f       = dataset->oloc.file;
        idx_info.pline   = &dataset->shared->dcpl_cache.pline;
        idx_info.layout  = &dataset->shared->layout.u.chunk;
        idx_info.storage = sc;

        assert(sc && sc->ops && sc->ops->is_open);
        if (sc->ops->is_open(&idx_info, &index_is_open) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to check if dataset chunk index is open");

        if (!index_is_open) {
            assert(sc->ops->open);
            if (sc->ops->open(&idx_info) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to open dataset chunk index");
        }

        /*
         * Load any other chunk index metadata that we can,
         * such as fixed array data blocks, while we know all
         * MPI ranks will do so with collective metadata reads
         * enabled
         */
        if (sc->ops->load_metadata && sc->ops->load_metadata(&idx_info) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to load additional chunk index metadata");
    }
#endif

done:
    if (file_space_normalized == true)
        if (H5S_hyper_denormalize_offset(dinfo->file_space, old_offset) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTSET, FAIL, "can't denormalize selection");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_io_init() */

/*-------------------------------------------------------------------------
 * Function:   H5D__struct_chunk_mdio_init
 *
 * Purpose:    Performs second phase of initialization for multi-dataset
 *             I/O.  Currently looks up chunk addresses and adds chunks to
 *             sel_pieces.
 *
 *             Only the on-disk chunk address is required from the
 *             structured-chunk lookup callback. Layout-specific lookup
 *             udata is therefore not requested and remains internal to
 *             H5D__struct_chunk_lookup().
 *
 * Return:     Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_mdio_init(H5D_io_info_t *io_info, H5D_dset_io_info_t *dinfo)
{
    H5SL_node_t      *piece_node; /* Current node in chunk skip list */
    H5D_piece_info_t *piece_info; /* Piece information for current piece */
    haddr_t          *addr[1];
    const hsize_t    *scaled[1];
    herr_t            ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Get first node in skip list.  Note we don't check for failure since NULL
     * simply indicates an empty skip list. */
    piece_node = H5D_CHUNK_GET_FIRST_NODE(dinfo);

    /* Iterate over skip list */
    while (piece_node) {
        /* Get piece info */
        if (NULL == (piece_info = (H5D_piece_info_t *)H5D_CHUNK_GET_NODE_INFO(dinfo, piece_node)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "couldn't get piece info from list");

        /* Get the info for the chunk in the file */
        scaled[0] = piece_info->scaled;
        addr[0]   = &piece_info->faddr;

        if (H5D__struct_chunk_lookup(dinfo->dset, 1, scaled, addr, NULL, NULL, NULL, NULL, NULL) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "error looking up chunk address");

        /* Add piece to MDIO operation if it has a file address */
        if (H5_addr_defined(piece_info->faddr)) {
            assert(io_info->sel_pieces);
            assert(io_info->pieces_added < io_info->piece_count);

            /* Add to sel_pieces and update pieces_added */
            io_info->sel_pieces[io_info->pieces_added++] = piece_info;

            if (piece_info->filtered_dset)
                io_info->filtered_pieces_added++;
        }

        /* Advance to next skip list node */
        piece_node = H5D_CHUNK_GET_NEXT_NODE(dinfo, piece_node);
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_mdio_init() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_io_term
 *
 * Purpose:    Destroy I/O operation information.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 * NOTE: No change from the legacy chunk version
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_io_term(H5D_io_info_t H5_ATTR_UNUSED *io_info, H5D_dset_io_info_t *di)
{
    H5D_chunk_map_t *fm;                  /* Convenience pointer to chunk map */
    herr_t           ret_value = SUCCEED; /*return value        */

    FUNC_ENTER_PACKAGE

    assert(di);

    /* Set convenience pointer */
    fm = di->layout_io_info.chunk_map;

    /* Single element I/O vs. multiple element I/O cleanup */
    if (fm->use_single) {
        /* Sanity checks */
        assert(fm->dset_sel_pieces == NULL);
        assert(fm->last_piece_info == NULL);
        assert(fm->single_piece_info);
        assert(fm->single_piece_info->fspace_shared);
        assert(fm->single_piece_info->mspace_shared);

        /* Reset the selection for the single element I/O */
        H5S_select_all(fm->single_space, true);
    } /* end if */
    else {
        /* Release the nodes on the list of selected pieces, or the last (only)
         * piece if the skiplist is not available */
        if (fm->dset_sel_pieces) {
            if (H5SL_free(fm->dset_sel_pieces, H5D__free_piece_info, NULL) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTNEXT, FAIL, "can't free dataset skip list");
        } /* end if */
        else if (fm->last_piece_info) {
            if (H5D__free_piece_info(fm->last_piece_info, NULL, NULL) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "can't free piece info");
            fm->last_piece_info = NULL;
        } /* end if */
    }     /* end else */

    /* Free the memory piece dataspace template */
    if (fm->mchunk_tmpl)
        if (H5S_close(fm->mchunk_tmpl) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                        "can't release memory chunk dataspace template");

    /* Free chunk map */
    di->layout_io_info.chunk_map = H5FL_FREE(H5D_chunk_map_t, di->layout_io_info.chunk_map);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_io_term() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_dest
 *
 * Purpose:     Free index structure
 *
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_dest(H5D_t *dset)
{
    H5D_chk_idx_info_t          idx_info; /* Chunked index info */
    H5O_storage_struct_chunk_t *storage   = &(dset->shared->layout.storage.u.struct_chunk);
    herr_t                      ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE_TAG(dset->oloc.addr)

    /* Sanity checks */
    assert(dset);
    H5D_STRUCT_CHUNK_STORAGE_INDEX_CHK(storage);

    /* Compose chunked index info struct */
    idx_info.f           = dset->oloc.file;
    idx_info.stc_pline   = &dset->shared->dcpl_cache.stc_pline;
    idx_info.stc_layout  = &dset->shared->layout.u.struct_chunk;
    idx_info.stc_storage = storage;

    /* Free any index structures */
    if (storage->ops->dest && (storage->ops->dest)(&idx_info) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "unable to release chunk index info");

done:
    FUNC_LEAVE_NOAPI_TAG(ret_value)
} /* end H5D__struct_chunk_dest() */

/*
 * Shared chunk cache layout callbacks for structured chunks
 */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_lookup
 *
 * Purpose:     Looks up chunk address and size on disk.
 *
 *              dset is the pointer to the dataset being operated on.
 *
 *              count is the number of chunks in this I/O request
 *
 *              scaled is an array of the scaled coordinates of the chunks in this I/O request
 *
 *              Each output array other than scaled is independently optional.
 *              A NULL output array, or a NULL element within one of the scalar
 *              output arrays, indicates that the corresponding value is not
 *              requested by the caller.
 *
 *              addr is an array of addresses of the on-disk chunk(s) (if available)
 *
 *              size is an array of encoded on-disk chunk sizes. For a chunk
 *              that is not allocated on disk, the returned size is 0.
 *
 *             defined_values_size is the number of encoded bytes occupied by
 *              the defined-value selection metadata at the beginning of the
 *              on-disk structured chunk.
 *
 *              For an allocated chunk, a returned value of 0 indicates that
 *              no separate defined-value selection metadata is stored and all
 *              values in the logical chunk are defined.
 *
 *              For an unallocated chunk, the chunk address is HADDR_UNDEF and
 *              defined_values_size is also returned as 0.
 *
 *              size_hint is the allocation size required for the encoded
 *              on-disk chunk buffer used by the raw read/decode path. The
 *              current implementation returns the encoded chunk size. It is
 *              not an estimate of the complete decoded resident allocation.
 *
 *              defined_values_size_hint is the allocation size required to
 *              read the encoded defined-values selection metadata. The
 *              current implementation returns defined_values_size. It is a
 *              byte size, not a defined-element count or decoded resident
 *              allocation.
 *
 *              A zero defined_values_size for an allocated chunk is a fast-path
 *              indication that all values are defined; no selection metadata
 *              needs to be read or passed to decode_defined_values().
 *
 *              _udata is an optional array of layout-specific output pointers.
 *              When supplied, each returned H5D_chunk_ud_t contains the lookup
 *              state required by subsequent structured-chunk callbacks such as
 *              decode, encode, insert, and vector I/O.
 *
 *              Ownership of each returned udata pointer is transferred to the
 *              caller. If _udata is NULL, this function retains ownership of
 *              its temporary lookup udata and releases it before returning,
 *              including on an error path.
 *
 *              If the structured-chunk index has not yet been allocated,
 *              lookup succeeds without opening the index. Each requested
 *              chunk is reported as unallocated with HADDR_UNDEF and zero
 *              sizes.
 *
 * Return:      Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_lookup(H5D_t *dset /*in*/, size_t count /*in*/, const hsize_t *scaled[] /*in*/,
                         haddr_t *addr[] /*out*/, hsize_t *size[] /*out*/,
                         hsize_t *defined_values_size[] /*out*/, size_t *size_hint[] /*out*/,
                         size_t *defined_values_size_hint[] /*out*/, void **_udata /*out*/)
{
    H5D_chunk_ud_t             *udata   = NULL;
    H5O_storage_struct_chunk_t *storage = &(dset->shared->layout.storage.u.struct_chunk);
    H5O_layout_struct_chunk_t  *layout  = &dset->shared->layout.u.struct_chunk;
    H5D_chk_idx_info_t          idx_info; /* Chunked index info */
    H5O_stc_pline_t            *pline;    /* I/O pipeline info */
    size_t                      i;
    haddr_t                     found_addr;
    hsize_t                     found_size;
    hsize_t                     found_defined_size;
    bool                        index_allocated;
    herr_t                      ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);
    assert(dset->shared->layout.type == H5D_STRUCT_CHUNK);

    pline = &(dset->shared->dcpl_cache.stc_pline);

    /* Compose chunked index info struct */
    idx_info.f           = dset->oloc.file;
    idx_info.stc_pline   = pline;
    idx_info.stc_layout  = layout;
    idx_info.stc_storage = storage;

    index_allocated = (storage->ops->is_space_alloc)(storage);

    for (i = 0; i < count; i++) {
        udata = NULL;

        /* Allocate zero-initialized udata */
        if (NULL == (udata = H5MM_calloc(sizeof(H5D_chunk_ud_t))))
            HGOTO_ERROR(H5E_ARGS, H5E_CANTALLOC, FAIL, "could not allocate space for chunk lookup udata");

        /* Set up udata */
        udata->common.stc_layout  = layout;
        udata->common.stc_storage = storage;
        udata->common.scaled      = scaled[i];

        /* Reset information about the chunk we are looking for */
        udata->chunk_block.offset = HADDR_UNDEF;
        udata->chunk_block.length = 0;

        /*
         * If the structured-chunk index has never been allocated, no
         * logical chunk can have on-disk storage. Leave the chunk address
         * undefined rather than invoking get_addr(), since some index
         * implementations require an allocated index address.
         */
        if (index_allocated) {
            /* chunk_idx is calculated in get_addr callback */
            if ((storage->ops->get_addr)(&idx_info, udata) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't query chunk address");
        }

        found_addr         = udata->chunk_block.offset;
        found_size         = 0;
        found_defined_size = 0;

        if (H5_addr_defined(found_addr)) {
            found_size         = udata->chunk_block.length;
            found_defined_size = udata->offset[1];
        }

        /* Each output is independently optional. */
        if (addr && addr[i])
            *addr[i] = found_addr;

        if (size && size[i])
            *size[i] = found_size;

        if (defined_values_size && defined_values_size[i])
            *defined_values_size[i] = found_defined_size;

        if (size_hint && size_hint[i]) {
            H5_CHECK_OVERFLOW(found_size, hsize_t, size_t);
            *size_hint[i] = (size_t)found_size;
        }

        if (defined_values_size_hint && defined_values_size_hint[i]) {
            H5_CHECK_OVERFLOW(found_defined_size, hsize_t, size_t);
            *defined_values_size_hint[i] = (size_t)found_defined_size;
        }

        /*
         * Unlike the scalar-output arrays above, _udata is an optional array of
         * output pointer values. Ownership of each returned lookup udata object is
         * transferred to the caller.
         */
        if (_udata) {
            _udata[i] = (void *)udata;
            udata     = NULL; /* Ownership transfers to caller */
        }
        else
            udata = H5MM_xfree(udata);

    } /* end count */

done:
    if (udata)
        udata = H5MM_xfree(udata);

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5D__struct_chunk_lookup() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_decode
 *
 * Purpose:     Decompresses/decodes the chunk from file format to memory cache format if necessary.
 *              Reallocs chunk buffer if necessary.
 *
 *              On entry, nbytes is the number of bytes used in the chunk buffer.
 *              On exit, it shall be set to the total number of bytes used (not allocated)
 *              across all buffers for this chunk.
 *
 *              On entry, alloc_size is the size of the chunk buffer.
 *              On exit, it shall be set to the total number of bytes allocated across all
 *              buffers for this chunk.
 *
 *              Optional, if not present, chunk is the same in cache as on disk.
 *
 *              partial_bound is true if the chunk was encoded with partial_bound set to true.
 *              If the dataset reported partial_bound_chunks_different_encoding as false,
 *              the setting of partial_bound is undefined.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 * NOTE: On entry: [chunk] is the pointer to the on disk file format chunk buffer
 *       On exit: [chunk] is the pointer to the chunk intermediate struct
 *
 *
 * Updated:     Added support for decoding three-section structured chunks
 *              containing variable-length data. The selection, fixed-value
 *              descriptor, and serialized VL heap-set sections are separated
 *              using the stored section boundaries and are independently
 *              unfiltered and checksum-verified as configured.
 *
 *              The VL section is decoded into a chunk-local H5HG heap set
 *              owned by the intermediate chunk. Section boundaries are
 *              validated before they are used so malformed offsets cannot
 *              cause size underflow or out-of-bounds section access.
 *
 *              After decoding, NBYTES describes only the logical selection
 *              and fixed-data bytes. ALLOC_SIZE also includes the resident
 *              allocation owned by the decoded VL heap set, obtained through
 *              the structured-chunk allocation helper. Partial-construction
 *              cleanup releases the heap set and all other resources created
 *              during decoding.
 *
 *              This update supersedes the earlier two-section limitation
 *              noted above while preserving two-section decoding for
 *              structured chunks without VL data.
 *
 *                                              -- AZO   9/20/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_decode(H5D_t *dset, size_t *nbytes /*in,out*/, size_t *alloc_size /*in,out*/,
                         bool partial_bound, void **chunk /*in,out*/, void *_udata)
{
    H5D_chunk_ud_t             *udata = (H5D_chunk_ud_t *)_udata;
    H5D_chunk_cache_mem_t      *chk   = NULL; /* Chunk's intermediate struct */
    H5O_stc_pline_t            *pline;        /* I/O pipeline info */
    hbool_t                     filtered = false;
    uint32_t                    stored_chksum;   /* Stored metadata checksum value */
    uint32_t                    computed_chksum; /* Computed metadata checksum value */
    void                       *tmp;
    const unsigned char        *sel_p;
    H5O_storage_struct_chunk_t *storage       = NULL; /* Structured-chunk storage information */
    void                       *vl_buf        = NULL;
    size_t                      vl_nbytes     = 0;
    size_t                      vl_alloc_size = 0;
    size_t                      fixed_end;
    size_t                      resident_alloc_size;
    hbool_t                     has_vlen_type                 = false;
    bool                        section_is_md[H5_SECTION_NUM] = {false};
    unsigned                    u;
    herr_t                      ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);

    storage = &dset->shared->layout.storage.u.struct_chunk;

    pline = &(dset->shared->dcpl_cache.stc_pline);
    if (pline && pline->tot_filt_nsects)
        filtered = true;

    /*
     * STRUCT_CHUNK_SECTION_COUNT_ASSUMPTION: The current format uses two
     * sections for fixed-size data and three for data containing VL values.
     * Revisit this inference if new section kinds, multiple VL sections,
     * or a dense VL layout are introduced.
     */
    has_vlen_type = (storage->nsects == H5_SECTION_NUM);

    /* Validate section boundaries before using them to split the chunk image */
    if (*alloc_size < *nbytes || udata->offset[H5_SECTION_FIXED] > *nbytes)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "invalid structured chunk section bounds");

    if (has_vlen_type) {
        if (udata->offset[H5_SECTION_VL] < udata->offset[H5_SECTION_FIXED] ||
            udata->offset[H5_SECTION_VL] > *nbytes)
            HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "invalid structured chunk VL section bounds");

        fixed_end = (size_t)udata->offset[H5_SECTION_VL];
    }
    else {
        fixed_end = *nbytes;
    }

    for (u = 0; u < storage->nsects_md; u++) {
        section_is_md[storage->seq_sects_md[u]] = true;
    }

    /* Allocate the chunk intermediate struct */
    if (NULL == (chk = H5MM_calloc(sizeof(H5D_chunk_cache_mem_t))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL,
                    "memory allocation failed for intermediate chunk struct");

    /* nbytes and alloc_size for encoded selection */
    chk->sel_nbytes = chk->sel_alloc_size = udata->offset[1];

    /* nbytes and alloc_size for data values */
    chk->data_nbytes     = fixed_end - chk->sel_nbytes;
    chk->data_alloc_size = chk->data_nbytes + (*alloc_size - *nbytes);

    if (has_vlen_type) {
        vl_nbytes = vl_alloc_size = *nbytes - fixed_end;
    }

    /* Allocate a buffer for the encoded selection */
    if (NULL == (chk->sel_buf = H5MM_malloc(chk->sel_alloc_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "memory allocation failed for encoded selection buffer");

    /* Copy over the encoded selection */
    H5MM_memcpy(chk->sel_buf, *chunk, chk->sel_nbytes);

    /* Allocate a buffer for the data values */
    if (chk->data_alloc_size > 0) {
        if (NULL == (chk->data_buf = H5MM_malloc(chk->data_alloc_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "memory allocation failed for data buffer");

        H5MM_memcpy(chk->data_buf, (uint8_t *)(*chunk) + chk->sel_nbytes, chk->data_nbytes);
    }

    /* Allocate a temporary buffer for the encoded VL heap set */
    if (vl_alloc_size > 0) {
        if (NULL == (vl_buf = H5MM_malloc(vl_alloc_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "memory allocation failed for encoded VL buffer");

        H5MM_memcpy(vl_buf, (uint8_t *)(*chunk) + fixed_end, vl_nbytes);
    }

    /* Decompress the encoded selection  & data values */
    if (filtered && !partial_bound) {
        H5Z_EDC_t              err_detect; /* Error detection info */
        H5Z_cb_t               filter_cb;  /* I/O filter callback function */
        unsigned               i;
        H5O_stc_filter_sect_t *filt_sect;

        /* Retrieve filter settings from API context */
        if (H5CX_get_err_detect(&err_detect) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get error detection info");
        if (H5CX_get_filter_cb(&filter_cb) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get I/O filter callback function");

        for (i = 0, filt_sect = &pline->filt_sects[0]; i < pline->tot_filt_nsects; i++, filt_sect++) {

            if (filt_sect->nused) {
                switch (filt_sect->seq_sect) {
                    case H5_SECTION_SELECTION:
                        if (H5Z_apply_filters(filt_sect->nused, filt_sect->filter, H5Z_FLAG_REVERSE,
                                              &udata->filt_mask[0], err_detect, filter_cb, &chk->sel_nbytes,
                                              &chk->sel_alloc_size, &chk->sel_buf) < 0)
                            HGOTO_ERROR(H5E_DATASET, H5E_CANTFILTER, FAIL, "output pipeline failed");
                        break;

                    case H5_SECTION_FIXED:
                        if (H5Z_apply_filters(filt_sect->nused, filt_sect->filter, H5Z_FLAG_REVERSE,
                                              &udata->filt_mask[1], err_detect, filter_cb, &chk->data_nbytes,
                                              &chk->data_alloc_size, &chk->data_buf) < 0)
                            HGOTO_ERROR(H5E_DATASET, H5E_CANTFILTER, FAIL, "output pipeline failed");
                        break;

                    case H5_SECTION_VL:
                        if (vl_nbytes > 0)
                            if (H5Z_apply_filters(filt_sect->nused, filt_sect->filter, H5Z_FLAG_REVERSE,
                                                  &udata->filt_mask[2], err_detect, filter_cb, &vl_nbytes,
                                                  &vl_alloc_size, &vl_buf) < 0)
                                HGOTO_ERROR(H5E_DATASET, H5E_CANTFILTER, FAIL, "output pipeline failed");
                        break;

                    case H5_SECTION_NUM:
                    default:
                        assert(0 && "Unknown action?!?");
                }
            } /* end if nused */
        }     /* end for */
    }
    /* Verify section checksums */
    if (section_is_md[H5_SECTION_SELECTION]) {
        if (chk->sel_nbytes < H5_SIZEOF_CHKSUM)
            HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "encoded selection is too small for checksum");
        if (H5F_get_checksums(chk->sel_buf, chk->sel_nbytes, &stored_chksum, &computed_chksum) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get checksums");
        if (stored_chksum != computed_chksum)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "checksums verification failed");

        chk->sel_nbytes -= H5_SIZEOF_CHKSUM;
    }

    if (section_is_md[H5_SECTION_FIXED] && chk->data_nbytes > 0) {
        if (chk->data_nbytes < H5_SIZEOF_CHKSUM)
            HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "encoded data is too small for checksum");
        if (H5F_get_checksums(chk->data_buf, chk->data_nbytes, &stored_chksum, &computed_chksum) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get checksums");
        if (stored_chksum != computed_chksum)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "checksums verification failed");

        chk->data_nbytes -= H5_SIZEOF_CHKSUM;
    }

    if (section_is_md[H5_SECTION_VL] && vl_nbytes > 0) {
        if (vl_nbytes <= H5_SIZEOF_CHKSUM)
            HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "encoded VL data is too small for checksum");
        if (H5F_get_checksums(vl_buf, vl_nbytes, &stored_chksum, &computed_chksum) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get checksums");
        if (stored_chksum != computed_chksum)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "checksums verification failed");

        vl_nbytes -= H5_SIZEOF_CHKSUM;
    }

    sel_p = chk->sel_buf;

    /* Decode the encoded selection to dataspace sel_space */
    if (NULL == (chk->sel_space = H5S_decode(&sel_p)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTDECODE, FAIL, "unable to decode dataspace");

    /* Decode the chunk-local VL heap set */
    if (vl_nbytes > 0) {
        if (NULL == (chk->vl_heapset = H5HG__decode_local_heapset(dset->oloc.file, vl_buf, vl_nbytes)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTDECODE, FAIL, "unable to decode chunk-local VL heap set");
    }

    /*
     * Release checksum space and unused filter-buffer capacity before
     * charging the decoded chunk to SCC. The VL heap set is retained
     * and its full resident allocation is counted separately below.
     */
    {
        void  *decoded_chunk  = chk;
        size_t decoded_nbytes = 0;

        if (H5D__struct_chunk_condense(dset, &decoded_nbytes, &decoded_chunk, udata) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to condense decoded structured chunk");

        assert(decoded_chunk == chk);
    }

    if (H5D__struct_chunk_get_alloc_size(chk, &resident_alloc_size) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to determine structured chunk allocation size");

    /* Return values on exit */
    *nbytes     = chk->sel_nbytes + chk->data_nbytes;
    *alloc_size = resident_alloc_size;

    tmp    = *chunk;
    *chunk = chk;
    chk    = NULL;
    tmp    = H5MM_xfree(tmp);

done:
    vl_buf = H5MM_xfree(vl_buf);

    if (chk) {
        if (chk->vl_heapset && H5HG__free_local_heapset(chk->vl_heapset) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free chunk-local VL heap set");

        if (chk->sel_space && H5S_close(chk->sel_space) < 0)
            HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to release decoded selection");

        chk->sel_buf  = H5MM_xfree(chk->sel_buf);
        chk->data_buf = H5MM_xfree(chk->data_buf);
        chk           = H5MM_xfree(chk);
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* H5D__struct_chunk_decode() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_decode_defined_values
 *
 * Purpose:     Decodes only the defined-value selection metadata from an
 *              encoded structured chunk into the structured-chunk intermediate
 *              memory representation.
 *
 *              The fixed-data section is not read or decoded. The resulting
 *              H5D_chunk_cache_mem_t therefore contains a decoded sel_space and
 *              selection-buffer state, while its data-buffer fields remain
 *              empty.
 *
 *              If filters apply to the selection section, they are reversed
 *              before checksum verification and dataspace decoding.
 *
 *              This callback is used when a caller needs the defined-value
 *              selection without materializing the complete structured chunk.
 *              It is optional at the SCC layout-callback level; if no
 *              defined-value metadata is stored for an allocated chunk, the
 *              caller may treat that chunk as fully defined without invoking
 *              this callback.
 *
 *              nbytes and alloc_size describe the caller's raw metadata
 *              buffer on entry. On successful return, they are updated to
 *              describe the decoded selection state held by the intermediate
 *              object.
 *
 *              partial_bound indicates whether the on-disk chunk used the
 *              partial-bound encoding path. When the layout reports that
 *              partial-bound chunks do not use different encoding, the value
 *              is not significant.
 *
 * Return:      Non-negative on success/Negative on failure
 *
 * NOTE: On entry, *chunk points to a caller-owned buffer containing only the
 *       encoded defined-value metadata read from disk.
 *
 *       On successful return, this callback consumes and frees that raw buffer
 *       and replaces *chunk with a metadata-only H5D_chunk_cache_mem_t object.
 *       The returned intermediate object is compatible with
 *       H5D__struct_chunk_evict(), which must be used to release it.
 *
 *       On failure, ownership is not transferred: *chunk continues to point
 *       to the original caller-owned raw buffer. The callback releases only
 *       allocations that it created internally.
 *
 *       This callback implements selection-only decoding for a
 *       caller that requests defined-value metadata without the
 *       fixed records or VL payloads. Its presence in the layout
 *       callback table does not establish that the current SCC
 *       read path invokes it; verify SCC wiring and test coverage
 *       before treating this as an exercised path.
 *
 * Updated:     Metadata-only decoding remains limited to the defined-value
 *              selection even when the encoded structured chunk also contains
 *              fixed descriptors and a chunk-local VL section. Filters for
 *              the fixed and VL sections are intentionally ignored because
 *              those value sections remain on disk until full decoding is
 *              requested.
 *
 *              The intermediate structure is zero initialized so DATA_BUF and
 *              VL_HEAPSET remain NULL and the result can safely be passed to
 *              the normal structured-chunk eviction callback. Failure cleanup
 *              also follows the complete chunk ownership rule and releases
 *              any internally created selection, buffers, or heap set.
 *
 *              Ownership of the caller's encoded buffer is transferred only
 *              after selection decoding succeeds. This preserves the original
 *              raw buffer on failure and prevents ambiguous ownership or
 *              double release by the SCC.
 *
 *                                              -- AZO   9/20/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_decode_defined_values(H5D_t *dset, size_t *nbytes /*in,out*/, size_t *alloc_size /*in,out*/,
                                        bool partial_bound, void **chunk /*in,out*/, void *_udata)
{
    H5D_chunk_ud_t        *udata = (H5D_chunk_ud_t *)_udata;
    H5D_chunk_cache_mem_t *chk   = NULL; /* Chunk's intermediate struct */
    H5O_stc_pline_t       *pline;        /* I/O pipeline info */
    hbool_t                filtered = false;
    uint32_t               stored_chksum;   /* Stored metadata checksum value */
    uint32_t               computed_chksum; /* Computed metadata checksum value */
    const unsigned char   *sel_p;
    void                  *raw_chunk = NULL;
    herr_t                 ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);
    assert(nbytes);
    assert(alloc_size);
    assert(chunk);
    assert(*chunk);
    assert(udata);

    raw_chunk = *chunk;

    pline = &(dset->shared->dcpl_cache.stc_pline);
    if (pline && pline->tot_filt_nsects)
        filtered = true;

    /*
     * Allocate a zero-initialized intermediate structure. The zero
     * initialization is important because metadata-only decoding does not
     * populate the data-value fields, but the object may later be released
     * through the generic structured-chunk evict callback.
     */
    if (NULL == (chk = H5MM_calloc(sizeof(H5D_chunk_cache_mem_t))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL,
                    "memory allocation failed for intermediate chunk struct");

    /* nbytes and alloc_size for encoded selection */
    chk->sel_nbytes = chk->sel_alloc_size = udata->offset[1];

    if (chk->sel_alloc_size == 0)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL,
                    "metadata-only decode called with zero-sized defined-value selection");

    /* Allocate a buffer for the encoded selection */
    if (NULL == (chk->sel_buf = H5MM_malloc(chk->sel_alloc_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "memory allocation failed for encoded selection buffer");

    /* Copy over the encoded selection */
    H5MM_memcpy(chk->sel_buf, raw_chunk, chk->sel_nbytes);

    /* Decompress the encoded selection */
    if (filtered && !partial_bound) {
        H5Z_EDC_t              err_detect; /* Error detection info */
        H5Z_cb_t               filter_cb;  /* I/O filter callback function */
        unsigned               i;
        H5O_stc_filter_sect_t *filt_sect;

        /* Retrieve filter settings from API context */
        if (H5CX_get_err_detect(&err_detect) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get error detection info");

        if (H5CX_get_filter_cb(&filter_cb) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get I/O filter callback function");

        for (i = 0, filt_sect = &pline->filt_sects[0]; i < pline->tot_filt_nsects; i++, filt_sect++) {

            if (filt_sect->nused) {
                switch (filt_sect->seq_sect) {
                    case H5_SECTION_SELECTION:
                        if (H5Z_apply_filters(filt_sect->nused, filt_sect->filter, H5Z_FLAG_REVERSE,
                                              &udata->filt_mask[0], err_detect, filter_cb, &chk->sel_nbytes,
                                              &chk->sel_alloc_size, &chk->sel_buf) < 0)
                            HGOTO_ERROR(H5E_DATASET, H5E_CANTFILTER, FAIL, "output pipeline failed");
                        break;

                    case H5_SECTION_FIXED:
                        break;

                    case H5_SECTION_VL:
                        /* metadata-only decode materializes only the selection section.
                         * Fixed and VL sections remain on disk and are intentionally ignored.
                         */
                        break;
                    case H5_SECTION_NUM:
                    default:
                        assert(0 && "Unknown action?!?");
                }
            } /* end if nused */

        } /* end for */
    }

    /*
     * The encoded selection must contain its metadata checksum before
     * H5F_get_checksums() examines the trailing checksum bytes.
     */
    if (chk->sel_nbytes < H5_SIZEOF_CHKSUM)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "encoded selection is smaller than checksum size");

    if (H5F_get_checksums(chk->sel_buf, chk->sel_nbytes, &stored_chksum, &computed_chksum) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get selection checksums");

    if (stored_chksum != computed_chksum)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "selection checksum verification failed");

    /*
     * Removing the checksum changes the bytes logically in use, but the
     * allocated buffer itself has not shrunk.
     */
    chk->sel_nbytes -= H5_SIZEOF_CHKSUM;

    sel_p = chk->sel_buf;

    /* Decode the defined-value selection */
    if (NULL == (chk->sel_space = H5S_decode(&sel_p)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTDECODE, FAIL, "unable to decode dataspace");

    /*
     * Commit ownership only after decoding has succeeded.
     *
     * Until this point, *chunk continues to reference the caller-owned raw
     * metadata buffer. Delaying the replacement makes ownership deterministic:
     * the caller retains the raw buffer on failure, while this callback consumes
     * it and returns the decoded object on success.
     */
    *nbytes     = chk->sel_nbytes;
    *alloc_size = chk->sel_alloc_size;

    raw_chunk = H5MM_xfree(raw_chunk);
    *chunk    = chk;
    chk       = NULL;

done:
    /*
     * On failure, *chunk still refers to the original caller-owned raw metadata
     * buffer. Release only the intermediate object and buffers allocated by this
     * callback.
     */
    if (chk) {
        if (chk->vl_heapset) {
            if (H5HG__free_local_heapset(chk->vl_heapset) < 0)
                HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free decoded chunk-local VL heap set");
            chk->vl_heapset = NULL;
        }
        if (chk->sel_space) {
            if (H5S_close(chk->sel_space) < 0)
                HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                            "unable to release decoded selection after decode failure");
            chk->sel_space = NULL;
        }

        chk->sel_buf  = H5MM_xfree(chk->sel_buf);
        chk->data_buf = H5MM_xfree(chk->data_buf);
        chk           = H5MM_xfree(chk);
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* H5D__struct_chunk_decode_defined_values() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_new_chunk
 *
 * Purpose:    Creates a new empty chunk.
 *             Does not insert into on disk chunk index.
 *
 *             If fill is true, writes the fill value to the chunk
 *             (unless this is a sparse chunk).
 *
 *             The number of bytes used is returned in *nbytes
 *             and the size of the chunk buffer is returned in *buf_size
 *
 * Return:    Non-negative on success/Negative on failure
 *
 * NOTE: On exit: [chunk] is the pointer to the chunk intermedidate struct
 *
 * Updated:     New structured chunks now initialize their chunk-local VL
 *              heap-set pointer to NULL. The heap set is created lazily when
 *              the first VL payload is stored, so an empty chunk owns no VL
 *              allocation.
 *
 *              Explicit initialization is required because the intermediate
 *              structure is allocated with H5MM_malloc() and is later handled
 *              by the common eviction and accounting paths, which use a NULL
 *              heap-set pointer to distinguish chunks without resident VL
 *              payload storage.
 *
 *                                              -- AZO   9/18/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_new_chunk(H5D_t *dset, bool fill, size_t *nbytes /*out*/, size_t *buf_size /*out*/,
                            void **chunk /*out*/, void **udata /*out*/)
{
    H5D_chunk_cache_mem_t *chk; /* Chunk's intermediate struct */
    H5D_chunk_ud_t        *uptr;

    herr_t ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);
    assert(dset->shared->layout.u.struct_chunk.stc_type == H5D_SPARSE_CHUNK);
    assert(!fill);

    /* Allocate the chunk's intermediate struct */
    if (NULL == (chk = H5MM_malloc(sizeof(H5D_chunk_cache_mem_t))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL,
                    "memory allocation failed for intermediate chunk struct");

    chk->sel_space  = NULL;
    chk->sel_buf    = NULL;
    chk->data_buf   = NULL;
    chk->vl_heapset = NULL;

    chk->sel_nbytes      = 0;
    chk->sel_alloc_size  = 0;
    chk->data_nbytes     = 0;
    chk->data_alloc_size = 0;

    *nbytes = *buf_size = 0;
    *chunk              = chk;

    /* Allocate udata */
    uptr = (H5D_chunk_ud_t *)H5MM_malloc(sizeof(H5D_chunk_ud_t));
    if (uptr == NULL)
        HGOTO_ERROR(H5E_ARGS, H5E_CANTALLOC, FAIL, "could not malloc space for udata");

    memset(uptr, 0, sizeof(H5D_chunk_ud_t));

    *udata = uptr;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_new_chunk() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_condense
 *
 * Purpose:    Reallocates buffers as necessary so the total allocated size of buffers
 *             for the chunk (alloc_size) is equal to the total number of bytes
 *             used (nbytes).
 *
 *             Optional, if not present the chunk cache will be more likely to
 *             evict chunks if there is wasted space in the buffers.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 * Updated:     Condensation now validates the used and allocated sizes of the
 *              selection and fixed-data buffers before modifying them. Each
 *              buffer is reduced to its logical used size, with zero-sized
 *              buffers released explicitly and successful reallocations
 *              published without losing the original pointer on failure.
 *
 *              The decoded chunk-local VL heap set is not condensed by this
 *              routine. Its allocations are independently owned and tracked
 *              by H5HG. Consequently, NBYTES continues to describe only the
 *              logical selection and fixed-data bytes, while the resident VL
 *              heap allocation remains part of the chunk's separately
 *              calculated SCC allocation size.
 *
 *                                              -- AZO   9/19/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_condense(H5D_t *dset, size_t *nbytes /*in, out*/, void **chunk /*in, out*/,
                           void H5_ATTR_UNUSED *udata)
{
    H5D_chunk_cache_mem_t *chk       = (H5D_chunk_cache_mem_t *)*chunk; /* Chunk's memory cache info */
    herr_t                 ret_value = SUCCEED;                         /* Return value */
    void                  *new_buf   = NULL;

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);
    assert(nbytes);
    assert(chunk);
    assert(*chunk);

    /*
     * The chunk-local H5HG heap set is deliberately not condensed here.
     * Only the selection and fixed-data buffers are managed by this callback.
     */
    if (chk->sel_nbytes > SIZE_MAX - chk->data_nbytes)
        HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "structured chunk used-size overflow");

    if ((chk->sel_alloc_size + chk->data_alloc_size) == (chk->sel_nbytes + chk->data_nbytes))
        /* Nothing to condense */
        HGOTO_DONE(SUCCEED);

    /*
     * Condense the resident selection buffer. A zero-byte buffer is released
     * explicitly because realloc(ptr, 0) is allowed to return NULL without
     * representing an allocation failure.
     */
    if (chk->sel_alloc_size != chk->sel_nbytes) {
        if (0 == chk->sel_nbytes) {
            chk->sel_buf = H5MM_xfree(chk->sel_buf);
        }
        else {
            if (NULL == (new_buf = H5MM_realloc(chk->sel_buf, chk->sel_nbytes)))
                HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                            "unable to condense structured chunk selection buffer");

            chk->sel_buf = new_buf;
            new_buf      = NULL;
        }

        chk->sel_alloc_size = chk->sel_nbytes;
    }

    /*
     * Condense the fixed-data/descriptor buffer. Publish the realloc result
     * only after realloc succeeds so the old pointer is not lost on failure.
     */
    if (chk->data_alloc_size != chk->data_nbytes) {
        if (0 == chk->data_nbytes) {
            chk->data_buf = H5MM_xfree(chk->data_buf);
        }
        else {
            if (NULL == (new_buf = H5MM_realloc(chk->data_buf, chk->data_nbytes)))
                HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                            "unable to condense structured chunk data buffer");

            chk->data_buf = new_buf;
            new_buf       = NULL;
        }

        chk->data_alloc_size = chk->data_nbytes;
    }

    /*
     * NBYTES describes logical bytes used by the selection and fixed-data
     * buffers. Decoded VL heap memory remains separate resident allocation
     * and is deliberately not added here.
     */
    if (chk->sel_nbytes > SIZE_MAX - chk->data_nbytes)
        HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "structured chunk used-size overflow");

    *nbytes = chk->sel_nbytes + chk->data_nbytes;
    *chunk  = chk;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_condense() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_encode
 *
 * Purpose:     Compresses/encodes the chunk as necessary.
 *              If chunk is the same as cache_buf, leaves *write_buf as NULL.
 *
 *              This function leaves chunk alone and allocates write_buf if necessary
 *              to hold compressed data, sets *write_size to the size of the data
 *              in write_buf, and sets *write_size_alloc to the size of write_buf,
 *              if it was allocated.
 *
 *              partial_bound is true if the chunk is partially outside the bounds
 *              of the dataset. If the dataset reported partial_bound_chunks_different_encoding
 *              as false, the setting of partial_bound is undefined.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 * NOTE: On entry: [chunk] points to the chunk intermediate struct
 * NOTE: On exit: [write_buf] points to the on disk file format chunk buffer
 *
 * NOTE: --chunk_encode callback: fill in udata: offset, unfilt_size, filt_mask,
 * NOTE: --chunk_insert callback: fill in udata: addr, nbytes, chunk_idx
 *
 *
 * Updated:     Added support for three-section VL structured chunks.
 *              For VL datatypes, the fixed section contains chunk-local
 *              descriptors and the VL section contains the serialized local
 *              heap set. Metadata checksums and configured filters are applied
 *              independently to the fixed and VL sections. The VL boundary is
 *              recorded in udata->offset[H5_SECTION_VL].
 *
 *              This function retains its non-destructive behavior by encoding
 *              the selection, fixed descriptors, and local heap set into
 *              temporary buffers before assembling the final chunk image. The
 *              decoded chunk and its local heap set remain owned by the caller.
 *
 *                                              -- AZO   9/16/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_encode(H5D_t *dset, hsize_t *write_size /*out*/, hsize_t *write_buf_alloc /*out*/,
                         bool partial_bound, const void *chunk, void *_udata, void **write_buf /*out*/)
{
    const H5D_chunk_cache_mem_t *chk   = (const H5D_chunk_cache_mem_t *)chunk; /* Chunk memory cache info */
    H5D_chunk_ud_t              *udata = (H5D_chunk_ud_t *)_udata;
    void                        *data_buf = NULL;
    uint8_t                     *p        = NULL;
    unsigned char               *sel_p    = NULL;
    size_t                       sel_nbytes, sel_alloc_size;
    size_t                       data_nbytes, data_alloc_size;
    H5O_stc_pline_t             *pline    = NULL; /* I/O pipeline info */
    hbool_t                      filtered = false;
    void                        *tot_buf  = NULL;
    hsize_t                      nelmts;
    size_t                       type_size;
    uint32_t                     metadata_chksum;
    void                        *vl_buf        = NULL; /* Temporary encoded VL section */
    uint8_t                     *vl_image      = NULL; /* Image returned by H5HG encoder */
    size_t                       vl_nbytes     = 0;
    size_t                       vl_alloc_size = 0;
    size_t                       write_nbytes  = 0;
    hbool_t                      has_vlen      = false;
    herr_t                       ret_value     = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);

    /*
     * STRUCT_CHUNK_SECTION_COUNT_ASSUMPTION: The current format uses two
     * sections for fixed-size data and three for data containing VL values.
     * Revisit this inference if new section kinds, multiple VL sections,
     * or a dense VL layout are introduced.
     */
    if (dset->shared->layout.storage.u.struct_chunk.nsects == 2) {
        has_vlen = false;
    }
    else if (dset->shared->layout.storage.u.struct_chunk.nsects == H5_SECTION_NUM) {
        has_vlen = true;
    }
    else
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "invalid structured chunk section count");

    pline = &(dset->shared->dcpl_cache.stc_pline);
    if (pline && pline->tot_filt_nsects)
        filtered = true;

    /* Determine size of selection dataspace */
    if (H5S_encode(chk->sel_space, &sel_p, &sel_nbytes) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to get encoded dataspace size");

    /* Allocate buffer for selection */
    sel_alloc_size = sel_nbytes + H5_SIZEOF_CHKSUM;
    if (NULL == (tot_buf = H5MM_malloc(sel_alloc_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "memory allocation failed for the chunk");

    sel_p = tot_buf;

    /* Encode the selection */
    if (H5S_encode(chk->sel_space, &sel_p, &sel_nbytes) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to encode dataspace");

    /* Compute metadata checksum for sel_space */
    metadata_chksum = H5_checksum_metadata(tot_buf, (size_t)sel_nbytes, 0);

    /* Encode metadata checksum for the selection */
    p = (uint8_t *)tot_buf + sel_nbytes;
    UINT32ENCODE(p, metadata_chksum);

    sel_nbytes += H5_SIZEOF_CHKSUM;

    /* Get the number of elements in the selection */
    nelmts    = H5S_GET_SELECT_NPOINTS(chk->sel_space);
    type_size = H5T_GET_SIZE(dset->shared->type);

    assert(nelmts * type_size == chk->data_nbytes);
    assert(chk->data_alloc_size >= chk->data_nbytes);

    data_nbytes = chk->data_nbytes;

    /*
     * For a VL chunk, the fixed section contains descriptors and therefore
     * is metadata. Add a checksum to those descriptor bytes.
     *
     * Fixed-size chunks retain the existing behavior and do not add a
     * checksum to H5_SECTION_FIXED.
     */
    if (has_vlen) {
        if (data_nbytes > SIZE_MAX - H5_SIZEOF_CHKSUM)
            HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "encoded fixed section size overflow");

        data_alloc_size = data_nbytes + H5_SIZEOF_CHKSUM;
    }
    else
        data_alloc_size = chk->data_alloc_size;

    if (data_alloc_size > 0) {
        if (NULL == (data_buf = H5MM_malloc(data_alloc_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "memory allocation failed for the chunk");

        if (data_nbytes > 0)
            H5MM_memcpy(data_buf, chk->data_buf, data_nbytes);
    }

    if (has_vlen) {
        metadata_chksum = H5_checksum_metadata(data_buf, data_nbytes, 0);

        p = (uint8_t *)data_buf + data_nbytes;
        UINT32ENCODE(p, metadata_chksum);

        data_nbytes += H5_SIZEOF_CHKSUM;
    }

    /*
     * Encode the chunk-local heap set as the third section. A NULL or
     * logically empty heap set produces a zero-length VL section.
     */
    if (has_vlen) {
        if (H5HG__encode_local_heapset(dset->oloc.file, chk->vl_heapset, &vl_image, &vl_nbytes) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTENCODE, FAIL, "unable to encode chunk-local VL heap set");

        vl_buf   = vl_image;
        vl_image = NULL;

        /*
         * A nonempty VL section is metadata and therefore receives its own
         * checksum. A zero-length section remains exactly zero bytes.
         */
        if (vl_nbytes > 0) {
            void  *new_vl_buf;
            size_t new_vl_size;

            if (vl_nbytes > SIZE_MAX - H5_SIZEOF_CHKSUM)
                HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "encoded VL section size overflow");

            new_vl_size = vl_nbytes + H5_SIZEOF_CHKSUM;

            if (NULL == (new_vl_buf = H5MM_realloc(vl_buf, new_vl_size)))
                HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to extend encoded VL section");

            vl_buf        = new_vl_buf;
            vl_alloc_size = new_vl_size;

            metadata_chksum = H5_checksum_metadata(vl_buf, vl_nbytes, 0);

            p = (uint8_t *)vl_buf + vl_nbytes;
            UINT32ENCODE(p, metadata_chksum);

            vl_nbytes += H5_SIZEOF_CHKSUM;
        }
    }

    /* Compression */
    if (filtered) {
        H5Z_EDC_t              err_detect; /* Error detection info */
        H5Z_cb_t               filter_cb;  /* I/O filter callback function */
        unsigned               i;
        H5O_stc_filter_sect_t *filt_sect;

        udata->unfilt_size[0] = sel_nbytes;
        udata->unfilt_size[1] = data_nbytes;

        if (has_vlen)
            udata->unfilt_size[2] = vl_nbytes;

        if (!partial_bound) {

            /* Retrieve filter settings from API context */
            if (H5CX_get_err_detect(&err_detect) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get error detection info");
            if (H5CX_get_filter_cb(&filter_cb) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get I/O filter callback function");

            for (i = 0, filt_sect = &pline->filt_sects[0]; i < pline->tot_filt_nsects; i++, filt_sect++) {

                if (filt_sect->nused) {
                    switch (filt_sect->seq_sect) {
                        case H5_SECTION_SELECTION:
                            if (H5Z_apply_filters(filt_sect->nused, filt_sect->filter, 0,
                                                  &udata->filt_mask[0], err_detect, filter_cb, &sel_nbytes,
                                                  &sel_alloc_size, &tot_buf) < 0)
                                HGOTO_ERROR(H5E_DATASET, H5E_CANTFILTER, FAIL, "output pipeline failed");
                            break;

                        case H5_SECTION_FIXED:
                            if (H5Z_apply_filters(filt_sect->nused, filt_sect->filter, 0,
                                                  &udata->filt_mask[1], err_detect, filter_cb, &data_nbytes,
                                                  &data_alloc_size, &data_buf) < 0)
                                HGOTO_ERROR(H5E_DATASET, H5E_CANTFILTER, FAIL, "output pipeline failed");
                            break;

                        case H5_SECTION_VL:
                            if (!has_vlen)
                                HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL,
                                            "VL filter configured for non-VL structured chunk");

                            /*
                             * A zero-length VL section has no encoded bytes and
                             * therefore has nothing to filter.
                             */
                            if (vl_nbytes > 0) {
                                if (H5Z_apply_filters(filt_sect->nused, filt_sect->filter, 0,
                                                      &udata->filt_mask[H5_SECTION_VL], err_detect, filter_cb,
                                                      &vl_nbytes, &vl_alloc_size, &vl_buf) < 0)
                                    HGOTO_ERROR(H5E_DATASET, H5E_CANTFILTER, FAIL,
                                                "VL-section filter pipeline failed");
                            }
                            break;

                        case H5_SECTION_NUM:
                        default:
                            assert(0 && "Unknown action?!?");
                    }
                } /* end if nused */

            } /* end for */
        }
    }

    if (sel_nbytes > SIZE_MAX - data_nbytes)
        HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "structured chunk encoded size overflow");

    write_nbytes = sel_nbytes + data_nbytes;

    if (has_vlen) {
        if (vl_nbytes > SIZE_MAX - write_nbytes)
            HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "structured chunk encoded size overflow");

        write_nbytes += vl_nbytes;
    }

    /*
     * Performance note: the encoded sections are assembled into one buffer
     * here, which copies the selection, fixed records, and VL section.
     * A future section-level vector write could submit their buffers and
     * offsets separately, if the SCC write interface supports that form.
     */

    /*
     * Build the final structured chunk image in a freshly allocated buffer.
     *
     * Do not grow tot_buf in place after filter processing. Filters may alter
     * the selection buffer allocation and size, and the fixed-data section may
     * be larger than the original selection-only allocation. Using a fresh
     * combined buffer avoids relying on realloc() after intermediate section
     * processing and prevents accidental overwrite of the allocation metadata.
     */
    {
        void *new_tot_buf = NULL;

        if (NULL == (new_tot_buf = H5MM_malloc(write_nbytes)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "memory allocation failed for the chunk");

        H5MM_memcpy(new_tot_buf, tot_buf, sel_nbytes);

        if (data_nbytes > 0)
            H5MM_memcpy((uint8_t *)new_tot_buf + sel_nbytes, data_buf, data_nbytes);

        if (has_vlen && vl_nbytes > 0)
            H5MM_memcpy((uint8_t *)new_tot_buf + sel_nbytes + data_nbytes, vl_buf, vl_nbytes);

        tot_buf     = H5MM_xfree(tot_buf);
        tot_buf     = new_tot_buf;
        new_tot_buf = NULL;
    }

    udata->offset[0] = 0; /* Filler */
    udata->offset[1] = sel_nbytes;

    if (has_vlen) {
        udata->offset[2] = sel_nbytes + data_nbytes;
    }

    *write_size      = (hsize_t)write_nbytes;
    *write_buf_alloc = (hsize_t)write_nbytes;
    *write_buf       = tot_buf;

done:
    if (data_buf)
        data_buf = H5MM_xfree(data_buf);

    if (vl_buf)
        vl_buf = H5MM_xfree(vl_buf);

    if (vl_image)
        vl_image = H5MM_xfree(vl_image);

    if (ret_value < 0 && tot_buf)
        tot_buf = H5MM_xfree(tot_buf);

    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_encode() */

/*------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_encode_in_place
 *
 * Purpose:     The same as H5D_struct_chunk_encode()  but does not preserve
 *              chunk buffer, encoding is performed in-place.
 *              Must free all other data used.
 *
 * Return:      Non-negative on success/Negative on failure
 *
 * NOTE: On entry: [chunk] points to the chunk intermediate struct
 * NOTE: On exit: [chunk] points to the on disk file format chunk buffer
 *
 * NOTE:  --chunk_encode: fill in udata: offset, unfilt_size, filt_mask,
 * NOTE:  --chunk_insert: fill in udata: addr, nbytes, chunk_idx
 *
 *
 *
 * Updated:     Added support for three-section VL structured chunks.
 *              For VL datatypes, the fixed section contains chunk-local
 *              descriptors and the VL section contains the serialized local
 *              heap set. Metadata checksums and configured filters are applied
 *              independently to the fixed and VL sections. The VL boundary is
 *              recorded in udata->offset[H5_SECTION_VL].
 *
 *              This function retains its destructive in-place behavior. After
 *              constructing the encoded selection/fixed/VL image, it releases
 *              the decoded selection, dataspace, local heap set, and chunk
 *              wrapper.
 *
 *                                              -- AZO   9/16/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_encode_in_place(H5D_t *dset, size_t *write_size /*out*/, bool partial_bound,
                                  void **chunk /*in,out*/, void *_udata)
{
    H5D_chunk_cache_mem_t *chk   = (H5D_chunk_cache_mem_t *)*chunk; /* Chunk memory cache info */
    H5D_chunk_ud_t        *udata = (H5D_chunk_ud_t *)_udata;
    H5O_stc_pline_t       *pline; /* I/O pipeline info */
    hbool_t                filtered = false;
    uint32_t               metadata_chksum;
    uint8_t               *p;
    unsigned char         *sel_p = NULL;
    H5D_chunk_cache_mem_t *tmp;
    hsize_t                nelmts;
    size_t                 type_size;
    void                  *vl_buf        = NULL; /* Temporary encoded VL section */
    uint8_t               *vl_image      = NULL; /* Image returned by H5HG encoder */
    size_t                 vl_nbytes     = 0;
    size_t                 vl_alloc_size = 0;
    size_t                 write_nbytes  = 0;
    hbool_t                has_vlen      = false;
    herr_t                 ret_value     = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);

    /*
     * STRUCT_CHUNK_SECTION_COUNT_ASSUMPTION: The current format uses two
     * sections for fixed-size data and three for data containing VL values.
     * Revisit this inference if new section kinds, multiple VL sections,
     * or a dense VL layout are introduced.
     */
    if (dset->shared->layout.storage.u.struct_chunk.nsects == 2) {
        has_vlen = false;
    }
    else if (dset->shared->layout.storage.u.struct_chunk.nsects == H5_SECTION_NUM) {
        has_vlen = true;
    }
    else
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "invalid structured chunk section count");

    pline = &(dset->shared->dcpl_cache.stc_pline);
    if (pline && pline->tot_filt_nsects)
        filtered = true;

    /* Determine size of selection dataspace */
    if (H5S_encode(chk->sel_space, &sel_p, &chk->sel_nbytes) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to get encoded dataspace size");

    chk->sel_alloc_size = chk->sel_nbytes;

    if (NULL == (chk->sel_buf = H5MM_realloc(chk->sel_buf, chk->sel_alloc_size + H5_SIZEOF_CHKSUM)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL,
                    "memory allocation failed for intermediate chunk struct");

    sel_p = chk->sel_buf;

    /* Encode the selection */
    if (H5S_encode(chk->sel_space, &sel_p, &chk->sel_nbytes) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to encode dataspace");

    /* Compute metadata checksum for chk->sel_space into chk->data_buf */
    metadata_chksum = H5_checksum_metadata(chk->sel_buf, (size_t)chk->sel_nbytes, 0);

    /* Encode metadata checksum for the selection */
    p = (uint8_t *)chk->sel_buf + chk->sel_nbytes;
    UINT32ENCODE(p, metadata_chksum);

    chk->sel_nbytes += H5_SIZEOF_CHKSUM;
    chk->sel_alloc_size += H5_SIZEOF_CHKSUM;

    /* Get the number of elements in the selection */
    nelmts    = H5S_GET_SELECT_NPOINTS(chk->sel_space);
    type_size = H5T_GET_SIZE(dset->shared->type);

    assert(nelmts * type_size == chk->data_nbytes);
    assert(chk->data_alloc_size >= chk->data_nbytes);

    /*
     * For a VL chunk, the fixed section contains descriptors and therefore
     * receives its own metadata checksum.
     */
    if (has_vlen) {
        if (chk->data_nbytes > SIZE_MAX - H5_SIZEOF_CHKSUM)
            HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "encoded fixed section size overflow");

        if (NULL == (chk->data_buf = H5MM_realloc(chk->data_buf, chk->data_nbytes + H5_SIZEOF_CHKSUM)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "unable to extend encoded fixed section");

        chk->data_alloc_size = chk->data_nbytes + H5_SIZEOF_CHKSUM;

        metadata_chksum = H5_checksum_metadata(chk->data_buf, chk->data_nbytes, 0);

        p = (uint8_t *)chk->data_buf + chk->data_nbytes;
        UINT32ENCODE(p, metadata_chksum);

        chk->data_nbytes += H5_SIZEOF_CHKSUM;
    }

    /*
     * Encode the chunk-local heap set as the third section. A NULL or
     * logically empty heap set produces a zero-length VL section.
     */
    if (has_vlen) {
        if (H5HG__encode_local_heapset(dset->oloc.file, chk->vl_heapset, &vl_image, &vl_nbytes) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTENCODE, FAIL, "unable to encode chunk-local VL heap set");

        vl_buf   = vl_image;
        vl_image = NULL;

        /*
         * A nonempty VL section is metadata and therefore receives its own
         * checksum. A zero-length section remains exactly zero bytes.
         */
        if (vl_nbytes > 0) {
            void  *new_vl_buf;
            size_t new_vl_size;

            if (vl_nbytes > SIZE_MAX - H5_SIZEOF_CHKSUM)
                HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "encoded VL section size overflow");

            new_vl_size = vl_nbytes + H5_SIZEOF_CHKSUM;

            if (NULL == (new_vl_buf = H5MM_realloc(vl_buf, new_vl_size)))
                HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to extend encoded VL section");

            vl_buf        = new_vl_buf;
            vl_alloc_size = new_vl_size;

            metadata_chksum = H5_checksum_metadata(vl_buf, vl_nbytes, 0);

            p = (uint8_t *)vl_buf + vl_nbytes;
            UINT32ENCODE(p, metadata_chksum);

            vl_nbytes += H5_SIZEOF_CHKSUM;
        }
    }

    /* Compression */
    if (filtered) {
        H5Z_EDC_t              err_detect; /* Error detection info */
        H5Z_cb_t               filter_cb;  /* I/O filter callback function */
        unsigned               i;
        H5O_stc_filter_sect_t *filt_sect;

        udata->unfilt_size[0] = chk->sel_nbytes;
        udata->unfilt_size[1] = chk->data_nbytes;

        if (has_vlen) {
            udata->unfilt_size[2] = vl_nbytes;
        }

        if (!partial_bound) {

            /* Retrieve filter settings from API context */
            if (H5CX_get_err_detect(&err_detect) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get error detection info");
            if (H5CX_get_filter_cb(&filter_cb) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get I/O filter callback function");

            for (i = 0, filt_sect = &pline->filt_sects[0]; i < pline->tot_filt_nsects; i++, filt_sect++) {

                if (filt_sect->nused) {
                    switch (filt_sect->seq_sect) {
                        case H5_SECTION_SELECTION:
                            if (H5Z_apply_filters(filt_sect->nused, filt_sect->filter, 0,
                                                  &udata->filt_mask[0], err_detect, filter_cb,
                                                  &chk->sel_nbytes, &chk->sel_alloc_size, &chk->sel_buf) < 0)
                                HGOTO_ERROR(H5E_DATASET, H5E_CANTFILTER, FAIL, "output pipeline failed");
                            break;

                        case H5_SECTION_FIXED:
                            if (H5Z_apply_filters(
                                    filt_sect->nused, filt_sect->filter, 0, &udata->filt_mask[1], err_detect,
                                    filter_cb, &chk->data_nbytes, &chk->data_alloc_size, &chk->data_buf) < 0)
                                HGOTO_ERROR(H5E_DATASET, H5E_CANTFILTER, FAIL, "output pipeline failed");
                            break;

                        case H5_SECTION_VL:
                            if (!has_vlen)
                                HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL,
                                            "VL filter configured for non-VL structured chunk");

                            /*
                             * A zero-length VL section has no encoded bytes and
                             * therefore has nothing to filter.
                             */
                            if (vl_nbytes > 0) {
                                if (H5Z_apply_filters(filt_sect->nused, filt_sect->filter, 0,
                                                      &udata->filt_mask[2], err_detect, filter_cb, &vl_nbytes,
                                                      &vl_alloc_size, &vl_buf) < 0)
                                    HGOTO_ERROR(H5E_DATASET, H5E_CANTFILTER, FAIL,
                                                "VL-section filter pipeline failed");
                            }
                            break;

                        case H5_SECTION_NUM:
                        default:
                            assert(0 && "Unknown action?!?");
                    }
                } /* end if nused */

            } /* end for */
        }
    }

    if (chk->sel_nbytes > SIZE_MAX - chk->data_nbytes)
        HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "structured chunk encoded size overflow");

    write_nbytes = chk->sel_nbytes + chk->data_nbytes;

    if (has_vlen) {
        if (vl_nbytes > SIZE_MAX - write_nbytes)
            HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "structured chunk encoded size overflow");

        write_nbytes += vl_nbytes;
    }

    /*
     * Assemble a contiguous encoded image by growing the fixed-data buffer,
     * shifting its contents, and copying the selection and VL sections.
     * These moves and copies remain necessary for the current contiguous
     * output interface. A future section-level vector-write interface could
     * submit the separate buffers without assembling this combined image.
     */
    if (NULL == (chk->data_buf = H5MM_realloc(chk->data_buf, write_nbytes)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "memory reallocation failed for data chunk");

    /* Shift data values to the right to provide space for encoded selection */
    memmove((uint8_t *)(chk->data_buf) + chk->sel_nbytes, chk->data_buf, chk->data_nbytes);

    H5MM_memcpy(chk->data_buf, chk->sel_buf, chk->sel_nbytes);

    if (has_vlen && vl_nbytes > 0) {
        H5MM_memcpy((uint8_t *)chk->data_buf + chk->sel_nbytes + chk->data_nbytes, vl_buf, vl_nbytes);
    }

    tmp = chk;

    *chunk           = chk->data_buf;
    *write_size      = write_nbytes;
    udata->offset[1] = chk->sel_nbytes;

    if (has_vlen) {
        udata->offset[2] = chk->sel_nbytes + chk->data_nbytes;
    }

    /* Free chk->sel_buf */
    chk->sel_buf    = H5MM_xfree(chk->sel_buf);
    chk->sel_nbytes = chk->sel_alloc_size = 0;

    /* Close chk->sel_space */
    if (chk->sel_space && H5S_close(chk->sel_space) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release dataspace for encoded selection");

    chk->sel_space = NULL;

    if (chk->vl_heapset) {
        if (H5HG__free_local_heapset(chk->vl_heapset) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free encoded chunk-local VL heap set");

        chk->vl_heapset = NULL;
    }

    tmp = H5MM_xfree(tmp);

done:
    if (vl_buf)
        vl_buf = H5MM_xfree(vl_buf);

    if (vl_image)
        vl_image = H5MM_xfree(vl_image);

    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_encode_in_place() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_evict
 *
 * Purpose:     Frees chunk and all memory referenced by it.
 *              Optional, if not present free() is simply used.
 *
 *              This callback releases both fully decoded structured chunks
 *              and metadata-only intermediate objects created by
 *              H5D__struct_chunk_decode_defined_values(). Fields that were
 *              not populated by metadata-only decoding are expected to be
 *              NULL/zero initialized.
 *
 *              The associated layout-specific udata object is also released.
 *
 * Return:      Non-negative on success/Negative on failure
 *
 * Updated:     Fully decoded structured chunks containing variable-length
 *              data own a chunk-local H5HG heap set in addition to their
 *              selection and fixed-data buffers. Full eviction now explicitly
 *              frees that heap set so all decoded VL payloads and associated
 *              heap allocations are released with the cached chunk.
 *
 *              The heap-set pointer may be NULL for non-VL chunks and for
 *              metadata-only intermediate objects, so cleanup remains valid
 *              for every structured-chunk representation accepted by this
 *              callback. The pointer is cleared after a successful release to
 *              make the ownership transition explicit.
 *
 *                                              -- AZO   9/17/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_evict(H5D_t *dset, void *chunk, void *udata)
{
    H5D_chunk_cache_mem_t *chk       = (H5D_chunk_cache_mem_t *)chunk; /* Chunk memory cache info */
    herr_t                 ret_value = SUCCEED;                        /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);
    assert(chunk);

    /* Free the sel_buf + data_buffer */
    chk->sel_buf  = H5MM_xfree(chk->sel_buf);
    chk->data_buf = H5MM_xfree(chk->data_buf);

    /* Close the encoded dataspace */
    if (chk->sel_space && H5S_close(chk->sel_space) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release dataspace for encoded selection");

    /*
     * Free the chunk-local VL heap set. Metadata-only decoded chunks and
     * structured chunks without VL data leave this field NULL.
     */
    if (chk->vl_heapset) {

        if (H5HG__free_local_heapset(chk->vl_heapset) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free chunk-local VL heap set");

        chk->vl_heapset = NULL;
    }

    /* Free the chunk memory cache info structure */
    chk = H5MM_xfree(chk);

    udata = H5MM_xfree(udata);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_evict() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_insert
 *
 * Purpose:     Inserts (or reinserts) count chunks into the chunk index if necessary.
 *              Old address and size (if any) of the chunks on disk are passed
 *              as addr and old_disk_size, the new size is passed in as new_disk_size.
 *
 *              This function resizes and reallocates on disk if necessary,
 *              returning the address of the chunks on disk in *addr.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 * NOTE: --chunk_encode callback: fill in udata: offset, unfilt_size, filt_mask,
 * NOTE: --chunk_insert callback: fill in udata: addr, nbytes, chunk_idx
 *
 * NOTE: [chunk] not used??
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_insert(H5D_t *dset, size_t count, const hsize_t *scaled[] /*in*/,
                         haddr_t *addr[] /*in,out*/, hsize_t old_disk_size[], hsize_t new_disk_size[],
                         void H5_ATTR_UNUSED *chunk[] /*in*/, void *_udata[])
{
    H5D_chunk_ud_t             *udata;
    H5D_chk_idx_info_t          idx_info; /* Chunked index info */
    H5O_storage_struct_chunk_t *storage = &(dset->shared->layout.storage.u.struct_chunk);
    H5O_layout_struct_chunk_t  *layout  = &(dset->shared->layout.u.struct_chunk);
    size_t                      i;
    H5D_chunk_ud_t             *my_udata;
    herr_t                      ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);
    assert(storage->idx_type != H5D_CHUNK_IDX_NONE);
    assert(storage->idx_type != H5D_CHUNK_IDX_BTREE);

    /* Compose chunked index info struct */
    idx_info.f           = dset->oloc.file;
    idx_info.stc_pline   = &dset->shared->dcpl_cache.stc_pline;
    idx_info.stc_layout  = layout;
    idx_info.stc_storage = storage;

    /* Allocage  my_udata */
    my_udata = (H5D_chunk_ud_t *)H5MM_malloc(sizeof(H5D_chunk_ud_t));
    if (my_udata == NULL)
        HGOTO_ERROR(H5E_ARGS, H5E_CANTALLOC, FAIL, "could not malloc space for udata");

    for (i = 0; i < count; i++) {
        bool need_alloc = true;

        memset(my_udata, 0, sizeof(H5D_chunk_ud_t));

        my_udata->common.stc_layout  = layout;
        my_udata->common.stc_storage = storage;
        my_udata->common.scaled      = scaled[i];

        my_udata->chunk_block.offset = HADDR_UNDEF;
        my_udata->chunk_block.length = 0;

        /* chunk_idx is calculated in get_addr callback */
        if ((storage->ops->get_addr)(&idx_info, my_udata) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't query chunk address");

        if (H5_addr_defined(*addr[i])) {
            assert(*addr[i] == my_udata->chunk_block.offset);
            assert(old_disk_size[i] == my_udata->chunk_block.length);

            if (old_disk_size[i] == new_disk_size[i])
                need_alloc = false;
            else {
                if (H5MF_xfree(dset->oloc.file, H5FD_MEM_DRAW, *addr[i], old_disk_size[i]) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "unable to free chunk");
            }
        }
        else
            assert(!H5_addr_defined(my_udata->chunk_block.offset));

        if (need_alloc) {
            *addr[i] = H5MF_alloc(dset->oloc.file, H5FD_MEM_DRAW, new_disk_size[i]);
            if (!H5_addr_defined(*addr[i]))
                HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "file allocation failed");
        }

        /* For dense chunk, no need to insert for non-filtered chunk with the same old/new sizes */
        /* For structured chunk, there is metadata for filtered and non-filtered chunk, so insert anyway */

        udata = (H5D_chunk_ud_t *)_udata[i];

        udata->chunk_block.offset = *(addr[i]);
        udata->chunk_block.length = new_disk_size[i];
        udata->chunk_idx          = my_udata->chunk_idx;
        udata->common.scaled      = scaled[i];

        if (storage->ops->insert) {
            if ((storage->ops->insert)(&idx_info, udata, dset) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTINSERT, FAIL, "unable to insert chunk addr into index");
        }

    } /* end for */

done:
    if (my_udata)
        H5MM_xfree(my_udata);

    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_insert() */

/*-------------------------------------------------------------------------
 * Function: H5D__struct_chunk_vector_read
 *
 * Purpose:
 *   Translate FILE_SPACE_IN for one structured chunk into physical file
 *   byte ranges that may be used by the SCC to construct a direct vector
 *   read operation.
 *
 *   This callback does not perform I/O and does not construct the
 *   corresponding memory-buffer vector. On success, OFFSETS and SIZES
 *   describe portions of the chunk's serialized on-disk value region.
 *   Each returned offset is an absolute file address derived from ADDR.
 *
 *   VECTOR_POSSIBLE is set to false when the on-disk representation cannot
 *   be accessed directly, such as when an applicable filter requires the
 *   complete encoded chunk to be read and decoded.
 *
 *   If the translation requires the chunk's defined-values selection and
 *   CHUNK does not provide it, REQUIRE_VALUES is set to true and no vector
 *   is returned. The caller may obtain or decode that metadata and retry.
 *
 *   A successful translation does not imply that an I/O operation has been
 *   submitted or completed. The caller owns the returned OFFSETS and SIZES
 *   arrays and is responsible for releasing them.
 *
 *   The vector-read and vector-write callbacks currently perform the same
 *   selection-to-file-range translation and are intentionally retained as
 *   separate callbacks while their eventual SCC read/write uses remain under
 *   development. Once those use cases and their eligibility requirements are
 *   established, their common implementation should be moved into one shared
 *   translation helper rather than maintained independently.
 *
 * Return:
 *   SUCCEED when eligibility is determined and, when possible, the vector
 *   is produced; FAIL on an internal translation or allocation error.
 *
 * Updated:     Structured chunks containing variable-length data cannot use
 *              direct vector I/O. Their fixed section contains chunk-local
 *              descriptors whose payloads are owned by the decoded chunk's
 *              H5HG heap set. Such reads must therefore pass through the
 *              resident structured-chunk representation and its chunk-local
 *              VL datatype-conversion callbacks.
 *
 *              The datatype is checked before constructing an I/O vector. If
 *              it contains VL data, VECTOR_POSSIBLE remains false so the SCC
 *              uses the decoded-value read path instead.
 *
 *                                              -- AZO   9/19/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_vector_read(H5D_t *dset, haddr_t addr, const H5S_t *file_space_in,
                              bool H5_ATTR_UNUSED partial_bound, void *chunk /*in*/,
                              size_t *vec_count /*out*/, haddr_t **offsets /*out*/, size_t **sizes /*out*/,
                              bool *vector_possible /*out*/, bool *require_values /*out*/,
                              void H5_ATTR_UNUSED *udata)
{
    H5D_chunk_cache_mem_t  *chk = (H5D_chunk_cache_mem_t *)chunk; /* Chunk memory cache info */
    H5O_stc_pline_t        *pline;                                /* I/O pipeline info */
    size_t                  elmt_size = 0;
    haddr_t                *vec_addrs = NULL;
    size_t                 *vec_sizes = NULL;
    hsize_t                 file_off[SEQ_LIST_LEN];
    size_t                  file_len[SEQ_LIST_LEN];
    size_t                  file_seq_i;
    size_t                  file_nseq;
    size_t                  io_len;
    size_t                  file_nelmts;
    hsize_t                 chk_nelmts;
    hssize_t                hss_nelmts;
    hsize_t                 projected_nelmts;
    size_t                  seq_nelem;
    H5S_sel_iter_t         *file_iter           = NULL;
    bool                    file_iter_init      = false;
    size_t                  vec_arr_nused       = 0;
    size_t                  vec_arr_nalloc      = VECTOR_LEN;
    H5S_t                  *serial_values_space = NULL;
    H5S_t                  *serial_file_space   = NULL;
    H5_flexible_const_ptr_t flex_selection;
    htri_t                  has_vlen_type = false;
    herr_t                  ret_value     = SUCCEED;

    FUNC_ENTER_PACKAGE

    *vec_count       = 0;
    *offsets         = NULL;
    *sizes           = NULL;
    *vector_possible = false;
    *require_values  = false;

    /* Sanity checks */
    assert(dset);

    /*
     * Chunk-local VL values cannot use the direct vector-I/O path.
     *
     * The fixed section contains descriptors whose payloads live in the
     * chunk-local H5HG heap set. Reads and writes must therefore pass through
     * the decoded structured-chunk representation and the H5T chunk-local VL
     * conversion callbacks.
     */
    if ((has_vlen_type = H5T_detect_class(dset->shared->type, H5T_VLEN, false)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "unable to detect VL datatype");

    if (has_vlen_type) {
        HGOTO_DONE(SUCCEED);
    }

    if (chk == NULL) {
        *require_values = true;
        HGOTO_DONE(SUCCEED);
    }

    pline = &(dset->shared->dcpl_cache.stc_pline);
    if (pline && pline->tot_filt_nsects) {
        /* true: a NOT-to-be-filtered-partial-edge chunk */
        /* false : a to-be-filtered-partial-edge-chunk */
        if (!partial_bound) {
            *vector_possible = false;
            HGOTO_DONE(SUCCEED);
        }
    }
    *vector_possible = true;

    assert(chk != NULL);
    assert(chk->sel_space != NULL);

    /* Get the number of elements in chk->sel_space */
    if ((hss_nelmts = (hssize_t)H5S_GET_SELECT_NPOINTS(chk->sel_space)) < 0)
        HGOTO_ERROR(H5E_VFL, H5E_CANTCOUNT, FAIL, "can't get number of elements selected");
    H5_CHECKED_ASSIGN(chk_nelmts, hsize_t, hss_nelmts, hssize_t);

    if (NULL == (serial_values_space = H5S_create_simple(1, &chk_nelmts, NULL)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, FAIL, "unable to create simple memory dataspace");

    flex_selection.cvp = file_space_in;

    if (H5S_select_project_intersection(chk->sel_space, serial_values_space, flex_selection.vp,
                                        &serial_file_space, true) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCLIP, FAIL,
                    "can't project the intersection of defined and requested values");

    /*
     * A successful projection may produce no dataspace when the requested
     * selection contains no values defined in this sparse chunk. This is a
     * successful translation to an empty vector.
     */
    if (serial_file_space == NULL)
        HGOTO_DONE(SUCCEED);

    /*
     * The iterator traverses the projected intersection, not file_space_in.
     * The original request may include values that are undefined in this chunk.
     */
    // if ((hss_nelmts = H5S_GET_SELECT_NPOINTS(serial_file_space)) < 0)
    //     HGOTO_ERROR(H5E_VFL, H5E_CANTCOUNT, FAIL, "can't get number of elements in projected file
    //     selection");
    // H5_CHECKED_ASSIGN(file_nelmts, size_t, hss_nelmts, hssize_t);

    projected_nelmts = H5S_GET_SELECT_NPOINTS(serial_file_space);
    H5_CHECKED_ASSIGN(file_nelmts, size_t, projected_nelmts, hsize_t);

    /*
     * Retain this defensive branch in case the projection API returns an
     * allocated dataspace carrying an empty selection.
     */
    if (file_nelmts == 0)
        HGOTO_DONE(SUCCEED);

    if (0 == (elmt_size = H5T_get_size(dset->shared->type)))
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADSIZE, FAIL, "datatype size invalid");

    if (NULL == (file_iter = H5FL_MALLOC(H5S_sel_iter_t)))
        HGOTO_ERROR(H5E_VFL, H5E_CANTALLOC, FAIL, "couldn't allocate file selection iterator");

    /* Initialize sequence lists for file space */
    if (H5S_select_iter_init(file_iter, serial_file_space, elmt_size, H5S_SEL_ITER_GET_SEQ_LIST_SORTED) < 0)
        HGOTO_ERROR(H5E_VFL, H5E_CANTINIT, FAIL, "can't initialize sequence list for file space");
    file_iter_init = true;

    /* Initialize values so sequence lists are retrieved on the first
     * iteration */
    file_seq_i = SEQ_LIST_LEN;
    file_nseq  = 0;

    if (NULL == (vec_addrs = H5MM_malloc(VECTOR_LEN * sizeof(*vec_addrs))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "memory allocation failed for vector addrs");

    if (NULL == (vec_sizes = H5MM_malloc(VECTOR_LEN * sizeof(*vec_sizes))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "memory allocation failed for vector sizes");

    /* Loop until all elements are processed */
    while (file_seq_i < file_nseq || file_nelmts > 0) {
        /* Fill/refill file sequence list if necessary */
        if (file_seq_i == SEQ_LIST_LEN) {
            if (H5S_SELECT_ITER_GET_SEQ_LIST(file_iter, SEQ_LIST_LEN, SIZE_MAX, &file_nseq, &seq_nelem,
                                             file_off, file_len) < 0)
                HGOTO_ERROR(H5E_INTERNAL, H5E_UNSUPPORTED, FAIL, "sequence length generation failed");
            assert(file_nseq > 0);

            file_nelmts -= seq_nelem;
            file_seq_i = 0;
        }
        assert(file_seq_i < file_nseq);

        /* Calculate length of this IO */
        io_len = file_len[file_seq_i];

        if (vec_arr_nused == vec_arr_nalloc) {
            size_t new_nalloc;
            void  *tmp_ptr;

            if (vec_arr_nalloc > SIZE_MAX / 2)
                HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "vector array element count overflow");

            new_nalloc = vec_arr_nalloc * 2;

            if (new_nalloc > SIZE_MAX / sizeof(*vec_addrs))
                HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                            "vector address array allocation size overflow");

            if (new_nalloc > SIZE_MAX / sizeof(*vec_sizes))
                HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "vector size array allocation size overflow");

            if (NULL == (tmp_ptr = H5MM_realloc(vec_addrs, new_nalloc * sizeof(*vec_addrs))))
                HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "memory reallocation failed for address list");
            vec_addrs = tmp_ptr;

            if (NULL == (tmp_ptr = H5MM_realloc(vec_sizes, new_nalloc * sizeof(*vec_sizes))))
                HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "memory reallocation failed for size list");
            vec_sizes = tmp_ptr;

            vec_arr_nalloc = new_nalloc;
        }

        /* Add this segment to vector read list */
        vec_addrs[vec_arr_nused] = addr + file_off[file_seq_i];
        vec_sizes[vec_arr_nused] = io_len;

        vec_arr_nused++;

        /* Update file sequence */
        if (io_len == file_len[file_seq_i])
            file_seq_i++;
        else {
            file_off[file_seq_i] += io_len;
            file_len[file_seq_i] -= io_len;
        }
    }

    *vec_count = vec_arr_nused;
    *offsets   = vec_addrs;
    *sizes     = vec_sizes;

    vec_addrs = NULL;
    vec_sizes = NULL;

done:
    vec_addrs = H5MM_xfree(vec_addrs);
    vec_sizes = H5MM_xfree(vec_sizes);

    /* Terminate and free iterators */
    if (file_iter) {
        if (file_iter_init && H5S_SELECT_ITER_RELEASE(file_iter) < 0)
            HGOTO_ERROR(H5E_INTERNAL, H5E_CANTFREE, FAIL, "can't release file selection iterator");
        file_iter = H5FL_FREE(H5S_sel_iter_t, file_iter);
    }

    if (serial_file_space) {
        if (H5S_close(serial_file_space) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "unable to close projected serial file space");
    }

    if (serial_values_space) {
        if (H5S_close(serial_values_space) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "unable to close serial values space");
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* H5D__struct_chunk_vector_read() */

/*-------------------------------------------------------------------------
 * Function: H5D__struct_chunk_vector_write
 *
 * Purpose:
 *   Translate FILE_SPACE_IN for one structured chunk into physical file
 *   byte ranges that may be used by the SCC to construct a direct vector
 *   write operation.
 *
 *   This callback does not perform I/O and does not construct the
 *   corresponding memory-buffer vector. On success, OFFSETS and SIZES
 *   describe existing portions of the chunk's serialized on-disk value
 *   region that correspond to the supplied selection.
 *
 *   VECTOR_POSSIBLE is set to false when direct in-place access is not
 *   compatible with the chunk's encoded representation. REQUIRE_VALUES is
 *   set when the translation cannot be completed without decoded
 *   defined-values metadata.
 *
 *   The returned ranges alone do not establish that a direct write is safe.
 *   The caller must additionally verify that the write does not change the
 *   chunk's defined-values metadata, encoded size, filter representation,
 *   index record, or SCC-resident authoritative state.
 *
 *   A successful translation does not imply that an I/O operation has been
 *   submitted or completed. The caller owns the returned OFFSETS and SIZES
 *   arrays and is responsible for releasing them.
 *
 *   The vector-read and vector-write callbacks currently perform the same
 *   selection-to-file-range translation and are intentionally retained as
 *   separate callbacks while their eventual SCC read/write uses remain under
 *   development. Once those use cases and their eligibility requirements are
 *   established, their common implementation should be moved into one shared
 *   translation helper rather than maintained independently.
 *
 * Return:
 *   SUCCEED when eligibility is determined and, when possible, the vector
 *   is produced; FAIL on an internal translation or allocation error.
 *
 * Updated:     Structured chunks containing variable-length data cannot use
 *              direct vector I/O. Their fixed section contains chunk-local
 *              descriptors whose payloads are owned by the decoded chunk's
 *              H5HG heap set. Such writes must therefore pass through the
 *              resident structured-chunk representation and its chunk-local
 *              VL datatype-conversion callbacks.
 *
 *              The datatype is checked before constructing an I/O vector. If
 *              it contains VL data, VECTOR_POSSIBLE remains false so the SCC
 *              uses the decoded-value write path instead.
 *
 *                                              -- AZO   9/19/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_vector_write(H5D_t *dset, haddr_t addr, const H5S_t *file_space_in, bool partial_bound,
                               void *chunk /*in*/, size_t *vec_count /*out*/, haddr_t **offsets /*out*/,
                               size_t **sizes /*out*/, bool *vector_possible /*out*/,
                               bool *require_values /*out*/, void H5_ATTR_UNUSED *udata)
{
    H5D_chunk_cache_mem_t  *chk       = (H5D_chunk_cache_mem_t *)chunk; /* Chunk memory cache info */
    size_t                  elmt_size = 0;
    haddr_t                *vec_addrs = NULL;
    size_t                 *vec_sizes = NULL;
    hsize_t                 file_off[SEQ_LIST_LEN];
    size_t                  file_len[SEQ_LIST_LEN];
    size_t                  file_seq_i;
    size_t                  file_nseq;
    size_t                  io_len;
    size_t                  file_nelmts;
    hsize_t                 chk_nelmts;
    hssize_t                hss_nelmts;
    hsize_t                 projected_nelmts;
    size_t                  seq_nelem;
    H5S_sel_iter_t         *file_iter           = NULL;
    bool                    file_iter_init      = false;
    size_t                  vec_arr_nused       = 0;
    size_t                  vec_arr_nalloc      = VECTOR_LEN;
    H5O_stc_pline_t        *pline               = NULL; /* I/O pipeline info */
    H5S_t                  *serial_values_space = NULL;
    H5S_t                  *serial_file_space   = NULL;
    H5_flexible_const_ptr_t flex_selection;
    htri_t                  has_vlen_type = false;
    herr_t                  ret_value     = SUCCEED;

    FUNC_ENTER_PACKAGE

    *vec_count       = 0;
    *offsets         = NULL;
    *sizes           = NULL;
    *vector_possible = false;
    *require_values  = false;

    /* Sanity checks */
    assert(dset);

    /*
     * Chunk-local VL values cannot use the direct vector-I/O path.
     *
     * The fixed section contains descriptors whose payloads live in the
     * chunk-local H5HG heap set. Reads and writes must therefore pass through
     * the decoded structured-chunk representation and the H5T chunk-local VL
     * conversion callbacks.
     */
    if ((has_vlen_type = H5T_detect_class(dset->shared->type, H5T_VLEN, false)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "unable to detect VL datatype");

    if (has_vlen_type) {
        HGOTO_DONE(SUCCEED);
    }

    if (chk == NULL) {
        *require_values = true;
        HGOTO_DONE(SUCCEED);
    }

    /* missing this see vector_read */
    pline = &(dset->shared->dcpl_cache.stc_pline);
    if (pline && pline->tot_filt_nsects) {
        /* true: a NOT-to-be-filtered-partial-edge chunk */
        /* false : a to-be-filtered-partial-edge-chunk */
        if (!partial_bound) {
            *vector_possible = false;
            HGOTO_DONE(SUCCEED);
        }
    }

    *vector_possible = true;

    assert(chk != NULL);
    assert(chk->sel_space != NULL);

    /* Get the number of elements in chk->sel_space */
    if ((hss_nelmts = (hssize_t)H5S_GET_SELECT_NPOINTS(chk->sel_space)) < 0)
        HGOTO_ERROR(H5E_VFL, H5E_CANTCOUNT, FAIL, "can't get number of elements selected");
    H5_CHECKED_ASSIGN(chk_nelmts, hsize_t, hss_nelmts, hssize_t);

    if (NULL == (serial_values_space = H5S_create_simple(1, &chk_nelmts, NULL)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, FAIL, "unable to create simple memory dataspace");

    flex_selection.cvp = file_space_in;

    if (H5S_select_project_intersection(chk->sel_space, serial_values_space, flex_selection.vp,
                                        &serial_file_space, true) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCLIP, FAIL,
                    "can't project the intersection of defined and requested values");

    /*
     * A successful projection may produce no dataspace when the requested
     * selection contains no values defined in this sparse chunk. This is a
     * successful translation to an empty vector.
     */
    if (serial_file_space == NULL)
        HGOTO_DONE(SUCCEED);

    /*
     * The iterator traverses the projected intersection, not file_space_in.
     * The original request may include values that are undefined in this chunk.
     */
    // if ((hss_nelmts = H5S_GET_SELECT_NPOINTS(serial_file_space)) < 0)
    //     HGOTO_ERROR(H5E_VFL, H5E_CANTCOUNT, FAIL, "can't get number of elements in projected file
    //     selection");
    // H5_CHECKED_ASSIGN(file_nelmts, size_t, hss_nelmts, hssize_t);
    projected_nelmts = H5S_GET_SELECT_NPOINTS(serial_file_space);
    H5_CHECKED_ASSIGN(file_nelmts, size_t, projected_nelmts, hsize_t);

    /*
     * Retain this defensive branch in case the projection API returns an
     * allocated dataspace carrying an empty selection.
     */
    if (file_nelmts == 0)
        HGOTO_DONE(SUCCEED);

    if (0 == (elmt_size = H5T_get_size(dset->shared->type)))
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADSIZE, FAIL, "datatype size invalid");

    if (NULL == (file_iter = H5FL_MALLOC(H5S_sel_iter_t)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate file iterator");

    if (H5S_select_iter_init(file_iter, serial_file_space, elmt_size, H5S_SEL_ITER_GET_SEQ_LIST_SORTED) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize file selection information");
    file_iter_init = true; /* file selection iteration info has been initialized */

    /* Initialize values so sequence lists are retrieved on the first
     * iteration */
    file_seq_i = SEQ_LIST_LEN;
    file_nseq  = 0;

    if (NULL == (vec_addrs = H5MM_malloc(VECTOR_LEN * sizeof(*vec_addrs))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "memory allocation failed for vector addrs");

    if (NULL == (vec_sizes = H5MM_malloc(VECTOR_LEN * sizeof(*vec_sizes))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "memory allocation failed for vector sizes");

    /* Loop until all elements are processed */
    while (file_seq_i < file_nseq || file_nelmts > 0) {
        /* Fill/refill file sequence list if necessary */
        if (file_seq_i == SEQ_LIST_LEN) {
            if (H5S_SELECT_ITER_GET_SEQ_LIST(file_iter, SEQ_LIST_LEN, SIZE_MAX, &file_nseq, &seq_nelem,
                                             file_off, file_len) < 0)
                HGOTO_ERROR(H5E_INTERNAL, H5E_UNSUPPORTED, FAIL, "sequence length generation failed");
            assert(file_nseq > 0);

            file_nelmts -= seq_nelem;
            file_seq_i = 0;
        }
        assert(file_seq_i < file_nseq);

        /* Calculate length of this IO */
        io_len = file_len[file_seq_i];

        if (vec_arr_nused == vec_arr_nalloc) {
            size_t new_nalloc;
            void  *tmp_ptr;

            if (vec_arr_nalloc > SIZE_MAX / 2)
                HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "vector array element count overflow");

            new_nalloc = vec_arr_nalloc * 2;

            if (new_nalloc > SIZE_MAX / sizeof(*vec_addrs))
                HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                            "vector address array allocation size overflow");

            if (new_nalloc > SIZE_MAX / sizeof(*vec_sizes))
                HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "vector size array allocation size overflow");

            if (NULL == (tmp_ptr = H5MM_realloc(vec_addrs, new_nalloc * sizeof(*vec_addrs))))
                HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "memory reallocation failed for address list");
            vec_addrs = tmp_ptr;

            if (NULL == (tmp_ptr = H5MM_realloc(vec_sizes, new_nalloc * sizeof(*vec_sizes))))
                HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "memory reallocation failed for size list");
            vec_sizes = tmp_ptr;

            vec_arr_nalloc = new_nalloc;
        }

        /* Add this segment to vector read list */
        vec_addrs[vec_arr_nused] = addr + file_off[file_seq_i];
        vec_sizes[vec_arr_nused] = io_len;

        vec_arr_nused++;

        /* Update file sequence */
        if (io_len == file_len[file_seq_i])
            file_seq_i++;
        else {
            file_off[file_seq_i] += io_len;
            file_len[file_seq_i] -= io_len;
        }
    }

    *vec_count = vec_arr_nused;
    *offsets   = vec_addrs;
    *sizes     = vec_sizes;

    vec_addrs = NULL;
    vec_sizes = NULL;

done:
    vec_addrs = H5MM_xfree(vec_addrs);
    vec_sizes = H5MM_xfree(vec_sizes);

    /* Terminate and free iterators */
    if (file_iter) {
        if (file_iter_init && H5S_SELECT_ITER_RELEASE(file_iter) < 0)
            HGOTO_ERROR(H5E_INTERNAL, H5E_CANTFREE, FAIL, "can't release file selection iterator");
        file_iter = H5FL_FREE(H5S_sel_iter_t, file_iter);
    }

    if (serial_file_space) {
        if (H5S_close(serial_file_space) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "unable to close projected serial file space");
    }

    if (serial_values_space) {
        if (H5S_close(serial_values_space) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "unable to close serial values space");
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* H5D__struct_chunk_vector_write() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_scatter_mem
 *
 * Purpose:     Scatters data from the chunk buffer into the memory buffer (in dset_info),
 *              performing type conversion if necessary.
 *              file_space's extent matches the chunk dimensions and the selection is within the chunk.
 *              mem_space's extent matches the entire memory buffer's and the selection within it is
 *              the selected values within the chunk, offset appropriately within the full extent.
 *              Optional, if not present, chunk is the same in memory as it is in cache, with the
 *              exception of type conversion (which will be handled by the H5SC layer).
 *              If the layout stores variable length data within the chunk this callback must be defined.
 *              TBD: the following description probably should not be here in the RFC:
 *              [partial_bound is true if the on-disk chunk was encoded with partial_bound set to true.
 *              If the dataset reported partial_bound_chunks_different_encoding as false,
 *              the setting of partial_bound is undefined.]
 *
 * Return:    Non-negative on success/Negative on failure
 *
 * NOTE: [chunk] is the pointer to the chunk intermediate struct
 * NOTE: This routine is modified from H5D__scatgath_read()
 *
 * NOTE: [udata] not used??
 *
 * Updated:     Added support for reading variable-length values stored in a
 *              structured chunk's local H5HG heap set. For VL datatypes, a
 *              private file-side datatype is prepared whose VL callbacks use
 *              a chunk-local conversion context rather than the normal
 *              file-wide VL storage backend.
 *
 *              VL reads use dedicated strip-mined conversion and background
 *              buffers and are forced out of place because conversion creates
 *              memory-side VL payloads. The fixed descriptors are gathered
 *              from the decoded chunk and converted through the chunk-local
 *              datatype path, which resolves their referenced payloads from
 *              the chunk's heap set.
 *
 *              The dedicated VL background buffer is passed to the conversion
 *              when required, including for compound datatypes containing VL
 *              members. This is necessary because compound conversion may
 *              require a valid background value for members that are not
 *              overwritten during conversion. The existing non-VL conversion
 *              and optimized compound paths remain unchanged.
 *
 *                                              -- AZO   9/19/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_scatter_mem(H5D_dset_io_info_t *dset_info, H5D_io_type_info_t *io_type_info,
                              const H5S_t *mem_space, const H5S_t *file_space, const void *chunk,
                              void H5_ATTR_UNUSED *udata)
{
    void           *buf;                    /* Local pointer to application buffer */
    void           *tmp_buf;                /* Buffer to use for type conversion */
    H5S_sel_iter_t *file_iter      = NULL;  /* Memory selection iteration info*/
    bool            file_iter_init = false; /* Memory selection iteration info has been initialized */
    H5S_sel_iter_t *mem_iter       = NULL;  /* Memory selection iteration info*/
    bool            mem_iter_init  = false; /* Memory selection iteration info has been initialized */
    H5S_sel_iter_t *bkg_iter       = NULL;  /* Background iteration info*/
    bool            bkg_iter_init  = false; /* Background iteration info has been initialized */
    H5S_sel_iter_t *sel_iter       = NULL;  /* Memory selection iteration info*/
    bool            sel_iter_init  = false; /* Memory selection iteration info has been initialized */
    hsize_t         nelmts         = 0;     /* Number of elements selected in file & memory dataspaces */
    hsize_t         smine_start;            /* Strip mine start loc */
    size_t          smine_nelmts;           /* Elements per strip   */
    bool            in_place_tconv = false; /* Whether to perform in-place type_conversion */
    size_t          mem_type_size;
    size_t          file_type_size;
    size_t          buf_off          = 0; /* Buffer offset for in-place type conversion */
    const H5D_chunk_cache_mem_t *chk = (const H5D_chunk_cache_mem_t *)chunk; /* Chunk's memory cache info */
    void                        *data_scat_buf = NULL;
    void                        *packed_buf    = NULL;
    hsize_t                      scat_buf_size;
    H5_flexible_const_ptr_t      flex_mspace;
    H5_flexible_const_ptr_t      flex_fspace;
    herr_t                       ret_value = SUCCEED; /* Return value     */

    H5T_t                *chunk_file_type = NULL;
    H5T_path_t           *chunk_tpath     = NULL;
    H5HG_local_heapset_t *read_heapset    = NULL;
    H5T_vlen_chunk_ctx_t  vl_ctx;
    size_t                ref_nbytes    = 0;
    htri_t                has_vlen_type = false;

    void  *vl_tconv_buf    = NULL;
    void  *vl_bkg_buf      = NULL;
    size_t vl_strip_nelmts = 0;
    size_t vl_type_size;

    FUNC_ENTER_PACKAGE

    assert(dset_info);
    assert(io_type_info);
    assert(mem_space);
    assert(file_space);
    assert(chk);

    /* Make certain that the number of elements in each selection is the same */
    nelmts = H5S_GET_SELECT_NPOINTS(mem_space);
    if (nelmts != H5S_GET_SELECT_NPOINTS(file_space))
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL,
                    "src and dest dataspaces have different number of elements selected");

    /* Check for NOOP read */
    if (nelmts == 0)
        HGOTO_DONE(SUCCEED);

    mem_type_size  = dset_info->type_info.dst_type_size;
    file_type_size = dset_info->type_info.src_type_size;

    /* On read, SRC_TYPE is the file-side datatype. */
    if ((has_vlen_type = H5T_detect_class(dset_info->type_info.src_type, H5T_VLEN, false)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "unable to detect VL datatype");

    if (has_vlen_type) {
        /*
         * Build a private file datatype whose VL nodes resolve through this
         * structured chunk rather than the normal file-wide blob backend.
         */
        if (H5D__struct_chunk_prepare_vlen_type(dset_info->dset, dset_info->type_info.src_type,
                                                dset_info->type_info.dst_type, true, &chunk_file_type,
                                                &chunk_tpath, &ref_nbytes) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to prepare chunk-local VL read conversion");

        H5_CHECK_OVERFLOW(nelmts, hsize_t, size_t);

        vl_strip_nelmts = MIN((size_t)nelmts, (size_t)1024);
        vl_type_size    = MAX(file_type_size, mem_type_size);

        if (vl_type_size == 0 || vl_strip_nelmts > SIZE_MAX / vl_type_size)
            HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL,
                        "chunk-local VL read conversion buffer size overflow");

        if (NULL == (vl_tconv_buf = H5MM_malloc(vl_strip_nelmts * vl_type_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                        "unable to allocate chunk-local VL read conversion buffer");

        if (dset_info->type_info.need_bkg != H5T_BKG_NO) {
            if (NULL == (vl_bkg_buf = H5MM_calloc(vl_strip_nelmts * vl_type_size)))
                HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                            "unable to allocate chunk-local VL read background buffer");
        }

        /*
         * READ does not create heap sets. Use a local pointer variable so the
         * H5T context can retain its existing H5HG_local_heapset_t ** interface
         * without casting away CHK's const qualification.
         */
        read_heapset = chk->vl_heapset;

        memset(&vl_ctx, 0, sizeof(vl_ctx));

        vl_ctx.f          = dset_info->dset->oloc.file;
        vl_ctx.heapset    = &read_heapset;
        vl_ctx.ref_nbytes = ref_nbytes;
    }

    flex_mspace.cvp = mem_space;
    flex_fspace.cvp = file_space;

    /* Set buf pointer (memory buffer in dset_info) it's the application buffer */
    buf = dset_info->buf.vp;

    /* Allocate the data_scat_buf: chunk size * element size */
    scat_buf_size = dset_info->layout->u.struct_chunk.size;
    if (NULL == (data_scat_buf = H5FL_BLK_MALLOC(scat_buf, scat_buf_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "memory allocation failed for scattered data buffer");
    memset(data_scat_buf, 0, scat_buf_size);

    /* Scatter dato to data_scat_buf according to chk->sel_space */
    if (chk->sel_space != NULL) {
        hsize_t sel_nelmts;

        /* Get the number of elements in the selection */
        sel_nelmts = H5S_GET_SELECT_NPOINTS(chk->sel_space);

        if (NULL == (sel_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate selection iterator");

        if (H5S_select_iter_init(sel_iter, chk->sel_space, file_type_size, H5S_SEL_ITER_GET_SEQ_LIST_SORTED) <
            0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize selection iter information");
        sel_iter_init = true;

        /* Scatter data values from chk->data_buf to data_scat_buf according to sel_space */
        if (H5D__scatter_mem(chk->data_buf, sel_iter, sel_nelmts, data_scat_buf /*out*/) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "mem scatter failed");

        if (sel_iter_init && H5S_SELECT_ITER_RELEASE(sel_iter) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "Can't release selection iterator");

        sel_iter_init = false;

        if (sel_iter)
            sel_iter = H5FL_FREE(H5S_sel_iter_t, sel_iter);
    }

    /*
     * If there is no data transform or type conversion then read directly
     * into the application's buffer.
     */

    if (!has_vlen_type && dset_info->type_info.is_xform_noop && dset_info->type_info.is_conv_noop) {

        size_t selected_nelmts;
        size_t packed_buf_size;
        size_t n;

        H5_CHECK_OVERFLOW(nelmts, hsize_t, size_t);
        selected_nelmts = (size_t)nelmts;

        if (selected_nelmts > (SIZE_MAX / file_type_size))
            HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "packed read buffer size overflow");

        packed_buf_size = selected_nelmts * file_type_size;

        if (NULL == (packed_buf = H5MM_malloc(packed_buf_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate packed read buffer");

        /*
         * Gather the selected chunk positions into a contiguous buffer.
         */
        if (NULL == (file_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate file selection iterator");

        if (H5S_select_iter_init(file_iter, flex_fspace.vp, file_type_size,
                                 H5S_SEL_ITER_GET_SEQ_LIST_SORTED) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize file selection iterator");

        file_iter_init = true;

        n = H5D__gather_mem(data_scat_buf, file_iter, selected_nelmts, packed_buf);
        if (n != selected_nelmts)
            HGOTO_ERROR(H5E_IO, H5E_READERROR, FAIL, "unable to gather selected chunk values");

        if (H5S_SELECT_ITER_RELEASE(file_iter) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "can't release file selection iterator");

        file_iter_init = false;
        file_iter      = H5FL_FREE(H5S_sel_iter_t, file_iter);

        /*
         * Scatter the contiguous values according to the application's
         * memory selection. This explicitly supports differing ranks.
         */
        if (NULL == (mem_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate memory selection iterator");

        if (H5S_select_iter_init(mem_iter, flex_mspace.vp, mem_type_size, H5S_SEL_ITER_GET_SEQ_LIST_SORTED) <
            0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize memory selection iterator");

        mem_iter_init = true;

        if (H5D__scatter_mem(packed_buf, mem_iter, selected_nelmts, buf /*out*/) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "unable to scatter values into read buffer");

        packed_buf = H5MM_xfree(packed_buf);
    }
    else { /* With type conversion */

        /* Check for in-place type conversion */
        if (io_type_info->may_use_in_place_tconv) {

            /* Make sure the memory type is not smaller than the file type, otherwise the memory buffer
               won't be big enough to serve as the type conversion buffer */
            if (mem_type_size >= file_type_size) {
                bool    is_contig;
                hsize_t sel_off;

                /* Check if the space is contiguous */
                if (H5S_select_contig_block(flex_mspace.vp, &is_contig, &sel_off, NULL) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "can't check if dataspace is contiguous");

                /* If the first sequence includes all the elements selected in this piece, it it contiguous */
                if (is_contig) {
                    H5_CHECK_OVERFLOW(sel_off, hsize_t, size_t);
                    in_place_tconv = true;
                    buf_off        = (size_t)sel_off * mem_type_size;
                }
            }
        }
        /* VL conversion creates memory-side payloads; use the conversion buffer. */
        if (has_vlen_type) {
            in_place_tconv = false;
        }

        /* Check if we should disable in-place type conversion for performance.  Do so if we can use the
         * optimized compound read function, and the either entire I/O operation can fit in the type
         * conversion buffer or we need to use a background buffer (and therefore could not do the I/O in one
         * operation with in-place conversion * anyways). */
        if (!has_vlen_type && H5D__SCATGATH_USE_CMPD_OPT_READ(dset_info, false) &&
            (dset_info->type_info.need_bkg || (nelmts <= dset_info->type_info.request_nelmts)))
            in_place_tconv = false;

        /* Allocate the iterators */
        if (NULL == (file_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate file iterator");
        if (NULL == (mem_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate memory iterator");
        if (NULL == (bkg_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate background iterator");

        /* Figure out the strip mine size. */
        if (H5S_select_iter_init(file_iter, flex_fspace.vp, dset_info->type_info.src_type_size,
                                 H5S_SEL_ITER_GET_SEQ_LIST_SORTED) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize file selection information");
        file_iter_init = true; /*file selection iteration info has been initialized */
        if (H5S_select_iter_init(mem_iter, flex_mspace.vp, dset_info->type_info.dst_type_size, 0) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize memory selection information");
        mem_iter_init = true; /*file selection iteration info has been initialized */
        if (H5S_select_iter_init(bkg_iter, flex_mspace.vp, dset_info->type_info.dst_type_size, 0) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL,
                        "unable to initialize background selection information");
        bkg_iter_init = true; /*file selection iteration info has been initialized */

        /* Start strip mining... */
        for (smine_start = 0; smine_start < nelmts; smine_start += smine_nelmts) {
            size_t n; /* Elements operated on */

            /* Determine strip mine size. First check if we're doing in-place type conversion */
            if (in_place_tconv) {
                /* If there is a background buffer, we cannot exceed request_nelmts. */
                assert(!H5D__SCATGATH_USE_CMPD_OPT_READ(dset_info, in_place_tconv));
                if (dset_info->type_info.need_bkg)
                    smine_nelmts = (size_t)MIN(dset_info->type_info.request_nelmts, (nelmts - smine_start));
                else {
                    assert(smine_start == 0);
                    smine_nelmts = nelmts;
                }

                /* Calculate buffer position in user buffer */
                tmp_buf = (uint8_t *)buf + buf_off + (smine_start * dset_info->type_info.dst_type_size);
            }
            else {
                /* Do type conversion using intermediate buffer */
                if (has_vlen_type) {
                    tmp_buf      = vl_tconv_buf;
                    smine_nelmts = (size_t)MIN((hsize_t)vl_strip_nelmts, nelmts - smine_start);
                }
                else {
                    tmp_buf = io_type_info->tconv_buf;
                    /* Go figure out how many elements to read from the file */
                    smine_nelmts = (size_t)MIN(dset_info->type_info.request_nelmts, nelmts - smine_start);
                }
            }

            /*
             * Gather the data from disk into the datatype conversion
             * buffer. Also gather data from application to background buffer
             * if necessary.
             */

            /* Fill background buffer here unless we will use H5D__compound_opt_read().  Must do this before
             * the read so the read buffer doesn't get wiped out if we're using in-place type conversion */
            if ((H5T_BKG_YES == dset_info->type_info.need_bkg) &&
                !H5D__SCATGATH_USE_CMPD_OPT_READ(dset_info, in_place_tconv)) {
                n = H5D__gather_mem(buf, bkg_iter, smine_nelmts,
                                    has_vlen_type ? vl_bkg_buf : io_type_info->bkg_buf);
                if (n != smine_nelmts)
                    HGOTO_ERROR(H5E_IO, H5E_READERROR, FAIL, "mem gather failed");
            }

            /*
             * Gather data from data_scat_buf to tmp_buf
             */
            n = H5D__gather_mem(data_scat_buf, file_iter, smine_nelmts, tmp_buf /*out*/);
            if (n != smine_nelmts)
                HGOTO_ERROR(H5E_IO, H5E_READERROR, FAIL, "mem gather failed");

            /* If the source and destination are compound types and subset of each other
             * and no conversion is needed, copy the data directly into user's buffer and
             * bypass the rest of steps.
             */
            if (!has_vlen_type && H5D__SCATGATH_USE_CMPD_OPT_READ(dset_info, in_place_tconv)) {
                if (H5D__compound_opt_read(smine_nelmts, mem_iter, &dset_info->type_info, tmp_buf,
                                           buf /*out*/) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "datatype conversion failed");
            } /* end if */
            else {
                /*
                 * Perform datatype conversion.
                 */
                if (has_vlen_type) {
                    if (H5D__struct_chunk_vlen_convert(&vl_ctx, chunk_tpath, chunk_file_type,
                                                       dset_info->type_info.dst_type, smine_nelmts, tmp_buf,
                                                       vl_bkg_buf) < 0)
                        HGOTO_ERROR(H5E_DATASET, H5E_CANTCONVERT, FAIL,
                                    "chunk-local VL read conversion failed");
                }
                else {
                    if (H5T_convert(dset_info->type_info.tpath, dset_info->type_info.src_type,
                                    dset_info->type_info.dst_type, smine_nelmts, (size_t)0, (size_t)0,
                                    tmp_buf, io_type_info->bkg_buf) < 0)
                        HGOTO_ERROR(H5E_DATASET, H5E_CANTCONVERT, FAIL, "datatype conversion failed");
                }

                /* Do the data transform after the conversion (since we're using type mem_type) */
                if (!dset_info->type_info.is_xform_noop) {
                    H5Z_data_xform_t *data_transform; /* Data transform info */

                    /* Retrieve info from API context */
                    if (H5CX_get_data_transform(&data_transform) < 0)
                        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get data transform info");

                    if (H5Z_xform_eval(data_transform, tmp_buf, smine_nelmts, dset_info->type_info.mem_type) <
                        0)
                        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "Error performing data transform");
                }

                /* Scatter the data into memory if this was not an in-place conversion */
                if (!in_place_tconv)
                    if (H5D__scatter_mem(tmp_buf, mem_iter, smine_nelmts, buf /*out*/) < 0)
                        HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "scatter failed");
            } /* end else */

        } /* end for */
    }

done:

    if (chunk_file_type) {
        if (H5T_close_real(chunk_file_type) < 0)
            HDONE_ERROR(H5E_DATATYPE, H5E_CANTCLOSEOBJ, FAIL,
                        "unable to release chunk-local VL file datatype");
    }
    /* Release selection iterators */
    if (file_iter_init && H5S_SELECT_ITER_RELEASE(file_iter) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "Can't release selection iterator");
    if (file_iter)
        file_iter = H5FL_FREE(H5S_sel_iter_t, file_iter);

    if (mem_iter_init && H5S_SELECT_ITER_RELEASE(mem_iter) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "Can't release selection iterator");
    if (mem_iter)
        mem_iter = H5FL_FREE(H5S_sel_iter_t, mem_iter);

    if (bkg_iter_init && H5S_SELECT_ITER_RELEASE(bkg_iter) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "Can't release selection iterator");
    if (bkg_iter)
        bkg_iter = H5FL_FREE(H5S_sel_iter_t, bkg_iter);

    /* Release resources */
    if (data_scat_buf)
        data_scat_buf = H5FL_BLK_FREE(scat_buf, data_scat_buf);

    if (packed_buf)
        packed_buf = H5MM_xfree(packed_buf);

    vl_tconv_buf = H5MM_xfree(vl_tconv_buf);
    vl_bkg_buf   = H5MM_xfree(vl_bkg_buf);

    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_scatter_mem() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_gather_mem
 *
 * Purpose:     Gathers data from the memory buffer (in dset_info) into the chunk buffer,
 *              performing type conversion if necessary.
 *              file_space's extent matches the chunk dimensions and the selection is within
 *              the chunk.
 *              mem_space's extent matches the entire memory buffer's and the selection within it
 *              is the selected values within the chunk, offset appropriately within the full extent.
 *              Defines selected values in the chunk.
 *              Optional, if not present, chunk is the same in memory as it is in cache, with the
 *              exception of type conversion (which will be handled by H5SC layer).
 *              If the layout stores variable length data within the chunk this callback must be defined.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *
 * NOTE: [chunk] is the pointer to the chunk intermediate struct
 * NOTE: This routine is modified from H5D__scatgath_write()
 * NOTE: [udata] not used??
 * NOTE: Not sure about my tracking of [nbytes], [alloc_size], and [alloc_size_total]
 *
 * Updated:    Adds serial conversion of memory-side VL values into fixed
 *             structured-chunk descriptors backed by a chunk-local H5HG
 *             heap set.
 *
 *             Heap-set changes are staged so failed conversion does not
 *             publish partially constructed VL storage. Existing fixed
 *             descriptors are gathered as conversion background so replacing
 *             a VL value also releases its previous chunk-local heap object.
 *
 *             The VL path supplies its own conversion and background buffers
 *             because the normal H5D conversion setup may classify logically
 *             identical source and destination datatypes as a no-op and
 *             provide zero strip-mining capacity. The resulting heap-set
 *             allocation is included in SCC resident-memory accounting.
 *
 *                                    -- AZO   09/17/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_gather_mem(H5D_dset_io_info_t *dset_info, H5D_io_type_info_t *io_type_info,
                             const H5S_t *mem_space, const H5S_t *file_space, size_t *nbytes /*in,out*/,
                             size_t *alloc_size /*in,out*/, size_t *alloc_size_total /*in,out*/, void *chunk,
                             void H5_ATTR_UNUSED *udata)
{

    const void             *buf;                    /* Local pointer to application buffer */
    void                   *tmp_buf;                /* Buffer to use for type conversion */
    H5S_sel_iter_t         *file_iter      = NULL;  /* Memory selection iteration info*/
    bool                    file_iter_init = false; /* Memory selection iteration info has been initialized */
    H5S_sel_iter_t         *mem_iter       = NULL;  /* Memory selection iteration info*/
    bool                    mem_iter_init  = false; /* Memory selection iteration info has been initialized */
    H5S_sel_iter_t         *bkg_iter       = NULL;  /* Memory selection iteration info*/
    H5S_sel_iter_t         *sel_iter       = NULL;  /* Memory selection iteration info*/
    bool                    sel_iter_init  = false; /* Memory selection iteration info has been initialized */
    bool                    bkg_iter_init  = false; /* Memory selection iteration info has been initialized */
    hsize_t                 smine_start;            /* Strip mine start loc */
    size_t                  smine_nelmts;           /* Elements per strip */
    hsize_t                 nelmts; /* Number of elements selected in file & memory dataspaces */
    size_t                  selected_nelmts;
    size_t                  mem_type_size;
    size_t                  file_type_size;
    size_t                  buf_off        = 0;     /* Buffer offset for in-place type conversion */
    bool                    in_place_tconv = false; /* Whether to perform in-place type_conversion */
    H5D_chunk_cache_mem_t  *chk            = (H5D_chunk_cache_mem_t *)chunk; /* Chunk's memory cache info */
    void                   *data_scat_buf  = NULL;
    void                   *packed_buf     = NULL;
    hsize_t                 scat_buf_size;
    H5_flexible_const_ptr_t flex_mspace;
    H5_flexible_const_ptr_t flex_fspace;
    H5T_t                  *chunk_file_type = NULL;   /* Chunk-file datatype using local VL references */
    H5T_path_t             *chunk_tpath     = NULL;   /* Conversion path to the chunk-file datatype */
    H5T_vlen_chunk_ctx_t    vl_ctx;                   /* Context used to create chunk-local VL objects */
    H5HG_local_heapset_t   *staged_heapset = NULL;    /* Working copy of the chunk-local VL heap set */
    H5HG_local_heapset_t   *old_heapset    = NULL;    /* Replaced VL heap set awaiting release */
    size_t                  ref_nbytes     = 0;       /* Encoded size of one chunk-local VL reference */
    htri_t                  has_vlen_type  = false;   /* Whether the destination datatype contains VL data */
    herr_t                  ret_value      = SUCCEED; /* Return value */

    void  *vl_tconv_buf    = NULL; /* Private conversion buffer for chunk-local VL writes */
    void  *vl_bkg_buf      = NULL; /* Existing file descriptors used during VL replacement */
    size_t vl_strip_nelmts = 0;    /* Maximum VL elements converted in one iteration */
    size_t vl_buf_size     = 0;    /* Allocation size of each private VL buffer */

    H5S_t                *staged_sel_space = NULL;
    H5S_t                *old_sel_space    = NULL;
    void                 *staged_data_buf  = NULL;
    void                 *old_data_buf     = NULL;
    H5D_chunk_cache_mem_t accounting_chk;
    size_t                staged_data_size = 0;
    size_t                new_nbytes       = 0;
    size_t                new_buffer_alloc = 0;
    size_t                new_total_alloc  = 0;
    htri_t                heapset_empty    = false;

    FUNC_ENTER_PACKAGE

    /* Sanity check */
    assert(dset_info);
    assert(io_type_info);
    assert(dset_info->mem_space);
    assert(dset_info->file_space);
    assert(dset_info->buf.cvp);
    assert(chk);

    /* Make certain that the number of elements in each selection is the same */
    nelmts = H5S_GET_SELECT_NPOINTS(mem_space);
    if (nelmts != H5S_GET_SELECT_NPOINTS(file_space))
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL,
                    "src and dest dataspaces have different number of elements selected");

    /* Check for NOOP write */
    if (nelmts == 0)
        HGOTO_DONE(SUCCEED);

    /*
     * H5S selection counts use hsize_t, while the gather/scatter helpers
     * consume size_t element counts.
     */
    H5_CHECK_OVERFLOW(nelmts, hsize_t, size_t);
    selected_nelmts = (size_t)nelmts;

    mem_type_size  = dset_info->type_info.src_type_size;
    file_type_size = dset_info->type_info.dst_type_size;

    /* Detect whether the file datatype has a VL component */
    if ((has_vlen_type = H5T_detect_class(dset_info->type_info.dst_type, H5T_VLEN, false)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "unable to detect vlen datatypes");

    if (has_vlen_type) {
        if (H5D__struct_chunk_prepare_vlen_type(dset_info->dset, dset_info->type_info.dst_type,
                                                dset_info->type_info.src_type, false, &chunk_file_type,
                                                &chunk_tpath, &ref_nbytes) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to prepare chunk-local VL write conversion");

        /*
         * This copies the entire heap set, including payloads that the write
         * will leave unchanged. It lets conversion modify private storage and
         * preserves the resident descriptors and heap set if staging fails.
         * Reducing that copy would require a way to stage only changed payloads
         * while keeping descriptor references and failure cleanup consistent.
         */
        if (H5HG__copy_local_heapset(dset_info->dset->oloc.file, chk->vl_heapset, &staged_heapset) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTCOPY, FAIL, "unable to copy chunk-local VL heap set");

        memset(&vl_ctx, 0, sizeof(vl_ctx));
        vl_ctx.f          = dset_info->dset->oloc.file;
        vl_ctx.heapset    = &staged_heapset;
        vl_ctx.ref_nbytes = ref_nbytes;

        /*
         * The normal I/O path may classify this operation as a no-op
         * conversion and therefore provide no conversion capacity. Chunk-local
         * VL storage still requires conversion from the memory representation
         * to fixed descriptors, so provide operation-local strip buffers.
         */
        vl_strip_nelmts = MIN(selected_nelmts, (size_t)1024);

        if (vl_strip_nelmts == 0)
            HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL,
                        "chunk-local VL write has an empty conversion strip");

        if (MAX(mem_type_size, file_type_size) == 0 ||
            vl_strip_nelmts > SIZE_MAX / MAX(mem_type_size, file_type_size))
            HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "chunk-local VL conversion buffer size overflow");

        vl_buf_size = vl_strip_nelmts * MAX(mem_type_size, file_type_size);

        if (NULL == (vl_tconv_buf = H5MM_malloc(vl_buf_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                        "unable to allocate chunk-local VL conversion buffer");

        if (NULL == (vl_bkg_buf = H5MM_malloc(vl_buf_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                        "unable to allocate chunk-local VL background buffer");
    }

    flex_mspace.cvp = mem_space;
    flex_fspace.cvp = file_space;

    /* Set buf pointer (memory buffer in dset_info) it's the application buffer */
    buf = dset_info->buf.cvp;

    /*
     * Scatter dato in chk->data_buf to data_scat_buf according to chk->sel_space
     */
    {
        hsize_t sel_nelmts;

        /* Allocate the data_scat_buf: chunk size * element size */
        scat_buf_size = dset_info->layout->u.struct_chunk.size;
        if (NULL == (data_scat_buf = H5FL_BLK_MALLOC(scat_buf, scat_buf_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                        "memory allocation failed for scattered data buffer");
        memset(data_scat_buf, 0, scat_buf_size);

        if (chk->sel_space != NULL) {
            /* Get the number of elements in the selection */
            sel_nelmts = H5S_GET_SELECT_NPOINTS(chk->sel_space);

            /* Initialize the iterator */
            if (NULL == (sel_iter = H5FL_MALLOC(H5S_sel_iter_t)))
                HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate selection iterator");

            if (H5S_select_iter_init(sel_iter, chk->sel_space, file_type_size,
                                     H5S_SEL_ITER_GET_SEQ_LIST_SORTED) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL,
                            "unable to initialize selection iter information");
            sel_iter_init = true;

            /* Scatter data */
            if (H5D__scatter_mem(chk->data_buf, sel_iter, sel_nelmts, data_scat_buf /*out*/) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "mem scatter failed");

            sel_iter_init = false;

            if (H5S_SELECT_ITER_RELEASE(sel_iter) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "unable to release selection iterator");

            sel_iter = H5FL_FREE(H5S_sel_iter_t, sel_iter);
        }
    }

    /*
     * If there is no data transform or type conversion then write directly
     * into the chunk buffer.
     */

    if (!has_vlen_type && dset_info->type_info.is_xform_noop && dset_info->type_info.is_conv_noop) {

        size_t packed_buf_size = 0;
        size_t n;

        if (selected_nelmts > (SIZE_MAX / file_type_size))
            HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "packed write buffer size overflow");

        packed_buf_size = selected_nelmts * file_type_size;

        if (NULL == (packed_buf = H5MM_malloc(packed_buf_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate packed write buffer");

        if (NULL == (mem_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate memory selection iterator");

        if (H5S_select_iter_init(mem_iter, flex_mspace.vp, file_type_size, H5S_SEL_ITER_GET_SEQ_LIST_SORTED) <
            0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize memory selection iterator");

        mem_iter_init = true;

        n = H5D__gather_mem(buf, mem_iter, selected_nelmts, packed_buf);
        if (n != selected_nelmts)
            HGOTO_ERROR(H5E_IO, H5E_WRITEERROR, FAIL, "unable to gather selected memory values");

        if (H5S_SELECT_ITER_RELEASE(mem_iter) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "can't release memory selection iterator");

        mem_iter_init = false;
        mem_iter      = H5FL_FREE(H5S_sel_iter_t, mem_iter);

        if (NULL == (file_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate file selection iterator");

        if (H5S_select_iter_init(file_iter, flex_fspace.vp, file_type_size,
                                 H5S_SEL_ITER_GET_SEQ_LIST_SORTED) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize file selection iterator");

        file_iter_init = true;

        if (H5D__scatter_mem(packed_buf, file_iter, selected_nelmts, data_scat_buf) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "unable to scatter values into chunk buffer");

        packed_buf = H5MM_xfree(packed_buf);
    }
    else { /* with type conversion */

        /* Check for in-place type conversion */
        if (io_type_info->may_use_in_place_tconv) {

            /* Make sure the memory type is not smaller than the file type, otherwise the memory buffer
               won't be big enough to serve as the type conversion buffer */
            if (mem_type_size >= file_type_size) {
                bool    is_contig;
                hsize_t sel_off;

                /* Check if the space is contiguous */
                if (H5S_select_contig_block(flex_mspace.vp, &is_contig, &sel_off, NULL) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "can't check if dataspace is contiguous");

                /* If the first sequence includes all the elements selected in this piece, it it contiguous */
                if (is_contig) {
                    H5_CHECK_OVERFLOW(sel_off, hsize_t, size_t);
                    in_place_tconv = true;
                    buf_off        = (size_t)sel_off * mem_type_size;
                }
            }
        }

        /* Chunk-local VL conversion must not modify the application buffer. */
        if (has_vlen_type)
            in_place_tconv = false;

        /* Check if we should disable in-place type conversion for performance.  Do so if we can use the
         * optimized compound write function, and either entire I/O operation can fit in the type conversion
         * buffer or we need to use a background buffer (and therefore could not do the I/O in one operation
         * with in-place conversion * anyways). */
        if (!has_vlen_type && in_place_tconv && H5D__SCATGATH_USE_CMPD_OPT_WRITE(dset_info, false) &&
            (dset_info->type_info.need_bkg || (nelmts <= dset_info->type_info.request_nelmts)))
            in_place_tconv = false;

        /* Allocate the iterators */
        if (NULL == (mem_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate memory iterator");

        if (NULL == (file_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate file iterator");

        if (NULL == (bkg_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate file iterator");

        /* Figure out the strip mine size. */
        if (H5S_select_iter_init(mem_iter, flex_mspace.vp, dset_info->type_info.src_type_size, 0) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize memory selection information");
        mem_iter_init = true; /*file selection iteration info has been initialized */

        if (H5S_select_iter_init(file_iter, flex_fspace.vp, file_type_size,
                                 H5S_SEL_ITER_GET_SEQ_LIST_SORTED) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL,
                        "unable to initialize background selection information");
        file_iter_init = true; /*file selection iteration info has been initialized */

        if (H5S_select_iter_init(bkg_iter, flex_fspace.vp, file_type_size, H5S_SEL_ITER_GET_SEQ_LIST_SORTED) <
            0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL,
                        "unable to initialize background selection information");
        bkg_iter_init = true; /*file selection iteration info has been initialized */

        /* Start strip mining... */
        for (smine_start = 0; smine_start < nelmts; smine_start += smine_nelmts) {
            size_t n; /* Elements operated on */

            /* Determine strip mine size. First check if we're doing in-place type conversion */
            if (in_place_tconv) {
                /* If there is a background buffer, we cannot exceed request_nelmts.  */
                assert(!H5D__SCATGATH_USE_CMPD_OPT_WRITE(dset_info, in_place_tconv));
                if (dset_info->type_info.need_bkg)
                    smine_nelmts = (size_t)MIN(dset_info->type_info.request_nelmts, (nelmts - smine_start));
                else {
                    assert(smine_start == 0);
                    smine_nelmts = nelmts;
                }

                /* Calculate buffer position in user buffer */
                /* Use "vp" field of union to twiddle away const.  OK because if we're doing this it means the
                 * user explicitly allowed us to modify this buffer via H5Pset_modify_write_buf(). */
                tmp_buf = (uint8_t *)dset_info->buf.vp + buf_off + (smine_start * mem_type_size);
            }
            else {
                /*
                 * Chunk-local VL conversion uses its own nonzero strip capacity.
                 * The normal request count may be zero when H5D classified the
                 * logical datatype conversion as a no-op.
                 */
                if (has_vlen_type) {
                    tmp_buf      = vl_tconv_buf;
                    smine_nelmts = MIN(vl_strip_nelmts, selected_nelmts - (size_t)smine_start);
                }
                else {
                    tmp_buf      = io_type_info->tconv_buf;
                    smine_nelmts = (size_t)MIN(dset_info->type_info.request_nelmts, nelmts - smine_start);
                }

                n = H5D__gather_mem(buf, mem_iter, smine_nelmts, tmp_buf /*out*/);

                if (n != smine_nelmts)
                    HGOTO_ERROR(H5E_IO, H5E_WRITEERROR, FAIL, "mem gather failed");
            }

            /* If the source and destination are compound types and the destination is
             * is a subset of the source and no conversion is needed, copy the data
             * directly from user's buffer and bypass the rest of steps.  If the source
             * is a subset of the destination, the optimization is done in conversion
             * function H5T_conv_struct_opt to protect the background data.
             */
            if (!has_vlen_type && H5D__SCATGATH_USE_CMPD_OPT_WRITE(dset_info, in_place_tconv)) {
                if (H5D__compound_opt_write(smine_nelmts, &dset_info->type_info, tmp_buf) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "datatype conversion failed");

            } /* end if */
            else {
                /*
                 * Chunk-local VL replacement always needs the previous file
                 * descriptors. Their delete callbacks release any H5HG objects
                 * being replaced, even when ordinary H5T conversion does not
                 * request a background buffer.
                 */
                if (has_vlen_type) {
                    n = H5D__gather_mem(data_scat_buf, bkg_iter, smine_nelmts, vl_bkg_buf /*out*/);
                    if (n != smine_nelmts)
                        HGOTO_ERROR(H5E_IO, H5E_READERROR, FAIL,
                                    "unable to gather VL background descriptors");
                }
                else if (H5T_BKG_YES == dset_info->type_info.need_bkg) {
                    n = H5D__gather_mem(data_scat_buf, bkg_iter, smine_nelmts, io_type_info->bkg_buf /*out*/);
                    if (n != smine_nelmts)
                        HGOTO_ERROR(H5E_IO, H5E_READERROR, FAIL, "file gather failed");
                }

                /* Do the data transform before the type conversion (since
                 * transforms must be done in the memory type). */
                if (!dset_info->type_info.is_xform_noop) {
                    H5Z_data_xform_t *data_transform; /* Data transform info */

                    /* Retrieve info from API context */
                    if (H5CX_get_data_transform(&data_transform) < 0)
                        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't get data transform info");

                    if (H5Z_xform_eval(data_transform, tmp_buf, smine_nelmts, dset_info->type_info.mem_type) <
                        0)
                        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "Error performing data transform");
                }

                /*
                 * Perform datatype conversion.
                 */
                if (has_vlen_type) {
                    if (H5D__struct_chunk_vlen_convert(&vl_ctx, chunk_tpath, dset_info->type_info.src_type,
                                                       chunk_file_type, smine_nelmts, tmp_buf,
                                                       vl_bkg_buf) < 0)
                        HGOTO_ERROR(H5E_DATASET, H5E_CANTCONVERT, FAIL,
                                    "chunk-local VL datatype conversion failed");
                }
                else if (H5T_convert(dset_info->type_info.tpath, dset_info->type_info.src_type,
                                     dset_info->type_info.dst_type, smine_nelmts, (size_t)0, (size_t)0,
                                     tmp_buf, io_type_info->bkg_buf) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTCONVERT, FAIL, "datatype conversion failed");
            } /* end else */

            /*
             * Scatter the data out to the data_scat_buffer.
             */
            if (H5D__scatter_mem(tmp_buf, file_iter, smine_nelmts, data_scat_buf /*out*/) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "scatter failed");

        } /* end for */

    } /* end if */

    /*
     * Build the replacement selection without changing the resident chunk.
     * Keep the existing ALL-selection behavior.
     */
    if (chk->sel_space) {
        if (H5S_GET_SELECT_TYPE(chk->sel_space) == H5S_SEL_ALL) {
            if (NULL == (staged_sel_space = H5S_copy(chk->sel_space, false, true)))
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy chunk selection");
        }
        else {
            if (NULL ==
                (staged_sel_space = H5S__combine_select(chk->sel_space, H5S_SELECT_OR, flex_fspace.vp)))
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTINIT, FAIL, "unable to combine chunk selections");
        }
    }
    else {
        if (NULL == (staged_sel_space = H5S_copy(file_space, false, true)))
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy write selection");
    }

    /*
     * Gather all resulting fixed records into a privately owned buffer.
     */
    {
        hsize_t sel_nelmts;
        size_t  staged_nelmts;
        size_t  gathered_nelmts;

        sel_nelmts = H5S_GET_SELECT_NPOINTS(staged_sel_space);

        H5_CHECK_OVERFLOW(sel_nelmts, hsize_t, size_t);
        staged_nelmts = (size_t)sel_nelmts;

        if (file_type_size == 0 || staged_nelmts > SIZE_MAX / file_type_size)
            HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "staged chunk data size overflow");

        staged_data_size = staged_nelmts * file_type_size;

        /*
         * A nonempty write must produce a nonempty resulting selection.
         */
        if (staged_nelmts == 0)
            HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "nonempty write produced an empty chunk selection");

        if (NULL == (staged_data_buf = H5MM_malloc(staged_data_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate staged chunk data");

        if (NULL == (sel_iter = H5FL_MALLOC(H5S_sel_iter_t)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "unable to allocate chunk selection iterator");

        if (H5S_select_iter_init(sel_iter, staged_sel_space, file_type_size,
                                 H5S_SEL_ITER_GET_SEQ_LIST_SORTED) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize chunk selection iterator");

        sel_iter_init = true;

        gathered_nelmts = H5D__gather_mem(data_scat_buf, sel_iter, staged_nelmts, staged_data_buf);

        if (gathered_nelmts != staged_nelmts)
            HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "unable to gather staged chunk data");

        sel_iter_init = false;

        if (H5S_SELECT_ITER_RELEASE(sel_iter) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "unable to release chunk selection iterator");

        sel_iter = H5FL_FREE(H5S_sel_iter_t, sel_iter);
    }

    /*
     * H5HG retains the outer manager when its final object is removed.
     * Normalize an empty staged heap set before publication.
     */
    if (has_vlen_type && staged_heapset) {
        if ((heapset_empty = H5HG__is_empty_local_heapset(staged_heapset)) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTGET, FAIL, "unable to query staged heap-set emptiness");

        if (heapset_empty) {
            H5HG_local_heapset_t *empty_heapset = staged_heapset;

            staged_heapset = NULL;

            if (H5HG__free_local_heapset(empty_heapset) < 0)
                HGOTO_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free empty staged heap set");
        }
    }

    /*
     * Calculate every reported size before changing resident state.
     * The accounting view borrows pointers; it owns no allocations.
     */
    accounting_chk = *chk;

    accounting_chk.sel_space       = staged_sel_space;
    accounting_chk.data_buf        = staged_data_buf;
    accounting_chk.data_nbytes     = staged_data_size;
    accounting_chk.data_alloc_size = staged_data_size;

    if (has_vlen_type) {
        accounting_chk.vl_heapset = staged_heapset;
    }

    if (accounting_chk.sel_nbytes > (SIZE_MAX - accounting_chk.data_nbytes))
        HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "structured chunk logical size overflow");

    new_nbytes = accounting_chk.sel_nbytes + accounting_chk.data_nbytes;

    if (accounting_chk.sel_alloc_size > SIZE_MAX - accounting_chk.data_alloc_size)
        HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "structured chunk buffer allocation overflow");

    new_buffer_alloc = accounting_chk.sel_alloc_size + accounting_chk.data_alloc_size;

    if (H5D__struct_chunk_get_alloc_size(&accounting_chk, &new_total_alloc) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to calculate staged chunk allocation");

    /*
     * Commit the matching selection, fixed records, and heap set.
     * No fallible construction or accounting remains in this block.
     */
    old_sel_space = chk->sel_space;
    old_data_buf  = chk->data_buf;

    chk->sel_space       = staged_sel_space;
    chk->data_buf        = staged_data_buf;
    chk->data_nbytes     = staged_data_size;
    chk->data_alloc_size = staged_data_size;

    staged_sel_space = NULL;
    staged_data_buf  = NULL;

    if (has_vlen_type) {
        old_heapset     = chk->vl_heapset;
        chk->vl_heapset = staged_heapset;
        staged_heapset  = NULL;
    }

    /*
     * Return sizes of the replacement state, rather than adding another
     * copy of the chunk's sizes to the caller's previous values.
     */
    *nbytes           = new_nbytes;
    *alloc_size       = new_buffer_alloc;
    *alloc_size_total = new_total_alloc;

done:
    if (chunk_file_type) {
        if (H5T_close_real(chunk_file_type) < 0)
            HDONE_ERROR(H5E_DATATYPE, H5E_CANTCLOSEOBJ, FAIL, "unable to close chunk-local VL file datatype");
    }

    /* Release selection iterators */
    if (mem_iter_init && H5S_SELECT_ITER_RELEASE(mem_iter) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "Can't release selection iterator");
    if (mem_iter)
        mem_iter = H5FL_FREE(H5S_sel_iter_t, mem_iter);

    if (file_iter_init && H5S_SELECT_ITER_RELEASE(file_iter) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "Can't release selection iterator");
    if (file_iter)
        file_iter = H5FL_FREE(H5S_sel_iter_t, file_iter);

    if (bkg_iter_init && H5S_SELECT_ITER_RELEASE(bkg_iter) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "Can't release selection iterator");
    if (bkg_iter)
        bkg_iter = H5FL_FREE(H5S_sel_iter_t, bkg_iter);

    if (sel_iter_init && H5S_SELECT_ITER_RELEASE(sel_iter) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "unable to release chunk selection iterator");

    if (sel_iter)
        sel_iter = H5FL_FREE(H5S_sel_iter_t, sel_iter);

    if (data_scat_buf)
        data_scat_buf = H5FL_BLK_FREE(scat_buf, data_scat_buf);

    if (packed_buf)
        packed_buf = H5MM_xfree(packed_buf);

    if (vl_tconv_buf)
        vl_tconv_buf = H5MM_xfree(vl_tconv_buf);

    if (vl_bkg_buf)
        vl_bkg_buf = H5MM_xfree(vl_bkg_buf);

    if (staged_heapset)
        if (H5HG__free_local_heapset(staged_heapset) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free staged chunk-local VL heap set");

    if (old_heapset)
        if (H5HG__free_local_heapset(old_heapset) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free replaced chunk-local VL heap set");

    if (staged_sel_space && H5S_close(staged_sel_space) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to release staged chunk selection");

    if (old_sel_space && H5S_close(old_sel_space) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to release replaced chunk selection");

    staged_data_buf = H5MM_xfree(staged_data_buf);
    old_data_buf    = H5MM_xfree(old_data_buf);

    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_gather_mem() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_fill
 *
 * Purpose:     Propagates the fill value into the selected elements of the chunk buffer,
 *              performing type conversion if necessary.
 *
 *              space's extent matches the chunk dimensions and the selection is
 *              within the chunk.
 *
 *              Optional, if not present, chunk is the same in memory as it is in cache,
 *              with the exception of type conversion (which will be handled
 *              by H5SC layer).
 *
 *              If the layout stores variable length data within the chunk
 *              this callback must be defined.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 * NOTE: chunk is pointer to the chunk intermediate struct
 *
 * NOTE: [io_type_info] not used??
 * NOTE: [udata] not used??
 *
 * Updated:    Adds serial support for filling structured chunks containing
 *             variable-length data.
 *
 *             For a VL datatype, the fixed-size descriptor buffer, selection,
 *             and chunk-local heap set are constructed as staged replacement
 *             state. They are installed in the chunk only after all fill
 *             elements have been converted successfully. If construction
 *             fails, the staged state is released and the existing chunk state
 *             remains intact.
 *
 *             Each non-empty VL fill element is converted independently so
 *             that every descriptor owns a distinct chunk-local heap object.
 *             The chunk-local heap allocation is also included in the SCC
 *
 *                                          - AZO     09/16/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_fill(H5D_dset_io_info_t *dset_info, H5D_io_type_info_t H5_ATTR_UNUSED *io_type_info,
                       H5S_t *space, size_t *nbytes /*in,out*/, size_t *alloc_size /*in,out*/,
                       size_t *alloc_size_total /*in,out*/, void *chunk, void H5_ATTR_UNUSED *udata)
{
    const H5O_fill_t      *fill = &(dset_info->dset->shared->dcpl_cache.fill); /* Fill value info */
    H5D_chunk_cache_mem_t *chk  = (H5D_chunk_cache_mem_t *)chunk;              /* Chunk's memory cache info */
    uint8_t                elmt_buf[H5T_ELEM_BUF_SIZE];                        /* Buffer for element data */
    uint8_t                bkg_elmt_buf[H5T_ELEM_BUF_SIZE]; /* Buffer for background data */
    H5T_t                 *chunk_file_type = NULL;  /* Chunk-file datatype using local VL references */
    H5T_path_t            *chunk_tpath     = NULL;  /* Conversion path to the chunk-file datatype */
    H5T_vlen_chunk_ctx_t   vl_ctx;                  /* Context used to create chunk-local VL objects */
    H5HG_local_heapset_t  *staged_heapset   = NULL; /* New VL heap set retained until conversion succeeds */
    H5HG_local_heapset_t  *old_heapset      = NULL; /* Replaced VL heap set awaiting release */
    H5S_t                 *staged_sel_space = NULL; /* New selection retained until conversion succeeds */
    H5S_t                 *old_sel_space    = NULL; /* Replaced selection awaiting release */
    void   *staged_data_buf = NULL; /* New descriptor buffer retained until conversion succeeds */
    void   *old_data_buf    = NULL; /* Replaced descriptor buffer awaiting release */
    size_t  ref_nbytes      = 0;    /* Encoded size of one chunk-local VL reference */
    size_t  buf_size;               /* Maximum source or destination element size */
    size_t  src_type_size;          /* Size of one source fill element */
    size_t  dst_type_size;          /* Size of one destination descriptor element */
    size_t  tot_buf_size;           /* Total allocation required for the data buffer */
    size_t  u;                      /* Fill element index */
    htri_t  has_vlen_type;          /* Whether the destination datatype contains VL data */
    hsize_t nelmts;                 /* Number of selected fill elements */
    herr_t  ret_value = SUCCEED;    /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity check */
    /* Check args */
    assert(space);

    src_type_size = dset_info->type_info.src_type_size;
    dst_type_size = dset_info->type_info.dst_type_size;

    buf_size = MAX(src_type_size, dst_type_size);

    /* Detect whether the datatype has a VL component */
    if ((has_vlen_type = H5T_detect_class(dset_info->dset->shared->type, H5T_VLEN, false)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "unable to detect vlen datatypes?");

    nelmts = H5S_GET_SELECT_NPOINTS(space);
    H5_CHECK_OVERFLOW(nelmts, hsize_t, size_t);

    if (has_vlen_type) {
        /*
         * The resident buffer always stores dataset-file records.
         *
         * During a read, type_info.dst_type_size describes the application's
         * memory datatype, which may be smaller than the file datatype.
         * Using that size here would underallocate the default-fill buffer
         * before scatter reads it using file-sized records.
         */
        if (0 == (dst_type_size = H5T_get_size(dset_info->dset->shared->type)))
            HGOTO_ERROR(H5E_DATATYPE, H5E_BADSIZE, FAIL,
                        "invalid file datatype size for structured chunk fill");

        if ((size_t)nelmts > (SIZE_MAX / dst_type_size))
            HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "structured chunk VL fill buffer size overflow");

        tot_buf_size = nelmts * dst_type_size;

        /*
         * Keep the new descriptor buffer separate from CHK until conversion
         * succeeds. On failure, the current chunk remains unchanged.
         */
        if (NULL == (staged_data_buf = H5MM_malloc(tot_buf_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                        "memory allocation failed for staged VL fill buffer");

        if (NULL == fill->buf) {
            /*
             * A zero VL descriptor represents an empty value and requires no
             * chunk-local heap object.
             */
            memset(staged_data_buf, 0, tot_buf_size);
        }
        else {
            if (buf_size > H5T_ELEM_BUF_SIZE)
                HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL,
                            "VL fill element exceeds conversion buffer size");

            if (H5D__struct_chunk_prepare_vlen_type(dset_info->dset, dset_info->type_info.dst_type,
                                                    dset_info->type_info.src_type, false, &chunk_file_type,
                                                    &chunk_tpath, &ref_nbytes) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL,
                            "unable to prepare chunk-local VL fill conversion");

            memset(&vl_ctx, 0, sizeof(vl_ctx));
            vl_ctx.f          = dset_info->dset->oloc.file;
            vl_ctx.heapset    = &staged_heapset;
            vl_ctx.ref_nbytes = ref_nbytes;

            /*
             * Convert each copy independently. Reusing one converted
             * descriptor would make several elements reference the same
             * uncounted heap object.
             */
            for (u = 0; u < (size_t)nelmts; u++) {

                memset(elmt_buf, 0, sizeof(elmt_buf));
                memset(bkg_elmt_buf, 0, sizeof(bkg_elmt_buf));

                H5MM_memcpy(elmt_buf, fill->buf, src_type_size);

                if (H5D__struct_chunk_vlen_convert(&vl_ctx, chunk_tpath, dset_info->type_info.src_type,
                                                   chunk_file_type, (size_t)1, elmt_buf, bkg_elmt_buf) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTCONVERT, FAIL, "chunk-local VL fill conversion failed");

                H5MM_memcpy((uint8_t *)staged_data_buf + (u * dst_type_size), elmt_buf, dst_type_size);
            }
        } /* end else */

        /*
         * Stage the matching selection as well. The selection, descriptor
         * buffer, and heap set describe one logical chunk state and must be
         * published together.
         */
        if (NULL == (staged_sel_space = H5S_copy(space, false, true)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to copy structured-chunk fill selection");

        /*
         * Save the old state for release after the new VL state has been
         * installed. No fallible construction remains before publication.
         */
        old_sel_space = chk->sel_space;
        old_data_buf  = chk->data_buf;
        old_heapset   = chk->vl_heapset;

        chk->sel_space       = staged_sel_space;
        chk->data_buf        = staged_data_buf;
        chk->vl_heapset      = staged_heapset;
        chk->data_nbytes     = tot_buf_size;
        chk->data_alloc_size = tot_buf_size;

        staged_sel_space = NULL;
        staged_data_buf  = NULL;
        staged_heapset   = NULL;

    } /* end if has_vlen */
    else {
        /*
         * Preserve the original fixed-size fill behavior. BUF_SIZE is used as
         * the conversion element width because it is the larger of the source
         * and destination datatype sizes.
         */

        tot_buf_size = nelmts * buf_size;

        if (NULL == (chk->data_buf = H5MM_realloc(chk->data_buf, tot_buf_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "memory reallocation failed for data buffer");

        if (fill->buf == NULL)
            memset(chk->data_buf, 0, tot_buf_size);
        else if (!has_vlen_type) { /* has fill value && not handling VL type yet */

            void *elmt_ptr = elmt_buf;     /* Pointer to element to use for fill value */
            void *bkg_ptr  = bkg_elmt_buf; /* Pointer to element to use for fill value */

            /* Copy the fill value to the buffer for conversion */
            H5MM_memcpy(elmt_ptr, fill->buf, buf_size);

            /* Perform datatype conversion */
            if (H5T_convert(dset_info->type_info.tpath, dset_info->type_info.src_type,
                            dset_info->type_info.dst_type, (size_t)1, (size_t)0, (size_t)0, elmt_ptr,
                            bkg_ptr) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTCONVERT, FAIL, "data type conversion failed");

            /* Replicate the fill value into the chunk buffer */
            H5VM_array_fill(chk->data_buf, elmt_ptr, buf_size, (size_t)nelmts);
        }

        if (chk->sel_space) {
            if (H5S_close(chk->sel_space) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release dataspace");

            chk->sel_space = NULL;
        }

        if (NULL == (chk->sel_space = H5S_copy(space, false, true)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to copy structured-chunk fill selection");

        chk->data_nbytes     = tot_buf_size;
        chk->data_alloc_size = tot_buf_size;

    } /* end else */

    /* Report the fixed descriptor/data allocation updated by this callback. */
    *nbytes += chk->data_nbytes;
    *alloc_size += chk->data_alloc_size;

    /*
     * Recalculate the complete SCC-resident allocation, including the
     * chunk-local VL heap set.
     */
    if (H5D__struct_chunk_get_alloc_size(chk, alloc_size_total) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL,
                    "unable to determine structured chunk resident allocation");

    /*
     * The replacement is now complete. Release the old VL chunk state that
     * was retained for rollback during construction.
     */
    if (old_sel_space) {
        if (H5S_close(old_sel_space) < 0)
            HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                        "unable to release replaced structured-chunk selection");

        old_sel_space = NULL;
    }

    old_data_buf = H5MM_xfree(old_data_buf);

    if (old_heapset) {
        if (H5HG__free_local_heapset(old_heapset) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to release replaced chunk-local VL heap set");

        old_heapset = NULL;
    }

done:
    if (chunk_file_type) {

        if (H5T_close_real(chunk_file_type) < 0)
            HDONE_ERROR(H5E_DATATYPE, H5E_CANTCLOSEOBJ, FAIL, "unable to close chunk-local VL fill datatype");
    }

    /*
     * These objects remain non-NULL only when VL construction failed before
     * publication.
     */
    if (staged_sel_space) {

        if (H5S_close(staged_sel_space) < 0)
            HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                        "unable to release staged structured-chunk selection");
    }

    if (staged_data_buf) {
        staged_data_buf = H5MM_xfree(staged_data_buf);
    }

    if (staged_heapset) {
        if (H5HG__free_local_heapset(staged_heapset) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to release staged chunk-local VL heap set");
    }

    /*
     * These remain non-NULL only if an error occurred after the new state was
     * published but before the replaced state was completely released.
     */
    if (old_sel_space && H5S_close(old_sel_space) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                    "unable to release replaced structured-chunk selection");

    if (old_data_buf) {
        old_data_buf = H5MM_xfree(old_data_buf);
    }

    if (old_heapset) {

        if (H5HG__free_local_heapset(old_heapset) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to release replaced chunk-local VL heap set");
    }

    FUNC_LEAVE_NOAPI(ret_value)

} /* H5D__struct_chunk_fill() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_defined values
 *
 * Purpose:     Queries the defined elements in the chunk.
 *              Selection may be passed as H5S_ALL.
 *              These selections are within the logical chunk.
 *
 *              Optional, if not present, all values are defined.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 * NOTE: [chunk] is pointer to the chunk intermediate struct
 *
 * NOTE: [udata] not used??
 *-------------------------------------------------------------------------
 */

static herr_t
H5D__struct_chunk_defined_values(H5D_t *dset, const H5S_t *selection, void *chunk,
                                 H5S_t **defined_values /*out*/, void H5_ATTR_UNUSED *udata)
{
    H5D_chunk_cache_mem_t *chk       = (H5D_chunk_cache_mem_t *)chunk; /* Chunk's memory cache info */
    H5S_t                 *tmp_space = NULL;
    H5S_sel_type           sel_type;
    H5S_sel_type           def_type;
    herr_t                 ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);
    assert(selection);
    assert(chunk);
    assert(defined_values);
    assert(chk->sel_space);

    *defined_values = NULL;

    if ((sel_type = H5S_GET_SELECT_TYPE(selection)) < H5S_SEL_NONE)
        HGOTO_ERROR(H5E_DATASPACE, H5E_BADSELECT, FAIL, "unable to get input selection type");

    if ((def_type = H5S_GET_SELECT_TYPE(chk->sel_space)) < H5S_SEL_NONE)
        HGOTO_ERROR(H5E_DATASPACE, H5E_BADSELECT, FAIL, "unable to get defined-values selection type");

    if (sel_type == H5S_SEL_ALL) {
        /* Caller wants all defined values in the chunk */
        if (NULL == (*defined_values = H5S_copy(chk->sel_space, false, true)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to copy defined-values dataspace");
    }
    else if (sel_type == H5S_SEL_HYPERSLABS) {
        /*
         * If all values in the chunk are defined, intersection with the
         * caller's hyperslab is just the caller's hyperslab selection.
         */
        if (def_type == H5S_SEL_ALL) {
            if (NULL == (*defined_values = H5S_copy(selection, false, true)))
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy selection dataspace");
        }
        else {
            /*
             * For non-ALL defined-value selections, combine on a copy of the
             * caller's selection.
             */
            if (NULL == (tmp_space = H5S_copy(selection, false, true)))
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy selection dataspace");

            if (NULL == (*defined_values = H5S__combine_select(tmp_space, H5S_SELECT_AND, chk->sel_space)))
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTINIT, FAIL,
                            "unable to intersect selection with defined values");

            if (H5S_close(tmp_space) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                            "unable to close temporary selection dataspace");
            tmp_space = NULL;
        }
    }
    else
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL,
                    "defined-values callback only supports H5S_ALL and hyperslab selections");

done:
    if (tmp_space && H5S_close(tmp_space) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to close temporary selection dataspace");

    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_defined_values() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_erase_values
 *
 * Purpose:     Erases the selected elements in the chunk, causing them to
 *              no longer be defined. If all values in the chunk are erased and
 *              the chunk should be deleted, sets *delete_chunk to true,
 *              causing the cache to delete the chunk from cache, free it in memory
 *              using H5SC_chunk_evict_t, and delete it on disk using H5SC_chunk_delete_t.
 *              These selections are within the logical chunk.
 *
 *              Optional, if not present, the fill value will be written to the selection
 *              using H5SC_chunk_fill_t.
 *
 * Return:      Non-negative on success/Negative on failure
 *
 * NOTE: chunk is pointer to the chunk intermediate struct
 *
 * NOTE: [udata] not used??
 *
 * Updated:    Adds support for erasing structured-chunk elements that
 *             contain variable-length data.
 *
 *             The fixed descriptor buffer and chunk-local VL heap set are
 *             copied before modification. VL payloads referenced by erased
 *             descriptors are recursively removed from the staged heap set,
 *             and the surviving fixed records are compacted in the staged
 *             buffer.
 *
 *             The staged buffer, selection, and heap set replace the existing
 *             chunk state only after deletion, compaction, validation, and
 *             resident-memory accounting succeed. Whole-chunk deletion
 *             continues to rely on normal chunk eviction to release the
 *             complete heap set.
 *
 *                                       -- AZO   09/16/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_erase_values(H5D_t *dset, const H5S_t *selection, size_t *nbytes /*in,out*/,
                               size_t *alloc_size /*in,out*/, void *chunk, bool *delete_chunk /*out*/,
                               void H5_ATTR_UNUSED *udata)
{
    H5D_chunk_cache_mem_t *chk = (H5D_chunk_cache_mem_t *)chunk; /* Chunk memory cache info */
    H5D_chunk_cache_mem_t  accounting_chk;                       /* Temporary staged accounting view */
    void                  *buf = chk->data_buf;        /* Staged fixed-record buffer being compacted */
    H5S_t                 *serial_values_space = NULL; /* Packed coordinate space for defined values */
    H5S_t                 *serial_erase_space  = NULL; /* Erase selection in packed coordinates */
    H5S_t                 *new_space           = NULL; /* Selection containing surviving values */
    H5S_t                 *old_space           = NULL; /* Replaced defined-value selection */

    H5S_sel_iter_t *erase_iter      = NULL;
    bool            erase_iter_init = false;

    hsize_t  chk_nelmts;
    hsize_t  erase_nelmts;
    hsize_t  projected_erase_nelmts;
    hssize_t hss_nelmts;
    size_t   elmt_size;
    size_t   chk_nbytes;
    size_t   dxpl_vec_size;
    size_t   vec_size;
    size_t  *len = NULL;
    hsize_t *off = NULL;
    size_t   nseq;
    size_t   nelem;
    size_t   curr_seq;
    size_t   src_off;
    size_t   dst_off;
    size_t   keep_bytes;
    size_t   prev_end_off;
    size_t   total_erased_bytes = 0;
    size_t   new_nelmts;

    void *staged_data_buf = NULL; /* New data buffer retained until erase succeeds */
    void *old_data_buf    = NULL; /* Replaced data buffer awaiting release */

    H5HG_local_heapset_t *staged_vl_heapset = NULL; /* Heap-set copy modified during VL erase */
    H5HG_local_heapset_t *old_vl_heapset    = NULL; /* Replaced heap set awaiting release */

    H5T_t                      *chunk_file_type = NULL; /* File datatype using local VL references */
    H5T_vlen_chunk_ctx_t        vl_ctx;                 /* Context for chunk-local VL deletion */
    const H5T_vlen_chunk_ctx_t *old_vl_ctx = NULL;      /* Previously active VL context */
    H5_flexible_const_ptr_t     flex_selection;

    bool   vl_ctx_active  = false; /* Whether VL_CTX is currently installed */
    htri_t has_vlen_type  = false; /* Whether the dataset datatype contains VL data */
    htri_t heapset_empty  = false; /* Whether staged deletion removed every VL object */
    size_t ref_nbytes     = 0;     /* Encoded size of one chunk-local VL reference */
    size_t new_alloc_size = 0;     /* Resident allocation after the staged erase */

    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(selection);
    assert(chunk);
    assert(delete_chunk);
    assert(nbytes);
    assert(alloc_size);

    *delete_chunk = false;

    /* Number of currently defined elements in the chunk */
    if ((hss_nelmts = (hssize_t)H5S_GET_SELECT_NPOINTS(chk->sel_space)) < 0)
        HGOTO_ERROR(H5E_VFL, H5E_CANTCOUNT, FAIL, "can't get number of defined elements in chunk");
    H5_CHECKED_ASSIGN(chk_nelmts, hsize_t, hss_nelmts, hssize_t);

    if (0 == (elmt_size = H5T_get_size(dset->shared->type)))
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADSIZE, FAIL, "datatype size invalid");

    chk_nbytes = chk->data_nbytes;

    /* Build a 1D serial dataspace for the packed defined values */
    if (NULL == (serial_values_space = H5S_create_simple(1, &chk_nelmts, NULL)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, FAIL, "unable to create serial values dataspace");

    flex_selection.cvp = selection;
    if (H5S_select_project_intersection(chk->sel_space, serial_values_space, flex_selection.vp,
                                        &serial_erase_space, true) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCLIP, FAIL,
                    "can't project erase selection into packed-value coordinates");

    /*
     * Use the projected erase selection for packed-buffer compaction.  The
     * requested erase selection can include elements that are not currently
     * defined in this sparse chunk.
     */
    if ((hss_nelmts = (hssize_t)H5S_GET_SELECT_NPOINTS(serial_erase_space)) < 0)
        HGOTO_ERROR(H5E_VFL, H5E_CANTCOUNT, FAIL,
                    "can't get number of projected elements selected for erase");
    H5_CHECKED_ASSIGN(projected_erase_nelmts, hsize_t, hss_nelmts, hssize_t);

    if (projected_erase_nelmts == 0)
        HGOTO_DONE(SUCCEED);

    /*
     * Whole-chunk eviction releases the complete chunk-local heap set, so
     * individual VL payloads do not need to be deleted here.
     */
    if (projected_erase_nelmts == chk_nelmts) {
        *delete_chunk = true;
        HGOTO_DONE(SUCCEED);
    }

    /*
     * Determine whether erased fixed records can own chunk-local VL payloads.
     */
    if ((has_vlen_type = H5T_detect_class(dset->shared->type, H5T_VLEN, false)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to determine whether datatype contains VL data");

    if (chk_nbytes > chk->data_alloc_size)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "structured chunk data size exceeds its allocation");

    /*
     * This copies all currently used fixed records even when few values
     * are erased. It keeps compaction off the resident buffer until the
     * replacement selection, records, and heap set are ready to commit.
     */
    if (chk->data_alloc_size > 0) {

        if (NULL == (staged_data_buf = H5MM_malloc(chk->data_alloc_size)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate staged erase data buffer");

        if (chk_nbytes > 0) {
            H5MM_memcpy(staged_data_buf, chk->data_buf, chk_nbytes);
        }
    }

    buf = staged_data_buf;

    /*
     * Copying the complete heap set also preserves payloads belonging to
     * surviving descriptors during staged deletion. This is a correctness
     * choice with a cost proportional to the copied heap set; selective
     * staging is a possible later optimization.
     */
    if (has_vlen_type) {
        if (H5HG__copy_local_heapset(dset->oloc.file, chk->vl_heapset, &staged_vl_heapset) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTCOPY, FAIL, "unable to copy chunk-local VL heap set for erase");
    }

    erase_nelmts = projected_erase_nelmts;

    if (NULL == (erase_iter = H5FL_MALLOC(H5S_sel_iter_t)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate selection iterator");

    if (H5S_select_iter_init(erase_iter, serial_erase_space, elmt_size, H5S_SEL_ITER_GET_SEQ_LIST_SORTED) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to initialize erase selection iterator");
    erase_iter_init = true;

    if (H5CX_get_vec_size(&dxpl_vec_size) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't retrieve I/O vector size");

    vec_size = (dxpl_vec_size > H5D_IO_VECTOR_SIZE) ? dxpl_vec_size : H5D_IO_VECTOR_SIZE;

    if (NULL == (len = H5FL_SEQ_MALLOC(size_t, vec_size)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate I/O length vector array");
    if (NULL == (off = H5FL_SEQ_MALLOC(hsize_t, vec_size)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTALLOC, FAIL, "can't allocate I/O offset vector array");

    /*
     * Build a private disk-located datatype whose VL nodes use the
     * chunk-local H5HG backend. This permits recursive deletion for VL values
     * nested inside arrays, compounds, or other VL sequences.
     */
    if (has_vlen_type) {
        if (H5D__struct_chunk_get_vlen_ref_size(dset, &ref_nbytes) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to determine chunk-local VL reference width");

        if (NULL == (chunk_file_type = H5T_copy(dset->shared->type, H5T_COPY_TRANSIENT)))
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTCOPY, FAIL, "unable to copy file datatype for VL erase");

        if (H5T_set_loc(chunk_file_type, H5F_VOL_OBJ(dset->oloc.file), H5T_LOC_DISK) < 0)
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL, "unable to configure VL erase datatype");

        if (H5T_get_size(chunk_file_type) != elmt_size)
            HGOTO_ERROR(H5E_DATATYPE, H5E_BADSIZE, FAIL,
                        "VL erase datatype has an unexpected file representation size");

        if (H5T_patch_vlen_chunk_local(chunk_file_type, ref_nbytes) < 0)
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL, "unable to install chunk-local VL erase backend");

        memset(&vl_ctx, 0, sizeof(vl_ctx));

        vl_ctx.f          = dset->oloc.file;
        vl_ctx.heapset    = &staged_vl_heapset;
        vl_ctx.ref_nbytes = ref_nbytes;

        old_vl_ctx    = H5T_set_vlen_chunk_ctx(&vl_ctx);
        vl_ctx_active = true;
    }

    /*
     * Compact the packed data buffer by copying surviving byte ranges
     * downward over erased byte ranges.
     */
    prev_end_off = 0;
    dst_off      = 0;

    while (erase_nelmts > 0) {
        if (H5S_SELECT_ITER_GET_SEQ_LIST(erase_iter, vec_size, erase_nelmts, &nseq, &nelem, off, len) < 0)
            HGOTO_ERROR(H5E_INTERNAL, H5E_UNSUPPORTED, FAIL, "sequence length generation failed");

        if (0 == nelem)
            HGOTO_ERROR(H5E_INTERNAL, H5E_CANTGET, FAIL, "erase selection iterator made no progress");

        for (curr_seq = 0; curr_seq < nseq; curr_seq++) {
            size_t seq_offset; /* Offset of one erased element within the sequence */

            H5_CHECKED_ASSIGN(src_off, size_t, off[curr_seq], hsize_t);

            if ((src_off < prev_end_off) || (src_off > chk_nbytes) ||
                (len[curr_seq] > (chk_nbytes - src_off)))
                HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL,
                            "erase sequence is outside the structured chunk data buffer");

            if (len[curr_seq] % elmt_size)
                HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL,
                            "erase sequence is not aligned to datatype elements");

            /*
             * Release every VL payload owned by the erased fixed records
             * before those records are compacted out of the staged buffer.
             */
            if (has_vlen_type) {

                /*
                 * Each removed VL object is passed to the heap backend separately.
                 * H5HG__remove_local() compacts surviving payload bytes after each
                 * removal, so erasing several objects can move heap contents repeatedly.
                 * Batched deletion and compaction are deferred performance work.
                 */
                for (seq_offset = 0; seq_offset < len[curr_seq]; seq_offset += elmt_size) {

                    if (H5T_vlen_delete_file_elmt((uint8_t *)buf + src_off + seq_offset, chunk_file_type) < 0)
                        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL,
                                    "unable to delete erased chunk-local VL value");
                }
            }

            /* Copy surviving bytes before this erased sequence. */
            if (src_off > prev_end_off) {
                keep_bytes = src_off - prev_end_off;
                if (dst_off != prev_end_off)
                    memmove((uint8_t *)buf + dst_off, (uint8_t *)buf + prev_end_off, keep_bytes);
                dst_off += keep_bytes;
            }

            /* Skip this erased sequence */
            prev_end_off = src_off + len[curr_seq];

            if (len[curr_seq] > SIZE_MAX - total_erased_bytes)
                HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "erased byte count overflows size_t");

            total_erased_bytes += len[curr_seq];

        } /* end for */

        if ((hsize_t)nelem > erase_nelmts)
            HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "erase iterator returned too many elements");

        erase_nelmts -= nelem;
    } /* end while */

    if (vl_ctx_active) {
        H5T_set_vlen_chunk_ctx(old_vl_ctx);
        vl_ctx_active = false;
    }

    /*
     * Member heaps are freed as their final objects are removed, but H5HG
     * deliberately retains the outer heap-set manager. Do not retain that
     * empty manager as part of the resident chunk state.
     */
    if (has_vlen_type && staged_vl_heapset) {
        if ((heapset_empty = H5HG__is_empty_local_heapset(staged_vl_heapset)) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTGET, FAIL,
                        "unable to determine whether staged VL heap set is empty");

        if (heapset_empty) {
            H5HG_local_heapset_t *empty_heapset = staged_vl_heapset;

            staged_vl_heapset = NULL;

            if (H5HG__free_local_heapset(empty_heapset) < 0)
                HGOTO_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to release empty staged VL heap set");
        }
    }

    /* Copy trailing surviving bytes after the last erased sequence */
    if (prev_end_off < chk_nbytes) {
        keep_bytes = chk_nbytes - prev_end_off;
        if (dst_off != prev_end_off)
            memmove((uint8_t *)buf + dst_off, (uint8_t *)buf + prev_end_off, keep_bytes);
        dst_off += keep_bytes;
    }

    /* Zero-fill the remainder of the buffer */
    if (dst_off < chk_nbytes)
        memset((uint8_t *)buf + dst_off, 0, chk_nbytes - dst_off);

    /* chk->data_nbytes -= total_erased_bytes; */

    {
        H5S_t  *full_chunk_space = NULL;
        H5S_t  *remain_space     = NULL;
        H5S_t  *intersect_base   = NULL;
        int     ndims;
        hsize_t dims[H5S_MAX_RANK];
        hsize_t start[H5S_MAX_RANK];
        hsize_t count[H5S_MAX_RANK];

        ndims = H5S_GET_EXTENT_NDIMS(chk->sel_space);
        if (ndims < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "unable to get chunk selection rank");

        if (H5S_get_simple_extent_dims(chk->sel_space, dims, NULL) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "unable to get chunk selection extent");

        memset(start, 0, sizeof(start));
        H5MM_memcpy(count, dims, sizeof(hsize_t) * (size_t)ndims);

        /*
         * Build an explicit hyperslab selection covering the full chunk extent.
         * This gives us a hyperslab-backed space to subtract from safely.
         */
        if (NULL == (full_chunk_space = H5S_create_simple((unsigned)ndims, dims, NULL)))
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCREATE, FAIL, "unable to create full chunk dataspace");

        if (H5S_select_hyperslab(full_chunk_space, H5S_SELECT_SET, start, NULL, count, NULL) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSELECT, FAIL, "unable to select full chunk hyperslab");

        /*
         * remain_space = full_chunk_space minus selection
         */
        if (NULL ==
            (remain_space = H5S__combine_select(full_chunk_space, H5S_SELECT_NOTB, flex_selection.vp)))
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTINIT, FAIL, "unable to compute remaining chunk selection");

        /*
         * Intersect the remaining region with the currently defined values.
         * This avoids assuming chk->sel_space itself is hyperslab-backed.
         */

        /* intersect_base must have the same selected-point count as chk->sel_space */
        if (NULL == (intersect_base = H5S_copy(chk->sel_space, false, false)))
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy chunk defined-value selection");

        if (H5S_select_project_intersection(chk->sel_space, intersect_base, remain_space, &new_space, true) <
            0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTINIT, FAIL,
                        "unable to intersect remaining region with defined values");

        if (H5S_close(intersect_base) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to close intersection base dataspace");
        intersect_base = NULL;

        if (H5S_close(remain_space) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                        "unable to close remaining selection dataspace");
        remain_space = NULL;

        if (H5S_close(full_chunk_space) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to close full chunk dataspace");
        full_chunk_space = NULL;
    }

    new_nelmts = (size_t)H5S_GET_SELECT_NPOINTS(new_space);
    assert((chk_nbytes - total_erased_bytes) == (new_nelmts * elmt_size));

    /*
     * Calculate SCC resident allocation against the staged heap set before
     * publishing any part of the replacement state.
     */
    accounting_chk = *chk;

    if (has_vlen_type)
        accounting_chk.vl_heapset = staged_vl_heapset;

    if (H5D__struct_chunk_get_alloc_size(&accounting_chk, &new_alloc_size) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL,
                    "unable to determine structured chunk allocation after erase");

    /*
     * Commit the new selection, compacted fixed records, and VL heap set as
     * one coherent structured-chunk state.
     */
    old_space    = chk->sel_space;
    old_data_buf = chk->data_buf;

    chk->sel_space   = new_space;
    chk->data_buf    = staged_data_buf;
    chk->data_nbytes = chk_nbytes - total_erased_bytes;

    new_space       = NULL;
    staged_data_buf = NULL;

    if (has_vlen_type) {
        old_vl_heapset    = chk->vl_heapset;
        chk->vl_heapset   = staged_vl_heapset;
        staged_vl_heapset = NULL;
    }

    *nbytes     = chk->data_nbytes;
    *alloc_size = new_alloc_size;

    /*
     * The replacement state is now resident. Release the old state retained
     * during staged construction.
     */
    if (old_space) {
        if (H5S_close(old_space) < 0)
            HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                        "unable to release replaced defined-value selection");

        old_space = NULL;
    }

    old_data_buf = H5MM_xfree(old_data_buf);

    if (old_vl_heapset) {
        if (H5HG__free_local_heapset(old_vl_heapset) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to release replaced chunk-local VL heap set");

        old_vl_heapset = NULL;
    }

done:
    if (vl_ctx_active) {
        H5T_set_vlen_chunk_ctx(old_vl_ctx);
    }

    if (chunk_file_type) {

        if (H5T_close_real(chunk_file_type) < 0)
            HDONE_ERROR(H5E_DATATYPE, H5E_CANTCLOSEOBJ, FAIL,
                        "unable to close chunk-local VL erase datatype");
    }

    if (old_space && H5S_close(old_space) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to release replaced selection");

    if (old_data_buf) {
        old_data_buf = H5MM_xfree(old_data_buf);
    }

    if (old_vl_heapset) {

        if (H5HG__free_local_heapset(old_vl_heapset) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to release replaced chunk-local VL heap set");
    }

    if (new_space && H5S_close(new_space) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "can't release new selection dataspace");
    if (erase_iter_init && H5S_SELECT_ITER_RELEASE(erase_iter) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "can't release selection iterator");
    if (erase_iter)
        erase_iter = H5FL_FREE(H5S_sel_iter_t, erase_iter);
    if (serial_values_space && H5S_close(serial_values_space) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "can't release serial values dataspace");
    if (serial_erase_space && H5S_close(serial_erase_space) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "can't release serial erase dataspace");
    if (len)
        len = H5FL_SEQ_FREE(size_t, len);
    if (off)
        off = H5FL_SEQ_FREE(hsize_t, off);

    /*
     * Non-NULL staged objects were never published, so failure cleanup can
     * release them without changing the resident chunk.
     */
    if (staged_data_buf) {
        staged_data_buf = H5MM_xfree(staged_data_buf);
    }

    if (staged_vl_heapset) {

        if (H5HG__free_local_heapset(staged_vl_heapset) < 0)
            HDONE_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to release staged erase heap set");
    }
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_erase_values() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_evict_values
 *
 * Purpose:     Frees the data values in the cached chunk and memory used by them
 *              (but does not reallocate - see H5SC_chunk_condense_t),
 *              but leaves the defined values intact.
 *
 *              Optional, if not present the entire chunk will be evicted.
 *
 * Return:      Non-negative on success/Negative on failure
 *
 * NOTE: chunk is pointer to the chunk intermediate struct
 *
 * NOTE: [udata] not used??
 *
 * NOTE:        This callback implements value-only eviction while retaining
 *              defined-value metadata. Registration in the callback table
 *              does not establish that SCC currently invokes this path;
 *              verify SCC wiring and test coverage before describing it
 *              as exercised.
 *
 * Updated:     For structured chunks containing variable-length data, the
 *              decoded chunk also owns a chunk-local H5HG heap set containing
 *              the resident VL payloads. Value eviction now obtains the heap
 *              set's cached allocation size, validates the SCC byte and
 *              allocation accounting, and frees the heap set before freeing
 *              the fixed-data buffer containing descriptors that reference it.
 *
 *              NBYTES is reduced only by DATA_NBYTES because decoded VL
 *              payloads are not part of the packed logical section size.
 *              ALLOC_SIZE is reduced by both DATA_ALLOC_SIZE and the resident
 *              allocation owned by the chunk-local VL heap set.
 *
 *                                              -- AZO   9/18/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_evict_values(H5D_t *dset, size_t *nbytes /*in,out*/, size_t *alloc_size /*in,out*/,
                               void *chunk, void H5_ATTR_UNUSED *udata)
{
    H5D_chunk_cache_mem_t *chk                 = (H5D_chunk_cache_mem_t *)chunk; /* Chunk memory cache info */
    size_t                 old_data_nbytes     = 0;
    size_t                 old_data_alloc_size = 0;
    size_t                 vl_alloc_size       = 0;
    herr_t                 ret_value           = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(dset);
    assert(nbytes);
    assert(alloc_size);
    assert(chunk);

    old_data_nbytes     = chk->data_nbytes;
    old_data_alloc_size = chk->data_alloc_size;

    /*
     * Obtain the VL allocation before freeing the heap set. This is an O(1)
     * cached query and is needed so SCC's resident allocation remains exact.
     */
    if (H5HG__get_local_heapset_alloc_size(chk->vl_heapset, &vl_alloc_size) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to get chunk-local VL allocation size");

    /*
     * Validate all accounting before changing the resident chunk.
     */
    if (*nbytes < old_data_nbytes)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "structured chunk resident byte count is inconsistent");

    if (*alloc_size < old_data_alloc_size)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL,
                    "structured chunk allocation accounting is inconsistent");

    if (vl_alloc_size > (*alloc_size - old_data_alloc_size))
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL,
                    "structured chunk VL allocation accounting is inconsistent");

    /*
     * Release the chunk-local payload storage before clearing the fixed
     * descriptors that reference it.
     */
    if (chk->vl_heapset) {
        if (H5HG__free_local_heapset(chk->vl_heapset) < 0)
            HGOTO_ERROR(H5E_HEAP, H5E_CANTFREE, FAIL, "unable to free chunk-local VL heap set");

        chk->vl_heapset = NULL;
    }

    /* Release the packed fixed-value / descriptor buffer. */
    chk->data_buf        = H5MM_xfree(chk->data_buf);
    chk->data_nbytes     = 0;
    chk->data_alloc_size = 0;

    /*
     * NBYTES intentionally excludes decoded H5HG allocations. They are
     * resident allocation rather than packed logical section bytes.
     */
    *nbytes -= old_data_nbytes;

    /*
     * ALLOC_SIZE includes both the packed fixed buffer and the decoded VL
     * heap set.
     */
    *alloc_size -= old_data_alloc_size;
    *alloc_size -= vl_alloc_size;

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* H5D__struct_chunk_evict_values() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_layout_query
 *
 * Purpose:     Queries data about the dataset from the layout client.
 *              The callback shall set the chunk dimensions in the chunk_dims array
 *              (the number of dimensions is the same as the rank of the dataset),
 *              whether encoding and decoding is necessary for chunks between cache
 *              and disk, and shall set whether chunks that are partially outside the bounds
 *              of the dataset are encoded differently (for example, they may not have
 *              filters applied).
 *              If *partial_bound_chunks_different_encoding is set to true,
 *              then chunks whose partial bound state changes will be re-encoded and
 *              re-inserted as necessary after the dataset extent changes to ensure
 *              they are encoded appropriately.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_layout_query(H5D_t *dset, hsize_t *chunk_dims, bool *encode_decode_necessary,
                               bool *partial_bound_chunks_different_encoding)
{
    herr_t ret_value = SUCCEED; /* Return value		*/

    FUNC_ENTER_PACKAGE

    /* Sanity check */
    assert(dset);

    /* Check for invalid chunk dimension rank */
    if (0 == dset->shared->layout.u.struct_chunk.ndims)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "no chunk information set?");
    if ((dset->shared->layout.u.struct_chunk.ndims - 1) != dset->shared->ndims)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "dimensionality of chunks doesn't match the dataspace");

    if (chunk_dims) {
        /* Get the chunk dimension sizes */
        for (unsigned u = 0; u < (dset->shared->layout.u.struct_chunk.ndims - 1); u++)
            chunk_dims[u] = dset->shared->layout.u.struct_chunk.dim[u];
    }

    /* For structured chunk:
         encoding and decoding is necessary for chunks between cache and disk */

    if (encode_decode_necessary)
        *encode_decode_necessary = true;

    if (partial_bound_chunks_different_encoding)
        *partial_bound_chunks_different_encoding =
            (dset->shared->layout.u.struct_chunk.flags & H5O_LAYOUT_CHUNK_DONT_FILTER_PARTIAL_BOUND_CHUNKS);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_layout_query() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_delete_chunk
 *
 * Purpose:     Removes the chunk from the index and deletes it on disk.
 *
 *              Only called if a chunk goes out of scope due to H5Dset_extent() or
 *              of H5SC_chunk_erase_values_t returns *delete_chunk == true.
 *
 * Return:    Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_delete_chunk(H5D_t *dset, const hsize_t *scaled /*in*/, haddr_t addr, hsize_t disk_size)
{
    H5D_chunk_ud_t              udata;
    H5D_chunk_common_ud_t       idx_udata; /* User data for index removal routine */
    H5O_storage_struct_chunk_t *storage = &(dset->shared->layout.storage.u.struct_chunk);
    H5D_chk_idx_info_t          idx_info;            /* Chunked index info */
    herr_t                      ret_value = SUCCEED; /* Return value		*/

    FUNC_ENTER_PACKAGE

    /* Sanity check */
    assert(dset);

    /* Compose chunked index info struct */
    idx_info.f           = dset->oloc.file;
    idx_info.stc_pline   = &dset->shared->dcpl_cache.stc_pline;
    idx_info.stc_layout  = &dset->shared->layout.u.struct_chunk;
    idx_info.stc_storage = &dset->shared->layout.storage.u.struct_chunk;

    /* Set up udata */
    udata.common.stc_layout  = idx_info.stc_layout;
    udata.common.stc_storage = idx_info.stc_storage;
    udata.common.scaled      = scaled;

    /* Reset information about the chunk we are looking for */
    udata.chunk_block.offset = HADDR_UNDEF;
    udata.chunk_block.length = 0;

    /* chunk_idx is calculated in get_addr callback */
    if ((storage->ops->get_addr)(&idx_info, &udata) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't query chunk address");

    /* Remove the chunk from disk, if present */
    if (H5_addr_defined(udata.chunk_block.offset)) {
        if (!H5_addr_eq(addr, udata.chunk_block.offset))
            HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "structured chunk address changed before deletion");

        if (udata.chunk_block.length != disk_size)
            HGOTO_ERROR(H5E_DATASET, H5E_SIZE_MISMATCH, FAIL,
                        "structured chunk size changed before deletion");

        idx_udata.scaled = udata.common.scaled;

        if ((storage->ops->remove)(&idx_info, &idx_udata) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTDELETE, FAIL, "unable to remove chunk entry from index");
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5D__struct_chunk_delete_chunk() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_bh_info
 *
 * Purpose:     Retrieve the amount of index storage for structured chunk dataset
 *
 * Return:      Success:        Non-negative
 *              Failure:        negative
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5D__struct_chunk_bh_info(const H5O_loc_t *loc, H5O_t *oh, H5O_layout_t *layout, hsize_t *index_size)
{
    H5D_chk_idx_info_t          idx_info;     /* Chunked index info */
    H5S_t                      *space = NULL; /* Dataset's dataspace */
    H5O_stc_pline_t             pline;        /* I/O pipeline message */
    H5O_storage_struct_chunk_t *sc = &(layout->storage.u.struct_chunk);
    htri_t                      exists;                /* Flag if header message of interest exists */
    bool                        idx_info_init = false; /* Whether the chunk index info has been initialized */
    bool                        pline_read    = false; /* Whether the I/O pipeline message was read */
    herr_t                      ret_value     = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(loc);
    assert(loc->file);
    assert(H5_addr_defined(loc->addr));
    assert(layout);
    H5D_STRUCT_CHUNK_STORAGE_INDEX_CHK(sc);
    assert(index_size);

    /* Check for I/O pipeline message */
    if ((exists = H5O_msg_exists_oh(oh, H5O_STC_PLINE_ID)) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to read object header");
    else if (exists) {
        if (NULL == H5O_msg_read_oh(loc->file, oh, H5O_STC_PLINE_ID, &pline))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't find I/O pipeline message");
        pline_read = true;
    } /* end else if */
    else
        memset(&pline, 0, sizeof(pline));

    /* Compose chunked index info struct */
    idx_info.f           = loc->file;
    idx_info.stc_pline   = &pline;
    idx_info.stc_layout  = &layout->u.struct_chunk;
    idx_info.stc_storage = sc;

    /* Get the dataspace for the dataset */
    if (NULL == (space = H5S_read(loc)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "unable to load dataspace info from dataset header");

    /* Allocate any indexing structures */
    if (sc->ops->init && (sc->ops->init)(&idx_info, space, loc->addr) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, FAIL, "can't initialize indexing information");
    idx_info_init = true;

    /* Get size of index structure */
    if (sc->ops->size && (sc->ops->size)(&idx_info, index_size) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to retrieve chunk index info");

done:
    /* Free resources, if they've been initialized */
    if (idx_info_init && sc->ops->dest && (sc->ops->dest)(&idx_info) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "unable to release chunk index info");
    if (pline_read && H5O_msg_reset(H5O_STC_PLINE_ID, &pline) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTRESET, FAIL, "unable to reset I/O pipeline message");
    if (space && H5S_close(space) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CLOSEERROR, FAIL, "unable to release dataspace");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5D__struct_chunk_bh_info() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_get_alloc_size
 *
 * Purpose:     Returns the resident allocation currently owned by a decoded
 *              structured chunk.
 *
 *              The total includes the selection buffer, fixed-data buffer,
 *              and any chunk-local VL heap-set allocation.
 *
 *              The VL heap-set size is obtained through the H5HG private
 *              interface so the structured-chunk code does not depend on
 *              the internal H5HG heap-set representation.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                                  -- AZO    9/13/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_get_alloc_size(const H5D_chunk_cache_mem_t *chk, size_t *alloc_size_out)
{
    size_t vl_alloc_size = 0;
    size_t total         = 0;
    herr_t ret_value     = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(chk);
    assert(alloc_size_out);

    *alloc_size_out = 0;

    if ((chk->sel_alloc_size) > (SIZE_MAX - chk->data_alloc_size))
        HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "structured chunk resident allocation size overflow");

    total = chk->sel_alloc_size + chk->data_alloc_size;

    if (H5HG__get_local_heapset_alloc_size(chk->vl_heapset, &vl_alloc_size) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to get chunk-local VL allocation size");

    if ((vl_alloc_size) > (SIZE_MAX - total))
        HGOTO_ERROR(H5E_DATASET, H5E_OVERFLOW, FAIL, "structured chunk resident allocation size overflow");

    total += vl_alloc_size;

    *alloc_size_out = total;

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5D__struct_chunk_get_alloc_size() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_get_vlen_ref_size
 *
 * Purpose:     Retrieves the reference-field width used by the normal
 *              file-side VL descriptor for this file.
 *
 *              Chunk-local VL storage preserves the existing descriptor
 *              width. The first four reference bytes are reinterpreted as
 *              a 16-bit stable heap slot followed by a 16-bit H5HG object
 *              index; any remaining bytes stay reserved.
 *
 * Return:      SUCCEED/FAIL
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_get_vlen_ref_size(H5D_t *dset, size_t *ref_nbytes)
{
    H5VL_file_cont_info_t cont_info = {H5VL_CONTAINER_INFO_VERSION, 0, 0, 0};
    H5VL_file_get_args_t  vol_cb_args;
    herr_t                ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(dset->oloc.file);
    assert(ref_nbytes);

    *ref_nbytes = 0;

    memset(&vol_cb_args, 0, sizeof(vol_cb_args));

    vol_cb_args.op_type                 = H5VL_FILE_GET_CONT_INFO;
    vol_cb_args.args.get_cont_info.info = &cont_info;

    /*
     * Use the same container information that normal H5T VL disk setup uses
     * when choosing the width of its file-side storage reference.
     */
    if (H5VL_file_get(H5F_VOL_OBJ(dset->oloc.file), &vol_cb_args, H5P_DATASET_XFER_DEFAULT, H5_REQUEST_NULL) <
        0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to retrieve file container information");

    /*
     * V1 needs at least four reference bytes:
     * two for the stable heap slot and two for the H5HG object index.
     */
    if (cont_info.blob_id_size < 4)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL,
                    "file VL reference field is too small for chunk-local encoding");

    *ref_nbytes = cont_info.blob_id_size;

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5D__struct_chunk_get_vlen_ref_size() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_prepare_vlen_type
 *
 * Purpose:     Creates an operation-local copy of the file-side datatype,
 *              installs the chunk-local VL callback class on that copy,
 *              and obtains the corresponding conversion path.
 *
 *              FILE_IS_SRC is true for reads and false for writes.
 *
 *              The shared dataset datatype is never modified.
 *
 * Return:      SUCCEED/FAIL
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_prepare_vlen_type(H5D_t *dset, const H5T_t *file_type, const H5T_t *other_type,
                                    bool file_is_src, H5T_t **chunk_file_type, H5T_path_t **chunk_tpath,
                                    size_t *ref_nbytes)
{
    H5T_t *tmp_type  = NULL;
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(file_type);
    assert(other_type);
    assert(chunk_file_type);
    assert(chunk_tpath);
    assert(ref_nbytes);

    *chunk_file_type = NULL;
    *chunk_tpath     = NULL;
    *ref_nbytes      = 0;

    /*
     * Recover the reference width already used by this file's ordinary
     * file-side VL descriptors.
     */
    if (H5D__struct_chunk_get_vlen_ref_size(dset, ref_nbytes) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to determine chunk-local VL reference width");

    /*
     * Work only on a private copy. H5T_patch_vlen_chunk_local() recursively
     * changes VL callback classes, so applying it to the shared dataset
     * datatype would contaminate unrelated chunks and I/O operations.
     */
    if (NULL == (tmp_type = H5T_copy(file_type, H5T_COPY_TRANSIENT)))
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTCOPY, FAIL,
                    "unable to copy file datatype for chunk-local VL conversion");

    if (H5T_set_loc(tmp_type, H5F_VOL_OBJ(dset->oloc.file), H5T_LOC_DISK) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
                    "unable to configure private chunk datatype for disk storage");

    if (H5T_get_size(tmp_type) != H5T_get_size(file_type))
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADSIZE, FAIL,
                    "SCC conversion datatype does not have file-side layout");

    if (H5T_patch_vlen_chunk_local(tmp_type, *ref_nbytes) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL, "unable to install chunk-local VL datatype backend");

    /*
     * H5T's conversion path must be found using the patched datatype.
     * Reusing the path that was built for the ordinary blob-backed file
     * datatype would defeat the purpose of changing the VL callback class.
     */
    if (file_is_src) {
        if (NULL == (*chunk_tpath = H5T_path_find(tmp_type, other_type)))
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
                        "unable to create chunk-local VL read conversion path");
    }
    else {
        if (NULL == (*chunk_tpath = H5T_path_find(other_type, tmp_type)))
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
                        "unable to create chunk-local VL write conversion path");
    }

    *chunk_file_type = tmp_type;
    tmp_type         = NULL;

done:
    if (tmp_type)
        if (H5T_close_real(tmp_type) < 0)
            HDONE_ERROR(H5E_DATATYPE, H5E_CANTCLOSEOBJ, FAIL,
                        "unable to release temporary chunk-local datatype");

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5D__struct_chunk_prepare_vlen_type() */

/*-------------------------------------------------------------------------
 * Function:    H5D__struct_chunk_vlen_convert
 *
 * Purpose:     Runs one datatype conversion while the supplied structured-
 *              chunk VL context is active.
 *
 *              The previously active context is always restored before
 *              returning, including when H5T_convert() fails.
 *
 * Return:      SUCCEED/FAIL
 *-------------------------------------------------------------------------
 */
static herr_t
H5D__struct_chunk_vlen_convert(const H5T_vlen_chunk_ctx_t *ctx, H5T_path_t *tpath, const H5T_t *src_type,
                               const H5T_t *dst_type, size_t nelmts, void *buf, void *bkg)
{
    const H5T_vlen_chunk_ctx_t *old_ctx   = NULL;
    herr_t                      ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(ctx);
    assert(tpath);
    assert(src_type);
    assert(dst_type);
    assert(buf);

    /*
     * The context is intentionally active only while H5T is interpreting
     * chunk-local VL descriptors.
     */
    old_ctx = H5T_set_vlen_chunk_ctx(ctx);

    if (H5T_convert(tpath, src_type, dst_type, nelmts, (size_t)0, (size_t)0, buf, bkg) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTCONVERT, FAIL, "chunk-local VL datatype conversion failed");

done:
    /* Always restore the previous context, including nested callers. */
    H5T_set_vlen_chunk_ctx(old_ctx);

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5D__struct_chunk_vlen_convert() */
