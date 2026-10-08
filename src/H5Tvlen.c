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
 * Module Info: This module contains the functionality for variable-length
 *      datatypes in the H5T interface.
 */

/****************/
/* Module Setup */
/****************/

#include "H5Tmodule.h" /* This source code file is part of the H5T module */

/***********/
/* Headers */
/***********/
#include "H5private.h"   /* Generic Functions    */
#include "H5CXprivate.h" /* API Contexts         */
#include "H5Eprivate.h"  /* Error handling       */
#include "H5Iprivate.h"  /* IDs                  */
#include "H5MMprivate.h" /* Memory management    */
#include "H5Tpkg.h"      /* Datatypes            */
#include "H5VLprivate.h" /* Virtual Object Layer */

#include "H5HGprivate.h" /* Chunk-local H5HG-style heap operations */

/****************/
/* Local Macros */
/****************/

/*
 * Number of reference bytes used by the V1 chunk-local VL encoding:
 * a 16-bit stable heap-slot index followed by a 16-bit H5HG object index.
 * Any additional bytes in the descriptor's reference field are reserved.
 */
#define H5T_VLEN_CHUNK_REF_SIZE 4

/******************/
/* Local Typedefs */
/******************/

/********************/
/* Package Typedefs */
/********************/

/********************/
/* Local Prototypes */
/********************/

/* Memory-based VL sequence callbacks */
static herr_t H5T__vlen_mem_seq_getlen(H5VL_object_t *file, const void *_vl, size_t *len);
static void  *H5T__vlen_mem_seq_getptr(void *_vl);
static herr_t H5T__vlen_mem_seq_isnull(const H5VL_object_t *file, void *_vl, bool *isnull);
static herr_t H5T__vlen_mem_seq_setnull(H5VL_object_t *file, void *_vl, void *_bg);
static herr_t H5T__vlen_mem_seq_read(H5VL_object_t *file, void *_vl, void *_buf, size_t len);
static herr_t H5T__vlen_mem_seq_write(H5VL_object_t *file, const H5T_vlen_alloc_info_t *vl_alloc_info,
                                      void *_vl, void *_buf, void *_bg, size_t seq_len, size_t base_size);

/* Memory-based VL string callbacks */
static herr_t H5T__vlen_mem_str_getlen(H5VL_object_t *file, const void *_vl, size_t *len);
static void  *H5T__vlen_mem_str_getptr(void *_vl);
static herr_t H5T__vlen_mem_str_isnull(const H5VL_object_t *file, void *_vl, bool *isnull);
static herr_t H5T__vlen_mem_str_setnull(H5VL_object_t *file, void *_vl, void *_bg);
static herr_t H5T__vlen_mem_str_read(H5VL_object_t *file, void *_vl, void *_buf, size_t len);
static herr_t H5T__vlen_mem_str_write(H5VL_object_t *file, const H5T_vlen_alloc_info_t *vl_alloc_info,
                                      void *_vl, void *_buf, void *_bg, size_t seq_len, size_t base_size);

/* Disk-based VL sequence (and string) callbacks */
static herr_t H5T__vlen_disk_getlen(H5VL_object_t *file, const void *_vl, size_t *len);
static herr_t H5T__vlen_disk_isnull(const H5VL_object_t *file, void *_vl, bool *isnull);
static herr_t H5T__vlen_disk_setnull(H5VL_object_t *file, void *_vl, void *_bg);
static herr_t H5T__vlen_disk_read(H5VL_object_t *file, void *_vl, void *_buf, size_t len);
static herr_t H5T__vlen_disk_write(H5VL_object_t *file, const H5T_vlen_alloc_info_t *vl_alloc_info, void *_vl,
                                   void *_buf, void *_bg, size_t seq_len, size_t base_size);
static herr_t H5T__vlen_disk_delete(H5VL_object_t *file, void *_vl);

/* Chunk-local VL context and descriptor-reference helpers. */
static herr_t H5T__vlen_chunk_get_ctx(const H5T_vlen_chunk_ctx_t **ctx);
static herr_t H5T__vlen_chunk_ref_decode(const H5T_vlen_chunk_ctx_t *ctx, const uint8_t *ref,
                                         uint16_t *heap_slot, uint16_t *obj_idx);
static herr_t H5T__vlen_chunk_ref_setnull(const H5T_vlen_chunk_ctx_t *ctx, uint8_t *ref);
static herr_t H5T__vlen_chunk_ref_isnull(const H5T_vlen_chunk_ctx_t *ctx, const uint8_t *ref, bool *isnull);

/* File-side VL callbacks for structured-chunk storage */
static herr_t H5T__vlen_chunk_getlen(H5VL_object_t *file, const void *_vl, size_t *seq_len);
static herr_t H5T__vlen_chunk_isnull(const H5VL_object_t *file, void *_vl, bool *isnull);
static herr_t H5T__vlen_chunk_setnull(H5VL_object_t *file, void *_vl, void *_bg);
static herr_t H5T__vlen_chunk_read(H5VL_object_t *file, void *_vl, void *_buf, size_t len);
static herr_t H5T__vlen_chunk_write(H5VL_object_t *file, const H5T_vlen_alloc_info_t *vl_alloc_info,
                                    void *_vl, void *_buf, void *_bg, size_t seq_len, size_t base_size);
static herr_t H5T__vlen_chunk_delete(H5VL_object_t *file, void *_vl);

/* Datatype visitor callback used to install the chunk-local VL backend */
static herr_t H5T__vlen_chunk_patch_cb(H5T_t *dt, void *op_data);

/*********************/
/* Public Variables */
/*********************/

/*********************/
/* Package Variables */
/*********************/

/*****************************/
/* Library Private Variables */
/*****************************/

/*******************/
/* Local Variables */
/*******************/

/*
 * Context used by the chunk-local file-side VL callbacks.
 *
 * Structured-chunk VL conversion is currently single-threaded, so one
 * H5T-local global pointer is sufficient. The structured-chunk path installs
 * the context for the chunk being converted and restores the previous
 * context when conversion completes.
 *
 * A threaded implementation can replace this pointer with thread-local
 * storage without changing the VL callback interface.
 */
static const H5T_vlen_chunk_ctx_t *H5T_vlen_chunk_ctx_g = NULL;

/* Class for VL sequences in memory */
static const H5T_vlen_class_t H5T_vlen_mem_seq_g = {
    H5T__vlen_mem_seq_getlen,  /* 'getlen' */
    H5T__vlen_mem_seq_getptr,  /* 'getptr' */
    H5T__vlen_mem_seq_isnull,  /* 'isnull' */
    H5T__vlen_mem_seq_setnull, /* 'setnull' */
    H5T__vlen_mem_seq_read,    /* 'read' */
    H5T__vlen_mem_seq_write,   /* 'write' */
    NULL                       /* 'delete' */
};

/* Class for VL strings in memory */
static const H5T_vlen_class_t H5T_vlen_mem_str_g = {
    H5T__vlen_mem_str_getlen,  /* 'getlen' */
    H5T__vlen_mem_str_getptr,  /* 'getptr' */
    H5T__vlen_mem_str_isnull,  /* 'isnull' */
    H5T__vlen_mem_str_setnull, /* 'setnull' */
    H5T__vlen_mem_str_read,    /* 'read' */
    H5T__vlen_mem_str_write,   /* 'write' */
    NULL                       /* 'delete' */
};

/* Class for both VL strings and sequences in file */
static const H5T_vlen_class_t H5T_vlen_disk_g = {
    H5T__vlen_disk_getlen,  /* 'getlen' */
    NULL,                   /* 'getptr' */
    H5T__vlen_disk_isnull,  /* 'isnull' */
    H5T__vlen_disk_setnull, /* 'setnull' */
    H5T__vlen_disk_read,    /* 'read' */
    H5T__vlen_disk_write,   /* 'write' */
    H5T__vlen_disk_delete   /* 'delete' */
};

/*
 * File-side VL callback class for structured-chunk storage.
 *
 * This parallels H5T_vlen_disk_g, but resolves VL payloads through the
 * H5HG heap set owned by the structured chunk currently being processed
 * instead of through the normal VOL blob backend.
 *
 * The descriptor is not a direct memory representation, so getptr is NULL.
 * The remaining callbacks use the active chunk-local context to resolve the
 * descriptor within the current chunk's heap set.
 */
static const H5T_vlen_class_t H5T_vlen_chunk_g = {
    H5T__vlen_chunk_getlen,  /* 'getlen' */
    NULL,                    /* 'getptr' */
    H5T__vlen_chunk_isnull,  /* 'isnull' */
    H5T__vlen_chunk_setnull, /* 'setnull' */
    H5T__vlen_chunk_read,    /* 'read' */
    H5T__vlen_chunk_write,   /* 'write' */
    H5T__vlen_chunk_delete   /* 'delete' */
};

/*-------------------------------------------------------------------------
 * Function:    H5Tvlen_create
 *
 * Purpose:     Create a new variable-length datatype based on the
 *              specified base datatype ID.
 *
 * Return:      Success:    ID of new VL datatype
 *              Failure:    H5I_INVALID_HID
 *
 *-------------------------------------------------------------------------
 */
hid_t
H5Tvlen_create(hid_t base_id)
{
    H5T_t *base = NULL; /*base datatype	*/
    H5T_t *dt   = NULL; /*new datatype	*/
    hid_t  ret_value;   /*return value			*/

    FUNC_ENTER_API(H5I_INVALID_HID)

    /* Check args */
    if (NULL == (base = (H5T_t *)H5I_object_verify(base_id, H5I_DATATYPE)))
        HGOTO_ERROR(H5E_ARGS, H5E_BADTYPE, H5I_INVALID_HID, "not an valid base datatype");

    /* Create up VL datatype */
    if ((dt = H5T__vlen_create(base)) == NULL)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, H5I_INVALID_HID, "invalid VL location");

    /* Register the type */
    if ((ret_value = H5I_register(H5I_DATATYPE, dt, true)) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREGISTER, H5I_INVALID_HID, "unable to register datatype");

done:
    FUNC_LEAVE_API(ret_value)
} /* end H5Tvlen_create() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_create
 *
 * Purpose:     Create a new variable-length datatype based on the
 *              specified base datatype.
 *
 * Return:      Success:    new VL datatype
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
H5T_t *
H5T__vlen_create(const H5T_t *base)
{
    H5T_t *dt        = NULL; /* New VL datatype */
    H5T_t *ret_value = NULL; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check args */
    assert(base);

    /* Build new type */
    if (NULL == (dt = H5T__alloc()))
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTALLOC, NULL, "memory allocation failed");
    dt->shared->type = H5T_VLEN;

    /*
     * Force conversions (i.e. memory to memory conversions should duplicate
     * data, not point to the same VL sequences)
     */
    dt->shared->force_conv = true;
    if (NULL == (dt->shared->parent = H5T_copy(base, H5T_COPY_ALL)))
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTCOPY, NULL, "can't copy base datatype");

    /* Inherit encoding version from base type */
    dt->shared->version = base->shared->version;

    /* This is a sequence, not a string */
    dt->shared->u.vlen.type = H5T_VLEN_SEQUENCE;

    /* Set up VL information */
    if (H5T_set_loc(dt, NULL, H5T_LOC_MEMORY) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, NULL, "invalid datatype location");

    /* Set return value */
    ret_value = dt;

done:
    if (!ret_value)
        if (dt && H5T_close_real(dt) < 0)
            HDONE_ERROR(H5E_DATATYPE, H5E_CANTRELEASE, NULL, "unable to release datatype info");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_create() */

/*-------------------------------------------------------------------------
 * Function: H5T__vlen_set_loc
 *
 * Purpose:	Sets the location of a VL datatype to be either on disk or in memory
 *
 * Return:
 *  One of two values on success:
 *      true - If the location of any vlen types changed
 *      false - If the location of any vlen types is the same
 *  <0 is returned on failure
 *
 *-------------------------------------------------------------------------
 */
htri_t
H5T__vlen_set_loc(H5T_t *dt, H5VL_object_t *file, H5T_loc_t loc)
{
    htri_t ret_value = false; /* Indicate success, but no location change */

    FUNC_ENTER_PACKAGE

    /* check parameters */
    assert(dt);
    assert(loc >= H5T_LOC_BADLOC && loc < H5T_LOC_MAXLOC);

    /* Only change the location if it's different */
    if (loc != dt->shared->u.vlen.loc || file != dt->shared->u.vlen.file) {
        switch (loc) {
            case H5T_LOC_MEMORY: /* Memory based VL datatype */
                assert(NULL == file);

                /* Mark this type as being stored in memory */
                dt->shared->u.vlen.loc = H5T_LOC_MEMORY;

                if (dt->shared->u.vlen.type == H5T_VLEN_SEQUENCE) {
                    /* Size in memory, disk size is different */
                    dt->shared->size = sizeof(hvl_t);

                    /* Set up the function pointers to access the VL sequence in memory */
                    dt->shared->u.vlen.cls = &H5T_vlen_mem_seq_g;
                } /* end if */
                else if (dt->shared->u.vlen.type == H5T_VLEN_STRING) {
                    /* Size in memory, disk size is different */
                    dt->shared->size = sizeof(char *);

                    /* Set up the function pointers to access the VL string in memory */
                    dt->shared->u.vlen.cls = &H5T_vlen_mem_str_g;
                } /* end else-if */
                else
                    assert(0 && "Invalid VL type");

                /* Release owned file */
                if (dt->shared->owned_vol_obj) {
                    if (H5VL_free_object(dt->shared->owned_vol_obj) < 0)
                        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTCLOSEOBJ, FAIL, "unable to close owned VOL object");
                    dt->shared->owned_vol_obj = NULL;
                } /* end if */

                /* Reset file pointer (since this VL is in memory) */
                dt->shared->u.vlen.file = NULL;
                break;

            /* Disk based VL datatype */
            case H5T_LOC_DISK: {
                H5VL_file_cont_info_t cont_info = {H5VL_CONTAINER_INFO_VERSION, 0, 0, 0};
                H5VL_file_get_args_t  vol_cb_args; /* Arguments to VOL callback */

                assert(file);

                /* Mark this type as being stored on disk */
                dt->shared->u.vlen.loc = H5T_LOC_DISK;

                /* Set up VOL callback arguments */
                vol_cb_args.op_type                 = H5VL_FILE_GET_CONT_INFO;
                vol_cb_args.args.get_cont_info.info = &cont_info;

                /* Get container info */
                if (H5VL_file_get(file, &vol_cb_args, H5P_DATASET_XFER_DEFAULT, H5_REQUEST_NULL) < 0)
                    HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL, "unable to get container info");

                /* The datatype size is equal to 4 bytes for the sequence length
                 * plus the size of a blob id */
                dt->shared->size = 4 + cont_info.blob_id_size;

                /* Set up the function pointers to access the VL information on disk */
                /* VL sequences and VL strings are stored identically on disk, so use the same functions */
                dt->shared->u.vlen.cls = &H5T_vlen_disk_g;

                /* Set file ID (since this VL is on disk) */
                dt->shared->u.vlen.file = file;

                /* dt now owns a reference to file */
                if (H5T_own_vol_obj(dt, file) < 0)
                    HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL, "can't give ownership of VOL object");
                break;
            }

            case H5T_LOC_BADLOC:
                /* Allow undefined location. In H5Odtype.c, H5O_dtype_decode sets undefined
                 * location for VL type and leaves it for the caller to decide.
                 */
                dt->shared->u.vlen.loc = H5T_LOC_BADLOC;

                /* Reset the function pointers to access the VL information */
                dt->shared->u.vlen.cls = NULL;

                /* Reset file pointer */
                dt->shared->u.vlen.file = NULL;
                break;

            case H5T_LOC_MAXLOC:
                /* MAXLOC is invalid */
            default:
                HGOTO_ERROR(H5E_DATATYPE, H5E_BADRANGE, FAIL, "invalid VL datatype location");
        } /* end switch */

        /* Indicate that the location changed */
        ret_value = true;
    } /* end if */

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_set_loc() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_seq_getlen
 *
 * Purpose:	Retrieves the length of a memory based VL element.
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_mem_seq_getlen(H5VL_object_t H5_ATTR_UNUSED *file, const void *_vl, size_t *len)
{
    hvl_t vl; /* User's hvl_t information */

    FUNC_ENTER_PACKAGE_NOERR

    assert(_vl);
    assert(len);

    /* Copy to ensure correct alignment */
    H5MM_memcpy(&vl, _vl, sizeof(hvl_t));

    *len = vl.len;

    FUNC_LEAVE_NOAPI(SUCCEED)
} /* end H5T__vlen_mem_seq_getlen() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_seq_getptr
 *
 * Purpose:	Retrieves the pointer for a memory based VL element.
 *
 * Return:	Non-NULL on success/NULL on failure
 *
 *-------------------------------------------------------------------------
 */
static void *
H5T__vlen_mem_seq_getptr(void *_vl)
{
    hvl_t vl; /* User's hvl_t information */

    FUNC_ENTER_PACKAGE_NOERR

    assert(_vl);

    /* Copy to ensure correct alignment */
    H5MM_memcpy(&vl, _vl, sizeof(hvl_t));

    FUNC_LEAVE_NOAPI(vl.p)
} /* end H5T__vlen_mem_seq_getptr() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_seq_isnull
 *
 * Purpose:	Checks if a memory sequence is the "null" sequence
 *
 * Return:	Non-negative on success / Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_mem_seq_isnull(const H5VL_object_t H5_ATTR_UNUSED *file, void *_vl, bool *isnull)
{
    hvl_t vl; /* User's hvl_t information */

    FUNC_ENTER_PACKAGE_NOERR

    assert(_vl);

    /* Copy to ensure correct alignment */
    H5MM_memcpy(&vl, _vl, sizeof(hvl_t));

    *isnull = ((vl.len == 0 || vl.p == NULL) ? true : false);

    FUNC_LEAVE_NOAPI(SUCCEED)
} /* end H5T__vlen_mem_seq_isnull() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_seq_setnull
 *
 * Purpose:	Sets a VL info object in memory to the "nil" value
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_mem_seq_setnull(H5VL_object_t H5_ATTR_UNUSED *file, void *_vl, void H5_ATTR_UNUSED *_bg)
{
    hvl_t vl; /* Temporary hvl_t to use during operation */

    FUNC_ENTER_PACKAGE_NOERR

    /* check parameters */
    assert(_vl);

    /* Set the "nil" hvl_t */
    vl.len = 0;
    vl.p   = NULL;

    /* Set pointer in user's buffer with memcpy, to avoid alignment issues */
    H5MM_memcpy(_vl, &vl, sizeof(hvl_t));

    FUNC_LEAVE_NOAPI(SUCCEED)
} /* end H5T__vlen_mem_seq_setnull() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_seq_read
 *
 * Purpose:	"Reads" the memory based VL sequence into a buffer
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_mem_seq_read(H5VL_object_t H5_ATTR_UNUSED *file, void *_vl, void *buf, size_t len)
{
    hvl_t vl; /* User's hvl_t information */

    FUNC_ENTER_PACKAGE_NOERR

    assert(buf);
    assert(_vl);

    /* Copy to ensure correct alignment */
    H5MM_memcpy(&vl, _vl, sizeof(hvl_t));
    assert(vl.p);

    H5MM_memcpy(buf, vl.p, len);

    FUNC_LEAVE_NOAPI(SUCCEED)
} /* end H5T__vlen_mem_seq_read() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_seq_write
 *
 * Purpose:	"Writes" the memory based VL sequence from a buffer
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_mem_seq_write(H5VL_object_t H5_ATTR_UNUSED *file, const H5T_vlen_alloc_info_t *vl_alloc_info,
                        void *_vl, void *buf, void H5_ATTR_UNUSED *_bg, size_t seq_len, size_t base_size)
{
    hvl_t  vl;                  /* Temporary hvl_t to use during operation */
    herr_t ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* check parameters */
    assert(_vl);
    assert(buf);

    if (seq_len) {
        size_t len = seq_len * base_size; /* Sequence size */

        /* Use the user's memory allocation routine if one is defined */
        if (vl_alloc_info->alloc_func != NULL) {
            /* Prepare & restore library for user callback */
            H5_BEFORE_USER_CB(FAIL)
                {
                    vl.p = (vl_alloc_info->alloc_func)(len, vl_alloc_info->alloc_info);
                }
            H5_AFTER_USER_CB(FAIL)
            if (NULL == vl.p)
                HGOTO_ERROR(H5E_DATATYPE, H5E_CANTALLOC, FAIL,
                            "application memory allocation routine failed for VL data");
        }    /* end if */
        else /* Default to system malloc */
            if (NULL == (vl.p = malloc(len)))
                HGOTO_ERROR(H5E_DATATYPE, H5E_CANTALLOC, FAIL, "memory allocation failed for VL data");

        /* Copy the data into the newly allocated buffer */
        H5MM_memcpy(vl.p, buf, len);
    } /* end if */
    else
        vl.p = NULL;

    /* Set the sequence length */
    vl.len = seq_len;

    /* Set pointer in user's buffer with memcpy, to avoid alignment issues */
    H5MM_memcpy(_vl, &vl, sizeof(hvl_t));

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_mem_seq_write() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_str_getlen
 *
 * Purpose:	Retrieves the length of a memory based VL string.
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_mem_str_getlen(H5VL_object_t H5_ATTR_UNUSED *file, const void *_vl, size_t *len)
{
    const char *s = NULL; /* Pointer to the user's string information */

    FUNC_ENTER_PACKAGE_NOERR

    assert(_vl);

    /* Copy to ensure correct alignment */
    H5MM_memcpy(&s, _vl, sizeof(char *));

    *len = strlen(s);

    FUNC_LEAVE_NOAPI(SUCCEED)
} /* end H5T__vlen_mem_str_getlen() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_str_getptr
 *
 * Purpose:	Retrieves the pointer for a memory based VL string.
 *
 * Return:	Non-NULL on success/NULL on failure
 *
 *-------------------------------------------------------------------------
 */
static void *
H5T__vlen_mem_str_getptr(void *_vl)
{
    char *s = NULL; /* Pointer to the user's string information */

    FUNC_ENTER_PACKAGE_NOERR

    assert(_vl);

    /* Copy to ensure correct alignment */
    H5MM_memcpy(&s, _vl, sizeof(char *));

    FUNC_LEAVE_NOAPI(s)
} /* end H5T__vlen_mem_str_getptr() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_str_isnull
 *
 * Purpose:	Checks if a memory string is a NULL pointer
 *
 * Return:	Non-negative on success / Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_mem_str_isnull(const H5VL_object_t H5_ATTR_UNUSED *file, void *_vl, bool *isnull)
{
    char *s = NULL; /* Pointer to the user's string information */

    FUNC_ENTER_PACKAGE_NOERR

    /* Copy to ensure correct alignment */
    H5MM_memcpy(&s, _vl, sizeof(char *));

    *isnull = (s == NULL ? true : false);

    FUNC_LEAVE_NOAPI(SUCCEED)
} /* end H5T__vlen_mem_str_isnull() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_str_setnull
 *
 * Purpose:	Sets a VL info object in memory to the "null" value
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_mem_str_setnull(H5VL_object_t H5_ATTR_UNUSED *file, void *_vl, void H5_ATTR_UNUSED *_bg)
{
    char *t = NULL; /* Pointer to temporary buffer allocated */

    FUNC_ENTER_PACKAGE_NOERR

    /* Set pointer in user's buffer with memcpy, to avoid alignment issues */
    H5MM_memcpy(_vl, &t, sizeof(char *));

    FUNC_LEAVE_NOAPI(SUCCEED)
} /* end H5T__vlen_mem_str_setnull() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_str_read
 *
 * Purpose:	"Reads" the memory based VL string into a buffer
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_mem_str_read(H5VL_object_t H5_ATTR_UNUSED *file, void *_vl, void *buf, size_t len)
{
    char *s; /* Pointer to the user's string information */

    FUNC_ENTER_PACKAGE_NOERR

    if (len > 0) {
        assert(buf);
        assert(_vl);

        /* Copy to ensure correct alignment */
        H5MM_memcpy(&s, _vl, sizeof(char *));
        H5MM_memcpy(buf, s, len);
    }

    FUNC_LEAVE_NOAPI(SUCCEED)
} /* end H5T__vlen_mem_str_read() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_mem_str_write
 *
 * Purpose:	"Writes" the memory based VL string from a buffer
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_mem_str_write(H5VL_object_t H5_ATTR_UNUSED *file, const H5T_vlen_alloc_info_t *vl_alloc_info,
                        void *_vl, void *buf, void H5_ATTR_UNUSED *_bg, size_t seq_len, size_t base_size)
{
    char  *t;                   /* Pointer to temporary buffer allocated */
    size_t len;                 /* Maximum length of the string to copy */
    herr_t ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* check parameters */
    assert(buf);

    /* Use the user's memory allocation routine if one is defined */
    if (vl_alloc_info->alloc_func != NULL) {
        /* Prepare & restore library for user callback */
        H5_BEFORE_USER_CB(FAIL)
            {
                t = (vl_alloc_info->alloc_func)((seq_len + 1) * base_size, vl_alloc_info->alloc_info);
            }
        H5_AFTER_USER_CB(FAIL)
        if (NULL == t)
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTALLOC, FAIL,
                        "application memory allocation routine failed for VL data");
    }    /* end if */
    else /* Default to system malloc */
        if (NULL == (t = (char *)malloc((seq_len + 1) * base_size)))
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTALLOC, FAIL, "memory allocation failed for VL data");

    /* 'write' the string into the buffer, with memcpy() */
    len = (seq_len * base_size);
    H5MM_memcpy(t, buf, len);
    t[len] = '\0';

    /* Set pointer in user's buffer with memcpy, to avoid alignment issues */
    H5MM_memcpy(_vl, &t, sizeof(char *));

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_mem_str_write() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_disk_getlen
 *
 * Purpose:	Retrieves the length of a disk based VL element.
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_disk_getlen(H5VL_object_t H5_ATTR_UNUSED *file, const void *_vl, size_t *seq_len)
{
    const uint8_t *vl = (const uint8_t *)_vl; /* Pointer to the user's hvl_t information */

    FUNC_ENTER_PACKAGE_NOERR

    /* Check parameters */
    assert(vl);
    assert(seq_len);

    /* Get length of sequence (different from blob size) */
    UINT32DECODE(vl, *seq_len);

    FUNC_LEAVE_NOAPI(SUCCEED)
} /* end H5T__vlen_disk_getlen() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_disk_isnull
 *
 * Purpose:	Checks if a disk VL info object is the "nil" object
 *
 * Return:	Non-negative on success / Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_disk_isnull(const H5VL_object_t *file, void *_vl, bool *isnull)
{
    H5VL_blob_specific_args_t vol_cb_args;                /* Arguments to VOL callback */
    uint8_t                  *vl        = (uint8_t *)_vl; /* Pointer to the user's hvl_t information */
    herr_t                    ret_value = SUCCEED;        /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check parameters */
    assert(file);
    assert(vl);
    assert(isnull);

    /* Skip the sequence's length */
    vl += 4;

    /* Set up VOL callback arguments */
    vol_cb_args.op_type             = H5VL_BLOB_ISNULL;
    vol_cb_args.args.is_null.isnull = isnull;

    /* Check if blob ID is "nil" */
    if (H5VL_blob_specific(file, vl, &vol_cb_args) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL, "unable to check if a blob ID is 'nil'");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_disk_isnull() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_disk_setnull
 *
 * Purpose:	Sets a VL info object on disk to the "nil" value
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_disk_setnull(H5VL_object_t *file, void *_vl, void *bg)
{
    H5VL_blob_specific_args_t vol_cb_args;                /* Arguments to VOL callback */
    uint8_t                  *vl        = (uint8_t *)_vl; /* Pointer to the user's hvl_t information */
    herr_t                    ret_value = SUCCEED;        /* Return value */

    FUNC_ENTER_PACKAGE

    /* check parameters */
    assert(file);
    assert(vl);

    /* Free heap object for old data */
    if (bg != NULL)
        /* Delete sequence in destination location */
        if (H5T__vlen_disk_delete(file, bg) < 0)
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL, "unable to remove background heap object");

    /* Set the length of the sequence */
    UINT32ENCODE(vl, 0);

    /* Set up VOL callback arguments */
    vol_cb_args.op_type = H5VL_BLOB_SETNULL;

    /* Set blob ID to "nil" */
    if (H5VL_blob_specific(file, vl, &vol_cb_args) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTSET, FAIL, "unable to set a blob ID to 'nil'");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_disk_setnull() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_disk_read
 *
 * Purpose:	Reads the disk based VL element into a buffer
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_disk_read(H5VL_object_t *file, void *_vl, void *buf, size_t len)
{
    const uint8_t *vl        = (const uint8_t *)_vl; /* Pointer to the user's hvl_t information */
    herr_t         ret_value = SUCCEED;              /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check parameters */
    assert(file);
    assert(vl);
    assert(buf);

    /* Skip the length of the sequence */
    vl += 4;

    /* Retrieve blob */
    if (H5VL_blob_get(file, vl, buf, len, NULL) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL, "unable to get blob");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_disk_read() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_disk_write
 *
 * Purpose:	Writes the disk based VL element from a buffer
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_disk_write(H5VL_object_t *file, const H5T_vlen_alloc_info_t H5_ATTR_UNUSED *vl_alloc_info,
                     void *_vl, void *buf, void *_bg, size_t seq_len, size_t base_size)
{
    uint8_t *vl        = (uint8_t *)_vl; /* Pointer to the user's hvl_t information */
    uint8_t *bg        = (uint8_t *)_bg; /* Pointer to the old data hvl_t */
    herr_t   ret_value = SUCCEED;        /* Return value */

    FUNC_ENTER_PACKAGE

    /* check parameters */
    assert(vl);
    assert(seq_len == 0 || buf);
    assert(file);

    /* Free heap object for old data, if non-NULL */
    if (bg != NULL)
        if (H5T__vlen_disk_delete(file, bg) < 0)
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL, "unable to remove background heap object");

    /* Set the length of the sequence */
    UINT32ENCODE(vl, seq_len);

    /* Store blob */
    if (H5VL_blob_put(file, buf, (seq_len * base_size), vl, NULL) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTSET, FAIL, "unable to put blob");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_disk_write() */

/*-------------------------------------------------------------------------
 * Function:	H5T__vlen_disk_delete
 *
 * Purpose:	Deletes a disk-based VL element
 *
 * Return:	Non-negative on success / Negative on failure
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_disk_delete(H5VL_object_t *file, void *_vl)
{
    uint8_t *vl        = (uint8_t *)_vl; /* Pointer to the user's hvl_t information */
    herr_t   ret_value = SUCCEED;        /* Return value */

    FUNC_ENTER_PACKAGE

    /* Check parameters */
    assert(file);

    /* Free heap object for old data */
    if (vl != NULL) {
        size_t seq_len; /* VL sequence's length */

        /* Get length of sequence */
        UINT32DECODE(vl, seq_len);

        /* Delete object, if length > 0 */
        if (seq_len > 0) {
            H5VL_blob_specific_args_t vol_cb_args; /* Arguments to VOL callback */

            /* Set up VOL callback arguments */
            vol_cb_args.op_type = H5VL_BLOB_DELETE;

            if (H5VL_blob_specific(file, vl, &vol_cb_args) < 0)
                HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL, "unable to delete blob");
        } /* end if */
    }     /* end if */

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_disk_delete() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_reclaim
 *
 * Purpose: Internal recursive routine to free VL datatypes
 *
 * Return:  Non-negative on success / Negative on failure
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5T__vlen_reclaim(void *elem, const H5T_t *dt, H5T_vlen_alloc_info_t *alloc_info)
{
    unsigned    u;                   /* Local index variable */
    H5MM_free_t free_func;           /* Free function */
    void       *free_info;           /* Free info */
    herr_t      ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_PACKAGE

    /* Sanity checks */
    assert(elem);
    assert(dt);
    assert(alloc_info);

    free_func = alloc_info->free_func;
    free_info = alloc_info->free_info;

    /* Check the datatype of this element */
    switch (dt->shared->type) {
        case H5T_ARRAY:
            /* Recurse on each element, if the array's base type is array, VL, enum or compound */
            if (H5T_IS_COMPOSITE(dt->shared->parent->shared->type)) {
                void *off; /* offset of field */

                /* Calculate the offset member and recurse on it */
                for (u = 0; u < dt->shared->u.array.nelem; u++) {
                    off = ((uint8_t *)elem) + u * (dt->shared->parent->shared->size);
                    if (H5T_reclaim_cb(off, dt->shared->parent, 0, NULL, alloc_info) < 0)
                        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTFREE, FAIL, "unable to free array element");
                } /* end for */
            }     /* end if */
            break;

        case H5T_COMPOUND:
            /* Check each field and recurse on VL, compound, enum or array ones */
            for (u = 0; u < dt->shared->u.compnd.nmembs; u++) {
                /* Recurse if it's VL, compound, enum or array */
                if (H5T_IS_COMPOSITE(dt->shared->u.compnd.memb[u].type->shared->type)) {
                    void *off; /* offset of field */

                    /* Calculate the offset member and recurse on it */
                    off = ((uint8_t *)elem) + dt->shared->u.compnd.memb[u].offset;
                    if (H5T_reclaim_cb(off, dt->shared->u.compnd.memb[u].type, 0, NULL, alloc_info) < 0)
                        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTFREE, FAIL, "unable to free compound field");
                } /* end if */
            }     /* end for */
            break;

        case H5T_VLEN:
            /* Recurse on the VL information if it's VL, compound, enum or array, then free VL sequence */
            if (dt->shared->u.vlen.type == H5T_VLEN_SEQUENCE) {
                hvl_t *vl = (hvl_t *)elem; /* Temp. ptr to the vl info */

                /* Check if there is anything actually in this sequence */
                if (vl->len != 0) {
                    /* Recurse if it's VL, array, enum or compound */
                    if (H5T_IS_COMPOSITE(dt->shared->parent->shared->type)) {
                        void *off; /* offset of field */

                        /* Calculate the offset of each array element and recurse on it */
                        while (vl->len > 0) {
                            off = ((uint8_t *)vl->p) + (vl->len - 1) * dt->shared->parent->shared->size;
                            if (H5T_reclaim_cb(off, dt->shared->parent, 0, NULL, alloc_info) < 0)
                                HGOTO_ERROR(H5E_DATATYPE, H5E_CANTFREE, FAIL, "unable to free VL element");
                            vl->len--;
                        } /* end while */
                    }     /* end if */

                    /* Free the VL sequence */
                    if (free_func != NULL)
                        (*free_func)(vl->p, free_info);
                    else
                        free(vl->p);
                } /* end if */
            }
            else if (dt->shared->u.vlen.type == H5T_VLEN_STRING) {
                /* Free the VL string */
                if (free_func != NULL)
                    (*free_func)(*(char **)elem, free_info);
                else
                    free(*(char **)elem);
            }
            else {
                assert(0 && "Invalid VL type");
            } /* end else */
            break;

        /* Don't do anything for simple types */
        case H5T_INTEGER:
        case H5T_FLOAT:
        case H5T_TIME:
        case H5T_STRING:
        case H5T_BITFIELD:
        case H5T_OPAQUE:
        case H5T_ENUM:
        case H5T_COMPLEX:
            break;

        /* Should never have these values */
        case H5T_REFERENCE:
        case H5T_NO_CLASS:
        case H5T_NCLASSES:
        default:
            HGOTO_ERROR(H5E_DATATYPE, H5E_BADRANGE, FAIL, "invalid VL datatype class");
            break;

    } /* end switch */

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_reclaim() */

/*-------------------------------------------------------------------------
 * Function:	H5T_vlen_reclaim_elmt
 *
 * Purpose: Alternative method to reclaim any VL data for a buffer element.
 *
 *          Use this function when the datatype is already available, but
 *          the allocation info is needed from the context before jumping
 *          into recursion.
 *
 * Return:	Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5T_vlen_reclaim_elmt(void *elem, const H5T_t *dt)
{
    H5T_vlen_alloc_info_t vl_alloc_info;       /* VL allocation info */
    herr_t                ret_value = SUCCEED; /* return value */

    assert(dt);
    assert(elem);

    FUNC_ENTER_NOAPI(FAIL)

    /* Get VL allocation info */
    if (H5CX_get_vlen_alloc_info(&vl_alloc_info) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL, "unable to retrieve VL allocation info");

    /* Recurse on buffer to free dynamic fields */
    if (H5T__vlen_reclaim(elem, dt, &vl_alloc_info) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTFREE, FAIL, "can't reclaim vlen elements");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5T_vlen_reclaim_elmt() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_get_ctx
 *
 * Purpose:     Retrieves and validates the active chunk-local VL context.
 *
 *              The context identifies the file, the current structured
 *              chunk's heap set, and the width of the descriptor reference
 *              field. The heap set itself may legitimately be NULL before
 *              the first payload is written.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                              -- AZO   08/26/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_get_ctx(const H5T_vlen_chunk_ctx_t **ctx)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(ctx);

    *ctx = H5T_vlen_chunk_ctx_g;

    if (NULL == *ctx)
        HGOTO_ERROR(H5E_DATATYPE, H5E_UNINITIALIZED, FAIL, "chunk-local VL context is not active");

    if (NULL == (*ctx)->f)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL, "chunk-local VL context has no file");

    /*
     * The chunk must provide an owning heap-set pointer. The heap set itself
     * may still be NULL because it is created lazily by the first insertion.
     */
    if (NULL == (*ctx)->heapset)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL, "chunk-local VL context has no heap-set pointer");

    if ((*ctx)->ref_nbytes < H5T_VLEN_CHUNK_REF_SIZE)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL, "chunk-local VL reference field is too small");

done:
    if (ret_value < 0)
        *ctx = NULL;

    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5T__vlen_chunk_get_ctx() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_ref_decode
 *
 * Purpose:     Decodes a version-1 chunk-local VL reference.
 *
 *              The first two bytes contain the stable heap-slot index and
 *              the next two bytes contain the per-heap H5HG object index.
 *              Any remaining reference bytes must be zero.
 *
 *              {0,0} is the canonical null reference. Heap slot zero is a
 *              valid non-null slot when the object index is nonzero.
 *              A nonzero heap slot with object index zero is malformed.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                              -- AZO   08/26/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_ref_decode(const H5T_vlen_chunk_ctx_t *ctx, const uint8_t *ref, uint16_t *heap_slot,
                           uint16_t *obj_idx)
{
    const uint8_t *p = ref;
    uint16_t       slot16;
    uint16_t       idx16;
    size_t         u;
    herr_t         ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(ctx);
    assert(ref);
    assert(heap_slot);
    assert(obj_idx);

    *heap_slot = 0;
    *obj_idx   = 0;

    if (ctx->ref_nbytes < H5T_VLEN_CHUNK_REF_SIZE)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL, "chunk-local VL reference field is too small");

    UINT16DECODE(p, slot16);
    UINT16DECODE(p, idx16);

    /* All bytes not used by the V1 composite reference are reserved. */
    for (u = H5T_VLEN_CHUNK_REF_SIZE; u < ctx->ref_nbytes; u++)
        if (ref[u] != 0)
            HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                        "chunk-local VL descriptor has nonzero reserved bytes");

    /*
     * Object index zero is reserved for null. Heap slot zero itself is a
     * normal stable heap slot, so {0, nonzero} is valid.
     */
    if ((0 == idx16) && (0 != slot16))
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL, "chunk-local VL descriptor has invalid null reference");

    *heap_slot = slot16;
    *obj_idx   = idx16;

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5T__vlen_chunk_ref_decode() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_ref_setnull
 *
 * Purpose:     Determines whether a V1 chunk-local VL reference is the
 *              canonical null reference.
 *
 *              The reference is decoded and fully validated first, including
 *              reserved bytes and the heap-slot/object-index combination.
 *              Only {heap slot = 0, object index = 0} is null. Heap slot zero
 *              with a nonzero object index is a normal reference to the first
 *              member heap.
 *
 *              Sequence length is not considered here. The caller handles
 *              the separate distinction between a null VL value and a valid
 *              zero-length value that owns a real heap object.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                           -- AZO   07/20/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_ref_setnull(const H5T_vlen_chunk_ctx_t *ctx, uint8_t *ref)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* Check arguments */
    assert(ctx);
    assert(ref);

    if (ctx->ref_nbytes < H5T_VLEN_CHUNK_REF_SIZE)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL, "chunk-local VL reference field is too small");

    memset(ref, 0, ctx->ref_nbytes);

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5T__vlen_chunk_ref_setnull() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_ref_isnull
 *
 * Purpose:     Determines whether a chunk-local VL reference field contains
 *              the null reference.
 *
 *              Nullness is determined from the local heap index, not from
 *              the sequence length. This distinction is required because a
 *              valid empty VL string or zero-length sequence has length zero
 *              but may still own a nonzero heap object index.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                           -- AZO   07/20/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_ref_isnull(const H5T_vlen_chunk_ctx_t *ctx, const uint8_t *ref, bool *isnull)
{
    uint16_t heap_slot = 0;
    uint16_t obj_idx   = 0;
    herr_t   ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(ctx);
    assert(ref);
    assert(isnull);

    *isnull = false;

    if (H5T__vlen_chunk_ref_decode(ctx, ref, &heap_slot, &obj_idx) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTDECODE, FAIL, "unable to decode chunk-local VL reference");

    *isnull = ((0 == heap_slot) && (0 == obj_idx));

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5T__vlen_chunk_ref_isnull() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_getlen
 *
 * Purpose:     Retrieves the logical sequence length from a file-side
 *              chunk-local VL descriptor.
 *
 *              Chunk-local descriptors retain the existing H5T disk layout:
 *              a four-byte sequence length followed by a storage reference.
 *              Only the meaning of the storage reference differs.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                           -- AZO   07/23/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_getlen(H5VL_object_t H5_ATTR_UNUSED *file, const void *_vl, size_t *seq_len)
{
    const uint8_t *vl = (const uint8_t *)_vl;

    FUNC_ENTER_PACKAGE_NOERR

    /* Check arguments */
    assert(vl);
    assert(seq_len);

    /* Decode the logical length, which is stored in base-type elements */
    UINT32DECODE(vl, *seq_len);

    FUNC_LEAVE_NOAPI(SUCCEED)

} /* end H5T__vlen_chunk_getlen() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_isnull
 *
 * Purpose:     Determines whether a file-side chunk-local VL descriptor
 *              contains the null reference.
 *
 *              The four-byte sequence length is skipped and nullness is
 *              determined from the local heap reference. A zero sequence
 *              length alone does not imply null because a valid zero-length
 *              VL value may still own a heap object.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                              -- AZO   07/23/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_isnull(const H5VL_object_t H5_ATTR_UNUSED *file, void *_vl, bool *isnull)
{
    const H5T_vlen_chunk_ctx_t *ctx       = NULL;
    const uint8_t              *p         = (const uint8_t *)_vl;
    size_t                      seq_len   = 0;
    bool                        ref_null  = false;
    herr_t                      ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(p);
    assert(isnull);

    *isnull = false;

    if (H5T__vlen_chunk_get_ctx(&ctx) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL, "unable to retrieve chunk-local VL context");

    UINT32DECODE(p, seq_len);

    if (H5T__vlen_chunk_ref_isnull(ctx, p, &ref_null) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL,
                    "unable to determine whether chunk-local VL reference is null");

    /*
     * A nonzero logical length cannot have the canonical null reference.
     * A zero logical length with a non-null reference is a valid empty VL.
     */
    if ((seq_len > 0) && ref_null)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                    "non-empty chunk-local VL descriptor has null reference");

    *isnull = ref_null;

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5T__vlen_chunk_isnull() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_setnull
 *
 * Purpose:     Sets a file-side chunk-local VL descriptor to null.
 *
 *              If a background descriptor is supplied, its referenced heap
 *              object is removed first. The destination descriptor is then
 *              written with a zero sequence length and an all-zero local
 *              reference field.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                               -- AZO   07/23/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_setnull(H5VL_object_t H5_ATTR_UNUSED *file, void *_vl, void *_bg)
{
    const H5T_vlen_chunk_ctx_t *ctx       = NULL;
    uint8_t                    *vl        = (uint8_t *)_vl;
    herr_t                      ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* Check arguments */
    assert(vl);

    /* Retrieve and validate the active chunk-local VL context */
    if (H5T__vlen_chunk_get_ctx(&ctx) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL, "unable to retrieve chunk-local VL context");

    /*
     * Release the object referenced by the old destination before publishing
     * the null descriptor. Once the descriptor is cleared, that object could
     * no longer be located for deletion.
     */
    if (_bg)
        if (H5T__vlen_chunk_delete(file, _bg) < 0)
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL,
                        "unable to remove background chunk-local VL object");

    /* Encode the null sequence length */
    UINT32ENCODE(vl, 0);

    /* Encode the null local reference after the sequence length */
    if (H5T__vlen_chunk_ref_setnull(ctx, vl) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTSET, FAIL, "unable to set chunk-local VL reference to null");

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5T__vlen_chunk_setnull() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_read
 *
 * Purpose:     Reads the payload referenced by a chunk-local file
 *              descriptor into H5T's conversion buffer.
 *
 *              The descriptor identifies the payload with a stable heap
 *              slot and a per-heap H5HG object index. The active chunk
 *              context supplies the heap set in which that reference is
 *              resolved.
 *
 *              H5HG is first called with a NULL destination to query the
 *              object's actual logical size. That size must exactly match
 *              LEN before the destination buffer is modified. A second
 *              call then copies the validated payload into the caller's
 *              buffer.
 *
 *              This deliberately uses the query/fill contract of
 *              H5HG__read_local_heapset() so a malformed or inconsistent
 *              descriptor is detected before any payload bytes are copied.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                              -- AZO   08/26/26
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_read(H5VL_object_t H5_ATTR_UNUSED *file, void *_vl, void *_buf, size_t len)
{
    const H5T_vlen_chunk_ctx_t *ctx       = NULL;
    const uint8_t              *vl        = (const uint8_t *)_vl;
    uint16_t                    heap_slot = 0;
    uint16_t                    obj_idx   = 0;
    size_t                      obj_size  = 0;
    size_t                      read_size = len;
    herr_t                      ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(vl);

    if (H5T__vlen_chunk_get_ctx(&ctx) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL, "unable to retrieve chunk-local VL context");

    if (H5T__vlen_chunk_ref_decode(ctx, vl + 4, &heap_slot, &obj_idx) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTDECODE, FAIL, "unable to decode chunk-local VL reference");

    if (0 == obj_idx)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL, "cannot read through a null chunk-local VL reference");

    if (NULL == *ctx->heapset)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                    "chunk-local VL descriptor refers to a missing heap set");

    /*
     * Validate the stored object size before touching the destination buffer.
     * A NULL destination requests only the object's logical size.
     */
    if (H5HG__read_local_heapset(ctx->f, *ctx->heapset, heap_slot, obj_idx, NULL, &obj_size) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL, "unable to retrieve chunk-local VL object size");

    if (obj_size != len)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                    "chunk-local VL object size does not match descriptor length");

    /*
     * A real zero-length payload has already been completely validated.
     * There are no bytes to copy.
     */
    if (0 == len)
        HGOTO_DONE(SUCCEED);

    if (NULL == _buf)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "chunk-local VL read requires a destination buffer");

    if (H5HG__read_local_heapset(ctx->f, *ctx->heapset, heap_slot, obj_idx, _buf, &read_size) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_READERROR, FAIL, "unable to read chunk-local VL object");

    assert(read_size == len);

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5T__vlen_chunk_read() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_write
 *
 * Purpose:     Stores a VL payload in the current structured chunk's H5HG
 *              heap set and publishes the corresponding file-side
 *              descriptor.
 *
 *              The logical byte count is computed from SEQ_LEN and
 *              BASE_SIZE. H5HG__insert_local_heapset() then selects or
 *              creates the appropriate member heap and returns a stable
 *              heap slot plus a per-heap H5HG object index.
 *
 *              A real H5HG object is created even when the payload size is
 *              zero. This preserves the distinction between a valid empty
 *              VL value, which has a nonzero object index, and the canonical
 *              null value, whose reference is {0,0}.
 *
 *              For replacement writes, the new object is inserted before
 *              the object referenced by the background descriptor is
 *              removed. The new descriptor is published only after that
 *              work succeeds. If an error occurs before publication, the
 *              newly inserted object is removed so that no unreachable
 *              heap object is leaked.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                               -- AZO   07/23/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_write(H5VL_object_t H5_ATTR_UNUSED               *file,
                      const H5T_vlen_alloc_info_t H5_ATTR_UNUSED *vl_alloc_info, void *_vl, void *_buf,
                      void *_bg, size_t seq_len, size_t base_size)
{
    const H5T_vlen_chunk_ctx_t *ctx          = NULL;
    uint8_t                    *vl           = (uint8_t *)_vl;
    uint8_t                    *p            = NULL;
    size_t                      payload_size = 0;
    uint16_t                    heap_slot    = 0;
    uint16_t                    obj_idx      = 0;
    bool                        inserted     = false;
    herr_t                      ret_value    = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(vl);

    if ((seq_len > 0) && (NULL == _buf))
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "non-empty chunk-local VL value has no source buffer");

    if (H5T__vlen_chunk_get_ctx(&ctx) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL, "unable to retrieve chunk-local VL context");

    if (seq_len > UINT32_MAX)
        HGOTO_ERROR(H5E_DATATYPE, H5E_OVERFLOW, FAIL, "VL sequence length does not fit in descriptor");

    if ((base_size > 0) && (seq_len > (SIZE_MAX / base_size)))
        HGOTO_ERROR(H5E_DATATYPE, H5E_OVERFLOW, FAIL, "chunk-local VL payload size overflows size_t");

    payload_size = seq_len * base_size;

    /*
     * Let the heap-set manager select or create the appropriate member heap.
     * SIZE == 0 is valid and creates a real zero-length H5HG object.
     */
    if (H5HG__insert_local_heapset(ctx->f, ctx->heapset, payload_size, _buf, &heap_slot, &obj_idx) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINSERT, FAIL, "unable to insert chunk-local VL object");

    inserted = true;

    /*
     * Object index zero is reserved for the null reference. Heap slot zero is
     * the first normal member heap and is therefore valid for a live object.
     */
    if (0 == obj_idx)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                    "chunk-local heap set returned reserved object index zero");

    /*
     * Preserve the old descriptor until replacement storage is known to
     * exist. This is important when _BG aliases _VL.
     */
    if (_bg)
        if (H5T__vlen_chunk_delete(file, _bg) < 0)
            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL,
                        "unable to remove background chunk-local VL object");

    /*
     * All fallible work is complete. Publish the new descriptor.
     */
    p = vl;
    UINT32ENCODE(p, seq_len);

    memset(p, 0, ctx->ref_nbytes);
    UINT16ENCODE(p, heap_slot);
    UINT16ENCODE(p, obj_idx);

    inserted = false;

done:
    /*
     * If replacement failed before publication, remove the newly inserted
     * object. The outer heap-set manager remains owned by the chunk.
     */
    if ((ret_value < 0) && inserted && ctx && ctx->heapset && *ctx->heapset)
        if (H5HG__remove_local_heapset(ctx->f, *ctx->heapset, heap_slot, obj_idx) < 0)
            HDONE_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL,
                        "unable to remove unreferenced chunk-local VL object");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T__vlen_chunk_write() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_delete
 *
 * Purpose:     Releases the H5HG object referenced by a file-side
 *              chunk-local VL descriptor.
 *
 *              The descriptor's composite reference identifies a stable
 *              member-heap slot and a per-heap object index. The active
 *              chunk context supplies the heap set in which that reference
 *              is resolved.
 *
 *              The canonical {0,0} reference requires no deletion and is
 *              valid only with a zero sequence length. A zero-length value
 *              with a nonzero object index is a real object and must still
 *              be removed.
 *
 *              H5HG__remove_local_heapset() owns member-heap cleanup,
 *              stable-slot handling, and trailing-hole trimming. This
 *              routine does not modify the descriptor and does not free the
 *              outer heap set, which remains owned by the structured chunk.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                               -- AZO   07/23/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_delete(H5VL_object_t H5_ATTR_UNUSED *file, void *_vl)
{
    const H5T_vlen_chunk_ctx_t *ctx       = NULL;
    const uint8_t              *p         = (const uint8_t *)_vl;
    size_t                      seq_len   = 0;
    uint16_t                    heap_slot = 0;
    uint16_t                    obj_idx   = 0;
    herr_t                      ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    if (NULL == p)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "chunk-local VL delete requires a descriptor");

    if (H5T__vlen_chunk_get_ctx(&ctx) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL, "unable to retrieve chunk-local VL context");

    UINT32DECODE(p, seq_len);

    if (H5T__vlen_chunk_ref_decode(ctx, p, &heap_slot, &obj_idx) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTDECODE, FAIL, "unable to decode chunk-local VL reference");

    /*
     * {0,0} is valid only for a zero-length null descriptor.
     */
    if (0 == obj_idx) {
        if (seq_len > 0)
            HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                        "non-empty chunk-local VL descriptor has null reference");

        HGOTO_DONE(SUCCEED);
    }

    if (NULL == *ctx->heapset)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                    "chunk-local VL descriptor refers to a missing heap set");

    if (H5HG__remove_local_heapset(ctx->f, *ctx->heapset, heap_slot, obj_idx) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL, "unable to remove chunk-local VL object");

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5T__vlen_chunk_delete() */

/*-------------------------------------------------------------------------
 * Function:    H5T__vlen_chunk_patch_cb
 *
 * Purpose:     H5T__visit() callback that switches each file-side H5T_VLEN
 *              node in a private datatype copy to the structured-chunk VL
 *              backend.
 *
 *              Non-VLEN nodes are left unchanged; H5T__visit() is
 *              responsible for traversing the containing datatype
 *              hierarchy.
 *
 *              Chunk-local VL descriptors deliberately retain the existing
 *              file-side descriptor width. The first four bytes remain the
 *              sequence length. V1 interprets the first four bytes of the
 *              following reference field as a 16-bit stable heap slot and
 *              a 16-bit H5HG object index; any additional reference bytes
 *              remain reserved.
 *
 *              This callback changes only the VL callback class. It does
 *              not resize the datatype or alter its H5T_LOC_DISK location,
 *              file pointer, or VOL ownership state. Those properties must
 *              remain unchanged because surrounding datatype layout was
 *              already computed using the existing file descriptor size.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                               -- AZO   07/24/26
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5T__vlen_chunk_patch_cb(H5T_t *dt, void *op_data)
{
    size_t ref_nbytes;
    size_t expected_size;
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    /* Check arguments supplied by H5T__visit() */
    assert(dt);
    assert(dt->shared);
    assert(op_data);

    ref_nbytes = *(const size_t *)op_data;

    /* Only variable-length datatype nodes require a different backend */
    if (H5T_VLEN != dt->shared->type)
        HGOTO_DONE(SUCCEED);

    /*
     * This backend operates on file-side descriptors, not hvl_t or char *
     * memory representations. Requiring a disk-located node also prevents
     * accidentally modifying the application's memory datatype.
     */
    if (H5T_LOC_DISK != dt->shared->u.vlen.loc)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                    "cannot install chunk-local backend on a non-disk VL datatype");

    /*
     * Overflow was checked by H5T_patch_vlen_chunk_local(), but calculate
     * the expected size locally to make the descriptor requirement clear.
     */
    expected_size = 4 + ref_nbytes;

    /*
     * Never resize the datatype here. Compound offsets, array element sizes,
     * and enclosing VL base-type sizes were calculated using the existing
     * file descriptor size. A mismatch means the caller supplied the wrong
     * reference-field width for this datatype.
     */
    if (dt->shared->size != expected_size)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                    "chunk-local VL reference size does not match file datatype");

    /*
     * Replace only the payload-access implementation. The location remains
     * H5T_LOC_DISK and the file/VOL ownership information remains untouched.
     */
    dt->shared->u.vlen.cls = &H5T_vlen_chunk_g;

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5T__vlen_chunk_patch_cb() */

/*-------------------------------------------------------------------------
 * Function:    H5T_patch_vlen_chunk_local
 *
 * Purpose:     Installs the structured chunk-local VL backend on every
 *              variable-length node in an operation-local copy of a
 *              file-side datatype.
 *
 *              The datatype may be a direct VL sequence/string or may
 *              contain VL values recursively through compounds, arrays,
 *              enums, or other VL base types. H5T__visit() performs the
 *              recursive traversal so that each H5T_VLEN node receives the
 *              chunk-local callback class.
 *
 *              This routine does not copy the datatype. The caller must pass
 *              a private operation-local copy and must not pass the dataset's
 *              shared datatype directly. Otherwise the normal global
 *              disk/blob backend could be changed for unrelated I/O.
 *
 *              REF_NBYTES is the width of the existing storage-reference
 *              field following the four-byte sequence length. V1 requires
 *              at least four bytes for a 16-bit stable heap slot followed
 *              by a 16-bit H5HG object index. Any additional reference
 *              bytes remain reserved.
 *
 * Return:      Non-negative on success / Negative on failure
 *
 *                                               -- AZO   07/24/26
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5T_patch_vlen_chunk_local(H5T_t *dt, size_t ref_nbytes)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    /* Check argument */
    assert(dt);
    assert(dt->shared);

    /*
     * V1 requires four reference bytes: two for the stable member-heap slot
     * and two for the per-heap H5HG object index.
     */
    if (ref_nbytes < H5T_VLEN_CHUNK_REF_SIZE)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL, "chunk-local VL reference field is too small");

    /*
     * Ensure that adding the four-byte sequence length cannot overflow.
     */
    if (ref_nbytes > (SIZE_MAX - 4))
        HGOTO_ERROR(H5E_DATATYPE, H5E_OVERFLOW, FAIL, "chunk-local VL descriptor size overflows size_t");

    /*
     * VL nodes are composite datatypes. Visit each composite before its
     * children and patch onlyh nodes whose internal class is H5T_VLEN.
     * Simple atomic leaf types do not need to be visited.
     */
    if (H5T__visit(dt, H5T_VISIT_COMPOSITE_FIRST, H5T__vlen_chunk_patch_cb, &ref_nbytes) < 0)
        HGOTO_ERROR(H5E_DATATYPE, H5E_BADITER, FAIL, "unable to install chunk-local VL backend");

done:
    FUNC_LEAVE_NOAPI(ret_value)

} /* end H5T_patch_vlen_chunk_local() */

/*-------------------------------------------------------------------------
 * Function:    H5T_set_vlen_chunk_ctx
 *
 * Purpose:     Installs the context used by chunk-local VL callbacks and
 *              returns the previously active context.
 *
 *              The structured-chunk path installs a per-chunk context
 *              immediately before invoking H5T conversion and restores the
 *              returned pointer when that conversion completes. Returning
 *              the previous context also permits properly nested uses.
 *
 *              The current implementation uses one H5T-global context
 *              pointer and therefore assumes structured-chunk VL conversion
 *              is single-threaded. The storage can later be changed to TLS
 *              without changing this interface.
 *
 * Return:      Previously active chunk-local VL context, possibly NULL.
 *
 *                                              -- AZO   08/26/26
 *-------------------------------------------------------------------------
 */
const H5T_vlen_chunk_ctx_t *
H5T_set_vlen_chunk_ctx(const H5T_vlen_chunk_ctx_t *ctx)
{
    const H5T_vlen_chunk_ctx_t *old_ctx;

    FUNC_ENTER_NOAPI_NOINIT_NOERR

    old_ctx              = H5T_vlen_chunk_ctx_g;
    H5T_vlen_chunk_ctx_g = ctx;

    FUNC_LEAVE_NOAPI(old_ctx)

} /* end H5T_set_vlen_chunk_ctx() */

/*-------------------------------------------------------------------------
 * Function:    H5T_vlen_delete_file_elmt
 *
 * Purpose:     Recursively deletes every file-side VL payload referenced by
 *              one element of DT.
 *
 *              Unlike H5T_vlen_reclaim_elmt(), this routine operates on the
 *              fixed-size file representation rather than hvl_t or char *
 *              memory representations.
 *
 *              For nested VL sequences, the containing payload is copied
 *              into temporary memory before its children are deleted. This
 *              is required because deleting a child may compact its member
 *              heap and move other heap objects.
 *
 * Return:      SUCCEED/FAIL
 *
 *                                                 -- AZO   09/15/26
 *-------------------------------------------------------------------------
 */
herr_t
H5T_vlen_delete_file_elmt(void *elem, const H5T_t *dt)
{
    void  *payload = NULL;
    size_t u;
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(elem);
    assert(dt);
    assert(dt->shared);

    switch (dt->shared->type) {
        case H5T_ARRAY:
            /*
             * File-side array elements are stored inline using the parent
             * datatype's fixed representation.
             */
            if (H5T_IS_COMPOSITE(dt->shared->parent->shared->type)) {
                size_t parent_size = dt->shared->parent->shared->size;

                for (u = 0; u < dt->shared->u.array.nelem; u++) {
                    if (H5T_vlen_delete_file_elmt((uint8_t *)elem + (u * parent_size), dt->shared->parent) <
                        0)
                        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL,
                                    "unable to delete VL payload from file-side array element");
                }
            }
            break;

        case H5T_COMPOUND:
            for (u = 0; u < dt->shared->u.compnd.nmembs; u++) {
                const H5T_t *member_type = dt->shared->u.compnd.memb[u].type;

                if (H5T_IS_COMPOSITE(member_type->shared->type))
                    if (H5T_vlen_delete_file_elmt((uint8_t *)elem + dt->shared->u.compnd.memb[u].offset,
                                                  member_type) < 0)
                        HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL,
                                    "unable to delete VL payload from file-side compound member");
            }
            break;

        case H5T_VLEN: {
            const H5T_vlen_class_t *cls;
            size_t                  seq_len = 0;
            size_t                  payload_size;
            bool                    is_null = false;

            if (H5T_LOC_DISK != dt->shared->u.vlen.loc)
                HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                            "VL delete helper requires a disk-located datatype");

            if (NULL == (cls = dt->shared->u.vlen.cls))
                HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL, "file-side VL datatype has no callback class");

            if (!cls->isnull || !cls->del)
                HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                            "file-side VL callback class cannot delete values");

            if ((*(cls->isnull))(dt->shared->u.vlen.file, elem, &is_null) < 0)
                HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL,
                            "unable to determine whether file-side VL value is null");

            if (is_null)
                break;

            /*
             * VL strings cannot contain nested datatype descriptors. A VL
             * sequence can, depending on its base datatype.
             */
            if (H5T_VLEN_SEQUENCE == dt->shared->u.vlen.type &&
                H5T_IS_COMPOSITE(dt->shared->parent->shared->type)) {

                size_t parent_size = dt->shared->parent->shared->size;

                if (!cls->getlen || !cls->read)
                    HGOTO_ERROR(H5E_DATATYPE, H5E_BADVALUE, FAIL,
                                "file-side VL callback class cannot read nested values");

                if ((*(cls->getlen))(dt->shared->u.vlen.file, elem, &seq_len) < 0)
                    HGOTO_ERROR(H5E_DATATYPE, H5E_CANTGET, FAIL,
                                "unable to retrieve nested file-side VL sequence length");

                if (parent_size > 0 && seq_len > SIZE_MAX / parent_size)
                    HGOTO_ERROR(H5E_DATATYPE, H5E_OVERFLOW, FAIL,
                                "nested file-side VL payload size overflows size_t");

                payload_size = seq_len * parent_size;

                if (payload_size > 0) {
                    if (NULL == (payload = H5MM_malloc(payload_size)))
                        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                                    "unable to allocate nested VL deletion buffer");

                    /*
                     * Copy the complete containing payload before deleting
                     * children. Heap deletion may compact resident heap data.
                     */
                    if ((*(cls->read))(dt->shared->u.vlen.file, elem, payload, payload_size) < 0)
                        HGOTO_ERROR(H5E_DATATYPE, H5E_READERROR, FAIL,
                                    "unable to read nested file-side VL payload");

                    for (u = 0; u < seq_len; u++)
                        if (H5T_vlen_delete_file_elmt((uint8_t *)payload + (u * parent_size),
                                                      dt->shared->parent) < 0)
                            HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL,
                                        "unable to delete nested file-side VL element");
                }
            } /* end if */

            /*
             * Delete the containing payload after all nested payloads have
             * been deleted.
             */
            if ((*(cls->del))(dt->shared->u.vlen.file, elem) < 0)
                HGOTO_ERROR(H5E_DATATYPE, H5E_CANTREMOVE, FAIL, "unable to delete file-side VL payload");

            break;
        }

        /*
         * Atomic datatypes contain no recursively owned VL payload.
         */
        case H5T_INTEGER:
        case H5T_FLOAT:
        case H5T_TIME:
        case H5T_STRING:
        case H5T_BITFIELD:
        case H5T_OPAQUE:
        case H5T_ENUM:
        case H5T_COMPLEX:
            break;

        case H5T_REFERENCE:
        case H5T_NO_CLASS:
        case H5T_NCLASSES:
        default:
            HGOTO_ERROR(H5E_DATATYPE, H5E_BADRANGE, FAIL,
                        "invalid datatype class during file-side VL deletion");
    }

done:
    payload = H5MM_xfree(payload);

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5T_vlen_delete_file_elmt() */