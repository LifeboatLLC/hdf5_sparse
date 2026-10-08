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

#ifndef H5HGprivate_H
#define H5HGprivate_H

/* Private headers needed by this file. */
#include "H5Fprivate.h" /* File access				*/

/* Information to locate object in global heap */
typedef struct H5HG_t {
    haddr_t addr; /*address of collection		*/
    size_t  idx;  /*object ID within collection	*/
} H5HG_t;

/* Typedef for heap in memory (defined in H5HGpkg.h) */
typedef struct H5HG_heap_t H5HG_heap_t;

/* Typedef for heapset in memory (defined in H5HGpkg.h) */
typedef struct H5HG_local_heapset_t H5HG_local_heapset_t;

/*
 * Limit global heap collections to the some reasonable size.  This is
 * fairly arbitrary, but needs to be small enough that no more than H5HG_MAXIDX
 * objects will be allocated from a single heap.
 */
#define H5HG_MAXSIZE 65536

/* If the module using this macro is allowed access to the private variables, access them directly */
#ifdef H5HG_MODULE
#define H5HG_ADDR(H)      ((H)->addr)
#define H5HG_SIZE(H)      ((H)->size)
#define H5HG_FREE_SIZE(H) ((H)->obj[0].size)
#else /* H5HG_MODULE */
#define H5HG_ADDR(H)      (H5HG_get_addr(H))
#define H5HG_SIZE(H)      (H5HG_get_size(H))
#define H5HG_FREE_SIZE(H) (H5HG_get_free_size(H))
#endif /* H5HG_MODULE */

/* Size of encoded global heap ID */
/* (size of file address + 32-bit integer) */
#define H5HG_HEAP_ID_SIZE(F) ((size_t)H5F_SIZEOF_ADDR(F) + sizeof(uint32_t))

/* Main global heap routines */
H5_DLL herr_t H5HG_insert(H5F_t *f, size_t size, const void *obj, H5HG_t *hobj /*out*/);
H5_DLL void  *H5HG_read(H5F_t *f, H5HG_t *hobj, void *object, size_t *buf_size /*out*/);
H5_DLL int    H5HG_link(H5F_t *f, const H5HG_t *hobj, int adjust);
H5_DLL herr_t H5HG_get_obj_size(H5F_t *f, H5HG_t *hobj, size_t *obj_size);
H5_DLL herr_t H5HG_remove(H5F_t *f, H5HG_t *hobj);

/* Support routines */
H5_DLL herr_t H5HG_extend(H5F_t *f, haddr_t addr, size_t need);

/* Query routines */
H5_DLL haddr_t H5HG_get_addr(const H5HG_heap_t *h);
H5_DLL size_t  H5HG_get_size(const H5HG_heap_t *h);
H5_DLL size_t  H5HG_get_free_size(const H5HG_heap_t *h);

/* Debugging functions */
H5_DLL herr_t H5HG_debug(H5F_t *f, haddr_t addr, FILE *stream, int indent, int fwidth);

/* Main local heap routines */
H5_DLL H5HG_heap_t *H5HG__create_local(H5F_t *f, size_t init_size);
H5_DLL herr_t H5HG__insert_local(H5F_t *f, H5HG_heap_t *heap, size_t size, const void *obj, size_t *idx_out);
H5_DLL herr_t H5HG__read_local(H5F_t *f, const H5HG_heap_t *heap, size_t idx, void *object, size_t *buf_size);
H5_DLL herr_t H5HG__remove_local(H5F_t *f, H5HG_heap_t *heap, size_t idx, hbool_t *heap_empty);
H5_DLL htri_t H5HG__is_empty_local(const H5HG_heap_t *heap);
H5_DLL herr_t H5HG__get_obj_size_local(H5F_t *f, const H5HG_heap_t *heap, size_t idx, size_t *obj_size);
H5_DLL herr_t H5HG__free_local(H5HG_heap_t *heap);

/* Chunk-local heap image encoding/decoding */
H5_DLL herr_t       H5HG__encode_local(H5F_t *f, const H5HG_heap_t *heap, uint8_t **image_out,
                                       size_t *image_len_out);
H5_DLL H5HG_heap_t *H5HG__decode_local(H5F_t *f, const void *image, size_t len);

/* Chunk-local heap-set routines */
H5_DLL H5HG_local_heapset_t *H5HG__create_local_heapset(void);
H5_DLL herr_t H5HG__insert_local_heapset(H5F_t *f, H5HG_local_heapset_t **heapset_ptr, size_t size,
                                         const void *obj, uint16_t *heap_slot_out, uint16_t *obj_idx_out);
H5_DLL herr_t H5HG__read_local_heapset(H5F_t *f, const H5HG_local_heapset_t *heapset, uint16_t heap_slot,
                                       uint16_t obj_idx, void *object, size_t *buf_size);
H5_DLL herr_t H5HG__remove_local_heapset(H5F_t *f, H5HG_local_heapset_t *heapset, uint16_t heap_slot,
                                         uint16_t obj_idx);
H5_DLL htri_t H5HG__is_empty_local_heapset(const H5HG_local_heapset_t *heapset);
H5_DLL herr_t H5HG__free_local_heapset(H5HG_local_heapset_t *heapset);
H5_DLL herr_t H5HG__copy_local_heapset(H5F_t *f, const H5HG_local_heapset_t *heapset,
                                       H5HG_local_heapset_t **copy_out);
H5_DLL herr_t H5HG__get_local_heapset_alloc_size(const H5HG_local_heapset_t *heapset, size_t *size_out);

/* Complete logical H5_SECTION_VL heap-set image encoding/decoding */
H5_DLL herr_t H5HG__encode_local_heapset(H5F_t *f, const H5HG_local_heapset_t *heapset, uint8_t **image_out,
                                         size_t *image_len_out);
H5_DLL H5HG_local_heapset_t *H5HG__decode_local_heapset(H5F_t *f, const void *image, size_t len);

#endif /* H5HGprivate_H */
