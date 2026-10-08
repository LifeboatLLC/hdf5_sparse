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

/*
 * Purpose:    Tests sparse storage based on the concept of structured chunk
 */
#define H5D_FRIEND /*suppress error about including H5Dpkg      */
#define H5D_TESTING
#define H5FD_FRIEND /*suppress error about including H5FDpkg      */
#define H5FD_TESTING

#define H5Z_FRIEND /*suppress error about including H5Zpkg      */

#include "testhdf5.h"
#include "H5srcdir.h"

#include "H5CXprivate.h" /* API Contexts                         */
#include "H5Iprivate.h"
#include "H5Pprivate.h"

#define H5F_FRIEND /*suppress error about including H5Fpkg */
#define H5F_TESTING
#include "H5Fpkg.h" /* File access                          */

#define H5S_FRIEND  /*suppress error about including H5Spkg */
#include "H5Spkg.h" /* Dataspace                            */

#define H5T_FRIEND  /*suppress error about including H5Tpkg */
#include "H5Tpkg.h" /* Datatype                             */

#define H5A_FRIEND  /*suppress error about including H5Apkg     */
#include "H5Apkg.h" /* Attributes                   */

/* Use in version bound test */
#define H5O_FRIEND  /*suppress error about including H5Opkg */
#include "H5Opkg.h" /* Object headers                       */

#include "H5Dpkg.h"
#include "H5FDpkg.h"
#include "H5VMprivate.h"
#include "H5Zpkg.h"

#define H5SC_FRIEND
#include "H5SCpkg.h"
#include "H5SCprivate.h"

#define H5HG_FRIEND /* Inspect local heap-set invariants in the unit test */
#include "H5HGpkg.h"
#include "H5HGprivate.h"

static const char *FILENAME[] = {"struct_chunk_api",             /* 0 */
                                 "struct_chunk_1d",              /* 1 */
                                 "struct_chunk_2d",              /* 2 */
                                 "struct_chunk_filter_1d",       /* 3 */
                                 "struct_chunk_filter_2d",       /* 4 */
                                 "struct_chunk_filter_register", /* 5 */
                                 "struct_chunk_vlen",            /* 6 */
                                 NULL};

#ifdef TBD
static const char *FILENAME_TBD[] = {"sparse",                    /* 0 */
                                     "sparse_direct_chunk",       /* 1 */
                                     "sparse_query_direct_chunk", /* 2 */
                                     "sparse_dense_api",          /* 3 */
                                     NULL};
#endif

#define FILENAME_BUF_SIZE 1024

#define EXT1_SPARSE_DSET "ext1_sparse_dset"
#define EXT2_SPARSE_DSET "ext2_sparse_dset"

#define SPARSE_DSET        "sparse_dset"
#define SPARSE_DSET2       "sparse_dset2"
#define SPARSE_FILTER_DSET "sparse_filter_dset"

#define CHUNKED_DSET "chunked_dset"

#define RANK     2
#define NX       10
#define NY       10
#define CHUNK_NX 5
#define CHUNK_NY 5

#define CHK_SINGLE 1
#define CHK_FA     2
#define CHK_EA     3

/* Size of a chunk */
#define CHK_SIZE (CHUNK_NX * CHUNK_NY * sizeof(int))

static herr_t test_struct_chunk_info_1d(hid_t fcpl, hid_t fapl, bool filtered, bool early, unsigned chk_type);
static herr_t test_struct_chunk_info_2d_bt2(hid_t fcpl, hid_t fapl, bool filtered, bool early);
static herr_t test_struct_chunk_extent_1d(hid_t fcpl, hid_t fapl, bool filtered, bool early);
static herr_t test_struct_chunk_extent_2d(hid_t fcpl, hid_t fapl, bool filtered, bool early, bool expand);

static herr_t test_struct_chunk_api(hid_t fcpl, hid_t fapl);
static herr_t test_struct_chunk_1d_single(hid_t fcpl, hid_t fapl, bool filtered, bool early);
static herr_t test_struct_chunk_2d_bt2(hid_t fcpl, hid_t fapl, bool filtered, bool early);
static herr_t test_struct_chunk_1d_fa(hid_t fcpl, hid_t fapl, bool filtered, bool early);
static herr_t test_struct_chunk_2d_ea(hid_t fcpl, hid_t fapl, bool filtered, bool early);
static herr_t test_struct_chunk_filter_register(hid_t fcpl, hid_t fapl);

static herr_t filter_class3_set_local(hid_t dcpl_id, hid_t type_id, hid_t H5_ATTR_UNUSED space_id,
                                      H5_section_type_t sec_type);
static size_t filter_class3(unsigned int flags, size_t cd_nelmts, const unsigned int *cd_values,
                            size_t nbytes, size_t *buf_size, void **buf);

static herr_t test_struct_chunk_vlen(hid_t fcpl, hid_t fapl, unsigned chk_type, bool filtered);
static herr_t test_struct_chunk_vlen_partial(hid_t fcpl, hid_t fapl, bool filtered);
static herr_t test_struct_chunk_vlen_churn(hid_t fcpl, hid_t fapl, bool filtered);
static herr_t test_struct_chunk_vlen_large(hid_t fcpl, hid_t fapl, unsigned chk_type, bool filtered);
static herr_t test_struct_chunk_vlen_compound(hid_t fcpl, hid_t fapl, bool filtered);
static herr_t test_struct_chunk_vlen_two_members(hid_t fcpl, hid_t fapl, bool filtered);

static herr_t test_struct_chunk_vlen_erase(hid_t fcpl, hid_t fapl, bool filtered);
static herr_t test_struct_chunk_vlen_empty_section(hid_t fcpl, hid_t fapl, bool filtered);
static herr_t test_struct_chunk_vlen_type_conversion(hid_t fcpl, hid_t fapl, bool filtered);
static herr_t test_struct_chunk_vlen_lifecycle(hid_t fcpl, hid_t fapl, unsigned chk_type, bool filtered);
static herr_t test_struct_chunk_vlen_stress(hid_t fcpl, hid_t fapl, unsigned chk_type, bool filtered,
                                            size_t iterations, uint64_t seed);
static herr_t test_struct_chunk_vlen_dense_emulation(hid_t fcpl, hid_t fapl, unsigned chk_type,
                                                     bool filtered);
static herr_t test_local_heapset_stable_slots(hid_t fcpl, hid_t fapl);
static herr_t test_local_heapset_growth(hid_t fcpl, hid_t fapl);
static herr_t test_local_heapset_slot_growth(hid_t fcpl, hid_t fapl);
static herr_t test_struct_chunk_vlen_eviction(hid_t fcpl, hid_t base_fapl, bool filtered);

#define VL_STRESS_NELMTS             64
#define VL_STRESS_MAX_LEN            257
#define VL_STRESS_DEFAULT_ITERATIONS 96

#define H5Z_FILTER_CLASS3 305
#define FILTER_PARAM      9 /* No particular meaning, just for checking */
#define FILTER_PARAM_MOD  3 /* No particular meaning, just for checking */

size_t filter_bytes_read    = 0;
size_t filter_bytes_written = 0;

/* This message derives from H5Z */
const H5Z_class3_t H5Z_TEST_CLASS3[1] = {{
    H5Z_CLASS_T_VERS, H5Z_FILTER_CLASS3, /* Filter id number        */
    1, 1, "test_class3",                 /* Filter name for debugging    */
    NULL,                                /* The "can apply" callback     */
    filter_class3_set_local,             /* The "set local" callback     */
    filter_class3,                       /* The actual filter function    */
}};

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_info_1d
 *
 * Purpose:     Verify H5Oget_native_info() for 1d dataset with
 *              fixed array or extensible array chunk index
 *
 * Return:      Success:        0
 *              Failure:        -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_info_1d(hid_t fcpl, hid_t fapl, bool filtered, bool early, unsigned chk_type)
{
    char  filename[FILENAME_BUF_SIZE]; /* File name */
    hid_t fid  = H5I_INVALID_HID;
    hid_t sid  = H5I_INVALID_HID;
    hid_t msid = H5I_INVALID_HID;
    hid_t dcpl = H5I_INVALID_HID;
    hid_t did  = H5I_INVALID_HID;

    hsize_t sg_dim[1]       = {30}; /* 1-d dataspace */
    hsize_t sg_chunk_dim[1] = {30}; /* Chunk size */

    hsize_t fa_dim[1]     = {30}; /* 1-d dataspace */
    hsize_t fa_max_dim[1] = {50};

    hsize_t ea_dim[1]     = {30}; /* 1-d dataspace */
    hsize_t ea_max_dim[1] = {H5S_UNLIMITED};

    hsize_t chunk_dim[1] = {5}; /* Chunk size */

    H5D_chunk_index_t idx_type; /* dataset chunk index type */

    int     wbuf[30]; /* Write buffer */
    int     wvals[9];
    hsize_t mdim[1];

    hsize_t start[1];
    hsize_t stride[1];
    hsize_t count[1];
    hsize_t block[1];

    unsigned int level        = 9;
    unsigned int cd_values[1] = {level};
    size_t       cd_nelmts    = 1;

    int      nfilters;
    unsigned options;

    H5O_native_info_t nat_info;

    H5F_libver_t low, high; /* File format bound */
    bool         fail_as_expected = false;

    TESTING("structured chunk with H5Oget_native_info() on 1d dataset with Single/Fixed/Extensible array "
            "chunk index type");

    if (H5Pget_libver_bounds(fapl, &low, &high) < 0)
        TEST_ERROR;

    /* Create a file */
    h5_fixname(FILENAME[1], fapl, filename, sizeof filename);
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create property list for dataset creation */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl, 1, (chk_type == CHK_SINGLE ? sg_chunk_dim : chunk_dim), H5D_SPARSE_CHUNK) <
        0)
        TEST_ERROR;

    if (early) {
        if (H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY) < 0)
            TEST_ERROR;
    }

    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_FIXED, &nfilters) < 0)
            TEST_ERROR;

        if (nfilters != 1)
            TEST_ERROR;

        if (H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) < 0)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    /* Create dataspace */
    if (chk_type == CHK_SINGLE) {
        if ((sid = H5Screate_simple(1, sg_dim, NULL)) < 0)
            TEST_ERROR;
    }
    else if (chk_type == CHK_FA) {
        if ((sid = H5Screate_simple(1, fa_dim, fa_max_dim)) < 0)
            TEST_ERROR;
    }
    else if (chk_type == CHK_EA) {
        if ((sid = H5Screate_simple(1, ea_dim, ea_max_dim)) < 0)
            TEST_ERROR;
    }

    /* Create dataset */

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, SPARSE_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    if (idx_type != (chk_type == CHK_SINGLE
                         ? H5D_CHUNK_IDX_SINGLE
                         : (chk_type == CHK_FA ? H5D_CHUNK_IDX_FARRAY : H5D_CHUNK_IDX_EARRAY)))
        FAIL_PUTS_ERROR("should be using the expected array chunk index");

    /* Starting at 4, select 3 blocks of size 2 each */
    /* Selection is across chunks and within the chunk */
    start[0]  = 4;
    stride[0] = 6;
    count[0]  = 3;
    block[0]  = 2;
    H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block);

    memset(wbuf, 0, sizeof(wbuf));

    /* Starting at 4, initialize 3 blocks of size 2 to the dataset */
    wbuf[4]  = 4;
    wbuf[5]  = 5;
    wbuf[10] = 10;
    wbuf[11] = 11;
    wbuf[16] = 16;
    wbuf[17] = 17;

    /* Starting at 1, select 3 blocks of size 1 each */
    start[0]  = 1;
    stride[0] = 6;
    count[0]  = 3;
    block[0]  = 1;
    H5Sselect_hyperslab(sid, H5S_SELECT_OR, start, stride, count, block);

    wbuf[1]  = 1;
    wbuf[7]  = 7;
    wbuf[13] = 13;

    /*
     * Use a compact memory space for the sparse file selection.  This avoids
     * relying on matching sparse selections in memory and file space while
     * still writing the same selected file elements.
     */
    mdim[0] = 9;
    if ((msid = H5Screate_simple(1, mdim, NULL)) < 0)
        TEST_ERROR;

    wvals[0] = 1;
    wvals[1] = 4;
    wvals[2] = 5;
    wvals[3] = 7;
    wvals[4] = 10;
    wvals[5] = 11;
    wvals[6] = 13;
    wvals[7] = 16;
    wvals[8] = 17;

    if (H5Dwrite(did, H5T_NATIVE_INT, msid, sid, H5P_DEFAULT, wvals) < 0)
        TEST_ERROR;

    if (H5Sclose(msid) < 0)
        TEST_ERROR;
    msid = H5I_INVALID_HID;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_CHUNKED) < 0)
        TEST_ERROR;

    if (H5Pset_chunk(dcpl, 1, (chk_type == CHK_SINGLE ? sg_chunk_dim : chunk_dim)) < 0)

        if (early) {
            if (H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY) < 0)
                TEST_ERROR;
        }

    if (filtered) {
        if (H5Pset_deflate(dcpl, 9) < 0)
            TEST_ERROR;

        if (H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) < 0)
            TEST_ERROR;
    }

    /* Create dataspace */
    if (chk_type == CHK_SINGLE) {
        if ((sid = H5Screate_simple(1, sg_dim, NULL)) < 0)
            TEST_ERROR;
    }
    else if (chk_type == CHK_FA) {
        if ((sid = H5Screate_simple(1, fa_dim, fa_max_dim)) < 0)
            TEST_ERROR;
    }
    else if (chk_type == CHK_EA) {
        if ((sid = H5Screate_simple(1, ea_dim, ea_max_dim)) < 0)
            TEST_ERROR;
    }

    /*
     * Create legacy chunked dataset
     * This is done just to compare the meta data size
     * between structured chunk and legacy chunked datasets
     */
    if ((did = H5Dcreate2(fid, CHUNKED_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Obtain the correct chunk indexing type */
    /* This may be v1-btree chunk index */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    /* Starting at 4, select 3 blocks of size 2 each */
    start[0]  = 4;
    stride[0] = 6;
    block[0]  = 2;
    H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block);

    memset(wbuf, 0, sizeof(wbuf));

    /* Starting at 1, select 1 blocks of size 1 each */
    start[0]  = 1;
    stride[0] = 6;
    count[0]  = 1;
    block[0]  = 1;
    H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block);

    wbuf[1] = 1;

    if (H5Dwrite(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
        TEST_ERROR;

    if ((did = H5Dopen2(fid, SPARSE_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Obtain the correct chunk indexing type */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    memset(&nat_info, 0, sizeof(nat_info));
    if (H5Oget_native_info(did, &nat_info, H5O_NATIVE_INFO_META_SIZE) < 0)
        TEST_ERROR;

    /* Verify the size of meta data */
    /* This may change as the implementation of structured chunk is still in progress */
    if (chk_type == CHK_SINGLE) {
        if (nat_info.meta_size.obj.index_size != 0)
            TEST_ERROR;
    }
    else if (chk_type == CHK_FA) {
        if (nat_info.meta_size.obj.index_size != (filtered ? 466 : 226))
            TEST_ERROR;
    }
    else if (chk_type == CHK_EA) {
        if (nat_info.meta_size.obj.index_size != (filtered ? 274 : 178))
            TEST_ERROR;
    }

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if ((did = H5Dopen2(fid, CHUNKED_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Obtain the correct chunk indexing type */
    /* This may be v1-btree chunk index */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    memset(&nat_info, 0, sizeof(nat_info));
    if (H5Oget_native_info(did, &nat_info, H5O_NATIVE_INFO_META_SIZE) < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

done:
error:
    H5E_BEGIN_TRY
    {
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Dclose(did);
        H5Fclose(fid);
    }
    H5E_END_TRY

    if (fail_as_expected) {
        PASSED();
        return SUCCEED;
    }

    return FAIL;
} /* end test_struct_chunk_info_1d() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_info_2d_bt2
 *
 * Purpose:     Verify H5Oget_native_info() for 2d dataset with
 *              v2-btree chunk index
 *
 * Return:      Success:        0
 *              Failure:        -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_info_2d_bt2(hid_t fcpl, hid_t fapl, bool filtered, bool early)
{
    char              filename[FILENAME_BUF_SIZE];    /* File name */
    hid_t             fid          = H5I_INVALID_HID; /* File ID */
    hid_t             sid          = H5I_INVALID_HID; /* Dataspace ID */
    hid_t             did          = H5I_INVALID_HID; /* Dataset ID */
    hid_t             dcpl         = H5I_INVALID_HID; /* Creation plist */
    hsize_t           dim[2]       = {10, 19};        /* 2-d dataspace (contains partial edge chunk) */
    hsize_t           dmax[2]      = {H5S_UNLIMITED, H5S_UNLIMITED}; /* maximum dimension */
    hsize_t           chunk_dim[2] = {5, 5};                         /* Chunk size */
    H5D_chunk_index_t idx_type;                                      /* dataset chunk index type     */

    int          wbuf[190]; /* Write buffer */
    hsize_t      start[2];
    hsize_t      stride[2];
    hsize_t      count[2];
    hsize_t      block[2];
    unsigned int level        = 9;
    unsigned int cd_values[1] = {level};
    size_t       cd_nelmts    = 1;

    int      nfilters;
    unsigned options;

    H5O_native_info_t nat_info;

    H5F_libver_t low, high; /* File format bound */
    bool         fail_as_expected = false;

    TESTING("structured chunk with H5Oget_native_info() on 2d dataset with bt2 chunk index");

    if (H5Pget_libver_bounds(fapl, &low, &high) < 0)
        TEST_ERROR;

    /* Create the file */
    h5_fixname(FILENAME[2], fapl, filename, sizeof filename);

    /* Create a new file. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl, 2, chunk_dim, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (early) {
        if (H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY) < 0)
            TEST_ERROR;
    }

    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_FIXED, &nfilters) < 0)
            TEST_ERROR;
        if (nfilters != 1)
            TEST_ERROR;

        if (H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) < 0)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    /* Create dataspace */
    if ((sid = H5Screate_simple(2, dim, dmax)) < 0)
        TEST_ERROR;

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, SPARSE_FILTER_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_BT2)
        FAIL_PUTS_ERROR("should be using version 2 btree chunk index");

    /* Starting at [3, 3], select 2 blocks of size 3x3 each */
    start[0]  = 3;
    start[1]  = 3;
    stride[0] = 4;
    stride[1] = 12;
    count[0]  = 1;
    count[1]  = 2;
    block[0]  = 3;
    block[1]  = 3;
    if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block) < 0)
        TEST_ERROR;

    memset(wbuf, 0, sizeof(wbuf));

    /* Initialize 2 3x3 blocks */
    wbuf[60] = 60;
    wbuf[61] = 61;
    wbuf[62] = 62;

    wbuf[72] = 72;
    wbuf[73] = 73;
    wbuf[74] = 74;

    wbuf[79] = 79;
    wbuf[80] = 80;
    wbuf[81] = 81;

    wbuf[91] = 91;
    wbuf[92] = 92;
    wbuf[93] = 93;

    wbuf[98]  = 98;
    wbuf[99]  = 99;
    wbuf[100] = 100;

    wbuf[110] = 110;
    wbuf[111] = 111;
    wbuf[112] = 112;

    if (H5Dwrite(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    /*
     * Create legacy chunked dataset
     */

    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_CHUNKED) < 0)
        TEST_ERROR;

    if (H5Pset_chunk(dcpl, 2, chunk_dim) < 0)
        TEST_ERROR;

    if (early) {
        if (H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY) < 0)
            TEST_ERROR;
    }

    if (filtered) {
        if (H5Pset_deflate(dcpl, 9) < 0)
            TEST_ERROR;

        if (H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) < 0)
            TEST_ERROR;
    }

    /* Create dataspace */
    if ((sid = H5Screate_simple(2, dim, dmax)) < 0)
        TEST_ERROR;

    /*
     * Create legacy chunked dataset
     * This is done just to compare the meta data size
     * between structured chunk and legacy chunked datasets
     */
    if ((did = H5Dcreate2(fid, CHUNKED_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Retrieve the chunk indexing type */
    /* This may be v1-btree or v2-btree chunk index */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    /* Starting at [3, 3], select 2 blocks of size 3x3 each */
    start[0]  = 3;
    start[1]  = 3;
    stride[0] = 4;
    stride[1] = 12;
    count[0]  = 1;
    count[1]  = 2;
    block[0]  = 3;
    block[1]  = 3;
    if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block) < 0)
        TEST_ERROR;

    memset(wbuf, 0, sizeof(wbuf));

    /* Initialize 2 3x3 blocks */
    wbuf[60] = 60;
    wbuf[61] = 61;
    wbuf[62] = 62;

    wbuf[72] = 72;
    wbuf[73] = 73;
    wbuf[74] = 74;

    wbuf[79] = 79;
    wbuf[80] = 80;
    wbuf[81] = 81;

    wbuf[91] = 91;
    wbuf[92] = 92;
    wbuf[93] = 93;

    wbuf[98]  = 98;
    wbuf[99]  = 99;
    wbuf[100] = 100;

    wbuf[110] = 110;
    wbuf[111] = 111;
    wbuf[112] = 112;

    if (H5Dwrite(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
        TEST_ERROR;

    /* Open the structured chunk dataset */
    if ((did = H5Dopen2(fid, SPARSE_FILTER_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Retrieve the chunk indexing type */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    memset(&nat_info, 0, sizeof(nat_info));
    if (H5Oget_native_info(did, &nat_info, H5O_NATIVE_INFO_META_SIZE) < 0)
        TEST_ERROR;

    if (nat_info.meta_size.obj.index_size != 2086)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Open the legacy chunked dataset */
    if ((did = H5Dopen2(fid, CHUNKED_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Retrieve the chunk indexing type */
    /* This may be v1-btree or v2-btree chunk index */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    memset(&nat_info, 0, sizeof(nat_info));
    if (H5Oget_native_info(did, &nat_info, H5O_NATIVE_INFO_META_SIZE) < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

done:
error:
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Fclose(fid);
    }
    H5E_END_TRY

    if (fail_as_expected) {
        PASSED();
        return SUCCEED;
    }

    return FAIL;
} /* test_struct_chunk_info_2d_bt2() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_extent_1d
 *
 * Purpose:     Verify H5Dset_extent() for:
 *                  --1d dataset with fixed array and extensible array chunk index
 *                  --Chunks were not written before H5Dset_extent()
 *              Expand or shrink for H5Dset_extent should succeed
 *
 * Return:      Success:        0
 *              Failure:        -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_extent_1d(hid_t fcpl, hid_t fapl, bool filtered, bool early)
{
    char  filename[FILENAME_BUF_SIZE]; /* File name */
    hid_t fid     = H5I_INVALID_HID;
    hid_t sid     = H5I_INVALID_HID;
    hid_t dcpl    = H5I_INVALID_HID;
    hid_t did     = H5I_INVALID_HID;
    hid_t new_sid = H5I_INVALID_HID;
    hid_t msid    = H5I_INVALID_HID;

    hsize_t fa_dim[1]     = {20}; /* 1-d dataspace */
    hsize_t fa_max_dim[1] = {50};

    hsize_t chunk_dim[1] = {5}; /* Chunk size */

    hsize_t ea_dim[1]     = {20}; /* 1-d dataspace */
    hsize_t ea_max_dim[1] = {H5S_UNLIMITED};
    int     status;

    H5D_chunk_index_t idx_type; /* dataset chunk index type */

    hsize_t shrink_dim[1] = {10};
    hsize_t expand_dim[1] = {50};

    int wbuf1[30]; /* Write buffer */
    int wbuf2[30]; /* Write buffer */
    int rbuf[30];  /* Read buffer */

    hsize_t start[1];
    hsize_t stride[1];
    hsize_t count[1];
    hsize_t block[1];

    unsigned int level        = 9;
    unsigned int cd_values[1] = {level};
    size_t       cd_nelmts    = 1;

    int      nfilters;
    unsigned options;

    H5F_libver_t low, high; /* File format bound */
    bool         fail_as_expected = false;

    TESTING("structured chunk with HD5set_extent() on 1d dataset");

    if (H5Pget_libver_bounds(fapl, &low, &high) < 0)
        TEST_ERROR;

    /* Create a file */
    h5_fixname(FILENAME[1], fapl, filename, sizeof filename);
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create dataspace */
    if ((sid = H5Screate_simple(1, fa_dim, fa_max_dim)) < 0)
        TEST_ERROR;

    /* Create property list for dataset creation */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl, 1, chunk_dim, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (early) {
        if (H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY) < 0)
            TEST_ERROR;
    }

    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_SHUFFLE, H5Z_FLAG_OPTIONAL, (size_t)0,
                           NULL) < 0)
            TEST_ERROR;

        if (H5Pset_filter2(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_SELECTION, &nfilters) < 0)
            TEST_ERROR;
        if (nfilters != 2)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_FIXED, &nfilters) < 0)
            TEST_ERROR;

        if (nfilters != 1)
            TEST_ERROR;

        if (H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) < 0)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    /* Create 1st dataset */

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, EXT1_SPARSE_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_FARRAY)
        FAIL_PUTS_ERROR("should be using fixed array chunk index");

    /* Expand case */
    H5E_BEGIN_TRY
    {
        status = H5Dset_extent(did, expand_dim);
    }
    H5E_END_TRY

    if (status < 0)
        TEST_ERROR;

    if ((new_sid = H5Dget_space(did)) < 0)
        TEST_ERROR;

    if (!early) {

        /* Starting at 20, select 1 blocks of size 3 each */
        start[0]  = 20;
        stride[0] = 1;
        count[0]  = 3;
        block[0]  = 1;
        H5Sselect_hyperslab(new_sid, H5S_SELECT_SET, start, stride, count, block);

        memset(wbuf1, 0, sizeof(wbuf1));
        wbuf1[20] = 20;
        wbuf1[21] = 21;
        wbuf1[22] = 22;

        if (H5Dwrite(did, H5T_NATIVE_INT, new_sid, new_sid, H5P_DEFAULT, wbuf1) < 0)
            TEST_ERROR;
    }

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Sclose(new_sid) < 0)
        TEST_ERROR;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    /* Create 2nd dataset */

    /* Create dataspace */
    if ((sid = H5Screate_simple(1, ea_dim, ea_max_dim)) < 0)
        TEST_ERROR;

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, EXT2_SPARSE_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_EARRAY)
        FAIL_PUTS_ERROR("should be using extensible array chunk index");

    /* Shrink case */
    H5E_BEGIN_TRY
    {
        status = H5Dset_extent(did, shrink_dim);
    }
    H5E_END_TRY

    if (status < 0)
        TEST_ERROR;

    if ((new_sid = H5Dget_space(did)) < 0)
        TEST_ERROR;

    if (!early) {

        /* Starting at 1, select 1 block of size 3 each */
        start[0]  = 1;
        stride[0] = 1;
        count[0]  = 3;
        block[0]  = 1;
        H5Sselect_hyperslab(new_sid, H5S_SELECT_SET, start, stride, count, block);

        memset(wbuf2, 0, sizeof(wbuf2));
        wbuf2[1] = 1;
        wbuf2[2] = 2;
        wbuf2[3] = 3;

        if (H5Dwrite(did, H5T_NATIVE_INT, new_sid, new_sid, H5P_DEFAULT, wbuf2) < 0)
            TEST_ERROR;
    }

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Sclose(new_sid) < 0)
        TEST_ERROR;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if (!early) {
        if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
            TEST_ERROR;

        if ((did = H5Dopen2(fid, EXT1_SPARSE_DSET, H5P_DEFAULT)) < 0)
            TEST_ERROR;

        if ((new_sid = H5Dget_space(did)) < 0)
            TEST_ERROR;

        /* Just read the selected chunk, otherwise H5SC_read didn't handle the case
           properly when reading in all chunks (which may or may not be allocated) */
        count[0] = 3;
        if ((msid = H5Screate_simple(1, count, NULL)) < 0)
            TEST_ERROR;
        start[0]  = 20;
        stride[0] = 1;
        count[0]  = 3;
        block[0]  = 1;
        H5Sselect_hyperslab(new_sid, H5S_SELECT_SET, start, stride, count, block);

        memset(rbuf, 0, sizeof(rbuf));

        if (H5Dread(did, H5T_NATIVE_INT, msid, new_sid, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* Verify data read */
        if (rbuf[0] != 20 || rbuf[1] != 21 || rbuf[2] != 22)
            TEST_ERROR;

        if (H5Dclose(did) < 0)
            TEST_ERROR;

        if (H5Sclose(msid) < 0)
            TEST_ERROR;

        if (H5Sclose(new_sid) < 0)
            TEST_ERROR;

        if ((did = H5Dopen2(fid, EXT2_SPARSE_DSET, H5P_DEFAULT)) < 0)
            TEST_ERROR;

        if ((new_sid = H5Dget_space(did)) < 0)
            TEST_ERROR;

        /* Just read the selected chunk, otherwise H5SC_read didn't handle the case
           properly when reading in all chunks (which may or may not be allocated) */
        if ((msid = H5Screate_simple(1, count, NULL)) < 0)
            TEST_ERROR;
        start[0]  = 1;
        stride[0] = 1;
        count[0]  = 3;
        block[0]  = 1;
        H5Sselect_hyperslab(new_sid, H5S_SELECT_SET, start, stride, count, block);

        memset(rbuf, 0, sizeof(rbuf));

        if (H5Dread(did, H5T_NATIVE_INT, msid, new_sid, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* Verify data read */
        if (rbuf[0] != 1 || rbuf[1] != 2 || rbuf[2] != 3)
            TEST_ERROR;

        if (H5Dclose(did) < 0)
            TEST_ERROR;

        if (H5Sclose(new_sid) < 0)
            TEST_ERROR;

        if (H5Sclose(msid) < 0)
            TEST_ERROR;

        if (H5Fclose(fid) < 0)
            TEST_ERROR;
    }

    PASSED();
    return SUCCEED;

done:
error:
    H5E_BEGIN_TRY
    {
        H5Sclose(sid);
        H5Sclose(new_sid);
        H5Sclose(msid);
        H5Pclose(dcpl);
        H5Dclose(did);
        H5Fclose(fid);
    }
    H5E_END_TRY

    if (fail_as_expected) {
        PASSED();
        return SUCCEED;
    }

    return FAIL;
} /* end test_struct_chunk_extent_1d() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_extent_2d
 *
 * Purpose:     Verify H5Dset_extent() for 2d dataset with v2-btree chunk index;
 *              there is a write before set_extent()
 *
 *              Expand and shrink are expected to succeed when structured
 *              chunk extent transitions are supported by the selected file
 *              format bounds. Older format bounds continue to reject
 *              structured chunk dataset creation.
 *
 * Return:      Success:        0
 *              Failure:        -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_extent_2d(hid_t fcpl, hid_t fapl, bool filtered, bool early, bool expand)
{
    char              filename[FILENAME_BUF_SIZE];    /* File name */
    hid_t             fid          = H5I_INVALID_HID; /* File ID */
    hid_t             sid          = H5I_INVALID_HID; /* Dataspace ID */
    hid_t             did          = H5I_INVALID_HID; /* Dataset ID */
    hid_t             dcpl         = H5I_INVALID_HID; /* Creation plist */
    hsize_t           dim[2]       = {10, 19};        /* 2-d dataspace (contains partial edge chunk) */
    hsize_t           dmax[2]      = {H5S_UNLIMITED, H5S_UNLIMITED}; /* maximum dimension */
    hsize_t           chunk_dim[2] = {5, 5};                         /* Chunk size */
    H5D_chunk_index_t idx_type;                                      /* dataset chunk index type     */

    hsize_t expand_dim[2] = {20, 29}; /* Chunk size */
    hsize_t shrink_dim[2] = {5, 9};   /* Chunk size */
    int     status;

    int          wbuf[190]; /* Write buffer */
    hsize_t      start[2];
    hsize_t      stride[2];
    hsize_t      count[2];
    hsize_t      block[2];
    unsigned int level        = 9;
    unsigned int cd_values[1] = {level};
    size_t       cd_nelmts    = 1;

    int      nfilters;
    unsigned options;

    H5F_libver_t low, high; /* File format bound */
    bool         fail_as_expected = false;

    TESTING("structured chunk with H5Dset_extent() on 2d dataset");

    if (H5Pget_libver_bounds(fapl, &low, &high) < 0)
        TEST_ERROR;

    /* Create the file */
    h5_fixname(FILENAME[2], fapl, filename, sizeof filename);

    /* Create a new file. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create dataspace */
    if ((sid = H5Screate_simple(2, dim, dmax)) < 0)
        TEST_ERROR;

    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl, 2, chunk_dim, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (early) {
        if (H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY) < 0)
            TEST_ERROR;
    }

    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;

        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_SHUFFLE, H5Z_FLAG_OPTIONAL, (size_t)0,
                           NULL) < 0)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_SELECTION, &nfilters) < 0)
            TEST_ERROR;
        if (nfilters != 2)
            TEST_ERROR;

        if (H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) < 0)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }
    else {
        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options == H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, SPARSE_FILTER_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_BT2)
        FAIL_PUTS_ERROR("should be using version 2 btree chunk index");

    /* Starting at [3, 3], select 2 blocks of size 3x3 each */
    start[0]  = 3;
    start[1]  = 3;
    stride[0] = 4;
    stride[1] = 12;
    count[0]  = 1;
    count[1]  = 2;
    block[0]  = 3;
    block[1]  = 3;
    if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block) < 0)
        TEST_ERROR;

    memset(wbuf, 0, sizeof(wbuf));

    /* Initialize 2 3x3 blocks */
    wbuf[60] = 60;
    wbuf[61] = 61;
    wbuf[62] = 62;

    wbuf[72] = 72;
    wbuf[73] = 73;
    wbuf[74] = 74;

    wbuf[79] = 79;
    wbuf[80] = 80;
    wbuf[81] = 81;

    wbuf[91] = 91;
    wbuf[92] = 92;
    wbuf[93] = 93;

    wbuf[98]  = 98;
    wbuf[99]  = 99;
    wbuf[100] = 100;

    wbuf[110] = 110;
    wbuf[111] = 111;
    wbuf[112] = 112;

    if (H5Dwrite(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    /* Expand case */
    H5E_BEGIN_TRY
    {
        status = H5Dset_extent(did, expand ? expand_dim : shrink_dim);
    }
    H5E_END_TRY

    if (status < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

done:
error:
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Fclose(fid);
    }
    H5E_END_TRY

    if (fail_as_expected) {
        PASSED();
        return SUCCEED;
    }

    return FAIL;
} /* test_struct_chunk_extent_2d() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_api
 *
 * Purpose:     Verify APIs for structured chunk layout:
 *              --H5Dget_create_plist()
 *              --H5Pget/set_layout()
 *              --H5Pget/set_struct_chunk()
 *
 * Return:      Success:        0
 *              Failure:        -1
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_api(hid_t fcpl, hid_t fapl)
{
    char         filename[FILENAME_BUF_SIZE]; /* File name */
    hid_t        fid              = H5I_INVALID_HID;
    hid_t        sid              = H5I_INVALID_HID;
    hid_t        sid2             = H5I_INVALID_HID;
    hid_t        dcpl             = H5I_INVALID_HID;
    hid_t        dcpl2            = H5I_INVALID_HID;
    hid_t        did              = H5I_INVALID_HID;
    hid_t        did2             = H5I_INVALID_HID;
    hsize_t      dim[1]           = {50};      /* 1-d dataspace */
    hsize_t      chunk_dim[1]     = {5};       /* 1-d Chunk size */
    hsize_t      dim2[2]          = {50, 100}; /* 2-d dataspace */
    hsize_t      chunk_dim2[2]    = {5, 10};   /* 2-d Chunk size */
    hsize_t      my_chunk_dim[2]  = {0, 0};
    hsize_t      my_chunk_dim2[2] = {0, 0};
    unsigned     my_flag;
    int          my_rank;
    H5D_layout_t my_layout;
    H5F_libver_t low, high; /* File format bound */
    bool         fail_as_expected = false;

    TESTING("structured chunk APIs");

    if (H5Pget_libver_bounds(fapl, &low, &high) < 0)
        TEST_ERROR;

    /* Create a file */
    h5_fixname(FILENAME[0], fapl, filename, sizeof filename);

    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create 1d dataspace */
    if ((sid = H5Screate_simple(1, dim, NULL)) < 0)
        TEST_ERROR;

    /* Create property list for compact dataset creation */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;
    if (H5Pset_struct_chunk(dcpl, 1, chunk_dim, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, SPARSE_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    /* Reopen dataset */
    if ((did = H5Dopen2(fid, SPARSE_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    if ((dcpl = H5Dget_create_plist(did)) < 0)
        TEST_ERROR;

    if ((my_rank = H5Pget_struct_chunk(dcpl, 2, my_chunk_dim, &my_flag)) != 1)
        TEST_ERROR;
    if (my_flag != H5D_SPARSE_CHUNK)
        TEST_ERROR;
    if (my_chunk_dim[0] != chunk_dim[0])
        TEST_ERROR;

    if ((my_layout = H5Pget_layout(dcpl)) != H5D_STRUCT_CHUNK)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    /* Create 2d dataspace */
    if ((sid2 = H5Screate_simple(2, dim2, NULL)) < 0)
        TEST_ERROR;

    if ((dcpl2 = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl2, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl2, 2, chunk_dim2, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if ((did2 = H5Dcreate2(fid, SPARSE_DSET2, H5T_NATIVE_INT, sid2, H5P_DEFAULT, dcpl2, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    if (H5Dclose(did2) < 0)
        TEST_ERROR;

    if (H5Sclose(sid2) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl2) < 0)
        TEST_ERROR;

    if ((did2 = H5Dopen2(fid, SPARSE_DSET2, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    if ((dcpl2 = H5Dget_create_plist(did2)) < 0)
        TEST_ERROR;

    if ((my_rank = H5Pget_struct_chunk(dcpl2, 2, my_chunk_dim2, &my_flag)) != 2)
        TEST_ERROR;
    if (my_flag != H5D_SPARSE_CHUNK)
        TEST_ERROR;
    if (my_chunk_dim2[0] != chunk_dim2[0] || my_chunk_dim2[1] != chunk_dim2[1])
        TEST_ERROR;

    if ((my_layout = H5Pget_layout(dcpl2)) != H5D_STRUCT_CHUNK)
        TEST_ERROR;

    if (H5Dclose(did2) < 0)
        TEST_ERROR;
    if (H5Pclose(dcpl2) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

done:
error:
    H5E_BEGIN_TRY
    {
        H5Sclose(sid);
        H5Sclose(sid2);
        H5Pclose(dcpl);
        H5Pclose(dcpl2);
        H5Dclose(did);
        H5Dclose(did2);
        H5Fclose(fid);
    }
    H5E_END_TRY

    if (fail_as_expected) {
        PASSED();
        return SUCCEED;
    }
    return FAIL;
} /* end test_struct_chunk_api() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_1d_single
 *
 * Purpose:     Verify writing and reading hyperslab selection to a
 *              structured chunk dataset, using single chunk index
 *
 * Return:      Success:        0
 *              Failure:        -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_1d_single(hid_t fcpl, hid_t fapl, bool filtered, bool early)
{
    char  filename[FILENAME_BUF_SIZE]; /* File name */
    hid_t fid  = H5I_INVALID_HID;
    hid_t sid  = H5I_INVALID_HID;
    hid_t dcpl = H5I_INVALID_HID;
    hid_t did  = H5I_INVALID_HID;

    hsize_t           dim[1]       = {20}; /* 1-d dataspace */
    hsize_t           chunk_dim[1] = {20}; /* Chunk size */
    H5D_chunk_index_t idx_type;            /* dataset chunk index type */

    int wbuf[20]; /* Write buffer */
    int rbuf[20]; /* Read buffer */

    hsize_t start[1];
    hsize_t stride[1];
    hsize_t count[1];
    hsize_t block[1];

    unsigned     i;
    unsigned int level        = 9;
    unsigned int cd_values[1] = {level};
    size_t       cd_nelmts    = 1;

    size_t       my_cd_nelmts = 1;
    unsigned int my_cd_value  = 0;

    int          nfilters;
    H5Z_filter_t filter_id;
    unsigned int flags;
    unsigned     options;

    H5F_libver_t low, high; /* File format bound */
    bool         fail_as_expected = false;

    TESTING("structured chunk 1d dataset with single chunk index");

    if (H5Pget_libver_bounds(fapl, &low, &high) < 0)
        TEST_ERROR;

    /* Create a file */
    h5_fixname(FILENAME[1], fapl, filename, sizeof filename);
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create dataspace */
    if ((sid = H5Screate_simple(1, dim, NULL)) < 0)
        TEST_ERROR;

    /* Create property list for dataset creation */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl, 1, chunk_dim, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (early) {
        if (H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY) < 0)
            TEST_ERROR;
    }

    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_SHUFFLE, H5Z_FLAG_OPTIONAL, (size_t)0,
                           NULL) < 0)
            TEST_ERROR;

        if (H5Pset_filter2(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_SELECTION, &nfilters) < 0)
            TEST_ERROR;
        if (nfilters != 2)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_FIXED, &nfilters) < 0)
            TEST_ERROR;

        if (nfilters != 1)
            TEST_ERROR;

        if (H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) < 0)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, SPARSE_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_SINGLE)
        FAIL_PUTS_ERROR("should be using single chunk index");

    /* Starting at 4, select 3 blocks of size 2 each */
    /* Selection is across chunks and within the chunk */
    start[0]  = 4;
    stride[0] = 6;
    count[0]  = 3;
    block[0]  = 2;
    H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block);

    memset(wbuf, 0, sizeof(wbuf));

    /* Starting at 4, initialize 3 blocks of size 2 to the dataset */
    wbuf[4]  = 4;
    wbuf[5]  = 5;
    wbuf[10] = 10;
    wbuf[11] = 11;
    wbuf[16] = 16;
    wbuf[17] = 17;

    /* Starting at 1, select 3 blocks of size 1 each */
    start[0]  = 1;
    stride[0] = 6;
    count[0]  = 3;
    block[0]  = 1;
    H5Sselect_hyperslab(sid, H5S_SELECT_OR, start, stride, count, block);

    /* Starting at 1, initialize 3 blocks of size 1 to the dataset */
    wbuf[1]  = 1;
    wbuf[7]  = 7;
    wbuf[13] = 13;

    if (H5Dwrite(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
        TEST_ERROR;

    if ((did = H5Dopen2(fid, SPARSE_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_SINGLE)
        FAIL_PUTS_ERROR("should be using single chunk index");

    if ((dcpl = H5Dget_create_plist(did)) < 0)
        TEST_ERROR;

    if (filtered) {
        /* Get filter info for section "selection", filter number 0 */
        if ((filter_id = H5Pget_filter3(dcpl, H5_SECTION_SELECTION, 0, NULL, &my_cd_nelmts, &my_cd_value,
                                        (size_t)0, NULL, NULL)) < 0)
            TEST_ERROR;
        if (filter_id != H5Z_FILTER_DEFLATE)
            TEST_ERROR;
        if (my_cd_nelmts != 1)
            TEST_ERROR;
        if (my_cd_value != 9)
            TEST_ERROR;

        /* Get filter info by filter number 1 for section "selection" */
        if ((filter_id = H5Pget_filter3(dcpl, H5_SECTION_SELECTION, 1, &flags, NULL, NULL, (size_t)0, NULL,
                                        NULL)) < 0)
            TEST_ERROR;
        if (filter_id != H5Z_FILTER_SHUFFLE)
            TEST_ERROR;
        if (flags != H5Z_FLAG_OPTIONAL)
            TEST_ERROR;

        /* Get filter info by filter id for section "fixed data" */
        if (H5Pget_filter_by_id3(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, &flags, &my_cd_nelmts,
                                 &my_cd_value, (size_t)0, NULL, NULL) < 0)
            TEST_ERROR;
        if (my_cd_nelmts != 1)
            TEST_ERROR;
        if (my_cd_value != 9)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }
    else {
        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options == H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    memset(rbuf, 0, sizeof(rbuf));
    if (H5Dread(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;

    /* Verify data read */
    for (i = 0; i < 20; i++)
        if (rbuf[i] != wbuf[i])
            TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Closing */
    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

done:
error:
    H5E_BEGIN_TRY
    {
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Dclose(did);
        H5Fclose(fid);
    }
    H5E_END_TRY

    if (fail_as_expected) {
        PASSED();
        return SUCCEED;
    }

    return FAIL;
} /* end test_struct_chunk_1d() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_2d_bt2
 *
 * Purpose:     Verify writing and reading hyperslab selection to a
 *              structured chunk dataset, using v2-btree chunk index
 *
 * Return:      Success:        0
 *              Failure:        -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_2d_bt2(hid_t fcpl, hid_t fapl, bool filtered, bool early)
{
    char              filename[FILENAME_BUF_SIZE];    /* File name */
    hid_t             fid          = H5I_INVALID_HID; /* File ID */
    hid_t             sid          = H5I_INVALID_HID; /* Dataspace ID */
    hid_t             did          = H5I_INVALID_HID; /* Dataset ID */
    hid_t             dcpl         = H5I_INVALID_HID; /* Creation plist */
    hsize_t           dim[2]       = {10, 19};        /* 2-d dataspace (contains partial edge chunk) */
    hsize_t           dmax[2]      = {H5S_UNLIMITED, H5S_UNLIMITED}; /* maximum dimension */
    hsize_t           chunk_dim[2] = {5, 5};                         /* Chunk size */
    H5D_chunk_index_t idx_type;                                      /* dataset chunk index type     */

    int          wbuf[190]; /* Write buffer */
    int          rbuf[190]; /* Read buffer */
    hsize_t      start[2];
    hsize_t      stride[2];
    hsize_t      count[2];
    hsize_t      block[2];
    unsigned     i;
    unsigned int level        = 9;
    unsigned int cd_values[1] = {level};
    size_t       cd_nelmts    = 1;

    size_t       my_cd_nelmts = 1;
    unsigned int my_cd_value  = 0;

    int          nfilters;
    H5Z_filter_t filter_id;
    unsigned int flags;
    unsigned     options;

    H5F_libver_t low, high; /* File format bound */
    bool         fail_as_expected = false;

    TESTING("structured chunk 2d dataset with bt2 chunk index");

    if (H5Pget_libver_bounds(fapl, &low, &high) < 0)
        TEST_ERROR;

    /* Create the file */
    h5_fixname(FILENAME[2], fapl, filename, sizeof filename);

    /* Create a new file. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create dataspace */
    if ((sid = H5Screate_simple(2, dim, dmax)) < 0)
        TEST_ERROR;

    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl, 2, chunk_dim, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (early) {
        if (H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY) < 0)
            TEST_ERROR;
    }

    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;

        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_SHUFFLE, H5Z_FLAG_OPTIONAL, (size_t)0,
                           NULL) < 0)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_SELECTION, &nfilters) < 0)
            TEST_ERROR;
        if (nfilters != 2)
            TEST_ERROR;

        if (H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) < 0)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }
    else {
        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options == H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, SPARSE_FILTER_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_BT2)
        FAIL_PUTS_ERROR("should be using version 2 btree chunk index");

    /* Starting at [3, 3], select 2 blocks of size 3x3 each */
    start[0]  = 3;
    start[1]  = 3;
    stride[0] = 4;
    stride[1] = 12;
    count[0]  = 1;
    count[1]  = 2;
    block[0]  = 3;
    block[1]  = 3;
    if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block) < 0)
        TEST_ERROR;

    memset(wbuf, 0, sizeof(wbuf));

    /* Initialize 2 3x3 blocks */
    wbuf[60] = 60;
    wbuf[61] = 61;
    wbuf[62] = 62;

    wbuf[72] = 72;
    wbuf[73] = 73;
    wbuf[74] = 74;

    wbuf[79] = 79;
    wbuf[80] = 80;
    wbuf[81] = 81;

    wbuf[91] = 91;
    wbuf[92] = 92;
    wbuf[93] = 93;

    wbuf[98]  = 98;
    wbuf[99]  = 99;
    wbuf[100] = 100;

    wbuf[110] = 110;
    wbuf[111] = 111;
    wbuf[112] = 112;

    if (H5Dwrite(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
        TEST_ERROR;

    if ((did = H5Dopen2(fid, SPARSE_FILTER_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_BT2)
        FAIL_PUTS_ERROR("should be using v2 btree chunk index");

    if ((dcpl = H5Dget_create_plist(did)) < 0)
        TEST_ERROR;

    if (filtered) {
        /* Get filter info by filter number 0 for section "selection" */
        if ((filter_id = H5Pget_filter3(dcpl, H5_SECTION_SELECTION, 0, NULL, &my_cd_nelmts, &my_cd_value,
                                        (size_t)0, NULL, NULL)) < 0)
            TEST_ERROR;
        if (filter_id != H5Z_FILTER_DEFLATE)
            TEST_ERROR;
        if (my_cd_nelmts != 1)
            TEST_ERROR;
        if (my_cd_value != 9)
            TEST_ERROR;

        /* Get filter info by filter number 1 for section "selection" */
        if ((filter_id = H5Pget_filter3(dcpl, H5_SECTION_SELECTION, 1, &flags, NULL, NULL, (size_t)0, NULL,
                                        NULL)) < 0)
            TEST_ERROR;
        if (filter_id != H5Z_FILTER_SHUFFLE)
            TEST_ERROR;
        if (flags != H5Z_FLAG_OPTIONAL)
            TEST_ERROR;

        H5E_BEGIN_TRY
        {
            /* Get filter info by filter id for section "fixed data" */
            filter_id = H5Pget_filter_by_id3(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, &flags,
                                             &my_cd_nelmts, &my_cd_value, (size_t)0, NULL, NULL);
        }
        H5E_END_TRY
        /* No filter for section "fixed data" */
        if (filter_id >= 0)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }
    else {
        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options == H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    memset(rbuf, 0, sizeof(rbuf));
    if (H5Dread(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;

    /* Verify data read */
    for (i = 0; i < 190; i++)
        if (rbuf[i] != wbuf[i])
            TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Closing */
    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

done:
error:
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Fclose(fid);
    }
    H5E_END_TRY

    if (fail_as_expected) {
        PASSED();
        return SUCCEED;
    }

    return FAIL;
} /* test_struct_chunk_2d() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_1d_fa()
 *
 * Purpose:     Verify writing and reading hyperslab selection to a
 *              structured chunk dataset, using fixed array chunk index
 *              Also verify the following APIs for structured chunk with filter:
 *              --H5Pset_filter2()
 *              --H5Pget_nfilters2()
 *              --H5Pget_filters3()
 *              --H5Pget_filter_by_id3()
 *
 * Return:      Success:        0
 *              Failure:        -1
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_1d_fa(hid_t fcpl, hid_t fapl, bool filtered, bool early)
{
    char              filename[FILENAME_BUF_SIZE];    /* File name */
    hid_t             fid          = H5I_INVALID_HID; /* File ID */
    hid_t             sid          = H5I_INVALID_HID; /* Dataspace ID */
    hid_t             did          = H5I_INVALID_HID; /* Dataset ID */
    hid_t             dcpl         = H5I_INVALID_HID; /* Creation plist */
    hsize_t           dim[1]       = {19};            /* 1-d dataspace (contains partial edge chunk) */
    hsize_t           chunk_dim[1] = {5};             /* Chunk size */
    H5D_chunk_index_t idx_type;                       /* dataset chunk index type */

    int          wbuf[19]; /* Write buffer */
    int          rbuf[19]; /* Read buffer */
    hsize_t      start[1];
    hsize_t      stride[1];
    hsize_t      count[1];
    hsize_t      block[1];
    unsigned     i;
    unsigned int level        = 9;
    unsigned int cd_values[1] = {level};
    size_t       cd_nelmts    = 1;

    size_t       my_cd_nelmts = 1;
    unsigned int my_cd_value  = 0;

    int          nfilters;
    H5Z_filter_t filter_id;
    unsigned int flags;
    unsigned     options;

    H5F_libver_t low, high; /* File format bound */
    bool         fail_as_expected = false;

    TESTING("structured chunk 1d dataset with fixed array chunk index");

    if (H5Pget_libver_bounds(fapl, &low, &high) < 0)
        TEST_ERROR;

    /* Create the file */
    h5_fixname(FILENAME[3], fapl, filename, sizeof filename);

    /* Create a new file. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create dataspace */
    if ((sid = H5Screate_simple(1, dim, NULL)) < 0)
        TEST_ERROR;

    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl, 1, chunk_dim, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (early) {
        if (H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY) < 0)
            TEST_ERROR;
    }

    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_SHUFFLE, H5Z_FLAG_OPTIONAL, (size_t)0,
                           NULL) < 0)
            TEST_ERROR;

        if (H5Pset_filter2(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_SELECTION, &nfilters) < 0)
            TEST_ERROR;
        if (nfilters != 2)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_FIXED, &nfilters) < 0)
            TEST_ERROR;

        if (nfilters != 1)
            TEST_ERROR;

        if (H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) < 0)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, SPARSE_FILTER_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_FARRAY)
        FAIL_PUTS_ERROR("should be using fixed array chunk index");

    /* Starting at 3, select 3 blocks of size 3 each */
    start[0]  = 3;
    stride[0] = 6;
    count[0]  = 3;
    block[0]  = 3;
    if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block) < 0)
        TEST_ERROR;

    /* Write sparse data to the dataset */
    memset(wbuf, 0, sizeof(wbuf));

    /* Starting at 3, initialize 3 blocks of size 3 and write to the dataset */
    wbuf[3] = 3;
    wbuf[4] = 4;
    wbuf[5] = 5;

    wbuf[9]  = 9;
    wbuf[10] = 10;
    wbuf[11] = 11;

    wbuf[15] = 15;
    wbuf[16] = 16;
    wbuf[17] = 17;

    if (H5Dwrite(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
        TEST_ERROR;

    if ((did = H5Dopen2(fid, SPARSE_FILTER_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_FARRAY)
        FAIL_PUTS_ERROR("should be using fixed array chunk index");

    if ((dcpl = H5Dget_create_plist(did)) < 0)
        TEST_ERROR;

    if (filtered) {
        /* Get filter info for section "selection", filter number 0 */
        if ((filter_id = H5Pget_filter3(dcpl, H5_SECTION_SELECTION, 0, NULL, &my_cd_nelmts, &my_cd_value,
                                        (size_t)0, NULL, NULL)) < 0)
            TEST_ERROR;
        if (filter_id != H5Z_FILTER_DEFLATE)
            TEST_ERROR;
        if (my_cd_nelmts != 1)
            TEST_ERROR;
        if (my_cd_value != 9)
            TEST_ERROR;

        /* Get filter info by filter number 1 for section "selection" */
        if ((filter_id = H5Pget_filter3(dcpl, H5_SECTION_SELECTION, 1, &flags, NULL, NULL, (size_t)0, NULL,
                                        NULL)) < 0)
            TEST_ERROR;
        if (filter_id != H5Z_FILTER_SHUFFLE)
            TEST_ERROR;
        if (flags != H5Z_FLAG_OPTIONAL)
            TEST_ERROR;

        /* Get filter info by filter id for section "fixed data" */
        if (H5Pget_filter_by_id3(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, &flags, &my_cd_nelmts,
                                 &my_cd_value, (size_t)0, NULL, NULL) < 0)
            TEST_ERROR;
        if (my_cd_nelmts != 1)
            TEST_ERROR;
        if (my_cd_value != 9)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }
    else {
        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options == H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    memset(rbuf, 0, sizeof(rbuf));
    if (H5Dread(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;

    /* Verify data read */
    for (i = 0; i < 19; i++)
        if (rbuf[i] != wbuf[i])
            TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Closing */
    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

done:
error:
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Fclose(fid);
    }
    H5E_END_TRY

    if (fail_as_expected) {
        PASSED();
        return SUCCEED;
    }

    return FAIL;
} /* test_struct_chunk_1d_fa() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_2d_ea()
 *
 * Purpose:     Verify writing and reading hyperslab selection to a
 *              structured chunk dataset, using extensible array chunk index
 *              Also verify the following APIs for structured chunk with filter:
 *              --H5Pset_filter2()
 *              --H5Pget_nfilters2()
 *              --H5Pget_filters3()
 *              --H5Pget_filter_by_id3()
 *
 * Return:      Success:        0
 *              Failure:        -1
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_2d_ea(hid_t fcpl, hid_t fapl, bool filtered, bool early)
{
    char              filename[FILENAME_BUF_SIZE];    /* File name */
    hid_t             fid          = H5I_INVALID_HID; /* File ID */
    hid_t             sid          = H5I_INVALID_HID; /* Dataspace ID */
    hid_t             did          = H5I_INVALID_HID; /* Dataset ID */
    hid_t             dcpl         = H5I_INVALID_HID; /* Creation plist */
    hsize_t           dim[2]       = {10, 19};        /* 2-d dataspace (contains partial edge chunk) */
    hsize_t           dmax[2]      = {10, H5S_UNLIMITED};
    hsize_t           chunk_dim[2] = {5, 5}; /* Chunk size */
    H5D_chunk_index_t idx_type;              /* dataset chunk index type */
    int               wbuf[190];             /* Write buffer */
    int               rbuf[190];             /* Read buffer */
    hsize_t           start[2];
    hsize_t           stride[2];
    hsize_t           count[2];
    hsize_t           block[2];
    unsigned          i;
    unsigned int      level        = 9;
    unsigned int      cd_values[1] = {level};
    size_t            cd_nelmts    = 1;

    size_t       my_cd_nelmts = 1;
    unsigned int my_cd_value  = 0;

    int          nfilters;
    H5Z_filter_t filter_id;
    unsigned int flags;
    unsigned     options;

    H5F_libver_t low, high; /* File format bound */
    bool         fail_as_expected = false;

    TESTING("structured chunk 2d dataset with extensible array chunk index");

    if (H5Pget_libver_bounds(fapl, &low, &high) < 0)
        TEST_ERROR;

    /* Create the file */
    h5_fixname(FILENAME[4], fapl, filename, sizeof filename);

    /* Create a new file. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create dataspace */
    if ((sid = H5Screate_simple(2, dim, dmax)) < 0)
        TEST_ERROR;

    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl, 2, chunk_dim, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (early) {
        if (H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY) < 0)
            TEST_ERROR;
    }

    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, cd_nelmts,
                           cd_values) < 0)
            TEST_ERROR;

        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_SHUFFLE, H5Z_FLAG_OPTIONAL, (size_t)0,
                           NULL) < 0)
            TEST_ERROR;

        if (H5Pget_nfilters2(dcpl, H5_SECTION_SELECTION, &nfilters) < 0)
            TEST_ERROR;
        if (nfilters != 2)
            TEST_ERROR;

        if (H5Pset_chunk_opts(dcpl, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) < 0)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }
    else {
        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options == H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, SPARSE_FILTER_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_EARRAY)
        FAIL_PUTS_ERROR("should be using extensible array chunk index");

    /* Starting at [3, 3], select 2 blocks of size 3x3 each */
    start[0]  = 3;
    start[1]  = 3;
    stride[0] = 4;
    stride[1] = 12;
    count[0]  = 1;
    count[1]  = 2;
    block[0]  = 3;
    block[1]  = 3;
    if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block) < 0)
        TEST_ERROR;

    memset(wbuf, 0, sizeof(wbuf));

    /* Initialize 2 3x3 blocks */
    wbuf[60] = 60;
    wbuf[61] = 61;
    wbuf[62] = 62;

    wbuf[72] = 72;
    wbuf[73] = 73;
    wbuf[74] = 74;

    wbuf[79] = 79;
    wbuf[80] = 80;
    wbuf[81] = 81;

    wbuf[91] = 91;
    wbuf[92] = 92;
    wbuf[93] = 93;

    wbuf[98]  = 98;
    wbuf[99]  = 99;
    wbuf[100] = 100;

    wbuf[110] = 110;
    wbuf[111] = 111;
    wbuf[112] = 112;

    if (H5Dwrite(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
        TEST_ERROR;

    if ((did = H5Dopen2(fid, SPARSE_FILTER_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Ensure we're using the correct chunk indexing scheme */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    if (idx_type != H5D_CHUNK_IDX_EARRAY)
        FAIL_PUTS_ERROR("should be using extensible array chunk index");

    if ((dcpl = H5Dget_create_plist(did)) < 0)
        TEST_ERROR;

    if (filtered) {
        /* Get filter info by filter number 0 for section "selection" */
        if ((filter_id = H5Pget_filter3(dcpl, H5_SECTION_SELECTION, 0, NULL, &my_cd_nelmts, &my_cd_value,
                                        (size_t)0, NULL, NULL)) < 0)
            TEST_ERROR;
        if (filter_id != H5Z_FILTER_DEFLATE)
            TEST_ERROR;
        if (my_cd_nelmts != 1)
            TEST_ERROR;
        if (my_cd_value != 9)
            TEST_ERROR;

        /* Get filter info by filter number 1 for section "selection" */
        if ((filter_id = H5Pget_filter3(dcpl, H5_SECTION_SELECTION, 1, &flags, NULL, NULL, (size_t)0, NULL,
                                        NULL)) < 0)
            TEST_ERROR;
        if (filter_id != H5Z_FILTER_SHUFFLE)
            TEST_ERROR;
        if (flags != H5Z_FLAG_OPTIONAL)
            TEST_ERROR;

        H5E_BEGIN_TRY
        {
            /* Get filter info by filter id for section "fixed data" */
            filter_id = H5Pget_filter_by_id3(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, &flags,
                                             &my_cd_nelmts, &my_cd_value, (size_t)0, NULL, NULL);
        }
        H5E_END_TRY
        /* No filter for section "fixed data" */
        if (filter_id >= 0)
            TEST_ERROR;

        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options != H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }
    else {
        if (H5Pget_chunk_opts(dcpl, &options) < 0)
            TEST_ERROR;

        if (options == H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS)
            TEST_ERROR;
    }

    memset(rbuf, 0, sizeof(rbuf));
    if (H5Dread(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;

    /* Verify data read */
    for (i = 0; i < 190; i++)
        if (rbuf[i] != wbuf[i])
            TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Closing */
    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

done:
error:
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Fclose(fid);
    }
    H5E_END_TRY

    if (fail_as_expected) {
        PASSED();
        return SUCCEED;
    }

    return FAIL;
} /* test_struct_chunk_2d_ea() */

/*-------------------------------------------------------------------------
 *  Function:    filter_class3_set_local
 *
 *  Purpose:     Set_local callback for H5Z_FiLTER_CLASS3 filter.
 *
 *  Return:      Success:        Data chunk size
 *              Failure:        0
 *-------------------------------------------------------------------------
 */
static herr_t
filter_class3_set_local(hid_t dcpl_id, hid_t H5_ATTR_UNUSED type_id, hid_t H5_ATTR_UNUSED space_id,
                        H5_section_type_t sec_type)
{
    unsigned flags;         /* Filter flags */
    size_t   cd_nelmts = 1; /* Number of filter parameters */
    unsigned cd_values[1];  /* Filter parameters */

    /* Get the filter's current parameters */
    if (H5Pget_filter_by_id3(dcpl_id, sec_type, H5Z_FILTER_CLASS3, &flags, &cd_nelmts, cd_values, (size_t)0,
                             NULL, NULL) < 0)
        return (FAIL);

    /* Check that the parameter values were passed along correctly */
    cd_values[0] = FILTER_PARAM_MOD;

    /* Modify the filter's parameters for this dataset */
    if (H5Pmodify_filter2(dcpl_id, sec_type, H5Z_FILTER_CLASS3, flags, cd_nelmts, cd_values) < 0)
        return (FAIL);

    return (SUCCEED);
} /* filter_class3_set_local() */

/*-------------------------------------------------------------------------
 *  Function:    filter_class3
 *
 *  Purpose:     This filter counts the number of bytes read and written,
 *               incrementing count_nbytes_read or count_nbytes_written as
 *               appropriate.
 *
 *  Return:      Success:        Data chunk size
 *              Failure:        0
 *-------------------------------------------------------------------------
 */
static size_t
filter_class3(unsigned int flags, size_t H5_ATTR_UNUSED cd_nelmts,
              const unsigned int H5_ATTR_UNUSED *cd_values, size_t nbytes, size_t H5_ATTR_UNUSED *buf_size,
              void H5_ATTR_UNUSED **buf)
{
    if (flags & H5Z_FLAG_REVERSE)
        filter_bytes_read += nbytes;
    else
        filter_bytes_written += nbytes;

    return nbytes;
} /* filter_class3() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_filter_register()
 *
 * Purpose:     Verify H5Zregister with H5Z_class3_t
 *              Also verify APIs:
 *              --H5Pmodify_filter2()
 *              --H5Premove_filter2()
 *
 *              Because structured chunks use SCC write-back caching, the
 *              test explicitly flushes the dataset before verifying that
 *              the registered filter processed encoded output.
 *
 * Return:      # of errors
 *
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_filter_register(hid_t fcpl, hid_t fapl)
{
    char    filename[FILENAME_BUF_SIZE];    /* File name */
    hid_t   fid          = H5I_INVALID_HID; /* File ID */
    hid_t   sid          = H5I_INVALID_HID; /* Dataspace ID */
    hid_t   did          = H5I_INVALID_HID; /* Dataset ID */
    hid_t   dcpl         = H5I_INVALID_HID; /* Creation plist */
    hsize_t dim[1]       = {10};            /* 1-d dataspace */
    hsize_t chunk_dim[1] = {5};             /* Chunk size */
    int     wbuf[10];                       /* Write buffer */
    int     rbuf[10];                       /* Read buffer */
    hsize_t start[1];
    hsize_t stride[1];
    hsize_t count[1];
    hsize_t block[1];

    unsigned int cd_values[1] = {FILTER_PARAM};
    size_t       cd_nelmts    = 1;
    int          nfilters;

    H5F_libver_t low, high; /* File format bound */
    bool         fail_as_expected = false;

    TESTING("structured chunk dataset with filter register");

    if (H5Pget_libver_bounds(fapl, &low, &high) < 0)
        TEST_ERROR;

    /* Create the file */
    h5_fixname(FILENAME[5], fapl, filename, sizeof filename);

    /* Create a new file. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create dataspace */
    if ((sid = H5Screate_simple(1, dim, NULL)) < 0)
        TEST_ERROR;

    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    if (H5Zregister(H5Z_TEST_CLASS3) < 0)
        TEST_ERROR;

    if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_CLASS3, H5Z_FLAG_OPTIONAL, cd_nelmts,
                       cd_values) < 0)
        TEST_ERROR;

    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl, 1, chunk_dim, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    H5E_BEGIN_TRY
    {
        did = H5Dcreate2(fid, SPARSE_FILTER_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    }
    H5E_END_TRY

    /* Should fail for high bound < latest format */
    if (high < H5F_LIBVER_LATEST) {
        if (did >= 0)
            TEST_ERROR;
        else {
            /* Fail as expected: clean up and return succeed */
            fail_as_expected = true;
            goto done;
        }
    }
    else if (did < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if ((dcpl = H5Dget_create_plist(did)) < 0)
        TEST_ERROR;

    if (H5Pget_filter_by_id3(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_CLASS3, NULL, &cd_nelmts, cd_values,
                             (size_t)0, NULL, NULL) < 0)
        TEST_ERROR;

    if (cd_nelmts != 1)
        TEST_ERROR;
    if (cd_values[0] != FILTER_PARAM_MOD)
        TEST_ERROR;

    filter_bytes_written = 0;
    filter_bytes_read    = 0;

    /* Starting at 3, select 1 block of size 3 */
    /* Selection is across 2 chunks */
    start[0]  = 3;
    stride[0] = 6;
    count[0]  = 1;
    block[0]  = 3;
    if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, stride, count, block) < 0)
        TEST_ERROR;

    /* Write sparse data to the dataset */
    memset(wbuf, 0, sizeof(wbuf));

    /* Initialize and write sparse data to the dataset */
    wbuf[3] = 1;
    wbuf[4] = 2;
    wbuf[5] = 3;

    if (H5Dwrite(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    /*
     * Structured chunks held by the SCC use write-back semantics. H5Dwrite()
     * may therefore leave the modified chunks dirty and resident without
     * immediately invoking the on-disk filter pipeline. Explicitly flush the
     * dataset before checking that the registered filter processed encoded
     * output.
     */
    if (H5Dflush(did) < 0)
        TEST_ERROR;

    if (!filter_bytes_written)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;
    did = H5I_INVALID_HID;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;
    fid = H5I_INVALID_HID;

    filter_bytes_read = 0;

    if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
        TEST_ERROR;

    if ((did = H5Dopen2(fid, SPARSE_FILTER_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    if ((dcpl = H5Dget_create_plist(did)) < 0)
        TEST_ERROR;

    memset(rbuf, 0, sizeof(rbuf));
    if (H5Dread(did, H5T_NATIVE_INT, sid, sid, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;

    if (!filter_bytes_read)
        TEST_ERROR;

    if (rbuf[3] != wbuf[3] || rbuf[4] != wbuf[4] || rbuf[5] != wbuf[5])
        TEST_ERROR;

    if (H5Premove_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_CLASS3) < 0)
        TEST_ERROR;

    if (H5Pget_nfilters2(dcpl, H5_SECTION_SELECTION, &nfilters) < 0)
        TEST_ERROR;

    if (nfilters)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Closing */
    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

done:
error:
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Fclose(fid);
    }
    H5E_END_TRY

    if (fail_as_expected) {
        PASSED();
        return SUCCEED;
    }

    return FAIL;
} /* test_struct_chunk_filter_register() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen
 *
 * Purpose:     Exercise the public VL read/write path with each structured chunk index.
 *
 * How it works: The shape selects single, fixed array, extensible array, or B-tree 2.
 *              Each of two passes writes all values, closes the file to discard cached
 *              chunks, reopens it, checks the index and each decoded integer, and
 *              reclaims H5Dread allocations. The second pass replaces existing heap
 *              objects.
 *
 * Coverage:    Emulates dense VL chunks through the sparse structured layout:
 *              dataset dimensions are exact multiples of chunk dimensions,
 *              and every position is written. Checks complete replacement
 *              and readback after reopening across all four indexes.
 *              Does not assert the internal selection encoding.
 *
 *
 * Parameters:  fcpl supplies file creation settings; fapl supplies file access
 *              settings. chk_type selects the chunk index; filtered enables an optional
 *              deflate filter on the VL section.
 *
 * Return:      SUCCEED after verification and cleanup, FAIL on any error.
 *
 * Note:        Read-side hvl_t pointers belong to HDF5 until H5Treclaim;
 *              write-side pointers refer to local test payload arrays.
 *
 *
 *                                               -- AZO   09/15/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen(hid_t fcpl, hid_t fapl, unsigned chk_type, bool filtered)
{
    char              filename[FILENAME_BUF_SIZE]; /* Test file path */
    hid_t             fid  = H5I_INVALID_HID;      /* File ID */
    hid_t             sid  = H5I_INVALID_HID;      /* Dataset dataspace ID */
    hid_t             dcpl = H5I_INVALID_HID;      /* Dataset creation property list ID */
    hid_t             did  = H5I_INVALID_HID;      /* Dataset ID */
    hid_t             tid  = H5I_INVALID_HID;      /* VL integer datatype ID */
    H5D_chunk_index_t idx_type;                    /* Index type found on the dataset */
    H5D_chunk_index_t expected_idx;                /* Index type required by this case */
    hsize_t           dims[2]       = {8, 4};      /* Dataset dimensions; adjusted for the index case */
    hsize_t           maxdims[2]    = {8, 4};      /* Maximum dimensions; adjusted for the index case */
    hsize_t           chunk_dims[2] = {4, 2};      /* Chunk dimensions; adjusted for the index case */
    hvl_t             wbuf[8];                     /* VL values written to the dataset */
    hvl_t             rbuf[8];                     /* VL values allocated by H5Dread() */
    int               values[8][3];                /* Integer payloads referenced by wbuf */
    unsigned int      level = 6;                   /* Optional VL-section deflate level */
    unsigned          rank;                        /* Rank selected for the index case */
    unsigned          nelems;                      /* Number of dataset elements in this case */
    unsigned          pass;                        /* Initial write or replacement pass */
    unsigned          i;                           /* Dataset element index */
    unsigned          j;                           /* Integer within a VL value */
    bool              read_needs_reclaim = false;  /* Whether rbuf owns VL memory to reclaim */

    /* Announce this case through the HDF5 test harness. */
    TESTING("structured chunk VL round trip");

    /* The same round-trip checks cover single chunk, fixed array,
     * extensible array, and B-tree 2 indexes.
     */
    /* Select a shape and maximum extent that triggers the requested index. */
    switch (chk_type) {
        case CHK_SINGLE:
            rank          = 1;
            nelems        = 4;
            dims[0]       = 4;
            maxdims[0]    = 4;
            chunk_dims[0] = 4;
            expected_idx  = H5D_CHUNK_IDX_SINGLE;
            break;

        case CHK_FA:
            rank         = 1;
            nelems       = 8;
            expected_idx = H5D_CHUNK_IDX_FARRAY;
            break;

        case CHK_EA:
            rank         = 1;
            nelems       = 8;
            maxdims[0]   = H5S_UNLIMITED;
            expected_idx = H5D_CHUNK_IDX_EARRAY;
            break;

        default: /* BT2: both dimensions unlimited */
            rank          = 2;
            nelems        = 8;
            dims[0]       = 2;
            maxdims[0]    = H5S_UNLIMITED;
            maxdims[1]    = H5S_UNLIMITED;
            chunk_dims[0] = 1;
            expected_idx  = H5D_CHUNK_IDX_BT2;
            break;
    }

    /* Build a driver-aware filename for this test file. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    /* Create a fresh file so earlier cases cannot supply data. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;
    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((sid = H5Screate_simple((int)rank, dims, maxdims)) < 0)
        TEST_ERROR;
    /* Build a variable-length datatype with the requested base type. */
    if ((tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;
    /* Start a dataset creation property list for layout and filters. */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;
    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, rank, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (filtered) {

        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    /* Create the dataset using the configured type, extent, and chunk layout. */
    if ((did = H5Dcreate2(fid, "vlen", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Ask the dataset which chunk-index implementation it actually chose. */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;
    /* Catch an unexpected index choice before exercising its records. */
    if (idx_type != expected_idx)
        TEST_ERROR;

    /* The second pass replaces every VL value written by the first pass. */
    /* Run each write/update state and verify it after closing the file. */
    for (pass = 0; pass < 2; pass++) {

        /* Prepare or check each dataset element against its expected value. */
        for (i = 0; i < nelems; i++) {

            wbuf[i].len = 1 + ((i + pass) % 3);
            wbuf[i].p   = values[i];

            /* Process every scalar inside this variable-length payload. */
            for (j = 0; j < wbuf[i].len; j++) {
                values[i][j] = (int)(100 * pass + 10 * i + j + 1);
            }
        }

        /* Dense emulation: this full-dataset write defines every position
         * in every chunk. Pass 1 replaces all previously defined values.
         */
        if (H5Dwrite(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0)
            TEST_ERROR;

        /* Force the next read through the persisted chunk and index. */
        /* Close the dataset to release its cached decoded chunks and persist changes. */
        if (H5Dclose(did) < 0)
            TEST_ERROR;
        /* Mark the handle closed so error cleanup cannot close it twice. */
        did = H5I_INVALID_HID;
        /* Close the file; an active pass reopens it to check stored data. */
        if (H5Fclose(fid) < 0)
            TEST_ERROR;
        /* The file handle is no longer owned by this function. */
        fid = H5I_INVALID_HID;

        /* Reopen the file to verify the serialized, rather than cached, state. */
        if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
            TEST_ERROR;
        /* Open the same dataset for the persisted-data check. */
        if ((did = H5Dopen2(fid, "vlen", H5P_DEFAULT)) < 0)
            TEST_ERROR;

        /* Ask the dataset which chunk-index implementation it actually chose. */
        if (H5D__layout_idx_type_test(did, &idx_type) < 0)
            TEST_ERROR;
        /* Catch an unexpected index choice before exercising its records. */
        if (idx_type != expected_idx)
            TEST_ERROR;

        /* Start with empty read descriptors, including safe NULL payload pointers. */
        memset(rbuf, 0, sizeof rbuf);
        /* Decode the chunk and convert its VL values into newly allocated memory. */
        if (H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;
        /* Enable error-path reclamation of any VL buffers returned by the read. */
        read_needs_reclaim = true;

        /* Prepare or check each dataset element against its expected value. */
        for (i = 0; i < nelems; i++) {

            if (rbuf[i].len != wbuf[i].len || !rbuf[i].p)
                TEST_ERROR;

            /* Process every scalar inside this variable-length payload. */
            for (j = 0; j < wbuf[i].len; j++) {

                if (((int *)rbuf[i].p)[j] != values[i][j])
                    TEST_ERROR;
            }
        }

        /* Release every payload H5Dread allocated in this memory dataspace. */
        if (H5Treclaim(tid, sid, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;
        /* The read buffer no longer owns allocated VL payloads. */
        read_needs_reclaim = false;
    } /* end for */

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;
    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;
    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;
    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;
    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;
    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* Report success only after all comparisons and normal cleanup succeed. */
    PASSED();
    return SUCCEED;

/* One shared error path closes whatever was created before the failed check. */
error:
    /* Suppress secondary HDF5 errors while cleaning up partial setup. */
    H5E_BEGIN_TRY
    {
        if (read_needs_reclaim)
            H5Treclaim(tid, sid, H5P_DEFAULT, rbuf);
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    /* Tell the harness that this case failed. */
    return FAIL;
} /* end test_struct_chunk_vlen() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_partial
 *
 * Purpose:     Exercise sparse writes that touch two different chunks.
 *
 * How it works: A two-element memory selection maps to positions 1 and 6 in an eight-
 *              element dataset. The first pass writes both; the second replaces
 *              position 1 and writes an empty VL value at 6. Each pass reopens and
 *              checks all eight positions, including those never written.
 *
 * Parameters:  fcpl supplies file creation settings; fapl supplies file access
 *              settings. filtered enables the optional section filter(s) for the second
 *              variant of this test.
 *
 * Return:      SUCCEED after verification and cleanup, FAIL on any error.
 *
 * Note:        Read-side hvl_t pointers belong to HDF5 until H5Treclaim;
 *              write-side pointers refer to local test payload arrays.
 *
 *                                               -- AZO   09/15/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_partial(hid_t fcpl, hid_t fapl, bool filtered)
{
    char         filename[FILENAME_BUF_SIZE];     /* Test file path */
    hid_t        fid           = H5I_INVALID_HID; /* File ID */
    hid_t        sid           = H5I_INVALID_HID; /* Full dataset dataspace ID */
    hid_t        file_sid      = H5I_INVALID_HID; /* Dataspace used to select dataset positions */
    hid_t        mem_sid       = H5I_INVALID_HID; /* Dataspace for the two-element write buffer */
    hid_t        dcpl          = H5I_INVALID_HID; /* Dataset creation property list ID */
    hid_t        did           = H5I_INVALID_HID; /* Dataset ID */
    hid_t        tid           = H5I_INVALID_HID; /* VL integer datatype ID */
    hid_t        defined_sid   = H5I_INVALID_HID;
    hsize_t      dims[1]       = {8};          /* Eight dataset positions */
    hsize_t      chunk_dims[1] = {4};          /* Four positions per chunk */
    hsize_t      mem_dims[1]   = {2};          /* Two values supplied by a partial write */
    hsize_t      start[1]      = {1};          /* First selected dataset position */
    hsize_t      stride[1]     = {5};          /* Distance from position 1 to position 6 */
    hsize_t      count[1]      = {2};          /* Number of selected positions */
    hsize_t      block[1]      = {1};          /* One element at each selected position */
    unsigned int level         = 6;            /* Optional VL-section deflate level */
    int          first[3]      = {11, 12, 13}; /* Initial payload at position 1 */
    int          second[2]     = {61, 62};     /* Initial payload at position 6 */
    int          replacement   = 99;           /* Replacement payload at position 1 */
    hvl_t        wbuf[2];                      /* Values supplied to a partial write */
    hvl_t        rbuf[8];                      /* Values returned by the full read */
    unsigned     pass;                         /* Initial or replacement verification pass */
    unsigned     i;                            /* Dataset position being checked */
    bool         reclaim_read = false;         /* Whether rbuf owns VL memory to reclaim */

    /* Announce this case through the HDF5 test harness. */
    TESTING("structured chunk sparse VL writes and replacement");

    /* Build a driver-aware filename for this test file. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    /* Create a fresh file so earlier cases cannot supply data. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;
    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((sid = H5Screate_simple(1, dims, NULL)) < 0)
        TEST_ERROR;
    /* Keep an independent selection so the full dataset extent stays available. */
    if ((file_sid = H5Scopy(sid)) < 0)
        TEST_ERROR;
    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((mem_sid = H5Screate_simple(1, mem_dims, NULL)) < 0)
        TEST_ERROR;
    /* Build a variable-length datatype with the requested base type. */
    if ((tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;
    /* Start a dataset creation property list for layout and filters. */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;
    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, 1, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (filtered) {

        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    /* Create the dataset using the configured type, extent, and chunk layout. */
    if ((did = H5Dcreate2(fid, "partial_vlen", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Map the two-element memory buffer to dataset positions 1 and 6.
     * The stride crosses the chunk boundary between positions 3 and 4.
     */
    /* Map the small memory buffer to the selected file position(s). */
    if (H5Sselect_hyperslab(file_sid, H5S_SELECT_SET, start, stride, count, block) < 0)
        TEST_ERROR;

    /* Run each write/update state and verify it after closing the file. */
    for (pass = 0; pass < 2; pass++) {
        if (pass == 0) {
            wbuf[0].len = 3;
            wbuf[0].p   = first;
            wbuf[1].len = 2;
            wbuf[1].p   = second;
        }
        else {
            /* Replace one payload and clear the other. */
            wbuf[0].len = 1;
            wbuf[0].p   = &replacement;
            wbuf[1].len = 0;
            wbuf[1].p   = NULL;
        }

        /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
        if (H5Dwrite(did, tid, mem_sid, file_sid, H5P_DEFAULT, wbuf) < 0)
            TEST_ERROR;

        /* Reopen to force the result through serialization and index lookup. */
        /* Close the dataset to release its cached decoded chunks and persist changes. */
        if (H5Dclose(did) < 0)
            TEST_ERROR;

        /* Mark the handle closed so error cleanup cannot close it twice. */
        did = H5I_INVALID_HID;

        /* Close the file; an active pass reopens it to check stored data. */
        if (H5Fclose(fid) < 0)
            TEST_ERROR;

        /* The file handle is no longer owned by this function. */
        fid = H5I_INVALID_HID;

        /* Reopen the file to verify the serialized, rather than cached, state. */
        if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
            TEST_ERROR;

        /* Open the same dataset for the persisted-data check. */
        if ((did = H5Dopen2(fid, "partial_vlen", H5P_DEFAULT)) < 0)
            TEST_ERROR;

        /* Positions 1 and 6 remain defined in both passes.
         * In pass 1, position 6 is defined-empty, not undefined.
         */
        if ((defined_sid = H5Dget_defined(did, H5S_ALL, H5P_DEFAULT)) < 0)
            TEST_ERROR;

        if (H5Sget_select_npoints(defined_sid) != 2)
            TEST_ERROR;

        for (i = 0; i < 8; i++) {

            hsize_t coord[1] = {(hsize_t)i};
            htri_t  selected;

            if ((selected = H5Sselect_intersect_block(defined_sid, coord, coord)) < 0)
                TEST_ERROR;

            if ((selected > 0) != (i == 1 || i == 6))
                TEST_ERROR;
        }

        if (H5Sclose(defined_sid) < 0)
            TEST_ERROR;

        defined_sid = H5I_INVALID_HID;

        /* Start with empty read descriptors, including safe NULL payload pointers. */
        memset(rbuf, 0, sizeof rbuf);

        /* Enable error-path reclamation of any VL buffers returned by the read. */
        reclaim_read = true;

        /* Decode the chunk and convert its VL values into newly allocated memory. */
        if (H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* Prepare or check each dataset element against its expected value. */
        for (i = 0; i < 8; i++) {

            if (i == 1) {
                if (rbuf[i].len != (pass == 0 ? 3 : 1) || !rbuf[i].p)
                    TEST_ERROR;

                if (pass == 0) {

                    if ((((int *)rbuf[i].p)[0] != 11) || (((int *)rbuf[i].p)[1] != 12) ||
                        (((int *)rbuf[i].p)[2] != 13))
                        TEST_ERROR;
                }
                else if (((int *)rbuf[i].p)[0] != 99)
                    TEST_ERROR;
            }
            else if (i == 6 && pass == 0) {
                if ((rbuf[i].len != 2 || !rbuf[i].p) || (((int *)rbuf[i].p)[0] != 61) ||
                    (((int *)rbuf[i].p)[1] != 62))
                    TEST_ERROR;
            }
            else if (rbuf[i].len != 0)
                TEST_ERROR;
        } /* end for */

        /* Release every payload H5Dread allocated in this memory dataspace. */
        if (H5Treclaim(tid, sid, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* The read buffer no longer owns allocated VL payloads. */
        reclaim_read = false;

    } /* end for */

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;
    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;

    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;

    if (H5Sclose(mem_sid) < 0)
        TEST_ERROR;
    mem_sid = H5I_INVALID_HID;

    if (H5Sclose(file_sid) < 0)
        TEST_ERROR;
    file_sid = H5I_INVALID_HID;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* Report success only after all comparisons and normal cleanup succeed. */
    PASSED();
    return SUCCEED;

/* One shared error path closes whatever was created before the failed check. */
error:
    /* Suppress secondary HDF5 errors while cleaning up partial setup. */
    H5E_BEGIN_TRY
    {
        if (reclaim_read && tid >= 0 && sid >= 0)
            H5Treclaim(tid, sid, H5P_DEFAULT, rbuf);

        H5Sclose(defined_sid);
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Sclose(mem_sid);
        H5Sclose(file_sid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    /* Tell the harness that this case failed. */
    return FAIL;
} /* end test_struct_chunk_vlen_partial() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_churn
 *
 * Purpose:     Exercise repeated changes to heap objects while neighbors stay live.
 *
 * How it works: Initialize both chunks, then replace one of eight elements 24 times
 *              with lengths 1, 32, and 7. Reopen after every write and compare all
 *              values to the current expected arrays. This detects accidental changes
 *              to unchanged descriptors.
 *
 * Parameters:  fcpl supplies file creation settings; fapl supplies file access
 *              settings. filtered enables the optional section filter(s) for the second
 *              variant of this test.
 *
 * Return:      SUCCEED after verification and cleanup, FAIL on any error.
 *
 * Note:        Read-side hvl_t pointers belong to HDF5 until H5Treclaim;
 *              write-side pointers refer to local test payload arrays.
 *
 *                                                -- AZO   09/15/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_churn(hid_t fcpl, hid_t fapl, bool filtered)
{
    char              filename[FILENAME_BUF_SIZE];     /* Test file path */
    hid_t             fid      = H5I_INVALID_HID;      /* File ID */
    hid_t             sid      = H5I_INVALID_HID;      /* Full dataset dataspace ID */
    hid_t             file_sid = H5I_INVALID_HID;      /* Dataspace selecting the updated position */
    hid_t             mem_sid  = H5I_INVALID_HID;      /* One-element update dataspace ID */
    hid_t             dcpl     = H5I_INVALID_HID;      /* Dataset creation property list ID */
    hid_t             did      = H5I_INVALID_HID;      /* Dataset ID */
    hid_t             tid      = H5I_INVALID_HID;      /* VL integer datatype ID */
    H5D_chunk_index_t idx_type;                        /* Index type found on the dataset */
    hsize_t           dims[1]       = {8};             /* Eight dataset positions */
    hsize_t           maxdims[1]    = {H5S_UNLIMITED}; /* Unlimited dimension for the index */
    hsize_t           chunk_dims[1] = {4};             /* Two chunks of four positions */
    hsize_t           one[1]        = {1};             /* Extent of the update selection */
    hsize_t           start[1];                        /* Position selected for this update */
    hvl_t             wbuf[8];                         /* Initial VL values */
    hvl_t             update;                          /* Replacement for one position */
    hvl_t             rbuf[8];                         /* Values read after reopening */
    int               values[8][32];                   /* Current expected payloads */
    size_t            lengths[8];                      /* Current expected VL lengths */
    unsigned int      level = 6;                       /* Optional VL-section deflate level */
    unsigned          pass;                            /* Replacement iteration */
    unsigned          i;                               /* Dataset position */
    unsigned          j;                               /* Integer within a VL value */
    unsigned          target;                          /* Position replaced on this iteration */
    bool              reclaim_read = false;            /* Whether rbuf owns VL memory to reclaim */

    /* Announce this case through the HDF5 test harness. */
    TESTING("structured chunk repeated VL replacement");

    /* Build a driver-aware filename for this test file. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    /* Create a fresh file so earlier cases cannot supply data. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((sid = H5Screate_simple(1, dims, maxdims)) < 0)
        TEST_ERROR;

    /* Keep an independent selection so the full dataset extent stays available. */
    if ((file_sid = H5Scopy(sid)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((mem_sid = H5Screate_simple(1, one, NULL)) < 0)
        TEST_ERROR;

    /* Build a variable-length datatype with the requested base type. */
    if ((tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;

    /* Start a dataset creation property list for layout and filters. */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, 1, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (filtered) {

        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    /* Create the dataset using the configured type, extent, and chunk layout. */
    if ((did = H5Dcreate2(fid, "churn_vlen", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Ask the dataset which chunk-index implementation it actually chose. */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    /* Catch an unexpected index choice before exercising its records. */
    if (idx_type != H5D_CHUNK_IDX_EARRAY)
        TEST_ERROR;

    /* Give every element an initial payload. */
    /* Prepare or check each dataset element against its expected value. */
    for (i = 0; i < 8; i++) {
        lengths[i]   = 2;
        values[i][0] = (int)(10 * i + 1);
        values[i][1] = (int)(10 * i + 2);
        wbuf[i].len  = lengths[i];
        wbuf[i].p    = values[i];
    }

    /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
    if (H5Dwrite(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    /* Run each write/update state and verify it after closing the file. */
    for (pass = 0; pass < 24; pass++) {

        /* The target rotation reaches every position in both chunks. Alternating
         * lengths exercises replacement by smaller and larger heap objects.
         */
        target          = (pass * 5) % 8;
        lengths[target] = (pass % 3 == 0) ? 1 : ((pass % 3 == 1) ? 32 : 7);

        /* Process every scalar inside this variable-length payload. */
        for (j = 0; j < lengths[target]; j++) {

            values[target][j] = (int)(1000 + 100 * pass + j);
        }

        start[0] = target;

        /* Map the small memory buffer to the selected file position(s). */
        if (H5Sselect_hyperslab(file_sid, H5S_SELECT_SET, start, NULL, one, NULL) < 0)
            TEST_ERROR;

        update.len = lengths[target];
        update.p   = values[target];

        /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
        if (H5Dwrite(did, tid, mem_sid, file_sid, H5P_DEFAULT, &update) < 0)
            TEST_ERROR;

        /* Discard the in-memory chunk before checking this update on disk. */
        /* Close the dataset to release its cached decoded chunks and persist changes. */
        if (H5Dclose(did) < 0)
            TEST_ERROR;

        /* Mark the handle closed so error cleanup cannot close it twice. */
        did = H5I_INVALID_HID;

        /* Close the file; an active pass reopens it to check stored data. */
        if (H5Fclose(fid) < 0)
            TEST_ERROR;

        /* The file handle is no longer owned by this function. */
        fid = H5I_INVALID_HID;

        /* Reopen the file to verify the serialized, rather than cached, state. */
        if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
            TEST_ERROR;

        /* Open the same dataset for the persisted-data check. */
        if ((did = H5Dopen2(fid, "churn_vlen", H5P_DEFAULT)) < 0)
            TEST_ERROR;

        /* Start with empty read descriptors, including safe NULL payload pointers. */
        memset(rbuf, 0, sizeof rbuf);

        /* Enable error-path reclamation of any VL buffers returned by the read. */
        reclaim_read = true;

        /* Decode the chunk and convert its VL values into newly allocated memory. */
        if (H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* Prepare or check each dataset element against its expected value. */
        for (i = 0; i < 8; i++) {

            if (rbuf[i].len != lengths[i] || !rbuf[i].p)
                TEST_ERROR;

            /* Process every scalar inside this variable-length payload. */
            for (j = 0; j < lengths[i]; j++) {

                if (((int *)rbuf[i].p)[j] != values[i][j])
                    TEST_ERROR;
            }
        } /* end for */

        /* Release every payload H5Dread allocated in this memory dataspace. */
        if (H5Treclaim(tid, sid, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;
        /* The read buffer no longer owns allocated VL payloads. */
        reclaim_read = false;

    } /* end for */

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;

    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;

    if (H5Sclose(mem_sid) < 0)
        TEST_ERROR;
    mem_sid = H5I_INVALID_HID;

    if (H5Sclose(file_sid) < 0)
        TEST_ERROR;
    file_sid = H5I_INVALID_HID;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* Report success only after all comparisons and normal cleanup succeed. */
    PASSED();
    return SUCCEED;

error:
    /* Suppress secondary HDF5 errors while cleaning up partial setup. */
    H5E_BEGIN_TRY
    {
        if (reclaim_read && tid >= 0 && sid >= 0)
            H5Treclaim(tid, sid, H5P_DEFAULT, rbuf);
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Sclose(mem_sid);
        H5Sclose(file_sid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    /* Tell the harness that this case failed. */
    return FAIL;

} /* end test_struct_chunk_vlen_churn() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_large
 *
 * Purpose:     Exercise large payload encoding, filtering, and replacement.
 *
 * How it works: Choose each index shape, generate a varied 131072-integer payload, and
 *              write it alongside smaller live payloads. Reopen and check all elements;
 *              then remove a large payload, shrink another where applicable, and
 *              repeat. The varied data challenges encoded size handling with deflate
 *              enabled.
 *
 * Parameters:  fcpl supplies file creation settings; fapl supplies file access
 *              settings. chk_type selects the chunk index; filtered enables an optional
 *              deflate filter on the VL section.
 *
 * Return:      SUCCEED after verification and cleanup, FAIL on any error.
 *
 * Note:        Read-side hvl_t pointers belong to HDF5 until H5Treclaim;
 *              write-side pointers refer to local test payload arrays.
 *
 *                                                -- AZO   09/15/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_large(hid_t fcpl, hid_t fapl, unsigned chk_type, bool filtered)
{
    char              filename[FILENAME_BUF_SIZE]; /* Test file path */
    hid_t             fid  = H5I_INVALID_HID;      /* File ID */
    hid_t             sid  = H5I_INVALID_HID;      /* Dataset dataspace ID */
    hid_t             tid  = H5I_INVALID_HID;      /* VL integer datatype ID */
    hid_t             dcpl = H5I_INVALID_HID;      /* Dataset creation property list ID */
    hid_t             did  = H5I_INVALID_HID;      /* Dataset ID */
    H5D_chunk_index_t idx_type;                    /* Index type found on the dataset */
    H5D_chunk_index_t expected_idx;                /* Index type required by this case */
    hsize_t           dims[2]       = {8, 4};      /* Dataset dimensions for the index case */
    hsize_t           maxdims[2]    = {8, 4};      /* Maximum dimensions for the index case */
    hsize_t           chunk_dims[2] = {4, 2};      /* Chunk dimensions for the index case */
    hvl_t             wbuf[8];                     /* Large and small values being written */
    hvl_t             rbuf[8];                     /* Values allocated by H5Dread() */
    int              *large = NULL;                /* Dynamically allocated large payload */
    int               small[8][2];                 /* Small payloads kept live beside it */
    int               replacement = 987654;        /* Value used during replacement */
    uint32_t          state       = 0x12345678;    /* State for varied large-payload data */
    unsigned int      level       = 6;             /* Optional VL-section deflate level */
    unsigned          rank;                        /* Dataset rank for the index case */
    unsigned          nelems;                      /* Number of elements in this case */
    unsigned          pass;                        /* Write and verification pass */
    unsigned          i;                           /* Dataset element index */
    size_t            j;                           /* Integer within a VL payload */
    bool              reclaim_read = false;        /* Whether rbuf owns VL memory to reclaim */

    /* Number of integers in the large VL payload. */
    const size_t large_nints = 131072;

    /* Announce this case through the HDF5 test harness. */
    TESTING("structured chunk large VL payload and replacement");

    /* Select a shape and maximum extent that triggers the requested index. */
    switch (chk_type) {
        case CHK_SINGLE:
            rank          = 1;
            nelems        = 4;
            dims[0]       = 4;
            maxdims[0]    = 4;
            chunk_dims[0] = 4;
            expected_idx  = H5D_CHUNK_IDX_SINGLE;
            break;

        case CHK_FA:
            rank         = 1;
            nelems       = 8;
            expected_idx = H5D_CHUNK_IDX_FARRAY;
            break;

        case CHK_EA:
            rank         = 1;
            nelems       = 8;
            maxdims[0]   = H5S_UNLIMITED;
            expected_idx = H5D_CHUNK_IDX_EARRAY;
            break;

        default: /* BT2: both dimensions unlimited */
            rank          = 2;
            nelems        = 8;
            dims[0]       = 2;
            maxdims[0]    = H5S_UNLIMITED;
            maxdims[1]    = H5S_UNLIMITED;
            chunk_dims[0] = 1;
            expected_idx  = H5D_CHUNK_IDX_BT2;
            break;
    }

    /* Build a driver-aware filename for this test file. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    if (NULL == (large = malloc(large_nints * sizeof(*large))))
        TEST_ERROR;

    /* Produce varied bytes so the optional filter does not reduce this
     * payload to a tiny image; this exercises large encoded chunk lengths.
     */
    /* Process every scalar inside this variable-length payload. */
    for (j = 0; j < large_nints; j++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        large[j] = (int)(state & UINT32_C(0x7fffffff));
    }

    /* Prepare or check each dataset element against its expected value. */
    for (i = 0; i < 8; i++) {
        small[i][0] = (int)(100 + i);
        small[i][1] = (int)(200 + i);
    }

    /* Create a fresh file so earlier cases cannot supply data. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((sid = H5Screate_simple((int)rank, dims, maxdims)) < 0)
        TEST_ERROR;

    /* Build a variable-length datatype with the requested base type. */
    if ((tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;

    /* Start a dataset creation property list for layout and filters. */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, rank, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (filtered) {

        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    /* Create the dataset using the configured type, extent, and chunk layout. */
    if ((did = H5Dcreate2(fid, "large_vlen", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Ask the dataset which chunk-index implementation it actually chose. */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    /* Catch an unexpected index choice before exercising its records. */
    if (idx_type != expected_idx)
        TEST_ERROR;

    /* Run each write/update state and verify it after closing the file. */
    for (pass = 0; pass < 2; pass++) {

        /* Prepare or check each dataset element against its expected value. */
        for (i = 0; i < nelems; i++) {
            wbuf[i].len = 0;
            wbuf[i].p   = NULL;
        }

        /* Keep another live object in each chunk while replacing the large one. */
        wbuf[1].len = 2;
        wbuf[1].p   = small[1];

        if (pass == 0) {
            wbuf[0].len = large_nints;
            wbuf[0].p   = large;

            if (nelems == 8) {
                wbuf[4].len = large_nints;
                wbuf[4].p   = large;
                wbuf[5].len = 2;
                wbuf[5].p   = small[5];
            }
        }
        else {
            /* Remove one large value; shrink the one in the second chunk. */
            if (nelems == 8) {
                wbuf[4].len = 1;
                wbuf[4].p   = &replacement;
                wbuf[5].len = 2;
                wbuf[5].p   = small[5];
            }
        }

        /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
        if (H5Dwrite(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0)
            TEST_ERROR;

        /* Close the dataset to release its cached decoded chunks and persist changes. */
        if (H5Dclose(did) < 0)
            TEST_ERROR;

        /* Mark the handle closed so error cleanup cannot close it twice. */
        did = H5I_INVALID_HID;

        /* Close the file; an active pass reopens it to check stored data. */
        if (H5Fclose(fid) < 0)
            TEST_ERROR;

        /* The file handle is no longer owned by this function. */
        fid = H5I_INVALID_HID;

        /* Reopen the file to verify the serialized, rather than cached, state. */
        if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
            TEST_ERROR;

        /* Open the same dataset for the persisted-data check. */
        if ((did = H5Dopen2(fid, "large_vlen", H5P_DEFAULT)) < 0)
            TEST_ERROR;

        /* Ask the dataset which chunk-index implementation it actually chose. */
        if (H5D__layout_idx_type_test(did, &idx_type) < 0)
            TEST_ERROR;

        /* Catch an unexpected index choice before exercising its records. */
        if (idx_type != expected_idx)
            TEST_ERROR;

        /* Start with empty read descriptors, including safe NULL payload pointers. */
        memset(rbuf, 0, sizeof rbuf);

        /* Enable error-path reclamation of any VL buffers returned by the read. */
        reclaim_read = true;

        /* Decode the chunk and convert its VL values into newly allocated memory. */
        if (H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* Prepare or check each dataset element against its expected value. */
        for (i = 0; i < nelems; i++) {

            if (rbuf[i].len != wbuf[i].len)
                TEST_ERROR;

            if (wbuf[i].len > 0) {

                if (!rbuf[i].p)
                    TEST_ERROR;

                /* Process every scalar inside this variable-length payload. */
                for (j = 0; j < wbuf[i].len; j++) {
                    if (((int *)rbuf[i].p)[j] != ((int *)wbuf[i].p)[j])
                        TEST_ERROR;
                }
            }
        }

        /* Release every payload H5Dread allocated in this memory dataspace. */
        if (H5Treclaim(tid, sid, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* The read buffer no longer owns allocated VL payloads. */
        reclaim_read = false;

    } /* end for */

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;
    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;

    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    free(large);

    /* Report success only after all comparisons and normal cleanup succeed. */
    PASSED();
    return SUCCEED;

/* One shared error path closes whatever was created before the failed check. */
error:
    /* Suppress secondary HDF5 errors while cleaning up partial setup. */
    H5E_BEGIN_TRY
    {
        if (reclaim_read && tid >= 0 && sid >= 0)
            H5Treclaim(tid, sid, H5P_DEFAULT, rbuf);
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    free(large);
    /* Tell the harness that this case failed. */
    return FAIL;

} /* end test_struct_chunk_vlen_large() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_compound
 *
 * Purpose:     Exercise fixed and VL members in one compound chunk.
 *
 * How it works: Construct a compound datatype with integer fields on both sides of one
 *              VL field. Check the initial persisted records; then update record 2,
 *              clear its VL field, reopen, and confirm the other live VL members and
 *              all fixed fields remain correct.
 *
 * Parameters:  fcpl supplies file creation settings; fapl supplies file access
 *              settings. filtered enables the optional section filter(s) for the second
 *              variant of this test.
 *
 * Return:      SUCCEED after verification and cleanup, FAIL on any error.
 *
 * Note:        Read-side hvl_t pointers belong to HDF5 until H5Treclaim;
 *              write-side pointers refer to local test payload arrays.
 *
 *                                                -- AZO   09/15/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_compound(hid_t fcpl, hid_t fapl, bool filtered)
{
    typedef struct compound_vl_record_t {
        int   before;
        hvl_t values;
        int   after;
    } compound_vl_record_t;

    char                 filename[FILENAME_BUF_SIZE];  /* Test file path */
    hid_t                fid      = H5I_INVALID_HID;   /* File ID */
    hid_t                sid      = H5I_INVALID_HID;   /* Full dataset dataspace ID */
    hid_t                file_sid = H5I_INVALID_HID;   /* Dataspace selecting the updated record */
    hid_t                mem_sid  = H5I_INVALID_HID;   /* One-record update dataspace ID */
    hid_t                dcpl     = H5I_INVALID_HID;   /* Dataset creation property list ID */
    hid_t                did      = H5I_INVALID_HID;   /* Dataset ID */
    hid_t                vl_tid   = H5I_INVALID_HID;   /* VL integer member datatype ID */
    hid_t                tid      = H5I_INVALID_HID;   /* Compound record datatype ID */
    H5D_chunk_index_t    idx_type;                     /* Index type found on the dataset */
    hsize_t              dims[1]       = {4};          /* Four compound records */
    hsize_t              chunk_dims[1] = {4};          /* All records share one chunk */
    hsize_t              one[1]        = {1};          /* Extent of the update selection */
    hsize_t              start[1]      = {2};          /* Record selected for replacement */
    unsigned int         level         = 6;            /* Optional section deflate level */
    int                  values0[2]    = {11, 12};     /* VL payload for record 0 */
    int                  values2[3]    = {31, 32, 33}; /* Initial VL payload for record 2 */
    int                  values3[1]    = {41};         /* VL payload for record 3 */
    compound_vl_record_t wbuf[4];                      /* Initial compound records */
    compound_vl_record_t update;                       /* Replacement for record 2 */
    compound_vl_record_t rbuf[4];                      /* Records read after reopening */
    unsigned             pass;                         /* Initial or replacement state */
    unsigned             i;                            /* Record index */
    size_t               j;                            /* Integer within a VL member */
    bool                 reclaim_read = false;         /* Whether rbuf owns VL memory to reclaim */

    /* Announce this case through the HDF5 test harness. */
    TESTING("structured chunk compound datatype with VL member");

    /* Build a driver-aware filename for this test file. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    /* Create a fresh file so earlier cases cannot supply data. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((sid = H5Screate_simple(1, dims, NULL)) < 0)
        TEST_ERROR;

    /* Keep an independent selection so the full dataset extent stays available. */
    if ((file_sid = H5Scopy(sid)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((mem_sid = H5Screate_simple(1, one, NULL)) < 0)
        TEST_ERROR;

    /* Build a variable-length datatype with the requested base type. */
    if ((vl_tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;

    /* Create a compound datatype matching the C record size. */
    if ((tid = H5Tcreate(H5T_COMPOUND, sizeof(compound_vl_record_t))) < 0)
        TEST_ERROR;

    /* Add a member at its real C-structure offset (HOFFSET handles padding). */
    if (H5Tinsert(tid, "before", HOFFSET(compound_vl_record_t, before), H5T_NATIVE_INT) < 0)
        TEST_ERROR;

    /* Add a member at its real C-structure offset (HOFFSET handles padding). */
    if (H5Tinsert(tid, "values", HOFFSET(compound_vl_record_t, values), vl_tid) < 0)
        TEST_ERROR;

    /* Add a member at its real C-structure offset (HOFFSET handles padding). */
    if (H5Tinsert(tid, "after", HOFFSET(compound_vl_record_t, after), H5T_NATIVE_INT) < 0)
        TEST_ERROR;

    /* Start a dataset creation property list for layout and filters. */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, 1, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (filtered) {

        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    /* Create the dataset using the configured type, extent, and chunk layout. */
    if ((did = H5Dcreate2(fid, "compound_vlen", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Ask the dataset which chunk-index implementation it actually chose. */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    /* Catch an unexpected index choice before exercising its records. */
    if (idx_type != H5D_CHUNK_IDX_SINGLE)
        TEST_ERROR;

    memset(wbuf, 0, sizeof wbuf);
    /* Prepare or check each dataset element against its expected value. */
    for (i = 0; i < 4; i++) {
        wbuf[i].before = (int)(100 + i);
        wbuf[i].after  = (int)(200 + i);
    }

    wbuf[0].values.len = 2;
    wbuf[0].values.p   = values0;
    wbuf[2].values.len = 3;
    wbuf[2].values.p   = values2;
    wbuf[3].values.len = 1;
    wbuf[3].values.p   = values3;
    /* Element 1 has an empty VL value. */

    /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
    if (H5Dwrite(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    /* Run each write/update state and verify it after closing the file. */
    for (pass = 0; pass < 2; pass++) {

        if (pass == 1) {
            /*
             * Clear element 2's VL value and change its fixed fields.
             * Elements 0 and 3 retain live VL payloads in the same chunk.
             */
            memset(&update, 0, sizeof update);
            update.before = 302;
            update.after  = 402;

            /* Map the small memory buffer to the selected file position(s). */
            if (H5Sselect_hyperslab(file_sid, H5S_SELECT_SET, start, NULL, one, NULL) < 0)
                TEST_ERROR;

            /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
            if (H5Dwrite(did, tid, mem_sid, file_sid, H5P_DEFAULT, &update) < 0)
                TEST_ERROR;

            wbuf[2] = update;
        }

        /* Close the dataset to release its cached decoded chunks and persist changes. */
        if (H5Dclose(did) < 0)
            TEST_ERROR;

        /* Mark the handle closed so error cleanup cannot close it twice. */
        did = H5I_INVALID_HID;

        /* Close the file; an active pass reopens it to check stored data. */
        if (H5Fclose(fid) < 0)
            TEST_ERROR;

        /* The file handle is no longer owned by this function. */
        fid = H5I_INVALID_HID;

        /* Reopen the file to verify the serialized, rather than cached, state. */
        if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
            TEST_ERROR;

        /* Open the same dataset for the persisted-data check. */
        if ((did = H5Dopen2(fid, "compound_vlen", H5P_DEFAULT)) < 0)
            TEST_ERROR;

        /* Start with empty read descriptors, including safe NULL payload pointers. */
        memset(rbuf, 0, sizeof rbuf);

        /* Enable error-path reclamation of any VL buffers returned by the read. */
        reclaim_read = true;

        /* Decode the chunk and convert its VL values into newly allocated memory. */
        if (H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* Prepare or check each dataset element against its expected value. */
        for (i = 0; i < 4; i++) {

            if ((rbuf[i].before != wbuf[i].before) || (rbuf[i].after != wbuf[i].after) ||
                (rbuf[i].values.len != wbuf[i].values.len))
                TEST_ERROR;

            if (wbuf[i].values.len > 0) {

                if (!rbuf[i].values.p)
                    TEST_ERROR;

                /* Process every scalar inside this variable-length payload. */
                for (j = 0; j < wbuf[i].values.len; j++) {

                    if (((int *)rbuf[i].values.p)[j] != ((int *)wbuf[i].values.p)[j])
                        TEST_ERROR;
                }
            }
        }

        /* Release every payload H5Dread allocated in this memory dataspace. */
        if (H5Treclaim(tid, sid, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;
        /* The read buffer no longer owns allocated VL payloads. */
        reclaim_read = false;

    } /* end for */

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;
    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;

    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;

    if (H5Tclose(vl_tid) < 0)
        TEST_ERROR;
    vl_tid = H5I_INVALID_HID;

    if (H5Sclose(mem_sid) < 0)
        TEST_ERROR;
    mem_sid = H5I_INVALID_HID;

    if (H5Sclose(file_sid) < 0)
        TEST_ERROR;
    file_sid = H5I_INVALID_HID;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* Report success only after all comparisons and normal cleanup succeed. */
    PASSED();
    return SUCCEED;

error:
    /* Suppress secondary HDF5 errors while cleaning up partial setup. */
    H5E_BEGIN_TRY
    {
        if (reclaim_read && tid >= 0 && sid >= 0)
            H5Treclaim(tid, sid, H5P_DEFAULT, rbuf);
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Tclose(vl_tid);
        H5Sclose(mem_sid);
        H5Sclose(file_sid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    /* Tell the harness that this case failed. */
    return FAIL;

} /* end test_struct_chunk_vlen_compound() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_two_members
 *
 * Purpose:     Exercise two independent VL descriptors in each compound record.
 *
 * How it works: Write integer and double sequences with a fixed tag, reopen, and
 *              compare all three fields. Two later partial writes alternately clear
 *              integers and doubles in record 1. Every pass verifies all records after
 *              persistence.
 *
 * Parameters:  fcpl supplies file creation settings; fapl supplies file access
 *              settings. filtered enables the optional section filter(s) for the second
 *              variant of this test.
 *
 * Return:      SUCCEED after verification and cleanup, FAIL on any error.
 *
 * Note:        Read-side hvl_t pointers belong to HDF5 until H5Treclaim;
 *              write-side pointers refer to local test payload arrays.
 *
 *                                                -- AZO   09/15/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_two_members(hid_t fcpl, hid_t fapl, bool filtered)
{
    typedef struct two_vl_record_t {
        hvl_t ints;
        int   tag;
        hvl_t doubles;
    } two_vl_record_t;

    char              filename[FILENAME_BUF_SIZE];         /* Test file path */
    hid_t             fid        = H5I_INVALID_HID;        /* File ID */
    hid_t             sid        = H5I_INVALID_HID;        /* Full dataset dataspace ID */
    hid_t             file_sid   = H5I_INVALID_HID;        /* Dataspace selecting the updated record */
    hid_t             mem_sid    = H5I_INVALID_HID;        /* One-record update dataspace ID */
    hid_t             dcpl       = H5I_INVALID_HID;        /* Dataset creation property list ID */
    hid_t             did        = H5I_INVALID_HID;        /* Dataset ID */
    hid_t             int_vl_tid = H5I_INVALID_HID;        /* VL integer member datatype ID */
    hid_t             dbl_vl_tid = H5I_INVALID_HID;        /* VL double member datatype ID */
    hid_t             tid        = H5I_INVALID_HID;        /* Compound record datatype ID */
    H5D_chunk_index_t idx_type;                            /* Index type found on the dataset */
    hsize_t           dims[1]       = {4};                 /* Four compound records */
    hsize_t           chunk_dims[1] = {4};                 /* All records share one chunk */
    hsize_t           one[1]        = {1};                 /* Extent of the update selection */
    hsize_t           start[1]      = {1};                 /* Record selected for both updates */
    unsigned int      level         = 6;                   /* Optional section deflate level */
    int               initial_ints[4][3];                  /* Initial integer VL payloads */
    double            initial_doubles[4][2];               /* Initial double VL payloads */
    double            new_doubles[3] = {71.5, 72.5, 73.5}; /* Replacement double payload */
    int               new_ints[2]    = {81, 82};           /* Replacement integer payload */
    two_vl_record_t   wbuf[4];                             /* Initial compound records */
    two_vl_record_t   update;                              /* Replacement for record 1 */
    two_vl_record_t   rbuf[4];                             /* Records read after each reopen */
    unsigned          pass;                                /* Both-present and two replacement states */
    unsigned          i;                                   /* Record index */
    size_t            j;                                   /* Element within a VL member */
    bool              reclaim_read = false;                /* Whether rbuf owns VL memory to reclaim */

    /* Announce this case through the HDF5 test harness. */
    TESTING("structured chunk with two VL members per compound element");

    /* Build a driver-aware filename for this test file. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    /* Create a fresh file so earlier cases cannot supply data. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((sid = H5Screate_simple(1, dims, NULL)) < 0)
        TEST_ERROR;

    /* Keep an independent selection so the full dataset extent stays available. */
    if ((file_sid = H5Scopy(sid)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((mem_sid = H5Screate_simple(1, one, NULL)) < 0)
        TEST_ERROR;

    /* Build a variable-length datatype with the requested base type. */
    if ((int_vl_tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;

    /* Build a variable-length datatype with the requested base type. */
    if ((dbl_vl_tid = H5Tvlen_create(H5T_NATIVE_DOUBLE)) < 0)
        TEST_ERROR;

    /* Create a compound datatype matching the C record size. */
    if ((tid = H5Tcreate(H5T_COMPOUND, sizeof(two_vl_record_t))) < 0)
        TEST_ERROR;

    /* Add a member at its real C-structure offset (HOFFSET handles padding). */
    if (H5Tinsert(tid, "ints", HOFFSET(two_vl_record_t, ints), int_vl_tid) < 0)
        TEST_ERROR;

    /* Add a member at its real C-structure offset (HOFFSET handles padding). */
    if (H5Tinsert(tid, "tag", HOFFSET(two_vl_record_t, tag), H5T_NATIVE_INT) < 0)
        TEST_ERROR;

    /* Add a member at its real C-structure offset (HOFFSET handles padding). */
    if (H5Tinsert(tid, "doubles", HOFFSET(two_vl_record_t, doubles), dbl_vl_tid) < 0)
        TEST_ERROR;

    /* Start a dataset creation property list for layout and filters. */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, 1, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (filtered) {

        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    /* Create the dataset using the configured type, extent, and chunk layout. */
    if ((did = H5Dcreate2(fid, "two_vl_members", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Ask the dataset which chunk-index implementation it actually chose. */
    if (H5D__layout_idx_type_test(did, &idx_type) < 0)
        TEST_ERROR;

    /* Catch an unexpected index choice before exercising its records. */
    if (idx_type != H5D_CHUNK_IDX_SINGLE)
        TEST_ERROR;

    memset(wbuf, 0, sizeof wbuf);

    /* Prepare or check each dataset element against its expected value. */
    for (i = 0; i < 4; i++) {

        /* Process every scalar inside this variable-length payload. */
        for (j = 0; j < 3; j++) {
            initial_ints[i][j] = (int)(10 * i + j + 1);
        }

        /* Process every scalar inside this variable-length payload. */
        for (j = 0; j < 2; j++) {
            initial_doubles[i][j] = (double)(10 * i + j) + 0.5;
        }

        wbuf[i].ints.len    = 3;
        wbuf[i].ints.p      = initial_ints[i];
        wbuf[i].tag         = (int)(100 + i);
        wbuf[i].doubles.len = 2;
        wbuf[i].doubles.p   = initial_doubles[i];
    }

    /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
    if (H5Dwrite(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    /* Run each write/update state and verify it after closing the file. */
    for (pass = 0; pass < 3; pass++) {

        if (pass > 0) {

            memset(&update, 0, sizeof update);

            if (pass == 1) {
                /* Clear only the integer VL member of element 1. */
                update.tag         = 201;
                update.doubles.len = 3;
                update.doubles.p   = new_doubles;
            }
            else {
                /* Restore its integers and clear only its double member. */
                update.ints.len = 2;
                update.ints.p   = new_ints;
                update.tag      = 301;
            }

            /* Map the small memory buffer to the selected file position(s). */
            if (H5Sselect_hyperslab(file_sid, H5S_SELECT_SET, start, NULL, one, NULL) < 0)
                TEST_ERROR;
            /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
            if (H5Dwrite(did, tid, mem_sid, file_sid, H5P_DEFAULT, &update) < 0)
                TEST_ERROR;

            wbuf[1] = update;
        }

        /* Check the persisted chunk, rather than the chunk still in cache. */
        /* Close the dataset to release its cached decoded chunks and persist changes. */
        if (H5Dclose(did) < 0)
            TEST_ERROR;

        /* Mark the handle closed so error cleanup cannot close it twice. */
        did = H5I_INVALID_HID;

        /* Close the file; an active pass reopens it to check stored data. */
        if (H5Fclose(fid) < 0)
            TEST_ERROR;

        /* The file handle is no longer owned by this function. */
        fid = H5I_INVALID_HID;

        /* Reopen the file to verify the serialized, rather than cached, state. */
        if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
            TEST_ERROR;

        /* Open the same dataset for the persisted-data check. */
        if ((did = H5Dopen2(fid, "two_vl_members", H5P_DEFAULT)) < 0)
            TEST_ERROR;

        /* Start with empty read descriptors, including safe NULL payload pointers. */
        memset(rbuf, 0, sizeof rbuf);

        /* Enable error-path reclamation of any VL buffers returned by the read. */
        reclaim_read = true;

        /* Decode the chunk and convert its VL values into newly allocated memory. */
        if (H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* Prepare or check each dataset element against its expected value. */
        for (i = 0; i < 4; i++) {

            if (rbuf[i].tag != wbuf[i].tag || rbuf[i].ints.len != wbuf[i].ints.len ||
                rbuf[i].doubles.len != wbuf[i].doubles.len)
                TEST_ERROR;

            if (wbuf[i].ints.len > 0 && !rbuf[i].ints.p)
                TEST_ERROR;
            if (wbuf[i].doubles.len > 0 && !rbuf[i].doubles.p)
                TEST_ERROR;

            /* Process every scalar inside this variable-length payload. */
            for (j = 0; j < wbuf[i].ints.len; j++) {

                if (((int *)rbuf[i].ints.p)[j] != ((int *)wbuf[i].ints.p)[j])
                    TEST_ERROR;
            }

            /* Process every scalar inside this variable-length payload. */
            for (j = 0; j < wbuf[i].doubles.len; j++) {
                if (((double *)rbuf[i].doubles.p)[j] != ((double *)wbuf[i].doubles.p)[j])
                    TEST_ERROR;
            }
        }

        /* Release every payload H5Dread allocated in this memory dataspace. */
        if (H5Treclaim(tid, sid, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* The read buffer no longer owns allocated VL payloads. */
        reclaim_read = false;

    } /* end for */

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;

    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;

    if (H5Tclose(dbl_vl_tid) < 0)
        TEST_ERROR;
    dbl_vl_tid = H5I_INVALID_HID;

    if (H5Tclose(int_vl_tid) < 0)
        TEST_ERROR;
    int_vl_tid = H5I_INVALID_HID;

    if (H5Sclose(mem_sid) < 0)
        TEST_ERROR;
    mem_sid = H5I_INVALID_HID;

    if (H5Sclose(file_sid) < 0)
        TEST_ERROR;
    file_sid = H5I_INVALID_HID;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* Report success only after all comparisons and normal cleanup succeed. */
    PASSED();
    return SUCCEED;

error:
    /* Suppress secondary HDF5 errors while cleaning up partial setup. */
    H5E_BEGIN_TRY
    {
        if (reclaim_read && tid >= 0 && sid >= 0)
            H5Treclaim(tid, sid, H5P_DEFAULT, rbuf);
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Tclose(dbl_vl_tid);
        H5Tclose(int_vl_tid);
        H5Sclose(mem_sid);
        H5Sclose(file_sid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    /* Tell the harness that this case failed. */
    return FAIL;

} /* end test_struct_chunk_vlen_two_members() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_erase
 *
 * Purpose:     Exercise erasure and metadata-only defined-value lookup.
 *
 * How it works: Write eight values across two chunks, erase two positions, reopen, and
 *              query the defined selection before reading payloads. Read a survivor
 *              from each chunk, erase the remaining defined selection, then reopen and
 *              verify it is empty. The filtered case covers selection and VL sections.
 *
 * Parameters:  fcpl supplies file creation settings; fapl supplies file access
 *              settings. filtered enables the optional section filter(s) for the second
 *              variant of this test.
 *
 * Return:      SUCCEED after verification and cleanup, FAIL on any error.
 *
 * Note:        Read-side hvl_t pointers belong to HDF5 until H5Treclaim;
 *              write-side pointers refer to local test payload arrays.
 *
 *                                                -- AZO   09/15/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_erase(hid_t fcpl, hid_t fapl, bool filtered)
{
    char  filename[FILENAME_BUF_SIZE]; /* Test file path */
    hid_t fid = H5I_INVALID_HID;       /* File ID */
    hid_t sid = H5I_INVALID_HID;       /* Full dataset dataspace ID */

    /* Remove the selected defined values and their live VL objects. */
    hid_t erase_sid = H5I_INVALID_HID; /* Selection passed to H5Derase() */

    /* Fetch the defined-value selection without requiring a VL payload read. */
    hid_t        defined_sid   = H5I_INVALID_HID; /* Selection returned by H5Dget_defined() */
    hid_t        mem_sid       = H5I_INVALID_HID; /* Memory dataspace for a surviving value */
    hid_t        dcpl          = H5I_INVALID_HID; /* Dataset creation property list ID */
    hid_t        did           = H5I_INVALID_HID; /* Dataset ID */
    hid_t        tid           = H5I_INVALID_HID; /* VL integer datatype ID */
    hsize_t      dims[1]       = {8};             /* Eight dataset positions */
    hsize_t      maxdims[1]    = {H5S_UNLIMITED}; /* Unlimited dataset dimension */
    hsize_t      chunk_dims[1] = {4};             /* Four positions per chunk */
    hsize_t      one[1]        = {1};             /* Extent for a single-value read */
    hsize_t      start[1]      = {1};             /* First position to erase */
    hsize_t      stride[1]     = {5};             /* Distance to the second erased position */
    hsize_t      count[1]      = {2};             /* Number of positions in the erase selection */
    int          values[8];                       /* Payload integers for the initial values */
    hvl_t        wbuf[8];                         /* Initial VL values */
    hvl_t        rvalue;                          /* One surviving value read for verification */
    unsigned int level = 6;                       /* Optional section deflate level */
    unsigned     i;                               /* Dataset position or selection index */
    bool         reclaim_read = false;            /* Whether rvalue owns VL memory to reclaim */

    /* Announce this case through the HDF5 test harness. */
    TESTING("structured chunk VL erase and defined-value reload");

    /* Build a driver-aware filename for this test file. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    /* Create a fresh file so earlier cases cannot supply data. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((sid = H5Screate_simple(1, dims, maxdims)) < 0)
        TEST_ERROR;

    /* Keep an independent selection so the full dataset extent stays available. */
    if ((erase_sid = H5Scopy(sid)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((mem_sid = H5Screate_simple(1, one, NULL)) < 0)
        TEST_ERROR;

    /* Build a variable-length datatype with the requested base type. */
    if ((tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;

    /* Start a dataset creation property list for layout and filters. */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, 1, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (filtered) {
        /*
         * Filter the selection as well as the heap image. The defined-value
         * query after reopening must decode the selection correctly without
         * needing to read the VL payload section.
         */
        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    /* Create the dataset using the configured type, extent, and chunk layout. */
    if ((did = H5Dcreate2(fid, "erase_vlen", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Start with one live VL payload at every position in two chunks. */
    /* Prepare or check each dataset element against its expected value. */
    for (i = 0; i < 8; i++) {
        values[i]   = (int)(100 + i);
        wbuf[i].len = 1;
        wbuf[i].p   = &values[i];
    }

    /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
    if (H5Dwrite(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    /* Erase positions 1 and 6, leaving live VL objects in both chunks. */
    /* Map the small memory buffer to the selected file position(s). */
    if (H5Sselect_hyperslab(erase_sid, H5S_SELECT_SET, start, stride, count, NULL) < 0)
        TEST_ERROR;
    /* Remove the selected defined values and their live VL objects. */
    if (H5Derase(did, erase_sid, H5P_DEFAULT) < 0)
        TEST_ERROR;

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* The file handle is no longer owned by this function. */
    fid = H5I_INVALID_HID;

    /* Reopen the file to verify the serialized, rather than cached, state. */
    if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
        TEST_ERROR;

    /* Open the same dataset for the persisted-data check. */
    if ((did = H5Dopen2(fid, "erase_vlen", H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /*
     * Query the selection before reading values. Six positions should
     * remain defined; the two erased positions must not be counted.
     */
    /* Fetch the defined-value selection without requiring a VL payload read. */
    if ((defined_sid = H5Dget_defined(did, H5S_ALL, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Verify the number of defined positions, independently of VL length. */
    if (H5Sget_select_npoints(defined_sid) != 6)
        TEST_ERROR;

    if (H5Sclose(defined_sid) < 0)
        TEST_ERROR;

    defined_sid = H5I_INVALID_HID;

    /* Read one surviving payload from each chunk. */
    /* Prepare or check each dataset element against its expected value. */
    for (i = 0; i < 2; i++) {

        start[0] = i == 0 ? 0 : 5;
        /* Map the small memory buffer to the selected file position(s). */
        if (H5Sselect_hyperslab(erase_sid, H5S_SELECT_SET, start, NULL, one, NULL) < 0)
            TEST_ERROR;

        /* Start with empty read descriptors, including safe NULL payload pointers. */
        memset(&rvalue, 0, sizeof rvalue);

        /* Enable error-path reclamation of any VL buffers returned by the read. */
        reclaim_read = true;

        /* Decode the chunk and convert its VL values into newly allocated memory. */
        if (H5Dread(did, tid, mem_sid, erase_sid, H5P_DEFAULT, &rvalue) < 0)
            TEST_ERROR;

        if (rvalue.len != 1 || !rvalue.p || ((int *)rvalue.p)[0] != values[start[0]])
            TEST_ERROR;

        /* Release every payload H5Dread allocated in this memory dataspace. */
        if (H5Treclaim(tid, mem_sid, H5P_DEFAULT, &rvalue) < 0)
            TEST_ERROR;

        /* The read buffer no longer owns allocated VL payloads. */
        reclaim_read = false;
    }

    /*
     * Erase exactly the currently defined selection. This removes the
     * remaining VL objects and should leave no defined values in either
     * chunk.
     */
    /* Fetch the defined-value selection without requiring a VL payload read. */
    if ((defined_sid = H5Dget_defined(did, H5S_ALL, H5P_DEFAULT)) < 0)
        TEST_ERROR;
    /* Remove the selected defined values and their live VL objects. */
    if (H5Derase(did, defined_sid, H5P_DEFAULT) < 0)
        TEST_ERROR;
    if (H5Sclose(defined_sid) < 0)
        TEST_ERROR;
    defined_sid = H5I_INVALID_HID;

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* The file handle is no longer owned by this function. */
    fid = H5I_INVALID_HID;

    /* Reopen the file to verify the serialized, rather than cached, state. */
    if ((fid = H5Fopen(filename, H5F_ACC_RDONLY, fapl)) < 0)
        TEST_ERROR;

    /* Open the same dataset for the persisted-data check. */
    if ((did = H5Dopen2(fid, "erase_vlen", H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Fetch the defined-value selection without requiring a VL payload read. */
    if ((defined_sid = H5Dget_defined(did, H5S_ALL, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Verify the number of defined positions, independently of VL length. */
    if (H5Sget_select_npoints(defined_sid) != 0)
        TEST_ERROR;

    if (H5Sclose(defined_sid) < 0)
        TEST_ERROR;
    defined_sid = H5I_INVALID_HID;

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;

    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;

    if (H5Sclose(mem_sid) < 0)
        TEST_ERROR;
    mem_sid = H5I_INVALID_HID;

    if (H5Sclose(erase_sid) < 0)
        TEST_ERROR;
    erase_sid = H5I_INVALID_HID;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* Report success only after all comparisons and normal cleanup succeed. */
    PASSED();
    return SUCCEED;

error:
    /* Suppress secondary HDF5 errors while cleaning up partial setup. */
    H5E_BEGIN_TRY
    {
        if (reclaim_read && tid >= 0 && mem_sid >= 0)
            H5Treclaim(tid, mem_sid, H5P_DEFAULT, &rvalue);
        H5Sclose(defined_sid);
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Sclose(mem_sid);
        H5Sclose(erase_sid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    /* Tell the harness that this case failed. */
    return FAIL;

} /* end test_struct_chunk_vlen_erase() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_empty_section
 *
 * Purpose:     Exercise an empty VL section without losing defined records.
 *
 * How it works: Write four defined records with empty VL values, give record 2 a three-
 *              integer payload, and clear it again. After each state transition,
 *              reopen, verify that four records remain defined, and check their decoded
 *              values.
 *
 * Parameters:  fcpl supplies file creation settings; fapl supplies file access
 *              settings. filtered enables the optional section filter(s) for the second
 *              variant of this test.
 *
 * Return:      SUCCEED after verification and cleanup, FAIL on any error.
 *
 * Note:        Read-side hvl_t pointers belong to HDF5 until H5Treclaim;
 *              write-side pointers refer to local test payload arrays.
 *
 *                                                -- AZO   09/17/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_empty_section(hid_t fcpl, hid_t fapl, bool filtered)
{
    char  filename[FILENAME_BUF_SIZE]; /* Test file path */
    hid_t fid = H5I_INVALID_HID;       /* File ID */
    hid_t sid = H5I_INVALID_HID;       /* Dataset dataspace ID */
    /* Fetch the defined-value selection without requiring a VL payload read. */
    hid_t        defined_sid   = H5I_INVALID_HID; /* Selection returned by H5Dget_defined() */
    hid_t        dcpl          = H5I_INVALID_HID; /* Dataset creation property list ID */
    hid_t        did           = H5I_INVALID_HID; /* Dataset ID */
    hid_t        tid           = H5I_INVALID_HID; /* VL integer datatype ID */
    hsize_t      dims[1]       = {4};             /* Four defined dataset positions */
    hsize_t      chunk_dims[1] = {4};             /* All positions share one chunk */
    hvl_t        wbuf[4];                         /* Values written for the current state */
    hvl_t        rbuf[4];                         /* Values read after reopening */
    int          payload[3] = {41, 42, 43};       /* Payload used in the nonempty state */
    unsigned int level      = 6;                  /* Optional VL-section deflate level */
    unsigned     pass;                            /* Empty, nonempty, and empty-again states */
    unsigned     i;                               /* Dataset position being checked */
    bool         reclaim_read = false;            /* Whether rbuf owns VL memory to reclaim */

    /* Announce this case through the HDF5 test harness. */
    TESTING("structured chunk empty VL section transitions");

    /* Build a driver-aware filename for this test file. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    /* Create a fresh file so earlier cases cannot supply data. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((sid = H5Screate_simple(1, dims, NULL)) < 0)
        TEST_ERROR;

    /* Build a variable-length datatype with the requested base type. */
    if ((tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;

    /* Start a dataset creation property list for layout and filters. */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, 1, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (filtered) {

        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    /* Create the dataset using the configured type, extent, and chunk layout. */
    if ((did = H5Dcreate2(fid, "empty_section_vlen", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Run each write/update state and verify it after closing the file. */
    for (pass = 0; pass < 3; pass++) {
        /*
         * Write four defined records. Only the middle pass has a nonempty
         * VL value. The other passes require no heap-set image on disk.
         */
        memset(wbuf, 0, sizeof wbuf);

        if (pass == 1) {
            wbuf[2].len = 3;
            wbuf[2].p   = payload;
        }

        /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
        if (H5Dwrite(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0)
            TEST_ERROR;

        /* Close the dataset to release its cached decoded chunks and persist changes. */
        if (H5Dclose(did) < 0)
            TEST_ERROR;

        /* Mark the handle closed so error cleanup cannot close it twice. */
        did = H5I_INVALID_HID;

        /* Close the file; an active pass reopens it to check stored data. */
        if (H5Fclose(fid) < 0)
            TEST_ERROR;

        /* The file handle is no longer owned by this function. */
        fid = H5I_INVALID_HID;

        /* Reopen the file to verify the serialized, rather than cached, state. */
        if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
            TEST_ERROR;

        /* Open the same dataset for the persisted-data check. */
        if ((did = H5Dopen2(fid, "empty_section_vlen", H5P_DEFAULT)) < 0)
            TEST_ERROR;

        /* Empty payloads are still defined dataset values. */
        /* Fetch the defined-value selection without requiring a VL payload read. */
        if ((defined_sid = H5Dget_defined(did, H5S_ALL, H5P_DEFAULT)) < 0)
            TEST_ERROR;

        /* Verify the number of defined positions, independently of VL length. */
        if (H5Sget_select_npoints(defined_sid) != 4)
            TEST_ERROR;

        if (H5Sclose(defined_sid) < 0)
            TEST_ERROR;

        defined_sid = H5I_INVALID_HID;

        /* Start with empty read descriptors, including safe NULL payload pointers. */
        memset(rbuf, 0, sizeof rbuf);

        /* Enable error-path reclamation of any VL buffers returned by the read. */
        reclaim_read = true;

        /* Decode the chunk and convert its VL values into newly allocated memory. */
        if (H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* Prepare or check each dataset element against its expected value. */
        for (i = 0; i < 4; i++) {

            if (pass == 1 && i == 2) {

                if (rbuf[i].len != 3 || !rbuf[i].p || ((int *)rbuf[i].p)[0] != payload[0] ||
                    ((int *)rbuf[i].p)[1] != payload[1] || ((int *)rbuf[i].p)[2] != payload[2])
                    TEST_ERROR;
            }
            else if (rbuf[i].len != 0)
                TEST_ERROR;
        }

        /* Release every payload H5Dread allocated in this memory dataspace. */
        if (H5Treclaim(tid, sid, H5P_DEFAULT, rbuf) < 0)
            TEST_ERROR;

        /* The read buffer no longer owns allocated VL payloads. */
        reclaim_read = false;

    } /* end for */

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;

    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* Report success only after all comparisons and normal cleanup succeed. */
    PASSED();
    return SUCCEED;

error:
    /* Suppress secondary HDF5 errors while cleaning up partial setup. */
    H5E_BEGIN_TRY
    {
        if (reclaim_read && tid >= 0 && sid >= 0)
            H5Treclaim(tid, sid, H5P_DEFAULT, rbuf);
        H5Sclose(defined_sid);
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    /* Tell the harness that this case failed. */
    return FAIL;

} /* end test_struct_chunk_vlen_empty_section() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_type_conversion
 *
 * Purpose:     Exercise base-type conversion through the local VL heap.
 *
 * How it works: Write native short sequences to a file VL int32 little-endian datatype,
 *              replace one sequence from native ints, then reopen and read into native
 *              ints. Compare every length and integer to verify conversion and
 *              preservation of the other heap objects.
 *
 * Parameters:  fcpl supplies file creation settings; fapl supplies file access
 *              settings. filtered enables the optional section filter(s) for the second
 *              variant of this test.
 *
 * Return:      SUCCEED after verification and cleanup, FAIL on any error.
 *
 * Note:        Read-side hvl_t pointers belong to HDF5 until H5Treclaim;
 *              write-side pointers refer to local test payload arrays.
 *
 *                                                -- AZO   09/17/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_type_conversion(hid_t fcpl, hid_t fapl, bool filtered)
{
    char         filename[FILENAME_BUF_SIZE];        /* Test file path */
    hid_t        fid             = H5I_INVALID_HID;  /* File ID */
    hid_t        sid             = H5I_INVALID_HID;  /* Full dataset dataspace ID */
    hid_t        file_sid        = H5I_INVALID_HID;  /* Dataspace selecting the updated value */
    hid_t        mem_sid         = H5I_INVALID_HID;  /* One-value update dataspace ID */
    hid_t        dcpl            = H5I_INVALID_HID;  /* Dataset creation property list ID */
    hid_t        did             = H5I_INVALID_HID;  /* Dataset ID */
    hid_t        file_tid        = H5I_INVALID_HID;  /* VL datatype with file-format integer base */
    hid_t        short_tid       = H5I_INVALID_HID;  /* VL datatype with native short base */
    hid_t        int_tid         = H5I_INVALID_HID;  /* VL datatype with native int base */
    hsize_t      dims[1]         = {4};              /* Four dataset positions */
    hsize_t      chunk_dims[1]   = {4};              /* All values share one chunk */
    hsize_t      one[1]          = {1};              /* Extent of the replacement selection */
    hsize_t      start[1]        = {1};              /* Position replaced using native ints */
    short        a[2]            = {11, 12};         /* Initial payload for position 0 */
    short        b[3]            = {21, 22, 23};     /* Initial payload for position 1 */
    short        c[1]            = {31};             /* Initial payload for position 2 */
    short        d[2]            = {41, 42};         /* Initial payload for position 3 */
    int          replacement[4]  = {51, 52, 53, 54}; /* Replacement for position 1 */
    const int    expected[4][4]  = {{11, 12, 0, 0},
                                    {51, 52, 53, 54},
                                    {31, 0, 0, 0},
                                    {41, 42, 0, 0}}; /* Expected integer values after conversion */
    const size_t expected_len[4] = {2, 4, 1, 2};     /* Expected VL lengths */
    hvl_t        wbuf[4];                            /* Initial VL values backed by short arrays */
    hvl_t        update;                             /* Replacement VL value backed by ints */
    hvl_t        rbuf[4];                            /* Converted values returned by H5Dread() */
    unsigned int level = 6;                          /* Optional VL-section deflate level */
    unsigned     i;                                  /* Dataset position */
    size_t       j;                                  /* Integer within a VL value */
    bool         reclaim_read = false;               /* Whether rbuf owns VL memory to reclaim */

    /* Announce this case through the HDF5 test harness. */
    TESTING("structured chunk VL base-type conversion and heap reload");

    /* Build a driver-aware filename for this test file. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    /* Create a fresh file so earlier cases cannot supply data. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((sid = H5Screate_simple(1, dims, NULL)) < 0)
        TEST_ERROR;

    /* Keep an independent selection so the full dataset extent stays available. */
    if ((file_sid = H5Scopy(sid)) < 0)
        TEST_ERROR;

    /* Create the dataspace that defines this dataset or memory buffer. */
    if ((mem_sid = H5Screate_simple(1, one, NULL)) < 0)
        TEST_ERROR;

    /* Build a variable-length datatype with the requested base type. */
    if ((file_tid = H5Tvlen_create(H5T_STD_I32LE)) < 0)
        TEST_ERROR;

    /* Build a variable-length datatype with the requested base type. */
    if ((short_tid = H5Tvlen_create(H5T_NATIVE_SHORT)) < 0)
        TEST_ERROR;

    /* Build a variable-length datatype with the requested base type. */
    if ((int_tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;

    /* Start a dataset creation property list for layout and filters. */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, 1, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (filtered) {

        /* Attach deflate to this named section only; OPTIONAL permits unfiltered output. */
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    /* Create the dataset using the configured type, extent, and chunk layout. */
    if ((did = H5Dcreate2(fid, "vlen_type_conversion", file_tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Force short -> file int conversion for four distinct VL payloads. */
    wbuf[0].len = 2;
    wbuf[0].p   = a;
    wbuf[1].len = 3;
    wbuf[1].p   = b;
    wbuf[2].len = 1;
    wbuf[2].p   = c;
    wbuf[3].len = 2;
    wbuf[3].p   = d;

    /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
    if (H5Dwrite(did, short_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    /*
     * Replace one payload using native ints. Its old heap object must be
     * removed without disturbing the other three descriptors or payloads.
     */
    /* Map the small memory buffer to the selected file position(s). */
    if (H5Sselect_hyperslab(file_sid, H5S_SELECT_SET, start, NULL, one, NULL) < 0)
        TEST_ERROR;

    update.len = 4;
    update.p   = replacement;

    /* Write the current state; HDF5 converts VL descriptors into chunk-local payloads. */
    if (H5Dwrite(did, int_tid, mem_sid, file_sid, H5P_DEFAULT, &update) < 0)
        TEST_ERROR;

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* The file handle is no longer owned by this function. */
    fid = H5I_INVALID_HID;

    /* Force heap-set decode and file int -> native int VL conversion. */
    /* Reopen the file to verify the serialized, rather than cached, state. */
    if ((fid = H5Fopen(filename, H5F_ACC_RDONLY, fapl)) < 0)
        TEST_ERROR;

    /* Open the same dataset for the persisted-data check. */
    if ((did = H5Dopen2(fid, "vlen_type_conversion", H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Start with empty read descriptors, including safe NULL payload pointers. */
    memset(rbuf, 0, sizeof rbuf);

    /* Enable error-path reclamation of any VL buffers returned by the read. */
    reclaim_read = true;

    /* Decode the chunk and convert its VL values into newly allocated memory. */
    if (H5Dread(did, int_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;

    /* Prepare or check each dataset element against its expected value. */
    for (i = 0; i < 4; i++) {

        if (rbuf[i].len != expected_len[i] || !rbuf[i].p)
            TEST_ERROR;

        /* Process every scalar inside this variable-length payload. */
        for (j = 0; j < expected_len[i]; j++) {

            if (((int *)rbuf[i].p)[j] != expected[i][j])
                TEST_ERROR;
        }
    }

    /* Release every payload H5Dread allocated in this memory dataspace. */
    if (H5Treclaim(int_tid, sid, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;

    /* The read buffer no longer owns allocated VL payloads. */
    reclaim_read = false;

    /* Close the dataset to release its cached decoded chunks and persist changes. */
    if (H5Dclose(did) < 0)
        TEST_ERROR;

    /* Mark the handle closed so error cleanup cannot close it twice. */
    did = H5I_INVALID_HID;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;

    if (H5Tclose(int_tid) < 0)
        TEST_ERROR;
    int_tid = H5I_INVALID_HID;

    if (H5Tclose(short_tid) < 0)
        TEST_ERROR;
    short_tid = H5I_INVALID_HID;

    if (H5Tclose(file_tid) < 0)
        TEST_ERROR;
    file_tid = H5I_INVALID_HID;

    if (H5Sclose(mem_sid) < 0)
        TEST_ERROR;
    mem_sid = H5I_INVALID_HID;

    if (H5Sclose(file_sid) < 0)
        TEST_ERROR;
    file_sid = H5I_INVALID_HID;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* Report success only after all comparisons and normal cleanup succeed. */
    PASSED();
    return SUCCEED;

error:
    /* Suppress secondary HDF5 errors while cleaning up partial setup. */
    H5E_BEGIN_TRY
    {
        if (reclaim_read && int_tid >= 0 && sid >= 0)
            H5Treclaim(int_tid, sid, H5P_DEFAULT, rbuf);
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(int_tid);
        H5Tclose(short_tid);
        H5Tclose(file_tid);
        H5Sclose(mem_sid);
        H5Sclose(file_sid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    /* Tell the harness that this case failed. */
    return FAIL;

} /* end test_struct_chunk_vlen_type_conversion() */

/*-------------------------------------------------------------------------
 * Function:    test_local_heapset_stable_slots
 *
 * Purpose:     Exercise the local heap-set implementation directly.
 *
 * How it works: Use the internal file object to insert three dedicated member heaps.
 *              Recompute resident allocation from their actual capacities and compare
 *              it with cached accounting. Remove the middle member, serialize and
 *              decode the directory with its hole, read surviving references, reuse the
 *              hole, and check the third reference remains stable.
 *
 * Parameters:  fcpl supplies file creation settings; fapl supplies file access
 *              settings. No filtered argument: this checks internal heap-set behavior
 *              independently of a dataset.
 *
 * Return:      SUCCEED after verification and cleanup, FAIL on any error.
 *
 * Note:        Read-side hvl_t pointers belong to HDF5 until H5Treclaim;
 *              write-side pointers refer to local test payload arrays.
 *
 *                                                -- AZO   09/17/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_local_heapset_stable_slots(hid_t fcpl, hid_t fapl)
{
    char                  filename[FILENAME_BUF_SIZE];             /* Test file path */
    hid_t                 fid       = H5I_INVALID_HID;             /* File ID */
    H5VL_object_t        *vol_obj   = NULL;                        /* VOL wrapper for the file ID */
    H5F_t                *f         = NULL;                        /* Internal file object for H5HG calls */
    H5HG_local_heapset_t *heapset   = NULL;                        /* Original local heap set */
    H5HG_local_heapset_t *decoded   = NULL;                        /* Heap set reconstructed from image */
    uint8_t              *payload   = NULL;                        /* Bytes inserted into member heaps */
    uint8_t              *readback  = NULL;                        /* Buffer for checking stored objects */
    uint8_t              *image     = NULL;                        /* Encoded heap-set image */
    size_t                image_len = 0;                           /* Length of encoded image */
    size_t                size      = H5HG_LOCAL_NORMAL_HEAP_SIZE; /* Payload size */
    size_t                bytes     = 0;                           /* Size reported by getter or read */
    size_t                expected  = 0;                           /* Allocation total recomputed by test */
    uint16_t              slot[3]   = {0, 0, 0};                   /* Stable heap slot for each object */
    uint16_t              index[3]  = {0, 0, 0};                   /* Object index within each member heap */
    unsigned              i;                                       /* Object or member-heap index */

    /* Announce this case through the HDF5 test harness. */
    TESTING("chunk-local VL heap-set stable slots and allocation accounting");

    /* Build a driver-aware filename for this test file. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);
    /* Create a fresh file so earlier cases cannot supply data. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    if (NULL == (vol_obj = (H5VL_object_t *)H5I_object_verify(fid, H5I_FILE)))
        TEST_ERROR;

    if (NULL == (f = (H5F_t *)H5VL_object_data(vol_obj)))
        TEST_ERROR;

    if (NULL == (payload = H5MM_malloc(size)))
        TEST_ERROR;

    if (NULL == (readback = H5MM_malloc(size)))
        TEST_ERROR;

    /* A payload as large as the normal heap limit gets a dedicated member. */
    /* Visit each member slot or each surviving reference, as appropriate. */
    for (i = 0; i < 3; i++) {

        memset(payload, (int)('A' + i), size);

        /* Insert a payload and capture its stable member slot and object index. */
        if (H5HG__insert_local_heapset(f, &heapset, size, payload, &slot[i], &index[i]) < 0)
            TEST_ERROR;

        if (slot[i] != i || index[i] == 0 || heapset->nlive != (size_t)(i + 1))
            TEST_ERROR;
    }

    if (heapset->nslots != 3)
        TEST_ERROR;

    /* The cached total includes the outer manager and every live member. */
    expected = sizeof(*heapset) + heapset->nalloc * sizeof(heapset->heaps[0]);

    /* Visit each member slot or each surviving reference, as appropriate. */
    for (i = 0; i < 3; i++) {

        H5HG_heap_t *member = heapset->heaps[i];

        if (!member)
            TEST_ERROR;
        expected += sizeof(*member) + member->size + member->nalloc * sizeof(member->obj[0]);
    }

    /* Compare the cached O(1) allocation total with recomputed resident bytes. */
    if (H5HG__get_local_heapset_alloc_size(heapset, &bytes) < 0 || bytes != expected)
        TEST_ERROR;

    /* Removing the only object in slot 1 releases that member. */
    /* Remove the member object; its directory slot becomes reusable. */
    if (H5HG__remove_local_heapset(f, heapset, slot[1], index[1]) < 0)
        TEST_ERROR;

    if (heapset->nlive != 2 || heapset->nslots != 3 || heapset->heaps[1] || !heapset->heaps[2])
        TEST_ERROR;

    /* Dedicated member size may include alignment; recompute from survivors. */
    expected = sizeof(*heapset) + heapset->nalloc * sizeof(heapset->heaps[0]);

    /* Visit each member slot or each surviving reference, as appropriate. */
    for (i = 0; i < 3; i++) {

        if (heapset->heaps[i]) {

            H5HG_heap_t *member = heapset->heaps[i];
            expected += sizeof(*member) + member->size + member->nalloc * sizeof(member->obj[0]);
        }
    }
    /* Compare the cached O(1) allocation total with recomputed resident bytes. */
    if (H5HG__get_local_heapset_alloc_size(heapset, &bytes) < 0 || bytes != expected)
        TEST_ERROR;

    /* Serialize before reusing the hole so the empty directory entry is tested. */
    /* Serialize the directory and members while slot 1 is still empty. */
    if (H5HG__encode_local_heapset(f, heapset, &image, &image_len) < 0)
        TEST_ERROR;

    if (!image || !image_len || NULL == (decoded = H5HG__decode_local_heapset(f, image, image_len)))
        TEST_ERROR;

    if (decoded->nlive != 2 || decoded->nslots != 3 || decoded->heaps[1] || !decoded->heaps[2])
        TEST_ERROR;

    /* Decoding may choose a different outer capacity; check its actual bytes. */
    expected = sizeof(*decoded) + decoded->nalloc * sizeof(decoded->heaps[0]);

    /* Visit each member slot or each surviving reference, as appropriate. */
    for (i = 0; i < 3; i++)
        if (decoded->heaps[i]) {
            H5HG_heap_t *member = decoded->heaps[i];
            expected += sizeof(*member) + member->size + member->nalloc * sizeof(member->obj[0]);
        }

    /* Compare the cached O(1) allocation total with recomputed resident bytes. */
    if (H5HG__get_local_heapset_alloc_size(decoded, &bytes) < 0 || bytes != expected)
        TEST_ERROR;

    /* The stored references still select the original first and third data. */
    /* Visit each member slot or each surviving reference, as appropriate. */
    for (i = 0; i < 3; i += 2) {
        size_t j;

        bytes = size;

        /* Resolve the stored slot/index pair and read the surviving payload. */
        if (H5HG__read_local_heapset(f, decoded, slot[i], index[i], readback, &bytes) < 0 || bytes != size)
            TEST_ERROR;

        /* Process every scalar inside this variable-length payload. */
        for (j = 0; j < size; j++) {
            if (readback[j] != (uint8_t)('A' + i))
                TEST_ERROR;
        }
    }

    /* The next insertion may reuse slot 1, but cannot renumber slot 2. */
    memset(payload, 'D', size);

    /* Insert a payload and capture its stable member slot and object index. */
    if (H5HG__insert_local_heapset(f, &decoded, size, payload, &slot[1], &index[1]) < 0)
        TEST_ERROR;

    if (slot[1] != 1 || decoded->nlive != 3 || decoded->nslots != 3)
        TEST_ERROR;

    bytes = size;

    /* Resolve the stored slot/index pair and read the surviving payload. */
    if (H5HG__read_local_heapset(f, decoded, slot[2], index[2], readback, &bytes) < 0 || bytes != size ||
        readback[0] != 'C')
        TEST_ERROR;

    /* Release the heap set and its owned member allocations. */
    if (H5HG__free_local_heapset(decoded) < 0)
        TEST_ERROR;

    decoded = NULL;

    /* Release the heap set and its owned member allocations. */
    if (H5HG__free_local_heapset(heapset) < 0)
        TEST_ERROR;

    heapset  = NULL;
    image    = H5MM_xfree(image);
    payload  = H5MM_xfree(payload);
    readback = H5MM_xfree(readback);

    /* Close the file; an active pass reopens it to check stored data. */
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* The file handle is no longer owned by this function. */
    fid = H5I_INVALID_HID;

    /* Report success only after all comparisons and normal cleanup succeed. */
    PASSED();
    return SUCCEED;

error:

    /* Suppress secondary HDF5 errors while cleaning up partial setup. */
    H5E_BEGIN_TRY
    {
        H5HG__free_local_heapset(decoded);
        H5HG__free_local_heapset(heapset);
        H5Fclose(fid);
    }
    H5E_END_TRY

    H5MM_xfree(image);
    H5MM_xfree(payload);
    H5MM_xfree(readback);

    /* Tell the harness that this case failed. */
    return FAIL;

} /* end test_local_heapset_stable_slots() */

/*-------------------------------------------------------------------------
 * Function:    test_local_heapset_growth
 *
 * Purpose:     Verify small initial allocation, relocation from exhausted and
 *              headerless tails, geometric growth through the normal limit,
 *              spill into a second member, object-table growth, and accounting.
 *              Decode midway, then continue growing using the original IDs.
 *
 * Updated:     Regression coverage for incremental local member allocation.
 *
 *                                                  -- AZO  09/24/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_local_heapset_growth(hid_t fcpl, hid_t fapl)
{
    char                  filename[FILENAME_BUF_SIZE];
    hid_t                 fid     = H5I_INVALID_HID;
    H5VL_object_t        *vol_obj = NULL;
    H5F_t                *f       = NULL;
    H5HG_local_heapset_t *heapset = NULL;
    H5HG_local_heapset_t *decoded = NULL;
    uint8_t              *payload = NULL, *readback = NULL, *image = NULL;
    uint16_t             *slots = NULL, *indices = NULL;
    size_t               *lengths  = NULL;
    size_t                capacity = 2 * (H5HG_LOCAL_NORMAL_HEAP_SIZE / H5HG_MINSIZE) + 8;
    size_t                bytes, expected, image_len, i, j, n, tail, old_size, old_nalloc;
    uint16_t              slot, index;

    TESTING("chunk-local VL incremental member growth and accounting");
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;
    if (NULL == (vol_obj = (H5VL_object_t *)H5I_object_verify(fid, H5I_FILE)))
        TEST_ERROR;
    if (NULL == (f = (H5F_t *)H5VL_object_data(vol_obj)))
        TEST_ERROR;
    if (NULL == (payload = H5MM_malloc(H5HG_MINSIZE)) || NULL == (readback = H5MM_malloc(H5HG_MINSIZE)) ||
        NULL == (slots = H5MM_malloc(capacity * sizeof(*slots))) ||
        NULL == (indices = H5MM_malloc(capacity * sizeof(*indices))) ||
        NULL == (lengths = H5MM_malloc(capacity * sizeof(*lengths))))
        TEST_ERROR;

    /* Tiny payloads must not allocate the normal limit up front. */
    payload[0] = 42;

    if (H5HG__insert_local_heapset(f, &heapset, 1, payload, &slot, &index) < 0 ||
        heapset->heaps[slot]->size != H5HG_MINSIZE)
        TEST_ERROR;
    if (H5HG__free_local_heapset(heapset) < 0)
        TEST_ERROR;
    heapset = NULL;

    for (tail = 0; tail <= 8; tail += 8) {
        n = 0;
        do {
            if (n >= capacity)
                TEST_ERROR;
            lengths[n] =
                n == 0 ? H5HG_MINSIZE - H5HG_SIZEOF_HDR(f) - H5HG_SIZEOF_OBJHDR(f) - tail : H5HG_MINSIZE / 2;
            memset(payload, (int)(n % 251), lengths[n]);
            old_size = heapset ? heapset->heaps[0]->size : 0;
            if (H5HG__insert_local_heapset(f, &heapset, lengths[n], payload, &slots[n], &indices[n]) < 0)
                TEST_ERROR;
            if (n == 0 && heapset->heaps[0]->obj[0].size != tail)
                TEST_ERROR;
            if (n > 0 && slots[n] == 0 && heapset->heaps[0]->size != old_size &&
                heapset->heaps[0]->size != MIN(old_size * 2, (size_t)H5HG_LOCAL_NORMAL_HEAP_SIZE))
                TEST_ERROR;
            if (slots[n] > 1 || heapset->heaps[0]->size > H5HG_LOCAL_NORMAL_HEAP_SIZE)
                TEST_ERROR;
            n++;

            /* Continue insertion into a decoded, still-growable member. */
            if (n == 3) {
                if (H5HG__encode_local_heapset(f, heapset, &image, &image_len) < 0 ||
                    NULL == (decoded = H5HG__decode_local_heapset(f, image, image_len)))
                    TEST_ERROR;
                if (H5HG__free_local_heapset(heapset) < 0)
                    TEST_ERROR;
                heapset = decoded;
                decoded = NULL;
                image   = H5MM_xfree(image);
            }
            expected = sizeof(*heapset) + heapset->nalloc * sizeof(heapset->heaps[0]);
            for (i = 0; i < heapset->nslots; i++) {
                H5HG_heap_t *member = heapset->heaps[i];
                if (member)
                    expected += sizeof(*member) + member->size + member->nalloc * sizeof(member->obj[0]);
            }
            if (H5HG__get_local_heapset_alloc_size(heapset, &bytes) < 0 || bytes != expected)
                TEST_ERROR;
        } while (slots[n - 1] == 0);

        if (heapset->heaps[0]->size != H5HG_LOCAL_NORMAL_HEAP_SIZE)
            TEST_ERROR;
        /* Verify every old reference after repeated image relocations. */
        for (i = 0; i < n; i++) {
            bytes = H5HG_MINSIZE;
            if (H5HG__read_local_heapset(f, heapset, slots[i], indices[i], readback, &bytes) < 0 ||
                bytes != lengths[i])
                TEST_ERROR;
            for (j = 0; j < bytes; j++)
                if (readback[j] != (uint8_t)(i % 251))
                    TEST_ERROR;
        }
        /* Remove an interior record; reuse its space without growing the image. */
        old_size = heapset->heaps[0]->size;
        if (H5HG__remove_local_heapset(f, heapset, slots[1], indices[1]) < 0 ||
            H5HG__insert_local_heapset(f, &heapset, lengths[1], payload, &slot, &index) < 0 || slot != 0 ||
            heapset->heaps[0]->size != old_size)
            TEST_ERROR;
        if (H5HG__free_local_heapset(heapset) < 0)
            TEST_ERROR;
        heapset = NULL;
    }

    /* Deleted IDs advance NUSED even when image space is reused. Exercise
     * table-only growth, checking allocation deltas after every insertion.
     */
    if (H5HG__insert_local_heapset(f, &heapset, 0, NULL, &slot, &index) < 0)
        TEST_ERROR;
    old_nalloc = heapset->heaps[0]->nalloc;

    for (i = 0; i < old_nalloc + 2; i++) {
        uint16_t temp_slot, temp_index;
        if (H5HG__insert_local_heapset(f, &heapset, 0, NULL, &temp_slot, &temp_index) < 0 ||
            H5HG__remove_local_heapset(f, heapset, temp_slot, temp_index) < 0)
            TEST_ERROR;
        expected = sizeof(*heapset) + heapset->nalloc * sizeof(heapset->heaps[0]) +
                   sizeof(*heapset->heaps[0]) + heapset->heaps[0]->size +
                   heapset->heaps[0]->nalloc * sizeof(heapset->heaps[0]->obj[0]);
        if (H5HG__get_local_heapset_alloc_size(heapset, &bytes) < 0 || bytes != expected)
            TEST_ERROR;
    }

    if (heapset->heaps[0]->nalloc <= old_nalloc || heapset->heaps[0]->size != H5HG_MINSIZE)
        TEST_ERROR;
    if (H5HG__remove_local_heapset(f, heapset, slot, index) < 0 || heapset->nlive != 0 ||
        heapset->alloc_size != sizeof(*heapset) + heapset->nalloc * sizeof(heapset->heaps[0]))
        TEST_ERROR;
    if (H5HG__free_local_heapset(heapset) < 0)
        TEST_ERROR;
    heapset  = NULL;
    payload  = H5MM_xfree(payload);
    readback = H5MM_xfree(readback);
    slots    = H5MM_xfree(slots);
    indices  = H5MM_xfree(indices);
    lengths  = H5MM_xfree(lengths);
    if (H5Fclose(fid) < 0)
        TEST_ERROR;
    PASSED();
    return SUCCEED;

error:
    H5E_BEGIN_TRY
    {
        H5HG__free_local_heapset(decoded);
        H5HG__free_local_heapset(heapset);
        H5Fclose(fid);
    }
    H5E_END_TRY
    H5MM_xfree(payload);
    H5MM_xfree(readback);
    H5MM_xfree(image);
    H5MM_xfree(slots);
    H5MM_xfree(indices);
    H5MM_xfree(lengths);
    return FAIL;
} /* end test_local_heapset_growth() */

/*-------------------------------------------------------------------------
 * Function:    test_local_heapset_slot_growth
 *
 * Purpose:     Force the outer heap-slot array to grow while dedicated member
 *              heaps remain live. Check stable references and independently
 *              recompute cached allocation after growth, decode, removal,
 *              hole reuse, and removal of the last object.
 *
 *                                                     -- AZO  09/29/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_local_heapset_slot_growth(hid_t fcpl, hid_t fapl)
{
    enum { NOBJECTS = 64 };
    char                  filename[FILENAME_BUF_SIZE];
    hid_t                 fid     = H5I_INVALID_HID;
    H5VL_object_t        *vol_obj = NULL;
    H5F_t                *f       = NULL;
    H5HG_local_heapset_t *heapset = NULL, *decoded = NULL;
    uint8_t              *payload = NULL, *readback = NULL, *image = NULL;
    uint16_t              slots[NOBJECTS], indices[NOBJECTS];
    size_t                size = H5HG_LOCAL_NORMAL_HEAP_SIZE;
    size_t                i, j, expected, bytes, image_len = 0, previous_capacity = 0;
    bool                  grew = false;

    TESTING("chunk-local VL heap-slot array growth and cleanup");
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0 ||
        NULL == (vol_obj = (H5VL_object_t *)H5I_object_verify(fid, H5I_FILE)) ||
        NULL == (f = (H5F_t *)H5VL_object_data(vol_obj)) || NULL == (payload = H5MM_malloc(size)) ||
        NULL == (readback = H5MM_malloc(size)))
        TEST_ERROR;

    for (i = 0; i < NOBJECTS; i++) {
        memset(payload, (int)(i + 1), size);
        if (H5HG__insert_local_heapset(f, &heapset, size, payload, &slots[i], &indices[i]) < 0 ||
            slots[i] != i || !indices[i] || heapset->nlive != i + 1 || heapset->nslots != i + 1 ||
            heapset->nalloc < heapset->nslots)
            TEST_ERROR;

        /* New heap sets start with one slot; exhausted capacity doubles,
         * capped at the descriptor-visible slot limit.
         */
        if (previous_capacity == 0) {
            if (heapset->nalloc != 1)
                TEST_ERROR;
        }
        else if (i + 1 <= previous_capacity) {
            if (heapset->nalloc != previous_capacity)
                TEST_ERROR;
        }
        else {
            size_t expected_capacity = previous_capacity > H5HG_LOCAL_MAX_HEAP_SLOTS / 2
                                           ? H5HG_LOCAL_MAX_HEAP_SLOTS
                                           : previous_capacity * 2;

            if (heapset->nalloc != expected_capacity)
                TEST_ERROR;
            grew = true;
        }

        previous_capacity = heapset->nalloc;

        expected = sizeof(*heapset) + heapset->nalloc * sizeof(heapset->heaps[0]);

        for (j = 0; j < heapset->nslots; j++) {

            H5HG_heap_t *member = heapset->heaps[j];

            if (!member)
                TEST_ERROR;

            expected += sizeof(*member) + member->size + member->nalloc * sizeof(member->obj[0]);
        }
        if (H5HG__get_local_heapset_alloc_size(heapset, &bytes) < 0 || bytes != expected)
            TEST_ERROR;

    } /* end for */

    if (!grew)
        TEST_ERROR;

    /* Decode a populated directory after its backing array has grown. */
    if (H5HG__encode_local_heapset(f, heapset, &image, &image_len) < 0 || !image || !image_len ||
        NULL == (decoded = H5HG__decode_local_heapset(f, image, image_len)))
        TEST_ERROR;

    if (H5HG__free_local_heapset(heapset) < 0)
        TEST_ERROR;

    heapset = decoded;
    decoded = NULL;
    image   = H5MM_xfree(image);

    if (heapset->nlive != NOBJECTS || heapset->nslots != NOBJECTS)
        TEST_ERROR;

    expected = sizeof(*heapset) + heapset->nalloc * sizeof(heapset->heaps[0]);

    for (i = 0; i < heapset->nslots; i++) {
        H5HG_heap_t *member = heapset->heaps[i];
        if (!member)
            TEST_ERROR;
        expected += sizeof(*member) + member->size + member->nalloc * sizeof(member->obj[0]);
    }

    if (H5HG__get_local_heapset_alloc_size(heapset, &bytes) < 0 || bytes != expected)
        TEST_ERROR;

    for (i = 0; i < NOBJECTS; i++) {

        bytes = size;

        if (H5HG__read_local_heapset(f, heapset, slots[i], indices[i], readback, &bytes) < 0 || bytes != size)
            TEST_ERROR;

        for (j = 0; j < bytes; j++) {
            if (readback[j] != (uint8_t)(i + 1))
                TEST_ERROR;
        }
    }

    /* Reuse an interior slot without moving its neighbors. */
    if (H5HG__remove_local_heapset(f, heapset, slots[7], indices[7]) < 0)
        TEST_ERROR;

    memset(payload, 201, size);

    if (H5HG__insert_local_heapset(f, &heapset, size, payload, &slots[7], &indices[7]) < 0 || slots[7] != 7 ||
        heapset->nslots != NOBJECTS)
        TEST_ERROR;

    expected = sizeof(*heapset) + heapset->nalloc * sizeof(heapset->heaps[0]);

    for (i = 0; i < heapset->nslots; i++) {
        H5HG_heap_t *member = heapset->heaps[i];
        if (!member)
            TEST_ERROR;
        expected += sizeof(*member) + member->size + member->nalloc * sizeof(member->obj[0]);
    }

    if (H5HG__get_local_heapset_alloc_size(heapset, &bytes) < 0 || bytes != expected)
        TEST_ERROR;

    for (i = 0; i < NOBJECTS; i++) {

        uint8_t expected_byte = (uint8_t)(i == 7 ? 201 : i + 1);

        bytes = size;
        if (H5HG__read_local_heapset(f, heapset, slots[i], indices[i], readback, &bytes) < 0 || bytes != size)
            TEST_ERROR;

        for (j = 0; j < bytes; j++) {
            if (readback[j] != expected_byte)
                TEST_ERROR;
        }
    }

    for (i = 0; i < NOBJECTS; i++) {
        if (H5HG__remove_local_heapset(f, heapset, slots[i], indices[i]) < 0)
            TEST_ERROR;
    }

    expected = sizeof(*heapset) + heapset->nalloc * sizeof(heapset->heaps[0]);

    if (heapset->nlive != 0 || H5HG__get_local_heapset_alloc_size(heapset, &bytes) < 0 || bytes != expected)
        TEST_ERROR;

    if (H5HG__free_local_heapset(heapset) < 0)
        TEST_ERROR;

    heapset  = NULL;
    payload  = H5MM_xfree(payload);
    readback = H5MM_xfree(readback);

    if (H5Fclose(fid) < 0)
        TEST_ERROR;
    PASSED();
    return SUCCEED;

error:
    H5E_BEGIN_TRY
    {
        H5HG__free_local_heapset(decoded);
        H5HG__free_local_heapset(heapset);
        H5Fclose(fid);
    }
    H5E_END_TRY
    H5MM_xfree(image);
    H5MM_xfree(payload);
    H5MM_xfree(readback);
    return FAIL;
} /* end test_local_heapset_slot_growth() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_eviction
 *
 * Purpose:     Verify that SCC can flush and evict dirty structured chunks
 *              containing VL data, then reconstruct their values from the
 *              stored chunk images.
 *
 *              The test writes four two-element chunks. After the first
 *              two chunks are resident, it temporarily lowers SCC's limits
 *              so subsequent writes must reclaim space. SCC's flush and
 *              eviction counters establish that the intended path ran;
 *              closing and reopening the file alone would not prove that.
 *
 *              Values are checked both after eviction while the file is
 *              still open and after a separate close/reopen. The filtered
 *              argument repeats the test with a filter on the VL section.
 *
 * Parameters:  fcpl        File creation property list.
 *              base_fapl   Base file access property list. The test copies
 *                          it before changing SCC configuration.
 *              filtered    Whether to apply deflate to the VL section.
 *
 * Return:      SUCCEED on success; FAIL on error.
 *
 *                                                  -- AZO  09/29/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_eviction(hid_t fcpl, hid_t base_fapl, bool filtered)
{
    char                 filename[FILENAME_BUF_SIZE];
    hid_t                fapl = H5I_INVALID_HID, fid = H5I_INVALID_HID;
    hid_t                sid = H5I_INVALID_HID, mem_sid = H5I_INVALID_HID;
    hid_t                file_sel = H5I_INVALID_HID, dcpl = H5I_INVALID_HID;
    hid_t                tid = H5I_INVALID_HID, did = H5I_INVALID_HID;
    H5SC_t              *cache   = NULL;
    H5SC_dset_header_t  *hdr     = NULL;
    H5SC__cache_config_t config  = {H5SC__CURR_SCC_VERSION, 1024 * 1024, 2 * 1024 * 1024};
    const hsize_t        dims[1] = {8}, chunk_dims[1] = {2};
    hsize_t              mem_dims[1] = {2}, count[1] = {2};
    hvl_t                wbuf[2], rbuf[8];
    int                  payload[2][64];
    size_t               old_q = 0, old_a = 0, old_min = 0;
    uint64_t             flush_before, evict_before;
    unsigned int         level          = 6;
    bool                 limits_changed = false, reclaim_read = false;

    TESTING("structured chunk VL dirty eviction and reload");

    h5_fixname(FILENAME[6], base_fapl, filename, sizeof(filename));

    /* Use a private FAPL so the cache configuration does not affect other
     * tests. Each dataset has eight elements arranged in four chunks.
     */
    if ((fapl = H5Pcopy(base_fapl)) < 0 || H5Pset_scc_config(fapl, &config) < 0 ||
        (fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0 ||
        (sid = H5Screate_simple(1, dims, NULL)) < 0 || (mem_sid = H5Screate_simple(1, mem_dims, NULL)) < 0 ||
        (file_sel = H5Scopy(sid)) < 0 || (tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0 ||
        (dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, 1, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    /* Exercise both the unfiltered VL section and its filtered encode/decode
     * path. Only the VL section receives the optional filter.
     */
    if (filtered && H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, 0, 1, &level) < 0)
        TEST_ERROR;

    if ((did = H5Dcreate2(fid, "vlen_eviction", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0 ||
        H5SC__get_cache_from_file_id(fid, &cache) < 0 || !cache)
        TEST_ERROR;

    /* Give every element a distinct length and contents. This detects
     * descriptor mix-ups as well as corruption of the heap payload itself.
     * The local payload array stays valid for the duration of each write.
     */
    for (unsigned chunk = 0; chunk < 4; chunk++) {
        hsize_t start[1] = {2 * chunk};

        for (unsigned e = 0; e < 2; e++) {
            wbuf[e].len = 32 + 2 * chunk + e;
            wbuf[e].p   = payload[e];

            for (size_t j = 0; j < wbuf[e].len; j++) {
                payload[e][j] = (int)(1000 * chunk + 100 * e + j);
            }
        }

        if (H5Sselect_hyperslab(file_sel, H5S_SELECT_SET, start, NULL, count, NULL) < 0 ||
            H5Dwrite(did, tid, mem_sid, file_sel, H5P_DEFAULT, wbuf) < 0)
            TEST_ERROR;

        if (chunk == 1) {
            /* The first two writes establish dirty resident chunks. Confirm
             * that this dataset is the cache's sole dataset and that its
             * resident size agrees with SCC's quiescent-size accounting.
             */
            hdr = cache->dset_lru_head_ptr;

            if (!hdr || hdr != cache->dset_lru_tail_ptr || hdr->chunk_lru_len < 2 ||
                cache->SCC_quiescent_size == 0 || hdr->curr_dset_size != cache->SCC_quiescent_size)
                TEST_ERROR;

            /* Preserve the original limits for both normal and error exits.
             * Removing the dataset's minimum reservation and setting each
             * cache limit to its current use makes the next admission
             * reclaim an unpinned dirty chunk.
             */
            old_q   = cache->SCC_quiescent_limit;
            old_a   = cache->SCC_active_limit;
            old_min = hdr->min_dset_size;

            hdr->min_dset_size         = 0;
            cache->SCC_quiescent_limit = cache->SCC_quiescent_size;
            cache->SCC_active_limit    = cache->SCC_quiescent_size;
            limits_changed             = true;

            /* Snapshot the counters after the first two writes. Subsequent
             * increases must therefore come from the cache pressure caused
             * by the remaining writes.
             */
            flush_before = cache->stats.scc_chunk_flush_count;
            evict_before = cache->stats.scc_evictions;
        }
    } /* end for */

    /* Require an actual dirty flush and eviction. Also confirm that the
     * dataset's resident-size total still agrees with SCC accounting after
     * chunks have been removed and admitted.
     */
    if (cache->stats.scc_chunk_flush_count <= flush_before || cache->stats.scc_evictions <= evict_before ||
        hdr->curr_dset_size != cache->SCC_quiescent_size)
        TEST_ERROR;

    /* Restore the normal limits before reading the entire dataset: that
     * read may need to bring all four chunks back into the cache.
     */
    hdr->min_dset_size         = old_min;
    cache->SCC_quiescent_limit = old_q;
    cache->SCC_active_limit    = old_a;
    limits_changed             = false;

    /* Reading while the file remains open verifies that values in evicted
     * chunks can be reconstructed and used by the original SCC instance.
     */
    memset(rbuf, 0, sizeof(rbuf));

    if (H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;

    reclaim_read = true;

    for (unsigned i = 0; i < 8; i++) {
        if (rbuf[i].len != 32 + i || !rbuf[i].p)
            TEST_ERROR;

        for (size_t j = 0; j < rbuf[i].len; j++) {
            if (((int *)rbuf[i].p)[j] != (int)(1000 * (i / 2) + 100 * (i % 2) + j))
                TEST_ERROR;
        }
    }

    if (H5Treclaim(tid, sid, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;

    reclaim_read = false;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    did = H5I_INVALID_HID;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    fid   = H5I_INVALID_HID;
    cache = NULL;
    hdr   = NULL;

    /* Reopen through a new file/SCC instance. This independently checks
     * that the flushed chunk images, including their VL heap sections,
     * persisted correctly.
     */
    if ((fid = H5Fopen(filename, H5F_ACC_RDONLY, fapl)) < 0 ||
        (did = H5Dopen2(fid, "vlen_eviction", H5P_DEFAULT)) < 0)
        TEST_ERROR;

    memset(rbuf, 0, sizeof(rbuf));

    if (H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;

    reclaim_read = true;

    for (unsigned i = 0; i < 8; i++) {
        if (rbuf[i].len != 32 + i || !rbuf[i].p)
            TEST_ERROR;

        for (size_t j = 0; j < rbuf[i].len; j++) {
            if (((int *)rbuf[i].p)[j] != (int)(1000 * (i / 2) + 100 * (i % 2) + j))
                TEST_ERROR;
        }
    }

    if (H5Treclaim(tid, sid, H5P_DEFAULT, rbuf) < 0)
        TEST_ERROR;
    reclaim_read = false;

    if (H5Dclose(did) < 0)
        TEST_ERROR;
    did = H5I_INVALID_HID;
    if (H5Fclose(fid) < 0)
        TEST_ERROR;
    fid = H5I_INVALID_HID;
    if (H5Pclose(dcpl) < 0 || H5Tclose(tid) < 0 || H5Sclose(file_sel) < 0 || H5Sclose(mem_sid) < 0 ||
        H5Sclose(sid) < 0 || H5Pclose(fapl) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

error:
    /* Restore limits before closing the file: hdr and cache are only valid
     * while the original SCC instance remains open. Reclaim a completed VL
     * read before closing its datatype and dataspace.
     */
    if (limits_changed) {
        hdr->min_dset_size         = old_min;
        cache->SCC_quiescent_limit = old_q;
        cache->SCC_active_limit    = old_a;
    }
    H5E_BEGIN_TRY
    {
        if (reclaim_read)
            H5Treclaim(tid, sid, H5P_DEFAULT, rbuf);
        H5Dclose(did);
        H5Fclose(fid);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Sclose(file_sel);
        H5Sclose(mem_sid);
        H5Sclose(sid);
        H5Pclose(fapl);
    }
    H5E_END_TRY
    return FAIL;
} /* end test_struct_chunk_vlen_eviction() */

/*
 * One record exercises several independent paths through H5T recursion:
 *
 *   compound
 *     +-- VL string
 *     +-- array[2] of VL integers
 *     +-- VL sequence of VL integer sequences
 *
 * TAG is an ordinary fixed-size member. Checking it also detects corruption
 * of neighboring fixed fields when variable-length members are converted.
 */
typedef struct vl_edge_record_t {
    int   tag;
    char *text;
    hvl_t array[2];
    hvl_t nested;
} vl_edge_record_t;

/*
 * Track allocations returned through the read-side VL memory manager.
 *
 * Keeping an explicit allocation list lets failure cleanup release any
 * remaining allocations without traversing a partially converted record.
 */
typedef struct vl_edge_alloc_t {
    void                   *ptr;
    struct vl_edge_alloc_t *next;
} vl_edge_alloc_t;

typedef struct vl_edge_mem_t {
    vl_edge_alloc_t *head;
    size_t           live;
    size_t           allocations;
    size_t           frees;
    bool             bad_free;
} vl_edge_mem_t;

/* Allocate one application-visible VL payload and record its ownership. */
static void *
vl_edge_allocate(size_t size, void *info)
{
    vl_edge_mem_t   *mem = (vl_edge_mem_t *)info;
    vl_edge_alloc_t *node;
    void            *ptr;

    /*
     * Return a usable allocation even if the conversion requests zero bytes.
     * Do not require any particular number of zero-byte allocation calls.
     */
    if (NULL == (ptr = malloc(size ? size : 1)))
        return NULL;

    if (NULL == (node = malloc(sizeof(*node)))) {
        free(ptr);
        return NULL;
    }

    node->ptr  = ptr;
    node->next = mem->head;
    mem->head  = node;

    mem->live++;
    mem->allocations++;

    return ptr;
}

/* Release a tracked payload and detect an unknown or duplicate free. */
static void
vl_edge_free(void *ptr, void *info)
{
    vl_edge_mem_t    *mem = (vl_edge_mem_t *)info;
    vl_edge_alloc_t **link;
    vl_edge_alloc_t  *node;

    if (!ptr)
        return;

    for (link = &mem->head; *link; link = &(*link)->next)
        if ((*link)->ptr == ptr)
            break;

    if (!*link) {
        mem->bad_free = true;
        return;
    }

    node  = *link;
    *link = node->next;

    free(node->ptr);
    free(node);

    mem->live--;
    mem->frees++;
}

/*
 * Error cleanup only: release allocations still owned by the tracker.
 * These releases are not counted as successful H5Treclaim activity.
 */
static void
vl_edge_discard_allocations(vl_edge_mem_t *mem)
{
    while (mem->head) {
        vl_edge_alloc_t *node = mem->head;

        mem->head = node->next;
        free(node->ptr);
        free(node);
    }

    mem->live = 0;
}

/* Compare one VL integer sequence without imposing a pointer value at len=0. */
static bool
vl_edge_equal_ints(const hvl_t *actual, const hvl_t *expected)
{
    size_t i;

    if (actual->len != expected->len)
        return false;

    if (actual->len && (!actual->p || !expected->p))
        return false;

    for (i = 0; i < expected->len; i++)
        if (((const int *)actual->p)[i] != ((const int *)expected->p)[i])
            return false;

    return true;
}

/*-------------------------------------------------------------------------
 * Function:    verify_struct_chunk_vlen_edges
 *
 * Purpose:     Compare defined membership and every field of eight compound
 *              records against a caller-owned reference model.
 *
 * Procedure:   Query defined membership before reading payloads. Check both
 *              its count and each coordinate. Read using a custom VL memory
 *              manager, compare strings and recursive sequences, then reclaim
 *              with the same datatype, memory dataspace, and DXPL.
 *
 *              Undefined positions are expected to return the default zero
 *              fill representation. Defined records may also contain null or
 *              empty VL values; membership distinguishes those cases.
 *
 * Ownership:   Expected records borrow the test's write-side storage.
 *              Returned read allocations belong to the local tracker and are
 *              normally released by H5Treclaim. On failure, the tracker frees
 *              any remaining allocations without traversing partial records.
 *
 * Return:      SUCCEED/FAIL.
 *
 *                                                      -- AZO  09/24/26
 *-------------------------------------------------------------------------
 */
static herr_t
verify_struct_chunk_vlen_edges(hid_t did, hid_t tid, hid_t sid, const vl_edge_record_t expected[8],
                               const bool defined[8], unsigned phase, unsigned reopened)
{
    vl_edge_record_t actual[8];
    vl_edge_record_t zero_record;
    vl_edge_mem_t    mem;
    hid_t            dxpl        = H5I_INVALID_HID;
    hid_t            defined_sid = H5I_INVALID_HID;
    hsize_t          coord[1];
    hssize_t         expected_count = 0;
    size_t           i              = 0;
    size_t           a;
    size_t           j;
    const char      *check = "setup";

    memset(actual, 0, sizeof(actual));
    memset(&zero_record, 0, sizeof(zero_record));
    memset(&mem, 0, sizeof(mem));

    /* Step 1: Check membership independently of the payload representation. */
    check = "defined membership";

    if ((defined_sid = H5Dget_defined(did, H5S_ALL, H5P_DEFAULT)) < 0)
        goto error;

    for (i = 0; i < 8; i++)
        if (defined[i])
            expected_count++;

    if (H5Sget_select_npoints(defined_sid) != expected_count)
        goto error;

    for (i = 0; i < 8; i++) {
        htri_t selected;

        coord[0] = (hsize_t)i;

        if ((selected = H5Sselect_intersect_block(defined_sid, coord, coord)) < 0)
            goto error;

        if ((selected > 0) != defined[i])
            goto error;
    }

    /* Step 2: Read through the caller-supplied VL allocation interface. */
    check = "read";

    if ((dxpl = H5Pcreate(H5P_DATASET_XFER)) < 0)
        goto error;

    if (H5Pset_vlen_mem_manager(dxpl, vl_edge_allocate, &mem, vl_edge_free, &mem) < 0)
        goto error;

    if (H5Dread(did, tid, sid, H5S_ALL, dxpl, actual) < 0)
        goto error;

    /* Step 3: Compare every coordinate, including untouched neighbors. */
    for (i = 0; i < 8; i++) {
        const vl_edge_record_t *want = defined[i] ? &expected[i] : &zero_record;
        const hvl_t            *actual_inner;
        const hvl_t            *expected_inner;

        check = "fixed member";
        if (actual[i].tag != want->tag)
            goto error;

        /*
         * Strings preserve the distinction between NULL and a non-NULL
         * empty string. strcmp alone would not establish that distinction.
         */
        check = "string nullness";
        if ((actual[i].text == NULL) != (want->text == NULL))
            goto error;

        check = "string contents";
        if (want->text && strcmp(actual[i].text, want->text) != 0)
            goto error;

        check = "array of VL sequences";
        for (a = 0; a < 2; a++)
            if (!vl_edge_equal_ints(&actual[i].array[a], &want->array[a]))
                goto error;

        check = "nested outer sequence";
        if (actual[i].nested.len != want->nested.len)
            goto error;

        if (actual[i].nested.len && !actual[i].nested.p)
            goto error;

        actual_inner   = (const hvl_t *)actual[i].nested.p;
        expected_inner = (const hvl_t *)want->nested.p;

        check = "nested inner sequence";
        for (j = 0; j < want->nested.len; j++)
            if (!vl_edge_equal_ints(&actual_inner[j], &expected_inner[j]))
                goto error;
    }

    /*
     * Step 4: Recursive reclaim must use the same allocation policy as read.
     * No exact allocation count is required: that is an implementation detail.
     */
    check = "recursive reclaim";

    if (H5Treclaim(tid, sid, dxpl, actual) < 0)
        goto error;

    if (mem.bad_free || mem.live != 0 || mem.head != NULL || mem.allocations != mem.frees)
        goto error;

    check = "verification cleanup";

    if (H5Sclose(defined_sid) < 0)
        goto error;
    defined_sid = H5I_INVALID_HID;

    if (H5Pclose(dxpl) < 0)
        goto error;
    dxpl = H5I_INVALID_HID;

    return SUCCEED;

error:
    fprintf(stderr,
            "VL EDGE VERIFY: phase=%u reopened=%u element=%zu "
            "check=%s live=%zu alloc=%zu free=%zu bad_free=%u\n",
            phase, reopened, i, check, mem.live, mem.allocations, mem.frees, (unsigned)mem.bad_free);

    H5Eprint2(H5E_DEFAULT, stderr);

    /*
     * Do not retry H5Treclaim after a partial reclaim failure. The allocation
     * list identifies precisely which test-managed allocations remain live.
     */
    vl_edge_discard_allocations(&mem);

    H5E_BEGIN_TRY
    {
        if (defined_sid >= 0)
            H5Sclose(defined_sid);
        if (dxpl >= 0)
            H5Pclose(dxpl);
    }
    H5E_END_TRY

    return FAIL;
}

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_edges
 *
 * Purpose:     Exercise recursive VL datatype correctness and sparse chunk
 *              lifecycle transitions beyond the numeric VL stress test.
 *
 * Parameters:  fcpl, fapl - Borrowed file creation/access property lists.
 *              filtered   - Request optional deflate independently on the
 *                           selection, fixed, and VL sections.
 *
 * Data model:  Eight compound records occupy two four-element chunks.
 *              Each record contains:
 *                - one fixed integer tag;
 *                - one VL string;
 *                - an array of two VL integer sequences;
 *                - one VL sequence of VL integer sequences.
 *
 *              Two independent versions of the records contain different
 *              tags, lengths, strings, and integer contents. Expected records
 *              and defined membership are updated only after an operation
 *              succeeds. The verifier checks all coordinates after every
 *              phase, including records outside the modified selection.
 *
 * Procedure:   Phase 0: Verify an unwritten dataset and default fill.
 *              Phase 1: Write the first complete version.
 *              Phase 2: Replace file positions 1 and 6 using memory positions
 *                       2 and 5 from the second version.
 *              Phase 3: Issue read/write operations with NONE selections and
 *                       verify that neither data nor membership changes.
 *              Phase 4: Erase positions 1 and 6.
 *              Phase 5: Repeat the same erase to test already-undefined input.
 *              Phase 6: Erase all positions.
 *              Phase 7: Recreate all records using the second version.
 *              Phase 8: Replace every record with an explicitly written zero
 *                       record. All positions remain defined.
 *              Phase 9: Restore the first complete version.
 *
 *              After each phase, verify the current state, close both dataset
 *              and file, reopen them, and verify again. The reopened verifier
 *              queries defined membership before reading payloads.
 *
 * Coverage:    Null versus empty strings; empty/nonempty sequence transitions;
 *              arrays containing VL; nested VL; compound member offsets;
 *              noncontiguous memory/file hyperslab mapping; untouched values;
 *              repeated erase; complete deletion and recreation; defined null
 *              records versus undefined positions; section filtering; persisted
 *              state; custom read allocation callbacks; recursive VL reclaim.
 *
 * Limits:      This test does not validate non-null custom fill, compound subset
 *              conversion, numeric base-type conversion, malformed file images,
 *              injected allocation failures, SCC byte totals, capacity-driven
 *              eviction, or direct invocation of encode_in_place.
 *
 *              Optional filters do not prove that compression was applied.
 *              Close/reopen proves persistence across that boundary, not which
 *              encoding callback was selected internally.
 *
 * Ownership:   Write-side descriptors borrow stack arrays and are never passed
 *              to H5Treclaim. The verification helper owns all returned read
 *              payloads and checks their release through a custom allocator.
 *
 * Return:      SUCCEED after all phases and cleanup; FAIL otherwise.
 *
 *                           -- AZO    09/20/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_edges(hid_t fcpl, hid_t fapl, bool filtered)
{
    char filename[FILENAME_BUF_SIZE];

    hid_t fid        = H5I_INVALID_HID;
    hid_t did        = H5I_INVALID_HID;
    hid_t sid        = H5I_INVALID_HID;
    hid_t file_sid   = H5I_INVALID_HID;
    hid_t mem_sid    = H5I_INVALID_HID;
    hid_t dcpl       = H5I_INVALID_HID;
    hid_t string_tid = H5I_INVALID_HID;
    hid_t seq_tid    = H5I_INVALID_HID;
    hid_t array_tid  = H5I_INVALID_HID;
    hid_t nested_tid = H5I_INVALID_HID;
    hid_t record_tid = H5I_INVALID_HID;

    hsize_t dims[1]       = {8};
    hsize_t maxdims[1]    = {H5S_UNLIMITED};
    hsize_t chunk_dims[1] = {4};
    hsize_t array_dims[1] = {2};
    hsize_t start[1];
    hsize_t stride[1];
    hsize_t count[1] = {2};

    vl_edge_record_t versions[2][8];
    vl_edge_record_t expected[8];
    vl_edge_record_t zeros[8];
    vl_edge_record_t untouched[8];
    vl_edge_record_t untouched_before[8];

    /*
     * Inner descriptors and payloads remain valid throughout every write.
     * Two array members and two nested children use distinct payload storage.
     */
    hvl_t inner[2][8][2];
    int   array_values[2][8][2][3];
    int   nested_values[2][8][2][3];
    char  strings[2][8][258];

    bool         defined[8] = {false};
    unsigned int level      = 6;
    unsigned     phase      = 0;
    unsigned     checkpoint;
    size_t       v, i, a, j;
    const char  *operation = "setup";

    TESTING("structured chunk recursive VL edge cases");

    memset(versions, 0, sizeof(versions));
    memset(expected, 0, sizeof(expected));
    memset(zeros, 0, sizeof(zeros));
    memset(inner, 0, sizeof(inner));

    /*
     * Step 1: Build deterministic, distinct source versions.
     *
     * Sequence lengths include 0, 1, 2, and 3. Outer nested lengths include
     * 0, 1, and 2. Empty sequences use NULL pointers; empty strings do not.
     */
    for (v = 0; v < 2; v++) {
        for (i = 0; i < 8; i++) {
            size_t string_kind = (i + v) % 4;
            size_t string_len;

            versions[v][i].tag = (int)(1000 + 100 * v + i);

            if (string_kind != 0) {
                string_len = string_kind == 1 ? 0 : string_kind == 2 ? 7 : 257;

                for (j = 0; j < string_len; j++)
                    strings[v][i][j] = (char)('a' + (i + v + j) % 26);

                strings[v][i][string_len] = '\0';
                versions[v][i].text       = strings[v][i];
            }

            for (a = 0; a < 2; a++) {
                size_t array_len = (i + v + a) % 4;
                size_t inner_len = (i + 2 * v + a + 1) % 4;

                for (j = 0; j < 3; j++) {
                    array_values[v][i][a][j] = (int)(10000 * v + 100 * i + 10 * a + j);

                    nested_values[v][i][a][j] = -(int)(1 + 10000 * v + 100 * i + 10 * a + j);
                }

                versions[v][i].array[a].len = array_len;
                versions[v][i].array[a].p   = array_len ? array_values[v][i][a] : NULL;

                inner[v][i][a].len = inner_len;
                inner[v][i][a].p   = inner_len ? nested_values[v][i][a] : NULL;
            }

            versions[v][i].nested.len = (i + v) % 3;
            versions[v][i].nested.p   = versions[v][i].nested.len ? inner[v][i] : NULL;
        }
    }

    /* Step 2: Construct the recursive datatype using actual C member offsets. */
    if ((string_tid = H5Tcopy(H5T_C_S1)) < 0)
        TEST_ERROR;
    if (H5Tset_size(string_tid, H5T_VARIABLE) < 0)
        TEST_ERROR;

    if ((seq_tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;

    if ((array_tid = H5Tarray_create2(seq_tid, 1, array_dims)) < 0)
        TEST_ERROR;

    if ((nested_tid = H5Tvlen_create(seq_tid)) < 0)
        TEST_ERROR;

    if ((record_tid = H5Tcreate(H5T_COMPOUND, sizeof(vl_edge_record_t))) < 0)
        TEST_ERROR;

    if (H5Tinsert(record_tid, "tag", HOFFSET(vl_edge_record_t, tag), H5T_NATIVE_INT) < 0)
        TEST_ERROR;

    if (H5Tinsert(record_tid, "text", HOFFSET(vl_edge_record_t, text), string_tid) < 0)
        TEST_ERROR;

    if (H5Tinsert(record_tid, "array", HOFFSET(vl_edge_record_t, array), array_tid) < 0)
        TEST_ERROR;

    if (H5Tinsert(record_tid, "nested", HOFFSET(vl_edge_record_t, nested), nested_tid) < 0)
        TEST_ERROR;

    /* Step 3: Create two sparse chunks with independent working selections. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;

    if ((sid = H5Screate_simple(1, dims, maxdims)) < 0)
        TEST_ERROR;
    if ((file_sid = H5Scopy(sid)) < 0)
        TEST_ERROR;
    if ((mem_sid = H5Screate_simple(1, dims, NULL)) < 0)
        TEST_ERROR;

    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, 1, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;

        if (H5Pset_filter2(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;

        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    if ((did = H5Dcreate2(fid, "vlen_edges", record_tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Step 4: Execute transitions and verify both resident and reopened states. */
    for (phase = 0; phase < 10; phase++) {
        switch (phase) {
            case 0:
                operation = "initial undefined state";
                break;

            case 1:
            case 7:
            case 9:
                operation = "full recursive write";
                v         = phase == 7 ? 1 : 0;

                if (H5Dwrite(did, record_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, versions[v]) < 0)
                    TEST_ERROR;

                for (i = 0; i < 8; i++) {
                    expected[i] = versions[v][i];
                    defined[i]  = true;
                }
                break;

            case 2:
                operation = "noncontiguous partial overwrite";

                /* File positions 1 and 6 span the two chunks. */
                start[0]  = 1;
                stride[0] = 5;
                if (H5Sselect_hyperslab(file_sid, H5S_SELECT_SET, start, stride, count, NULL) < 0)
                    TEST_ERROR;

                /* Memory positions 2 and 5 must map to file positions 1 and 6. */
                start[0]  = 2;
                stride[0] = 3;
                if (H5Sselect_hyperslab(mem_sid, H5S_SELECT_SET, start, stride, count, NULL) < 0)
                    TEST_ERROR;

                if (H5Dwrite(did, record_tid, mem_sid, file_sid, H5P_DEFAULT, versions[1]) < 0)
                    TEST_ERROR;

                expected[1] = versions[1][2];
                expected[6] = versions[1][5];
                defined[1] = defined[6] = true;
                break;

            case 3:
                operation = "NONE selection read and write";

                if (H5Sselect_none(file_sid) < 0 || H5Sselect_none(mem_sid) < 0)
                    TEST_ERROR;

                if (H5Dwrite(did, record_tid, mem_sid, file_sid, H5P_DEFAULT, versions[1]) < 0)
                    TEST_ERROR;

                /*
                 * A NONE read must not touch the destination, including its
                 * existing pointer fields. These pointers borrow source data.
                 */
                memcpy(untouched, versions[0], sizeof(untouched));
                memcpy(untouched_before, untouched, sizeof(untouched));

                if (H5Dread(did, record_tid, mem_sid, file_sid, H5P_DEFAULT, untouched) < 0)
                    TEST_ERROR;

                if (memcmp(untouched, untouched_before, sizeof(untouched)) != 0)
                    TEST_ERROR;
                break;

            case 4:
            case 5:
                operation = phase == 4 ? "partial erase" : "repeated erase";

                start[0]  = 1;
                stride[0] = 5;
                if (H5Sselect_hyperslab(file_sid, H5S_SELECT_SET, start, stride, count, NULL) < 0)
                    TEST_ERROR;

                if (H5Derase(did, file_sid, H5P_DEFAULT) < 0)
                    TEST_ERROR;

                defined[1] = defined[6] = false;
                break;

            case 6:
                operation = "complete erase";

                if (H5Sselect_all(file_sid) < 0)
                    TEST_ERROR;
                if (H5Derase(did, file_sid, H5P_DEFAULT) < 0)
                    TEST_ERROR;

                memset(defined, 0, sizeof(defined));
                break;

            case 8:
                operation = "replace with defined null records";

                if (H5Dwrite(did, record_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, zeros) < 0)
                    TEST_ERROR;

                memcpy(expected, zeros, sizeof(expected));
                for (i = 0; i < 8; i++)
                    defined[i] = true;
                break;

            default:
                TEST_ERROR;
        }

        for (checkpoint = 0; checkpoint < 2; checkpoint++) {
            if (checkpoint == 1) {
                operation = "close/reopen checkpoint";

                if (H5Dclose(did) < 0)
                    TEST_ERROR;
                did = H5I_INVALID_HID;

                if (H5Fclose(fid) < 0)
                    TEST_ERROR;
                fid = H5I_INVALID_HID;

                if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
                    TEST_ERROR;

                if ((did = H5Dopen2(fid, "vlen_edges", H5P_DEFAULT)) < 0)
                    TEST_ERROR;
            }

            if (verify_struct_chunk_vlen_edges(did, record_tid, sid, expected, defined, phase, checkpoint) <
                0)
                TEST_ERROR;
        }
    }

    /* Step 5: Close locally owned IDs; never reclaim borrowed write payloads. */
    operation = "final cleanup";

#define VL_EDGE_CLOSE(call, id)                                                                              \
    do {                                                                                                     \
        if ((call)(id) < 0)                                                                                  \
            TEST_ERROR;                                                                                      \
        (id) = H5I_INVALID_HID;                                                                              \
    } while (0)

    VL_EDGE_CLOSE(H5Dclose, did);
    VL_EDGE_CLOSE(H5Fclose, fid);
    VL_EDGE_CLOSE(H5Sclose, mem_sid);
    VL_EDGE_CLOSE(H5Sclose, file_sid);
    VL_EDGE_CLOSE(H5Sclose, sid);
    VL_EDGE_CLOSE(H5Pclose, dcpl);
    VL_EDGE_CLOSE(H5Tclose, record_tid);
    VL_EDGE_CLOSE(H5Tclose, nested_tid);
    VL_EDGE_CLOSE(H5Tclose, array_tid);
    VL_EDGE_CLOSE(H5Tclose, seq_tid);
    VL_EDGE_CLOSE(H5Tclose, string_tid);

#undef VL_EDGE_CLOSE

    PASSED();
    return SUCCEED;

error:
    fprintf(stderr, "VL EDGE TEST: filtered=%u phase=%u operation=%s\n", (unsigned)filtered, phase,
            operation);

    H5Eprint2(H5E_DEFAULT, stderr);

    H5E_BEGIN_TRY
    {
        if (did >= 0)
            H5Dclose(did);
        if (fid >= 0)
            H5Fclose(fid);
        if (mem_sid >= 0)
            H5Sclose(mem_sid);
        if (file_sid >= 0)
            H5Sclose(file_sid);
        if (sid >= 0)
            H5Sclose(sid);
        if (dcpl >= 0)
            H5Pclose(dcpl);
        if (record_tid >= 0)
            H5Tclose(record_tid);
        if (nested_tid >= 0)
            H5Tclose(nested_tid);
        if (array_tid >= 0)
            H5Tclose(array_tid);
        if (seq_tid >= 0)
            H5Tclose(seq_tid);
        if (string_tid >= 0)
            H5Tclose(string_tid);
    }
    H5E_END_TRY

    return FAIL;
} /* end test_struct_chunk_vlen_edges() */

/*-------------------------------------------------------------------------
 * Function:    struct_chunk_vlen_stress_rand
 *
 * Purpose:     Advance a caller-owned xorshift64* generator and return its
 *              multiplied output. The stored state is the unmultiplied
 *              xorshift result, which seeds the next call.
 *
 * Parameters:  state - Valid pointer to a nonzero 64-bit generator state;
 *                      updated in place. The stress test substitutes a fixed
 *                      nonzero state when its input seed is zero.
 *
 * Notes:       Uses no process-global random state. The same seed and call
 *              sequence reproduce the same choices within this test.
 *              This generator is for test data, not cryptographic use.
 *
 * Return:      Next deterministic 64-bit output value.
 *
 *                                              --AZO    09/20/26
 *-------------------------------------------------------------------------
 */
static uint64_t
struct_chunk_vlen_stress_rand(uint64_t *state)
{
    /* Keep each case independent of other tests that use random numbers. */
    uint64_t x = *state;

    /* Advance the nonzero state with the xorshift64* shift sequence. */
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;

    /* Scramble the output without storing the multiplied value as state. */
    return x * UINT64_C(2685821657736338717);

} /* struct_chunk_vlen_stress_rand() */

/*-------------------------------------------------------------------------
 * Function:    struct_chunk_vlen_stress_value
 *
 * Purpose:     Regenerate one native-int payload item from compact model
 *              metadata instead of storing a second copy of every VL value.
 *
 * Parameters:  tag     - Tag recorded for the most recent successful write.
 *              element - Flattened, row-major dataset element index.
 *              item    - Item index within that element's VL sequence.
 *
 * Notes:       Both payload construction and readback verification use this
 *              function. Including the element and item helps expose misplaced
 *              descriptors, stale values, and incorrect item ordering; this
 *              pattern is not a collision-free encoding. Arithmetic is unsigned
 *              and the result is masked to 31 bits for the native-int payloads
 *              used by this test on platforms with at least 32-bit int.
 *
 * Return:      Deterministic nonnegative payload value; no state is changed.
 *
 *                                               --AZO    09/20/26
 *-------------------------------------------------------------------------
 */
static int
struct_chunk_vlen_stress_value(uint64_t tag, size_t element, size_t item)
{
    /* Regenerate the same item without advancing the operation generator. */
    return (int)((tag * UINT64_C(1315423911) + element * UINT64_C(2654435761) + item) & UINT64_C(0x7fffffff));

} /* end struct_chunk_vlen_stress_value() */

/*-------------------------------------------------------------------------
 * Function:    verify_struct_chunk_vlen_stress
 *
 * Purpose:     Compare the complete dataset against the caller's reference
 *              model, including coordinates untouched by the last operation.
 *
 * Parameters:  did, tid - Open dataset and its VL-of-native-int memory type.
 *              sid      - Full-extent, all-selected dataspace for reclaiming
 *                         the complete read buffer; never the operation space.
 *              rank     - One or two, matching the dataset geometry.
 *              dims     - Current dimensions, whose product is VL_STRESS_NELMTS.
 *              defined  - Whether each flattened dataset coordinate is defined.
 *              lengths  - Expected VL length at each defined coordinate.
 *              tags     - Payload-generation tag for each defined coordinate.
 *
 * Procedure:   Step 1: Allocate the read buffer and inspect defined membership.
 *              Check both the total count and each coordinate: an equal count
 *              alone cannot detect values assigned to the wrong locations.
 *              Step 2: Read and compare all VL values.
 *              Undefined coordinates must read as empty fill values. Defined
 *              zero-length values are distinguished by membership, even though
 *              their returned lengths also equal zero. Compare every item of
 *              every nonempty defined value against the generated model.
 *              Step 3: Reclaim read payloads and release temporary resources.
 *
 * Ownership:   Caller IDs and model arrays are borrowed. This helper owns its
 *              returned defined-selection dataspace and descriptor buffer.
 *              After a successful read, reclaim payload allocations with TID
 *              and the all-selected SID before freeing the descriptor buffer.
 *
 * Return:      SUCCEED when all comparisons and normal cleanup succeed;
 *              FAIL with diagnostics identifying the failed check.
 *
 *                                            --AZO   09/20/26
 *-------------------------------------------------------------------------
 */
static herr_t
verify_struct_chunk_vlen_stress(hid_t did, hid_t tid, hid_t sid, unsigned rank, const hsize_t *dims,
                                const bool *defined, const size_t *lengths, const uint64_t *tags)
{
    hid_t    defined_sid      = H5I_INVALID_HID;
    hvl_t   *rbuf             = NULL;
    hsize_t  coord[2]         = {0, 0};
    hsize_t  expected_defined = 0;
    hssize_t actual_defined;
    size_t   i;
    size_t   j;
    bool     reclaim_read = false;

    /* Step 1: Allocate the read buffer and inspect defined membership. */
    if (NULL == (rbuf = H5MM_calloc(VL_STRESS_NELMTS * sizeof(*rbuf)))) {
        fprintf(stderr, "VL VERIFY: unable to allocate read buffer\n");
        goto error;
    }

    if ((defined_sid = H5Dget_defined(did, H5S_ALL, H5P_DEFAULT)) < 0) {
        fprintf(stderr, "VL VERIFY: H5Dget_defined failed\n");
        H5Eprint2(H5E_DEFAULT, stderr);
        goto error;
    }

    /* Derive the expected count from the model rather than cached totals. */
    for (i = 0; i < VL_STRESS_NELMTS; i++)
        if (defined[i])
            expected_defined++;

    if ((actual_defined = H5Sget_select_npoints(defined_sid)) < 0) {
        fprintf(stderr, "VL VERIFY: H5Sget_select_npoints failed\n");
        H5Eprint2(H5E_DEFAULT, stderr);
        goto error;
    }

    if (actual_defined != (hssize_t)expected_defined) {
        fprintf(stderr, "VL VERIFY: defined-count mismatch: expected=%llu actual=%lld\n",
                (unsigned long long)expected_defined, (long long)actual_defined);
        goto error;
    }

    for (i = 0; i < VL_STRESS_NELMTS; i++) {
        htri_t selected;

        /* Map the flat model index back to a one-cell coordinate block. */
        if (rank == 1) {
            coord[0] = (hsize_t)i;
        }
        else {
            coord[0] = (hsize_t)i / dims[1];
            coord[1] = (hsize_t)i % dims[1];
        }

        if ((selected = H5Sselect_intersect_block(defined_sid, coord, coord)) < 0) {
            fprintf(stderr, "VL VERIFY: selection intersection failed at element=%zu\n", i);
            H5Eprint2(H5E_DEFAULT, stderr);
            goto error;
        }

        if ((selected > 0) != defined[i]) {
            fprintf(stderr,
                    "VL VERIFY: defined-selection mismatch at element=%zu: "
                    "expected=%u actual=%u\n",
                    i, (unsigned)defined[i], (unsigned)(selected > 0));
            goto error;
        }
    }

    /* Step 2: Read and compare all VL values. */
    if (H5Dread(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf) < 0) {
        fprintf(stderr, "VL VERIFY: H5Dread failed\n");
        H5Eprint2(H5E_DEFAULT, stderr);
        goto error;
    }
    /* A successful read transfers responsibility for reclaiming its payloads. */
    reclaim_read = true;

    for (i = 0; i < VL_STRESS_NELMTS; i++) {
        /* Membership was checked separately, so empty fill is unambiguous. */
        if (!defined[i]) {
            if (rbuf[i].len != 0) {
                fprintf(stderr, "VL VERIFY: undefined element=%zu has length=%zu\n", i, (size_t)rbuf[i].len);
                goto error;
            }

            continue;
        }

        if (rbuf[i].len != lengths[i]) {
            fprintf(stderr,
                    "VL VERIFY: length mismatch at element=%zu: "
                    "expected=%zu actual=%zu\n",
                    i, lengths[i], (size_t)rbuf[i].len);
            goto error;
        }

        if (lengths[i] > 0 && !rbuf[i].p) {
            fprintf(stderr, "VL VERIFY: element=%zu has length=%zu but NULL payload\n", i, lengths[i]);
            goto error;
        }

        /* Check item order as well as descriptor length and pointer validity. */
        for (j = 0; j < lengths[i]; j++) {
            int expected = struct_chunk_vlen_stress_value(tags[i], i, j);
            int actual   = ((int *)rbuf[i].p)[j];

            if (actual != expected) {
                fprintf(stderr,
                        "VL VERIFY: payload mismatch at element=%zu item=%zu: "
                        "expected=%d actual=%d length=%zu\n",
                        i, j, expected, actual, lengths[i]);
                goto error;
            }
        }
    }

    /* Step 3: Reclaim read payloads and release temporary resources. */
    if (H5Treclaim(tid, sid, H5P_DEFAULT, rbuf) < 0) {
        fprintf(stderr, "VL VERIFY: H5Treclaim failed\n");
        H5Eprint2(H5E_DEFAULT, stderr);
        goto error;
    }
    reclaim_read = false;

    if (H5Sclose(defined_sid) < 0) {
        fprintf(stderr, "VL VERIFY: H5Sclose failed\n");
        H5Eprint2(H5E_DEFAULT, stderr);
        goto error;
    }

    defined_sid = H5I_INVALID_HID;
    rbuf        = H5MM_xfree(rbuf);

    return SUCCEED;

error:
    /* Best-effort cleanup of owned resources; retain the primary diagnostic. */
    H5E_BEGIN_TRY
    {
        if (reclaim_read)
            H5Treclaim(tid, sid, H5P_DEFAULT, rbuf);

        if (defined_sid >= 0) {
            H5Sclose(defined_sid);
        }
    }
    H5E_END_TRY

    rbuf = H5MM_xfree(rbuf);

    return FAIL;
} /* end verify_struct_chunk_vlen_stress() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_stress
 *
 * Purpose:     Exercise structured-chunk VL writes, replacements, erasure,
 *              defined membership, filtering, and persistence against an
 *              independent per-coordinate model across repeated operations.
 *
 * Parameters:  fcpl, fapl - Borrowed file creation/access property lists.
 *              chk_type   - CHK_SINGLE, CHK_FA, CHK_EA, or the default BT2 case.
 *              filtered   - Request optional deflate on selection, fixed, and
 *                           VL sections. Optional filtering is not a guarantee
 *                           that every serialized section is compressed.
 *              iterations - Number of individual operations, not whole cycles.
 *              seed       - Reproducible generator input; zero uses a fixed
 *                           nonzero generator state.
 *
 * Procedure:   Step 1: Initialize the reference model and dataset geometry.
 *              Begin with every coordinate undefined. Use 64 elements: a 1-D
 *              extent for single/FA/EA indexes or an 8-by-8 extent for BT2.
 *              Choose chunk sizes and maximum extents for the requested index.
 *
 *              Step 2: Create the structured-chunk dataset and working spaces.
 *              Keep a full-extent SID for readback/reclaim, a separate file
 *              selection, and a four-element contiguous memory dataspace.
 *              Configure sparse structured chunks, VL native integers, optional
 *              section filters, and verify the index selected by the library.
 *
 *              Step 3: Select the next operation in the eight-operation cycle.
 *              Cycle positions are zero-based and repeat as follows:
 *                0 - Write four values across a chunk boundary (both row and
 *                    column boundaries in BT2; within one chunk for SINGLE).
 *                1 - Shift that region by one element/column and write again,
 *                    replacing overlaps while possibly defining new locations.
 *                2 - Choose a second four-element region and write it.
 *                3 - Replace the same second region with new VL values.
 *                4 - Erase that second region.
 *                5 - Erase the original, unshifted region from position 0.
 *                6 - Write one randomly chosen element.
 *                7 - Erase that same element.
 *              Rank-one four-element regions are intervals; rank-two regions
 *              are 2-by-2 rectangles. All operations use hyperslabs, including
 *              the one-element operations. No H5Sselect_elements coverage is
 *              provided. A partial final cycle is allowed.
 *
 *              Step 4: Match the memory selection to the file selection.
 *              Select the first one or four descriptors in memory and check
 *              that both selections contain the expected number of elements.
 *
 *              Step 5: Execute the operation and update the reference model.
 *              Writes mix empty, short, and longer payloads up to 257 integers
 *              with the current constants. Generate contents from a tag plus
 *              element/item indexes. Commit expected membership, lengths, and
 *              tags only after H5Dwrite succeeds. After successful H5Derase,
 *              mark selected coordinates undefined, including already-empty
 *              locations. A written empty VL value remains defined.
 *
 *              Step 6: Verify the complete dataset after the operation.
 *              Check defined membership, every VL length, and every payload
 *              item, including values outside the operation's selection.
 *
 *              Step 7: Check persistence every 31 operations.
 *              Close the dataset and file, reopen for writing, and compare the
 *              entire model again. Repeat Steps 3 through 7 until all requested
 *              operations have run. Closing crosses a serialization/index-load
 *              boundary without assuming capacity-driven SCC eviction.
 *
 *              Step 8: Reopen read-only and verify the final dataset.
 *              Always perform a final close/reopen and full-model comparison,
 *              even if the final operation was not a periodic checkpoint.
 *
 *              Step 9: Release resources and report the result.
 *              Close every locally owned ID. On failure, report the case, seed,
 *              zero-based iteration, phase, operation, and model count before
 *              best-effort cleanup with secondary HDF5 errors suppressed.
 *
 * Coverage:    Exercises VL data through the structured-chunk/SCC path but does
 *              not force SCC pressure eviction, validate cached allocation byte
 *              totals, test point selections, or inject allocation failures.
 *              Full readback must not be interpreted as proof of eviction.
 *
 * Controls:    main() reads HDF5_STRUCT_CHUNK_VL_STRESS_ITERS (default 96) and
 *              HDF5_STRUCT_CHUNK_VL_STRESS_SEED and passes them to this routine.
 *              A zero iteration setting disables invocation in the harness.
 *              Use a larger count for endurance runs; retain the seed, case,
 *              and test version when reproducing a failure.
 *
 * Ownership:   Write descriptors borrow this function's stack payload arrays;
 *              they are not reclaimed with H5Treclaim. The verification helper
 *              owns and reclaims each successful read's returned VL payloads.
 *
 * Updated:     Documents the hyperslab operation cycle, model transitions,
 *              readback ownership, and close/reopen checkpoints with matching
 *              numbered steps in the function body.
 *
 * Return:      SUCCEED/FAIL.
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_stress(hid_t fcpl, hid_t fapl, unsigned chk_type, bool filtered, size_t iterations,
                              uint64_t seed)
{
    char              filename[FILENAME_BUF_SIZE];
    hid_t             fid     = H5I_INVALID_HID;
    hid_t             sid     = H5I_INVALID_HID;
    hid_t             op_sid  = H5I_INVALID_HID;
    hid_t             mem_sid = H5I_INVALID_HID;
    hid_t             dcpl    = H5I_INVALID_HID;
    hid_t             did     = H5I_INVALID_HID;
    hid_t             tid     = H5I_INVALID_HID;
    H5D_chunk_index_t idx_type;
    H5D_chunk_index_t expected_idx;
    hsize_t           dims[2]         = {VL_STRESS_NELMTS, 1};
    hsize_t           maxdims[2]      = {VL_STRESS_NELMTS, 1};
    hsize_t           chunk_dims[2]   = {8, 1};
    hsize_t           mem_dims[1]     = {4};
    hsize_t           start[2]        = {0, 0};
    hsize_t           count[2]        = {1, 1};
    hsize_t           slab_start[2]   = {0, 0}; /* Base region retained for overlap */
    hsize_t           repeat_start[2] = {0, 0}; /* Region reused for replacement and erase */
    size_t            elements[4];              /* File indices in transfer order */
    size_t            single_element = 0;       /* Reused by single-element erase */
    bool              defined[VL_STRESS_NELMTS];
    size_t            lengths[VL_STRESS_NELMTS];
    uint64_t          tags[VL_STRESS_NELMTS];
    int               payload[4][VL_STRESS_MAX_LEN];
    hvl_t             wvalues[4];
    size_t            write_lengths[4];
    uint64_t          write_tags[4];
    unsigned int      level = 6;
    unsigned          rank;
    size_t            iter     = 0;
    size_t            ndefined = 0;
    uint64_t          rng;
    size_t            nselected = 0;
    const char       *operation = "setup";
    const char       *phase     = "create";

    TESTING("structured chunk VL model-based smoke/stress");

    /* Step 1: Initialize the reference model and dataset geometry. */
    memset(defined, 0, sizeof defined);
    memset(lengths, 0, sizeof lengths);
    memset(tags, 0, sizeof tags);

    rng = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);

    /* Unlimited dimensions distinguish EA and BT2 from the fixed FA case. */
    switch (chk_type) {
        case CHK_SINGLE:
            rank          = 1;
            chunk_dims[0] = VL_STRESS_NELMTS;
            expected_idx  = H5D_CHUNK_IDX_SINGLE;
            break;
        case CHK_FA:
            rank         = 1;
            expected_idx = H5D_CHUNK_IDX_FARRAY;
            break;
        case CHK_EA:
            rank         = 1;
            maxdims[0]   = H5S_UNLIMITED;
            expected_idx = H5D_CHUNK_IDX_EARRAY;
            break;
        default:
            rank    = 2;
            dims[0] = dims[1] = 8;
            maxdims[0] = maxdims[1] = H5S_UNLIMITED;
            chunk_dims[0]           = 2;
            chunk_dims[1]           = 4;
            expected_idx            = H5D_CHUNK_IDX_BT2;
            break;
    }

    /* Step 2: Create the structured-chunk dataset and working spaces. */
    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);

    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;
    if ((sid = H5Screate_simple((int)rank, dims, maxdims)) < 0)
        TEST_ERROR;
    if ((op_sid = H5Scopy(sid)) < 0)
        TEST_ERROR;
    if ((mem_sid = H5Screate_simple(1, mem_dims, NULL)) < 0)
        TEST_ERROR;
    if ((tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, rank, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    /* Configure each independently stored section with the same filter. */
    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
        if (H5Pset_filter2(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }

    if ((did = H5Dcreate2(fid, "vlen_stress", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    if (H5D__layout_idx_type_test(did, &idx_type) < 0 || idx_type != expected_idx)
        TEST_ERROR;

    /* Steps 3 through 7 repeat once per API operation. */
    for (iter = 0; iter < iterations; iter++) {
        uint64_t random   = struct_chunk_vlen_stress_rand(&rng);
        unsigned step     = (unsigned)(iter % 8);
        bool     do_write = (step < 4 || step == 6);
        size_t   i;
        size_t   j;

        /* Step 3: Select the next operation in the eight-operation cycle. */
        phase = "resident operation";

        /*
         * Preserve the base region for the cycle. Shifting the next write by
         * one column replaces existing values while also defining new ones.
         * For multi-chunk indexes the base region crosses a chunk boundary;
         * in rank two it crosses both a row and a column chunk boundary.
         */
        if (step == 0) {
            if (rank == 1) {
                slab_start[0] = 8 * (1 + (hsize_t)(random % 7)) - 2;
                slab_start[1] = 0;
            }
            else {
                slab_start[0] = 1 + 2 * (hsize_t)(random % 3);
                slab_start[1] = 3;
            }
        }

        if (step == 0 || step == 1 || step == 5) {
            operation = step == 0   ? "crossing hyperslab write"
                        : step == 1 ? "overlapping hyperslab replacement"
                                    : "hyperslab erase";
            nselected = 4;
            start[0]  = slab_start[0];
            start[1]  = slab_start[1];
            count[0]  = rank == 1 ? 4 : 2;
            count[1]  = rank == 1 ? 1 : 2;
            start[rank - 1] += (step == 1 ? 1 : 0);

            if (H5Sselect_hyperslab(op_sid, H5S_SELECT_SET, start, NULL, count, NULL) < 0)
                TEST_ERROR;

            /* Hyperslabs transfer in row-major coordinate order. */
            for (i = 0; i < nselected; i++)
                elements[i] = rank == 1
                                  ? (size_t)start[0] + i
                                  : ((size_t)start[0] + i / 2) * (size_t)dims[1] + (size_t)start[1] + i % 2;
        }
        else if (step >= 2 && step <= 4) {
            operation = step == 2   ? "repeated-region hyperslab write"
                        : step == 3 ? "same-region hyperslab replacement"
                                    : "same-region hyperslab erase";
            nselected = 4;

            /*
             * This test uses hyperslab selections for these transfers.
             * Retain one region across the write, replacement, and erase so
             * each operation exercises the same four descriptor locations.
             *
             * Choose an in-bounds interval in rank one or a two-by-two
             * rectangle in rank two. Crossing chunk boundaries is exercised
             * separately by the opening hyperslabs in each cycle.
             */
            if (step == 2) {
                if (rank == 1) {
                    repeat_start[0] = (hsize_t)(random % (dims[0] - 3));
                    repeat_start[1] = 0;
                }
                else {
                    repeat_start[0] = (hsize_t)(random % (dims[0] - 1));
                    repeat_start[1] = (hsize_t)(struct_chunk_vlen_stress_rand(&rng) % (dims[1] - 1));
                }
            }

            start[0] = repeat_start[0];
            start[1] = repeat_start[1];
            count[0] = rank == 1 ? 4 : 2;
            count[1] = rank == 1 ? 1 : 2;

            if (H5Sselect_hyperslab(op_sid, H5S_SELECT_SET, start, NULL, count, NULL) < 0)
                TEST_ERROR;

            /*
             * Match the contiguous memory values to the hyperslab's
             * row-major traversal order when updating the reference model.
             */
            for (i = 0; i < nselected; i++)
                elements[i] = rank == 1
                                  ? (size_t)start[0] + i
                                  : ((size_t)start[0] + i / 2) * (size_t)dims[1] + (size_t)start[1] + i % 2;
        }
        else {
            /* Retain the chosen element so cycle position 7 erases position 6. */
            operation = step == 6 ? "single-element write" : "single-element erase";
            nselected = 1;
            if (step == 6)
                single_element = (size_t)(random % VL_STRESS_NELMTS);
            elements[0] = single_element;
            start[0]    = rank == 1 ? (hsize_t)single_element : (hsize_t)single_element / dims[1];
            start[1]    = rank == 1 ? 0 : (hsize_t)single_element % dims[1];
            count[0] = count[1] = 1;
            if (H5Sselect_hyperslab(op_sid, H5S_SELECT_SET, start, NULL, count, NULL) < 0)
                TEST_ERROR;
        }

        /* Step 4: Match the memory selection to the file selection. */
        /*
         * Use a separate, contiguous memory selection. The verifier uses SID,
         * whose all-elements selection must remain intact for VL reclamation.
         */
        {
            hsize_t mem_start[1] = {0};
            hsize_t mem_count[1] = {(hsize_t)nselected};

            if (H5Sselect_hyperslab(mem_sid, H5S_SELECT_SET, mem_start, NULL, mem_count, NULL) < 0)
                TEST_ERROR;
            if (H5Sget_select_npoints(op_sid) != (hssize_t)nselected ||
                H5Sget_select_npoints(mem_sid) != (hssize_t)nselected)
                TEST_ERROR;
        }

        /* Step 5: Execute the operation and update the reference model. */
        if (do_write) {
            /*
             * Cycle through empty, short, and larger payloads independently of
             * the random coordinates. Empty VL values remain defined values;
             * only H5Derase removes a coordinate from the defined selection.
             */
            for (i = 0; i < nselected; i++) {
                size_t   len;
                uint64_t tag = struct_chunk_vlen_stress_rand(&rng) ^ (uint64_t)iter ^ seed;

                switch ((unsigned)((iter + i) % 8)) {
                    case 0:
                        len = 0;
                        break;
                    case 1:
                        len = 1;
                        break;
                    case 2:
                        len = 2 + (size_t)(tag % 7);
                        break;
                    case 3:
                        len = 31;
                        break;
                    case 4:
                        len = 64;
                        break;
                    case 5:
                        len = 127;
                        break;
                    case 6:
                        len = VL_STRESS_MAX_LEN;
                        break;
                    default:
                        len = 1 + (size_t)(tag % VL_STRESS_MAX_LEN);
                        break;
                }
                for (j = 0; j < len; j++)
                    payload[i][j] = struct_chunk_vlen_stress_value(tag, elements[i], j);

                /* Write buffers borrow stack storage; empty values need no payload. */
                wvalues[i].len   = len;
                wvalues[i].p     = len ? payload[i] : NULL;
                write_lengths[i] = len;
                write_tags[i]    = tag;
            }

            if (H5Dwrite(did, tid, mem_sid, op_sid, H5P_DEFAULT, wvalues) < 0)
                TEST_ERROR;

            /* Publish model changes only after the complete API call succeeds. */
            for (i = 0; i < nselected; i++) {
                size_t element = elements[i];

                if (!defined[element])
                    ndefined++;
                defined[element] = true;
                lengths[element] = write_lengths[i];
                tags[element]    = write_tags[i];
            }
        }
        else {
            if (H5Derase(did, op_sid, H5P_DEFAULT) < 0)
                TEST_ERROR;

            /* A mixed selection may include already-undefined coordinates. */
            for (i = 0; i < nselected; i++) {
                size_t element = elements[i];

                if (defined[element])
                    ndefined--;
                defined[element] = false;
                lengths[element] = 0;
                tags[element]    = 0;
            }
        }

        /* Step 6: Verify the complete dataset after the operation. */
        /* Check all coordinates, including those outside the current selection. */
        phase = "resident verification";
        if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
            TEST_ERROR;

        /* Step 7: Check persistence every 31 operations. */
        if ((iter + 1) % 31 == 0) {
            /*
             * The resident model was just checked above. Closing every object
             * on the file and reopening checks serialization and index reload;
             * this checkpoint is independent of SCC capacity-driven eviction.
             */
            phase = "periodic close/reopen";
            if (H5Dclose(did) < 0)
                TEST_ERROR;
            did = H5I_INVALID_HID;

            if (H5Fclose(fid) < 0)
                TEST_ERROR;
            fid = H5I_INVALID_HID;

            if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
                TEST_ERROR;

            if ((did = H5Dopen2(fid, "vlen_stress", H5P_DEFAULT)) < 0)
                TEST_ERROR;

            if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0) {
                printf("    VL stress state became incorrect after close/reopen\n");
                TEST_ERROR;
            }
        }

    } /* end for */

    /* Step 8: Reopen read-only and verify the final dataset. */
    /* The final verification always crosses a file-close boundary. */
    phase = "final read-only reopen";
    if (H5Dclose(did) < 0)
        TEST_ERROR;
    did = H5I_INVALID_HID;
    if (H5Fclose(fid) < 0)
        TEST_ERROR;
    fid = H5I_INVALID_HID;
    if ((fid = H5Fopen(filename, H5F_ACC_RDONLY, fapl)) < 0)
        TEST_ERROR;
    if ((did = H5Dopen2(fid, "vlen_stress", H5P_DEFAULT)) < 0)
        TEST_ERROR;
    if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
        TEST_ERROR;

    /* Step 9: Release resources and report the result. */
    /* Invalidate each released ID so error cleanup only has outstanding owners. */
    phase = "cleanup";
    if (H5Dclose(did) < 0)
        TEST_ERROR;
    did = H5I_INVALID_HID;
    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;
    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;
    if (H5Sclose(mem_sid) < 0)
        TEST_ERROR;
    mem_sid = H5I_INVALID_HID;
    if (H5Sclose(op_sid) < 0)
        TEST_ERROR;
    op_sid = H5I_INVALID_HID;
    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;
    if (H5Fclose(fid) < 0)
        TEST_ERROR;
    fid = H5I_INVALID_HID;

    PASSED();
    return SUCCEED;

error:
    /* Step 9 (failure path): Report context and release remaining resources. */
    printf("    VL stress failure: index=%u filtered=%u iteration=%zu seed=%llu\n", chk_type,
           (unsigned)filtered, iter, (unsigned long long)seed);
    printf("    phase=%s operation=%s selected=%zu expected_defined=%zu\n", phase, operation, nselected,
           ndefined);
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Sclose(mem_sid);
        H5Sclose(op_sid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    return FAIL;
} /* end test_struct_chunk_vlen_stress() */

/*-------------------------------------------------------------------------
 * Function:    struct_chunk_vlen_lifecycle_write
 *
 * Purpose:     Write one value through the public dataset API. Generate each
 *              item from coordinate/tag/item metadata, then immediately free
 *              the input payload. Later checks must read HDF5-owned storage;
 *              they cannot accidentally compare against a retained input.
 *
 * Ownership:   IDs are borrowed. The helper owns its temporary input buffer.
 *              The caller updates the reference model only after success.
 *-------------------------------------------------------------------------
 */
static herr_t
struct_chunk_vlen_lifecycle_write(hid_t did, hid_t tid, hid_t op_sid, hid_t mem_sid, unsigned rank,
                                  const hsize_t *dims, size_t element, size_t length, uint64_t tag)
{
    hsize_t start[2] = {0, 0};
    hsize_t count[2] = {1, 1};
    hvl_t   value    = {0, NULL};
    size_t  j;
    herr_t  status = FAIL;

    start[0] = rank == 1 ? (hsize_t)element : (hsize_t)element / dims[1];
    start[1] = rank == 1 ? 0 : (hsize_t)element % dims[1];
    if (H5Sselect_hyperslab(op_sid, H5S_SELECT_SET, start, NULL, count, NULL) < 0)
        goto done;

    /* Guard the allocation independently of the lengths used by this test. */
    if (length > SIZE_MAX / sizeof(int))
        goto done;
    if (length > 0) {
        if (NULL == (value.p = H5MM_malloc(length * sizeof(int))))
            goto done;
        for (j = 0; j < length; j++)
            ((int *)value.p)[j] = struct_chunk_vlen_stress_value(tag, element, j);
    }
    value.len = length;
    status    = H5Dwrite(did, tid, mem_sid, op_sid, H5P_DEFAULT, &value);

done:
    H5MM_xfree(value.p);
    return status;
} /* end struct_chunk_vlen_lifecycle_write() */

/*-------------------------------------------------------------------------
 * Function:    struct_chunk_vlen_lifecycle_checkpoint
 *
 * Purpose:     Check the entire model, close BOTH dataset and file, reopen,
 *              verify the index type, and check the entire model again.
 *              A flush alone would leave decoded cache state available and
 *              could hide an encode/decode or index-persistence defect.
 *
 * Ownership:   Replaces the caller's file/dataset IDs, invalidating each ID
 *              immediately after closing it. Other IDs and arrays are borrowed.
 *-------------------------------------------------------------------------
 */
static herr_t
struct_chunk_vlen_lifecycle_checkpoint(hid_t *fid, hid_t *did, const char *filename, hid_t fapl, hid_t tid,
                                       hid_t sid, unsigned rank, const hsize_t *dims,
                                       H5D_chunk_index_t expected_idx, const bool *defined,
                                       const size_t *lengths, const uint64_t *tags, bool read_only)
{
    H5D_chunk_index_t actual_idx;

    if (verify_struct_chunk_vlen_stress(*did, tid, sid, rank, dims, defined, lengths, tags) < 0)
        return FAIL;
    if (H5Dclose(*did) < 0)
        return FAIL;
    *did = H5I_INVALID_HID;
    if (H5Fclose(*fid) < 0)
        return FAIL;
    *fid = H5I_INVALID_HID;
    if ((*fid = H5Fopen(filename, read_only ? H5F_ACC_RDONLY : H5F_ACC_RDWR, fapl)) < 0)
        return FAIL;
    if ((*did = H5Dopen2(*fid, "vlen_lifecycle", H5P_DEFAULT)) < 0)
        return FAIL;
    if (H5D__layout_idx_type_test(*did, &actual_idx) < 0 || actual_idx != expected_idx)
        return FAIL;
    return verify_struct_chunk_vlen_stress(*did, tid, sid, rank, dims, defined, lengths, tags);
} /* end struct_chunk_vlen_lifecycle_checkpoint() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_lifecycle
 *
 * Purpose:     Verify that structured chunks preserve VL values and defined
 *              selections through repeated growth, replacement, erasure,
 *              empty values, and reuse of previously removed storage.
 *
 *              The test uses a dataset of VL integer sequences. CHK_TYPE
 *              selects the dataset geometry for the single-chunk, fixed
 *              array, extensible array, or v2 B-tree index. The actual index
 *              type is checked after creation. When FILTERED is true,
 *              mandatory deflate filters are applied to the selection,
 *              fixed-data, and VL sections.
 *
 *              Each of two lifecycle passes performs the following steps:
 *
 *              1. Populate every coordinate with a small VL value,
 *                 including some defined values of length zero.
 *
 *              2. Replace eight values in the first chunk with increasing
 *                 ordinary payloads, then with payloads larger than the
 *                 configured normal member-heap limit. Keeping all eight
 *                 oversized values live together exercises multiple member
 *                 heaps and the heap-set slot array.
 *
 *              3. Replace selected large values with smaller or empty
 *                 values, erase other interior positions, and write new
 *                 oversized values into the erased positions. Verify the
 *                 untouched values throughout these changes.
 *
 *              4. Replace all remaining nonempty values with empty values,
 *                 keeping one nonempty payload until the final replacement.
 *                 This checks the transition to a dataset whose values are
 *                 all still defined but have no sequence elements.
 *
 *              5. Erase the complete first chunk, then erase the complete
 *                 dataset. Reopen the empty dataset before starting the
 *                 next pass.
 *
 *              After both passes, repopulate the dataset in reverse
 *              coordinate order and perform a final read-only reopen.
 *              Reverse insertion also exercises insertion before existing
 *              records in the packed fixed-data buffer.
 *
 *              Expected membership, sequence lengths, and payload contents
 *              are tracked independently of readback. The verifier checks
 *              the complete dataset after individual changes, and file
 *              close/reopen checkpoints check persistence at each major
 *              stage. Defined empty values must remain distinguishable
 *              from erased, undefined positions.
 *
 *              Payload sizes are derived from H5HG_LOCAL_NORMAL_HEAP_SIZE.
 *              This test checks the observable integration of heap growth
 *              and removal with descriptors, selections, and dataset I/O.
 *              It does not inspect private SCC allocation counters, prove
 *              that a reallocation moved storage, or require retained heap
 *              capacity to shrink after each removal. Direct heap tests
 *              check allocation and capacity invariants separately.
 *
 *              The write helper releases its application buffers, and the
 *              verifier reclaims VL read buffers. This function closes the
 *              identifiers it creates; FCPL and FAPL are borrowed from the
 *              caller and remain open.
 *
 * Coverage:    Also tests dense emulation through the sparse layout.
 *              Completing population defines every chunk position;
 *              the verifier checks exact membership and VL contents,
 *              including after reopening and when all values are empty.
 *
 *
 * Return:      SUCCEED on success / FAIL on error
 *
 *                                                      -- AZO  09/28/26
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_lifecycle(hid_t fcpl, hid_t fapl, unsigned chk_type, bool filtered)
{
    char              filename[FILENAME_BUF_SIZE];
    hid_t             fid     = H5I_INVALID_HID;
    hid_t             did     = H5I_INVALID_HID;
    hid_t             tid     = H5I_INVALID_HID;
    hid_t             sid     = H5I_INVALID_HID;
    hid_t             op_sid  = H5I_INVALID_HID;
    hid_t             mem_sid = H5I_INVALID_HID;
    hid_t             dcpl    = H5I_INVALID_HID;
    H5D_chunk_index_t expected_idx, actual_idx;
    hsize_t           dims[2]                   = {VL_STRESS_NELMTS, 1};
    hsize_t           maxdims[2]                = {VL_STRESS_NELMTS, 1};
    hsize_t           chunk_dims[2]             = {16, 1};
    hsize_t           one[1]                    = {1};
    hsize_t           start[2]                  = {0, 0};
    hsize_t           count[2]                  = {1, 1};
    bool              defined[VL_STRESS_NELMTS] = {false};
    size_t            lengths[VL_STRESS_NELMTS] = {0};
    uint64_t          tags[VL_STRESS_NELMTS]    = {0};
    /* All these coordinates belong to the first chunk in every index case. */
    const size_t grown[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    unsigned int level    = 6;
    unsigned     rank, cycle;
    size_t       i     = 0, k, length;
    uint64_t     tag   = 0;
    const char  *phase = "setup";

    TESTING("structured chunk VL grow/erase/empty/reuse lifecycle");

    /* Geometry selects the actual index; verify it rather than assume it. */
    switch (chk_type) {
        case CHK_SINGLE:
            rank          = 1;
            chunk_dims[0] = VL_STRESS_NELMTS;
            expected_idx  = H5D_CHUNK_IDX_SINGLE;
            break;
        case CHK_FA:
            rank         = 1;
            expected_idx = H5D_CHUNK_IDX_FARRAY;
            break;
        case CHK_EA:
            rank         = 1;
            maxdims[0]   = H5S_UNLIMITED;
            expected_idx = H5D_CHUNK_IDX_EARRAY;
            break;
        default:
            rank    = 2;
            dims[0] = dims[1] = 8;
            maxdims[0] = maxdims[1] = H5S_UNLIMITED;
            chunk_dims[0]           = 2;
            chunk_dims[1]           = 8;
            expected_idx            = H5D_CHUNK_IDX_BT2;
            break;
    }

    h5_fixname(FILENAME[6], fapl, filename, sizeof(filename));
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0 ||
        (sid = H5Screate_simple((int)rank, dims, maxdims)) < 0 || (op_sid = H5Scopy(sid)) < 0 ||
        (mem_sid = H5Screate_simple(1, one, NULL)) < 0 || (tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0 ||
        (dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* Set the chunk shape and sparse storage policy. */
    if (H5Pset_struct_chunk(dcpl, rank, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    /* Mandatory filters prevent a filtered case from silently skipping them. */
    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, 0, 1, &level) < 0 ||
            H5Pset_filter2(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, 0, 1, &level) < 0 ||
            H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, 0, 1, &level) < 0)
            TEST_ERROR;
    }
    if ((did = H5Dcreate2(fid, "vlen_lifecycle", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;
    if (H5D__layout_idx_type_test(did, &actual_idx) < 0 || actual_idx != expected_idx)
        TEST_ERROR;
    if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
        TEST_ERROR;

    for (cycle = 0; cycle < 2; cycle++) {
        /* Phase 1: Repopulate a genuinely erased dataset, including empty VL. */
        phase = "small values";
        for (i = 0; i < VL_STRESS_NELMTS; i++) {
            length = i % 7 == 0 ? 0 : 1 + i % 11;
            tag++;
            if (struct_chunk_vlen_lifecycle_write(did, tid, op_sid, mem_sid, rank, dims, i, length, tag) < 0)
                TEST_ERROR;
            defined[i] = true;
            lengths[i] = length;
            tags[i]    = tag;
            if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
                TEST_ERROR;
        }

        /* Every position is now defined, including zero-length VL values.
         * This is the emulated dense state; verify it across close/reopen.
         */

        if (struct_chunk_vlen_lifecycle_checkpoint(&fid, &did, filename, fapl, tid, sid, rank, dims,
                                                   expected_idx, defined, lengths, tags, false) < 0)
            TEST_ERROR;

        /* Phases 2/3: Grow ordinary members, then force multiple oversized
         * members to coexist. Other coordinates retain their small values.
         */
        for (unsigned stage = 0; stage < 2; stage++) {
            phase = stage == 0 ? "ordinary member growth" : "oversized member/slot growth";
            for (k = 0; k < 8; k++) {
                i      = grown[k];
                length = stage == 0 ? (H5HG_LOCAL_NORMAL_HEAP_SIZE / (32 * sizeof(int))) * (k + 1) + 1
                                    : H5HG_LOCAL_NORMAL_HEAP_SIZE / sizeof(int) + 33 + k;
                tag++;
                if (struct_chunk_vlen_lifecycle_write(did, tid, op_sid, mem_sid, rank, dims, i, length, tag) <
                    0)
                    TEST_ERROR;
                lengths[i] = length;
                tags[i]    = tag;
                if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
                    TEST_ERROR;
            }
            if (struct_chunk_vlen_lifecycle_checkpoint(&fid, &did, filename, fapl, tid, sid, rank, dims,
                                                       expected_idx, defined, lengths, tags, false) < 0)
                TEST_ERROR;
        }

        /* Phase 4: Shrink interior payloads; the last two large neighbors
         * stay live to expose descriptor damage and accidental slot renumbering.
         */
        phase = "smaller replacements";
        for (k = 1; k < 6; k += 2) {
            i      = grown[k];
            length = k == 3 ? 0 : 2 + k;
            tag++;
            if (struct_chunk_vlen_lifecycle_write(did, tid, op_sid, mem_sid, rank, dims, i, length, tag) < 0)
                TEST_ERROR;
            lengths[i] = length;
            tags[i]    = tag;
            if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
                TEST_ERROR;
        }
        if (struct_chunk_vlen_lifecycle_checkpoint(&fid, &did, filename, fapl, tid, sid, rank, dims,
                                                   expected_idx, defined, lengths, tags, false) < 0)
            TEST_ERROR;

        /* Phase 5: Erase different interior records, creating undefined holes.
         * Verification includes large survivors, small neighbors, and the
         * defined-empty replacement from the preceding phase.
         */
        phase = "interior erases";
        for (k = 0; k < 6; k += 2) {
            i        = grown[k];
            start[0] = rank == 1 ? (hsize_t)i : (hsize_t)i / dims[1];
            start[1] = rank == 1 ? 0 : (hsize_t)i % dims[1];
            count[0] = count[1] = 1;
            if (H5Sselect_hyperslab(op_sid, H5S_SELECT_SET, start, NULL, count, NULL) < 0 ||
                H5Derase(did, op_sid, H5P_DEFAULT) < 0)
                TEST_ERROR;
            defined[i] = false;
            lengths[i] = 0;
            tags[i]    = 0;
            if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
                TEST_ERROR;
        }
        if (struct_chunk_vlen_lifecycle_checkpoint(&fid, &did, filename, fapl, tid, sid, rank, dims,
                                                   expected_idx, defined, lengths, tags, false) < 0)
            TEST_ERROR;

        /* Phase 6: Reuse the erased positions with NEW oversized contents.
         * Stale descriptors or surviving references damaged by reuse fail
         * the whole-dataset comparison after each write.
         */
        phase = "reinsert after interior erases";
        for (k = 0; k < 6; k += 2) {
            i      = grown[k];
            length = H5HG_LOCAL_NORMAL_HEAP_SIZE / sizeof(int) + 65 + k;
            tag++;
            if (struct_chunk_vlen_lifecycle_write(did, tid, op_sid, mem_sid, rank, dims, i, length, tag) < 0)
                TEST_ERROR;
            defined[i] = true;
            lengths[i] = length;
            tags[i]    = tag;
            if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
                TEST_ERROR;
        }
        if (struct_chunk_vlen_lifecycle_checkpoint(&fid, &did, filename, fapl, tid, sid, rank, dims,
                                                   expected_idx, defined, lengths, tags, false) < 0)
            TEST_ERROR;

        /* Phase 7: Empty values individually, retaining position 15 as the
         * last nonempty payload. This tests cleanup while other descriptors
         * remain defined, then the exact transition from one payload to none.
         * Empty values must not disappear from the defined selection.
         */
        phase = "empty all but the last payload";
        for (i = 0; i < VL_STRESS_NELMTS; i++) {
            if (i == grown[7])
                continue;
            tag++;
            if (struct_chunk_vlen_lifecycle_write(did, tid, op_sid, mem_sid, rank, dims, i, 0, tag) < 0)
                TEST_ERROR;
            lengths[i] = 0;
            tags[i]    = tag;
            if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
                TEST_ERROR;
        }
        if (struct_chunk_vlen_lifecycle_checkpoint(&fid, &did, filename, fapl, tid, sid, rank, dims,
                                                   expected_idx, defined, lengths, tags, false) < 0)
            TEST_ERROR;

        phase = "last payload becomes defined-empty";
        i     = grown[7];
        tag++;
        if (struct_chunk_vlen_lifecycle_write(did, tid, op_sid, mem_sid, rank, dims, i, 0, tag) < 0)
            TEST_ERROR;
        lengths[i] = 0;
        tags[i]    = tag;
        /* Reopening checks the zero-length VL section and its filter handling. */
        if (struct_chunk_vlen_lifecycle_checkpoint(&fid, &did, filename, fapl, tid, sid, rank, dims,
                                                   expected_idx, defined, lengths, tags, false) < 0)
            TEST_ERROR;

        /* Phase 8: Erase one complete chunk while leaving other chunks alone.
         * SINGLE has only one chunk, so this erases the complete dataset there.
         */
        phase    = "complete first-chunk erase";
        start[0] = start[1] = 0;
        count[0]            = chunk_dims[0];
        count[1]            = chunk_dims[1];
        if (H5Sselect_hyperslab(op_sid, H5S_SELECT_SET, start, NULL, count, NULL) < 0 ||
            H5Derase(did, op_sid, H5P_DEFAULT) < 0)
            TEST_ERROR;
        for (i = 0; i < (chk_type == CHK_SINGLE ? VL_STRESS_NELMTS : 16); i++)
            defined[i] = false;
        if (struct_chunk_vlen_lifecycle_checkpoint(&fid, &did, filename, fapl, tid, sid, rank, dims,
                                                   expected_idx, defined, lengths, tags, false) < 0)
            TEST_ERROR;

        /* Phase 9: Erase everything, including already undefined coordinates.
         * Reopen the empty dataset before the next cycle recreates storage.
         */
        phase = "complete dataset erase";
        if (H5Sselect_all(op_sid) < 0 || H5Derase(did, op_sid, H5P_DEFAULT) < 0)
            TEST_ERROR;
        memset(defined, 0, sizeof(defined));
        if (struct_chunk_vlen_lifecycle_checkpoint(&fid, &did, filename, fapl, tid, sid, rank, dims,
                                                   expected_idx, defined, lengths, tags, false) < 0)
            TEST_ERROR;
    } /* end for */

    /* Finish with nonempty data, not only an empty file. Reverse coordinate
     * order exercises insertion before existing packed records after deletion.
     */
    phase = "final repopulation";
    for (k = 0; k < VL_STRESS_NELMTS; k++) {
        i      = VL_STRESS_NELMTS - 1 - k;
        length = 3 + i % 17;
        tag++;
        if (struct_chunk_vlen_lifecycle_write(did, tid, op_sid, mem_sid, rank, dims, i, length, tag) < 0)
            TEST_ERROR;
        defined[i] = true;
        lengths[i] = length;
        tags[i]    = tag;
        if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
            TEST_ERROR;
    }
    phase = "final read-only reopen";
    if (struct_chunk_vlen_lifecycle_checkpoint(&fid, &did, filename, fapl, tid, sid, rank, dims, expected_idx,
                                               defined, lengths, tags, true) < 0)
        TEST_ERROR;

    phase = "cleanup";
    if (H5Dclose(did) < 0)
        TEST_ERROR;
    did = H5I_INVALID_HID;
    if (H5Fclose(fid) < 0)
        TEST_ERROR;
    fid = H5I_INVALID_HID;
    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;
    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;
    if (H5Sclose(mem_sid) < 0)
        TEST_ERROR;
    mem_sid = H5I_INVALID_HID;
    if (H5Sclose(op_sid) < 0)
        TEST_ERROR;
    op_sid = H5I_INVALID_HID;
    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;
    PASSED();
    return SUCCEED;

error:
    fprintf(stderr, "VL lifecycle failure: index=%u filtered=%u phase=%s element=%zu tag=%llu\n", chk_type,
            (unsigned)filtered, phase, i, (unsigned long long)tag);
    H5Eprint2(H5E_DEFAULT, stderr);
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Fclose(fid);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Sclose(mem_sid);
        H5Sclose(op_sid);
        H5Sclose(sid);
    }
    H5E_END_TRY
    return FAIL;
} /* end test_struct_chunk_vlen_lifecycle() */

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_vlen_dense_emulation
 *
 * Purpose:     Explicitly test dense population in the sparse structured-chunk
 *              representation. This does not select a dedicated dense layout.
 *
 * Procedure:   Write every position, including defined empty VL values.
 *              Replace the first and last record of every chunk while keeping
 *              all positions defined. Replace the entire dataset with empty
 *              values, then restore nonempty values with different payloads.
 *              After every phase, check the defined count of EACH chunk and
 *              every coordinate's membership, VL length, and payload. Repeat
 *              those checks after closing BOTH dataset and file and reopening.
 *
 * Dependencies: VL_STRESS_NELMTS (64), verify_struct_chunk_vlen_stress(),
 *               struct_chunk_vlen_stress_value(), and
 *               struct_chunk_vlen_lifecycle_write(), already in this file.
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_vlen_dense_emulation(hid_t fcpl, hid_t fapl, unsigned chk_type, bool filtered)
{
    char              filename[FILENAME_BUF_SIZE];
    hid_t             fid         = H5I_INVALID_HID;
    hid_t             did         = H5I_INVALID_HID;
    hid_t             sid         = H5I_INVALID_HID;
    hid_t             op_sid      = H5I_INVALID_HID;
    hid_t             mem_sid     = H5I_INVALID_HID;
    hid_t             dcpl        = H5I_INVALID_HID;
    hid_t             tid         = H5I_INVALID_HID;
    hid_t             defined_sid = H5I_INVALID_HID;
    H5D_chunk_index_t idx_type;
    H5D_chunk_index_t expected_idx;
    hsize_t           dims[2]       = {VL_STRESS_NELMTS, 1};
    hsize_t           maxdims[2]    = {VL_STRESS_NELMTS, 1};
    hsize_t           chunk_dims[2] = {8, 1};
    hsize_t           mem_dims[1]   = {1};
    hsize_t           start[2];
    hsize_t           count[2] = {1, 1};
    hvl_t             wbuf[VL_STRESS_NELMTS];
    int               payload[VL_STRESS_NELMTS][7];
    bool              defined[VL_STRESS_NELMTS];
    size_t            lengths[VL_STRESS_NELMTS];
    uint64_t          tags[VL_STRESS_NELMTS];
    const char       *phase_names[] = {"full population", "partial replacement", "all defined-empty",
                                       "restore nonempty"};
    unsigned          rank;
    unsigned          phase      = 0;
    unsigned          checkpoint = 0;
    unsigned int      level      = 6;
    hsize_t           x, y;
    size_t            i, j;

    TESTING("structured chunk VL dense emulation and defined membership");

    switch (chk_type) {
        case CHK_SINGLE:
            rank          = 1;
            chunk_dims[0] = VL_STRESS_NELMTS;
            expected_idx  = H5D_CHUNK_IDX_SINGLE;
            break;
        case CHK_FA:
            rank         = 1;
            expected_idx = H5D_CHUNK_IDX_FARRAY;
            break;
        case CHK_EA:
            rank         = 1;
            maxdims[0]   = H5S_UNLIMITED;
            expected_idx = H5D_CHUNK_IDX_EARRAY;
            break;
        default:
            rank    = 2;
            dims[0] = dims[1] = 8;
            maxdims[0] = maxdims[1] = H5S_UNLIMITED;
            chunk_dims[0]           = 2;
            chunk_dims[1]           = 4;
            expected_idx            = H5D_CHUNK_IDX_BT2;
            break;
    }

    h5_fixname(FILENAME[6], fapl, filename, sizeof filename);
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, fcpl, fapl)) < 0)
        TEST_ERROR;
    if ((sid = H5Screate_simple((int)rank, dims, maxdims)) < 0)
        TEST_ERROR;
    if ((op_sid = H5Scopy(sid)) < 0)
        TEST_ERROR;
    if ((mem_sid = H5Screate_simple(1, mem_dims, NULL)) < 0)
        TEST_ERROR;
    if ((tid = H5Tvlen_create(H5T_NATIVE_INT)) < 0)
        TEST_ERROR;
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;
    if (H5Pset_struct_chunk(dcpl, rank, chunk_dims, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;
    if (filtered) {
        if (H5Pset_filter2(dcpl, H5_SECTION_SELECTION, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
        if (H5Pset_filter2(dcpl, H5_SECTION_FIXED, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
        if (H5Pset_filter2(dcpl, H5_SECTION_VL, H5Z_FILTER_DEFLATE, H5Z_FLAG_OPTIONAL, 1, &level) < 0)
            TEST_ERROR;
    }
    if ((did = H5Dcreate2(fid, "vlen_dense_emulation", tid, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    for (phase = 0; phase < 4; phase++) {
        if (phase != 1) {
            /* Full-dataset writes establish density directly. A zero-length
             * sequence is still a defined value, not a missing position.
             */
            for (i = 0; i < VL_STRESS_NELMTS; i++) {
                lengths[i]  = phase == 0 ? i % 5 : (phase == 2 ? 0 : 1 + i % 7);
                tags[i]     = UINT64_C(1000) + phase * 100 + i;
                wbuf[i].len = lengths[i];
                wbuf[i].p   = lengths[i] ? payload[i] : NULL;
                for (j = 0; j < lengths[i]; j++)
                    payload[i][j] = struct_chunk_vlen_stress_value(tags[i], i, j);
            }
            if (H5Dwrite(did, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0)
                TEST_ERROR;
            for (i = 0; i < VL_STRESS_NELMTS; i++)
                defined[i] = true;
        }
        else {
            /* Update both ends of every chunk. The model retains all other
             * values, so the verifier also checks untouched neighbors.
             */
            for (x = 0; x < dims[0]; x += chunk_dims[0])
                for (y = 0; y < dims[1]; y += chunk_dims[1]) {
                    size_t   first = rank == 1 ? (size_t)x : (size_t)(x * dims[1] + y);
                    size_t   last  = rank == 1
                                         ? (size_t)(x + chunk_dims[0] - 1)
                                         : (size_t)((x + chunk_dims[0] - 1) * dims[1] + y + chunk_dims[1] - 1);
                    uint64_t tag   = UINT64_C(9000) + first;

                    if (struct_chunk_vlen_lifecycle_write(did, tid, op_sid, mem_sid, rank, dims, first, 7,
                                                          tag) < 0)
                        TEST_ERROR;
                    lengths[first] = 7;
                    tags[first]    = tag;
                    if (struct_chunk_vlen_lifecycle_write(did, tid, op_sid, mem_sid, rank, dims, last, 0, 0) <
                        0)
                        TEST_ERROR;
                    lengths[last] = 0;
                    tags[last]    = 0;
                }
        }

        for (checkpoint = 0; checkpoint < 2; checkpoint++) {
            if (H5D__layout_idx_type_test(did, &idx_type) < 0 || idx_type != expected_idx)
                TEST_ERROR;

            /* Explicit per-chunk density checks, before any payload read at
             * this checkpoint. All shapes contain only complete chunks.
             */
            for (x = 0; x < dims[0]; x += chunk_dims[0])
                for (y = 0; y < dims[1]; y += chunk_dims[1]) {
                    start[0] = x;
                    start[1] = y;
                    if (H5Sselect_hyperslab(op_sid, H5S_SELECT_SET, start, NULL, count, chunk_dims) < 0)
                        TEST_ERROR;
                    if ((defined_sid = H5Dget_defined(did, op_sid, H5P_DEFAULT)) < 0)
                        TEST_ERROR;
                    if (H5Sget_select_npoints(defined_sid) != (hssize_t)(chunk_dims[0] * chunk_dims[1]))
                        TEST_ERROR;
                    if (H5Sclose(defined_sid) < 0)
                        TEST_ERROR;
                    defined_sid = H5I_INVALID_HID;
                }

            /* This helper independently asserts all 64 memberships, all
             * lengths and payload items, and reclaims the read allocations.
             */
            if (verify_struct_chunk_vlen_stress(did, tid, sid, rank, dims, defined, lengths, tags) < 0)
                TEST_ERROR;

            if (checkpoint == 0) {
                if (H5Dclose(did) < 0)
                    TEST_ERROR;
                did = H5I_INVALID_HID;
                if (H5Fclose(fid) < 0)
                    TEST_ERROR;
                fid = H5I_INVALID_HID;
                if ((fid = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
                    TEST_ERROR;
                if ((did = H5Dopen2(fid, "vlen_dense_emulation", H5P_DEFAULT)) < 0)
                    TEST_ERROR;
            }
        }
    }

    if (H5Dclose(did) < 0)
        TEST_ERROR;
    did = H5I_INVALID_HID;
    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    dcpl = H5I_INVALID_HID;
    if (H5Tclose(tid) < 0)
        TEST_ERROR;
    tid = H5I_INVALID_HID;
    if (H5Sclose(mem_sid) < 0)
        TEST_ERROR;
    mem_sid = H5I_INVALID_HID;
    if (H5Sclose(op_sid) < 0)
        TEST_ERROR;
    op_sid = H5I_INVALID_HID;
    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    sid = H5I_INVALID_HID;
    if (H5Fclose(fid) < 0)
        TEST_ERROR;
    fid = H5I_INVALID_HID;

    PASSED();
    return SUCCEED;

error:
    fprintf(stderr, "Dense emulation: index case=%u filtered=%u phase=%s checkpoint=%u\n", chk_type,
            (unsigned)filtered, phase_names[phase < 4 ? phase : 3], checkpoint);
    H5E_BEGIN_TRY
    {
        H5Sclose(defined_sid);
        H5Dclose(did);
        H5Pclose(dcpl);
        H5Tclose(tid);
        H5Sclose(mem_sid);
        H5Sclose(op_sid);
        H5Sclose(sid);
        H5Fclose(fid);
    }
    H5E_END_TRY
    return FAIL;
} /* end test_struct_chunk_vlen_dense_emulation() */

#ifdef TBD

/*-------------------------------------------------------------------------
 * Function:    test_struct_chunk_api_defined_erase
 *
 * Purpose:     Verify APIs for handling sparse data:
 *              --H5Dget_defined()
 *              --H5Derase()
 *
 * Return:      Success:        0
 *              Failure:        -1
 *-------------------------------------------------------------------------
 */
static herr_t
test_struct_chunk_api_defined_erase(hid_t fapl)
{
    char    filename[FILENAME_BUF_SIZE]; /* File name */
    hid_t   fid          = H5I_INVALID_HID;
    hid_t   sid          = H5I_INVALID_HID;
    hid_t   sid1         = H5I_INVALID_HID;
    hid_t   sid2         = H5I_INVALID_HID;
    hid_t   dcpl         = H5I_INVALID_HID;
    hid_t   did          = H5I_INVALID_HID;
    hsize_t dim[1]       = {50}; /* 1-d dataspace */
    hsize_t chunk_dim[1] = {5};  /* Chunk size */
    int     wbuf[50];            /* Write buffer */
    herr_t  ret;

    TESTING("APIs for handling sparse data");

    /* Create a file */
    h5_fixname(FILENAME_TBD[0], fapl, filename, sizeof filename);
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        TEST_ERROR;

    /* Create dataspace */
    if ((sid = H5Screate_simple(1, dim, NULL)) < 0)
        TEST_ERROR;

    /* Create property list for compact dataset creation */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* TBD: need to set to H5D_SPARSE_CHUNK */
    if (H5Pset_layout(dcpl, H5D_STRUCT_CHUNK) < 0)
        TEST_ERROR;

    if (H5Pset_struct_chunk(dcpl, 1, chunk_dim, H5D_SPARSE_CHUNK) < 0)
        TEST_ERROR;

    if ((did = H5Dcreate2(fid, SPARSE_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Write sparse data to the dataset */
    memset(wbuf, 0, sizeof(wbuf));

    /* Initialize and write sparse data to the dataset */
    wbuf[1]  = 1;
    wbuf[12] = 12;
    wbuf[13] = 13;
    wbuf[14] = 14;
    wbuf[22] = 22;
    wbuf[23] = 23;
    wbuf[24] = 24;
    wbuf[48] = 48;
    wbuf[49] = 49;
    if (H5Dwrite(did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0)
        TEST_ERROR;

    /* Get defined elements */
    /* TBD: Verify that dataset with H5D_SPARSE_CHUNK layout will succeed; otherwise fail */
    if ((sid1 = H5Dget_defined(did, H5S_ALL, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* TBD: Verify defined elements in sid1 are as expected */

    /* Erase all defined elements */
    /* TBD: Verify that dataset with H5D_SPARSE_CHUNK layout will succeed; otherwise fail */
    /* Since it is not supported yet, it is expected to fail */
    H5E_BEGIN_TRY
    {
        ret = H5Derase(did, sid1, H5P_DEFAULT);
    }
    H5E_END_TRY
    if (ret >= 0)
        TEST_ERROR;

    /* Call H5Dget_defined() again after H5Derase() */
    if ((sid2 = H5Dget_defined(did, H5S_ALL, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* TBD: Verify nothing is defined in sid2 */

    if (H5Sclose(sid1) < 0)
        TEST_ERROR;
    if (H5Sclose(sid2) < 0)
        TEST_ERROR;

    /* Closing */
    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

error:
    H5E_BEGIN_TRY
    {
        H5Sclose(sid);
        H5Sclose(sid1);
        H5Sclose(sid2);
        H5Pclose(dcpl);
        H5Dclose(did);
        H5Fclose(fid);
    }
    H5E_END_TRY

    return FAIL;
} /* end test_struct_chunk_api_defined_erase() */

/*-------------------------------------------------------------------------
 * Function:    test_sparse_direct_chunk
 *
 * Purpose:     Verify APIs for direct chunk I/O on structured chunk:
 *                  --H5Dwrite_struct_chunk()
 *                  --H5Dread_struct_chunk()
 *
 * Return:      # of errors
 *
 *-------------------------------------------------------------------------
 */
static herr_t
test_sparse_direct_chunk(hid_t fapl)
{
    char  filename[FILENAME_BUF_SIZE]; /* File name */
    hid_t fid  = H5I_INVALID_HID;
    hid_t did  = H5I_INVALID_HID;
    hid_t sid  = H5I_INVALID_HID;
    hid_t dcpl = H5I_INVALID_HID;

    hsize_t dims[2]       = {NX, NY};
    hsize_t maxdims[2]    = {H5S_UNLIMITED, H5S_UNLIMITED};
    hsize_t chunk_dims[2] = {CHUNK_NX, CHUNK_NY};

    int     buf[NX][NY];
    size_t  encode_size;
    hsize_t start[2], block[2], count[2];

    hsize_t                 wr_offset[2] = {0, 0};
    H5D_struct_chunk_info_t wr_chk_info;
    uint16_t                wr_filter_mask[2] = {0, 0};
    size_t                  wr_section_size[2];
    void                   *wr_buf[2];
    unsigned char          *wr_buf0;
    int                    *wr_buf1;

    hsize_t                 rd_offset[2] = {5, 5};
    H5D_struct_chunk_info_t rd_chk_info;
    uint16_t                rd_filter_mask[2] = {0, 0};
    size_t                  rd_section_size[2];
    void                   *rd_buf[2];
    unsigned char          *rd_buf0;
    int                    *rd_buf1;

    TESTING("APIs for direct chunk I/O on structured chunks");

    SKIPPED();
    return 0;

    /* Create a file */
    h5_fixname(FILENAME_TBD[1], fapl, filename, sizeof filename);
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        TEST_ERROR;

    /*
     * Create the data space with unlimited dimensions.
     */
    if ((sid = H5Screate_simple(RANK, dims, maxdims)) < 0)
        TEST_ERROR;

    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* TBD: need to set to H5D_SPARSE_CHUNK */
    if (H5Pset_layout(dcpl, H5D_CHUNKED) < 0)
        TEST_ERROR;

    if (H5Pset_chunk(dcpl, RANK, chunk_dims) < 0)
        TEST_ERROR;

    /*
     * Create a new dataset within the file using dcpl
     */
    if ((did = H5Dcreate2(fid, SPARSE_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    start[0] = 3;
    start[1] = 2;
    block[0] = 2;
    block[1] = 3;
    count[0] = count[1] = 1;
    /* Select the 2x3 block in chunk index 0 for writing */
    if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, NULL, count, block) < 0)
        TEST_ERROR;

    /* Get the encoded size for the selection */
    if (H5Sencode2(sid, NULL, &encode_size, H5P_DEFAULT) < 0)
        TEST_ERROR;

    /* Set up section size for section 0 and section 1 */
    wr_section_size[0] = encode_size;
    wr_section_size[1] = block[0] * block[1] * sizeof(int);

    /* Allocate buffers for section 0 (encoded selection) and section 1 (data) */
    if ((wr_buf0 = (unsigned char *)calloc((size_t)1, encode_size)) == NULL)
        TEST_ERROR;
    if ((wr_buf1 = (int *)calloc((size_t)1, wr_section_size[1])) == NULL)
        TEST_ERROR;

    /* Encode selection into the buffer for section 0 */
    if (H5Sencode2(sid, wr_buf0, &encode_size, H5P_DEFAULT) < 0)
        TEST_ERROR;

    /* Set up data into the buffer for section 1 */
    wr_buf1[0] = 32;
    wr_buf1[1] = 33;
    wr_buf1[2] = 34;
    wr_buf1[3] = 42;
    wr_buf1[4] = 43;
    wr_buf1[5] = 44;

    /* Set up the buffer for H5D_write_struct_chunk() */
    wr_buf[0] = wr_buf0;
    wr_buf[1] = wr_buf1;

    wr_chk_info.type              = 4; /* should be H5D_SPARSE_CHUNK */
    wr_chk_info.num_sections      = 2;
    wr_chk_info.filter_mask       = wr_filter_mask;
    wr_chk_info.section_size      = wr_section_size;
    wr_chk_info.section_orig_size = wr_section_size;

    /* Write the structured chunk at offset [0,0]: chunk index 0 */
    if (H5Dwrite_struct_chunk(did, H5P_DEFAULT, wr_offset, &wr_chk_info, wr_buf) < 0)
        TEST_ERROR;

    /* Read the whole dataset */
    if (H5Dread(did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0)
        TEST_ERROR;
    /* TBD: Verify buf read has data as in wr_buf1[] at location wr_buf0[] */

    if (H5Dclose(did) < 0)
        TEST_ERROR;

    if (H5Sclose(sid) < 0)
        TEST_ERROR;

    if ((did = H5Dopen2(fid, SPARSE_DSET, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;

    if ((sid = H5Dget_space(did)) == H5I_INVALID_HID)
        TEST_ERROR;

    /* Select the 2x1 block in chunk index 3 for reading */
    start[0] = 5;
    start[1] = 5;
    block[0] = 2;
    block[1] = 1;
    count[0] = count[1] = 1;
    if (H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, NULL, count, block) < 0)
        TEST_ERROR;

    if (H5Sencode2(sid, NULL, &encode_size, H5P_DEFAULT) < 0)
        TEST_ERROR;

    rd_section_size[0] = encode_size;
    rd_section_size[1] = block[0] * block[1] * sizeof(int);

    /* Allocate buffers for section 0 (encoded selection) and section 1 (data) */
    if ((rd_buf0 = (unsigned char *)calloc((size_t)1, encode_size)) == NULL)
        TEST_ERROR;
    if ((rd_buf1 = (int *)calloc((size_t)1, rd_section_size[1])) == NULL)
        TEST_ERROR;

    rd_buf[0] = rd_buf0;
    rd_buf[1] = rd_buf1;

    rd_chk_info.type              = 4; /* should be H5D_SPARSE_CHUNK */
    rd_chk_info.num_sections      = 2;
    rd_chk_info.filter_mask       = rd_filter_mask;
    rd_chk_info.section_size      = rd_section_size;
    rd_chk_info.section_orig_size = rd_section_size;

    /* Read the structured chunk at offset [5,5] */
    if (H5Dread_struct_chunk(did, H5P_DEFAULT, rd_offset, &rd_chk_info, rd_buf) < 0)
        TEST_ERROR;
    /* Verify rd_chk_info and rd_buf are the same as wr_chk_info and wr_buf */

    /*
     * Close/release resources.
     */
    if (H5Dclose(did) < 0)
        TEST_ERROR;
    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    PASSED();
    return SUCCEED;

error:
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Fclose(fid);
    }
    H5E_END_TRY

    H5_FAILED();
    return FAIL;

} /* test_sparse_direct_chunk() */

/*-------------------------------------------------------------------------
 * Function:    verify_get_struct_chunk_info (helper function)
 *
 * Purpose:     Verifies that H5Dget_struct_chunk_info returns correct
 *              values for a chunk.
 *
 * Return:      SUCCEED/FAIL
 *-------------------------------------------------------------------------
 */
static herr_t
verify_get_struct_chunk_info(hid_t did, hid_t sid, hsize_t chk_index,
                             const hsize_t H5_ATTR_UNUSED          *exp_offset,
                             H5D_struct_chunk_info_t H5_ATTR_UNUSED exp_chunk_info[],
                             hsize_t H5_ATTR_UNUSED                 exp_chk_size)
{
    hsize_t                 out_offset[2] = {0, 0}; /* Buffer to get offset coordinates */
    hsize_t                 out_chk_size  = 0;      /* Size of an allocated/written chunk */
    haddr_t                 out_addr      = 0;      /* Address of an allocated/written chunk */
    H5D_struct_chunk_info_t out_chunk_info[50];

    /* Get info of the chunk specified by chk_index */
    if (H5Dget_struct_chunk_info(did, sid, chk_index, out_offset, out_chunk_info, &out_addr, &out_chk_size) <
        0)
        TEST_ERROR;

#ifdef TBD

    /* Verify info from H5Dget_struct_chunk_info() with expected chunk info */

    if (out_offset[0] != exp_offset[0])
        FAIL_PUTS_ERROR("unexpected offset[0]");
    if (out_offset[1] != exp_offset[1])
        FAIL_PUTS_ERROR("unexpected offset[1]");

    Compare out_chunk_info with exp_chunk_info

        if (HADDR_UNDEF == out_addr) FAIL_PUTS_ERROR("address cannot be HADDR_UNDEF");

    if (out_chk_size != exp_chk_size)
        FAIL_PUTS_ERROR("unexpected chunk size");

#endif

    /* For now, just return SUCCEED */

    return SUCCEED;

error:
    return FAIL;
} /* verify_get_struct_chunk_info() */

/*-------------------------------------------------------------------------
 *
 * Function:    verify_get_struct_chunk_info_by_coord (helper function)
 *
 * Purpose:     Verifies that H5Dget_struct_chunk_info_by_coord returns correct
 *              values for a chunk.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */
static herr_t
verify_get_struct_chunk_info_by_coord(hid_t did, hsize_t *offset,
                                      H5D_struct_chunk_info_t H5_ATTR_UNUSED exp_chunk_info[],
                                      hsize_t H5_ATTR_UNUSED                 exp_chk_size)
{
    hsize_t                 out_chk_size = 0; /* Size of an allocated/written chunk */
    haddr_t                 out_addr     = 0; /* Address of an allocated/written chunk */
    H5D_struct_chunk_info_t out_chunk_info[50];

    /* Get info of the chunk at logical coordinates specified by offset */
    if (H5Dget_struct_chunk_info_by_coord(did, offset, out_chunk_info, &out_addr, &out_chk_size) < 0)
        TEST_ERROR;

#ifdef TBD
    {
        /* Verify info from H5Dget_struct_chunk_info_by_coord() with expected chunk info */

        if (HADDR_UNDEF == out_addr)
            FAIL_PUTS_ERROR("address cannot be HADDR_UNDEF");

        Compare out_chunk_info with exp_chunk_info

            if (out_chk_size != exp_chk_size) FAIL_PUTS_ERROR("unexpected chunk size");
    }
#endif

    /* For now, just return SUCCEED */

    return SUCCEED;

error:
    return FAIL;
} /* verify_get_struct_chunk_info_by_coord() */

typedef struct struct_chunk_iter_info_t {
    hsize_t                  offset[2];
    H5D_struct_chunk_info_t *chunk_info;
    haddr_t                  addr;
    hsize_t                  chunk_size;
} struct_chunk_iter_info_t;

typedef struct struct_chunk_iter_udata_t {
    struct_chunk_iter_info_t *struct_chunk_info;
    int                       last_index;
} struct_chunk_iter_udata_t;

static int
iter_cb_struct(const hsize_t *offset, H5D_struct_chunk_info_t *chunk_info, haddr_t *addr, hsize_t *chunk_size,
               void *op_data)
{
    struct_chunk_iter_udata_t *cidata = (struct_chunk_iter_udata_t *)op_data;
    int                        idx    = cidata->last_index + 1;

    cidata->struct_chunk_info[idx].offset[0]  = offset[0];
    cidata->struct_chunk_info[idx].offset[1]  = offset[1];
    cidata->struct_chunk_info[idx].chunk_info = chunk_info;
    cidata->struct_chunk_info[idx].addr       = *addr;
    cidata->struct_chunk_info[idx].chunk_size = *chunk_size;

    cidata->last_index++;

    return H5_ITER_CONT;
} /* iter_cb_struct() */

/*-------------------------------------------------------------------------
 * Function:    test_sparse_direct_chunk_query
 *
 * Purpose:     Verify APIs for direct chunk I/O query on structured chunk:
 *                  --H5Dget_struct_chunk_info()
 *                  --H5Dget_struct_chunk_info_by_coord()
 *                  --H5Dstruct_chunk_iter()
 *
 * Return:      # of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_sparse_direct_chunk_query(hid_t fapl)
{
    char    filename[FILENAME_BUF_SIZE];          /* File name */
    hid_t   fid           = H5I_INVALID_HID;      /* File ID */
    hid_t   sid           = H5I_INVALID_HID;      /* Dataspace ID */
    hid_t   did           = H5I_INVALID_HID;      /* Dataset ID */
    hid_t   dcpl          = H5I_INVALID_HID;      /* Creation plist */
    hsize_t dims[2]       = {NX, NY};             /* Dataset dimensions */
    hsize_t chunk_dims[2] = {CHUNK_NX, CHUNK_NY}; /* Chunk dimensions */

    struct_chunk_iter_info_t  chunk_infos[2]; /* Chunk infos filled up by iterator */
    struct_chunk_iter_udata_t udata;          /* udata for iteration */
    H5D_struct_chunk_info_t   chk_info;

    uint16_t filter_mask[2]  = {0, 0};
    hsize_t  offset[2]       = {0, 0};
    size_t   section_size[2] = {32, 48};
    void    *write_buf[2];
    hsize_t  in0[4] = {3, 2, 4, 4};              /* Encoded coordinates: [3,2] - [4,4] */
    hsize_t  in1[6] = {66, 69, 72, 96, 99, 102}; /* Data: 66,69,72,96,99,102 */

    TESTING("APIs for direct chunk I/O query on structured chunk");

    SKIPPED();
    return 0;

    /* Create the file */
    h5_fixname(FILENAME_TBD[2], fapl, filename, sizeof filename);

    /* Create a new file. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Create dataspace */
    if ((sid = H5Screate_simple(RANK, dims, NULL)) < 0)
        TEST_ERROR;

    /* Enable chunking */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* TBD: need to set to H5D_SPARSE_CHUNK */
    if (H5Pset_layout(dcpl, H5D_CHUNKED) < 0)
        TEST_ERROR;

    if (H5Pset_chunk(dcpl, RANK, chunk_dims) < 0)
        TEST_ERROR;

    /* Create a new dataset using dcpl creation properties */
    did = H5Dcreate2(fid, SPARSE_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (did < 0)
        TEST_ERROR;

    write_buf[0] = in0;
    write_buf[1] = in1;

    chk_info.type              = 4; /* should be H5D_SPARSE_CHUNK */
    chk_info.num_sections      = 2;
    chk_info.filter_mask       = filter_mask;
    chk_info.section_size      = section_size;
    chk_info.section_orig_size = section_size;

    /* Write the structured chunk at offset */
    if (H5Dwrite_struct_chunk(did, H5P_DEFAULT, offset, &chk_info, write_buf) < 0)
        TEST_ERROR;

    /* Verify info of the first and only chunk via H5Dget_struct_chunk_info() */
    if (verify_get_struct_chunk_info(did, H5S_ALL, 0, offset, &chk_info, CHK_SIZE) == FAIL)
        FAIL_PUTS_ERROR("Verification H5Dget_struct_chunk_info failed\n");

    offset[0] = CHUNK_NX;
    offset[1] = CHUNK_NY;

    /* Write the structured chunk at offset */
    if (H5Dwrite_struct_chunk(did, H5P_DEFAULT, offset, &chk_info, write_buf) < 0)
        TEST_ERROR;

    /* Verify info of the chunk at offset [CHUNK_NX,CHUNK_NY] via H5Dget_struct_chunk_info_by_coord() */
    if (verify_get_struct_chunk_info_by_coord(did, offset, &chk_info, CHK_SIZE) == FAIL)
        FAIL_PUTS_ERROR("Verification of H5Dget_struct_chunk_info_by_coord failed\n");

    /* For now, H5Dstruct_chunk_iter() just returns SUCCEED without actual iteration */
    udata.struct_chunk_info = chunk_infos;
    udata.last_index        = -1;
    if (H5Dstruct_chunk_iter(did, H5P_DEFAULT, &iter_cb_struct, &udata) < 0)
        TEST_ERROR;

    /* Release resource */
    if (H5Dclose(did) < 0)
        TEST_ERROR;
    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* Remove the test file */
    HDremove(filename);

    PASSED();
    return SUCCEED;

error:
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Fclose(fid);
    }
    H5E_END_TRY

    return FAIL;
} /* test_sparse_direct_chunk_query() */

typedef struct chunk_iter_info_t {
    hsize_t  offset[2];
    unsigned filter_mask;
    haddr_t  addr;
    hsize_t  size;
} chunk_iter_info_t;

typedef struct chunk_iter_udata_t {
    chunk_iter_info_t *chunk_info;
    int                last_index;
} chunk_iter_udata_t;

static int
iter_cb(const hsize_t *offset, unsigned filter_mask, haddr_t addr, hsize_t size, void *op_data)
{
    chunk_iter_udata_t *cidata = (chunk_iter_udata_t *)op_data;
    int                 idx    = cidata->last_index + 1;

    cidata->chunk_info[idx].offset[0]   = offset[0];
    cidata->chunk_info[idx].offset[1]   = offset[1];
    cidata->chunk_info[idx].filter_mask = filter_mask;
    cidata->chunk_info[idx].addr        = addr;
    cidata->chunk_info[idx].size        = size;

    cidata->last_index++;

    return H5_ITER_CONT;
} /* iter_cb() */

/*-------------------------------------------------------------------------
 * Function:    test_dense_chunk_api_on_sparse()
 *
 * Purpose: Verify the following dense chunk APIs will fail for
 *          H5D_SPARSE_CHUNK layout:
 *             --H5Dwrite_chunk()
 *             --H5Dget_chunk_info()
 *             --H5Dget_chunk_info_by_coord()
 *             --H5Dchunk_iter()
 *          Verify the following dense chunk APIs will succeed for
 *          H5D_SPARSE_CHUNK layout:
 *              --H5Dread_chunk()
 *              --H5Dget_chunk_storage_size()
 *              --H5Dget_num_chunks()
 *
 * Return:      # of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_dense_chunk_api_on_sparse(hid_t fapl)
{
    char               filename[FILENAME_BUF_SIZE];          /* File name */
    hid_t              fid           = H5I_INVALID_HID;      /* File ID */
    hid_t              sid           = H5I_INVALID_HID;      /* Dataspace ID */
    hid_t              did           = H5I_INVALID_HID;      /* Dataset ID */
    hid_t              dcpl          = H5I_INVALID_HID;      /* Creation plist */
    hsize_t            dims[2]       = {NX, NY};             /* Dataset dimensions */
    hsize_t            chunk_dims[2] = {CHUNK_NX, CHUNK_NY}; /* Chunk dimensions */
    chunk_iter_info_t  chunk_infos[2];
    chunk_iter_udata_t udata;
    hsize_t            nchunks = 0;
    hsize_t            chunk_nbytes;
    hsize_t            offset[2] = {0, 0};
    int                direct_buf[CHUNK_NX][CHUNK_NY];
    haddr_t            addr    = 0;
    uint32_t           filters = 0;

    TESTING("APIs for direct chunk I/O: dense chunk functions on sparse layout");

    /* Create the file */
    h5_fixname(FILENAME_TBD[3], fapl, filename, sizeof filename);

    /* Create a new file. */
    if ((fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)) < 0)
        TEST_ERROR;

    /* Create dataspace */
    if ((sid = H5Screate_simple(RANK, dims, NULL)) < 0)
        TEST_ERROR;

    /* Enable chunking */
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        TEST_ERROR;

    /* TBD: need to set to H5D_SPARSE_CHUNK */
    if (H5Pset_layout(dcpl, H5D_CHUNKED) < 0)
        TEST_ERROR;

    /* The layout is set to H5D_CHUNKED as a side-effect */
    if (H5Pset_chunk(dcpl, RANK, chunk_dims) < 0)
        TEST_ERROR;

    /* Create a new dataset using dcpl creation properties */
    did = H5Dcreate2(fid, SPARSE_DSET, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (did < 0)
        TEST_ERROR;

    H5E_BEGIN_TRY
    {
        H5Dwrite_chunk(did, H5P_DEFAULT, 0, offset, CHK_SIZE, direct_buf);
    }
    H5E_END_TRY
    /* TBD: set return status and verify that it should fail */

    H5E_BEGIN_TRY
    {
        H5Dget_chunk_info(did, H5S_ALL, 0, NULL, NULL, &addr, NULL);
    }
    H5E_END_TRY
    /* TBD: set return status and verify that it should fail */

    H5E_BEGIN_TRY
    {
        H5Dget_chunk_info_by_coord(did, offset, NULL, &addr, NULL);
    }
    H5E_END_TRY
    /* TBD: set return status and verify that it should fail */

    H5E_BEGIN_TRY
    {
        udata.chunk_info = chunk_infos;
        udata.last_index = -1;
        H5Dchunk_iter(did, H5P_DEFAULT, &iter_cb, &udata);
    }
    H5E_END_TRY
    /* TBD: set return status and verify that it should fail */

    H5Dread_chunk(did, H5P_DEFAULT, offset, &filters, direct_buf);
    /* TBD: should succeed */

    H5Dget_num_chunks(did, sid, &nchunks);
    /* TBD: should succeed */

    H5Dget_chunk_storage_size(did, offset, &chunk_nbytes);
    /* TBD: should succeed */

    /* Release resource */
    if (H5Dclose(did) < 0)
        TEST_ERROR;
    if (H5Sclose(sid) < 0)
        TEST_ERROR;
    if (H5Pclose(dcpl) < 0)
        TEST_ERROR;
    if (H5Fclose(fid) < 0)
        TEST_ERROR;

    /* Remove the test file */
    HDremove(filename);

    PASSED();
    return SUCCEED;

error:
    H5E_BEGIN_TRY
    {
        H5Dclose(did);
        H5Sclose(sid);
        H5Pclose(dcpl);
        H5Fclose(fid);
    }
    H5E_END_TRY

    return FAIL;
} /* test_dense_chunk_api_on_sparse() */

#endif /* TBD */

/*-------------------------------------------------------------------------
 * Function:    main
 *
 * Purpose:     Tests for structured chunk layout
 *              Some are copied and modified from:
 *               --test/dsets.c
 *               --test/direct_chunk.c
 *               --test/chunk_info.c
 *
 *
 * *
 * Return:      EXIT_SUCCESS/EXIT_FAILURE
 *
 *-------------------------------------------------------------------------
 */
int
main(void)
{
    unsigned     paged;
    unsigned     filtered;
    unsigned     early;
    int          nerrors = 0;
    const char  *driver_name;
    bool         contig_addr_vfd; /* Whether VFD used has a contiguous address space */
    bool         driver_is_default_compatible;
    hid_t        fcpl        = H5I_INVALID_HID;
    hid_t        page_fcpl   = H5I_INVALID_HID;
    hid_t        fapl        = H5I_INVALID_HID;
    hid_t        libver_fapl = H5I_INVALID_HID;
    H5F_libver_t low, high; /* File format bounds */

    /* Don't run this test using certain file drivers */
    driver_name = h5_get_test_driver_name();

    /* Current VFD that does not support contiguous address space */
    contig_addr_vfd = (bool)(strcmp(driver_name, "split") != 0 && strcmp(driver_name, "multi") != 0);

    /* Testing setup */
    h5_test_init();

    fapl = h5_fileaccess();

    if (h5_driver_is_default_vfd_compatible(fapl, &driver_is_default_compatible) < 0)
        TEST_ERROR;

    /* create a file creation property list */
    if ((fcpl = H5Pcreate(H5P_FILE_CREATE)) < 0)
        TEST_ERROR;

    if ((page_fcpl = H5Pcopy(fcpl)) < 0)
        TEST_ERROR;

    /* Set file space strategy to paged aggregation and persisting free-space */
    if (H5Pset_file_space_strategy(page_fcpl, H5F_FSPACE_STRATEGY_PAGE, true, (hsize_t)1) < 0)
        TEST_ERROR;

    /* Test with paged aggregation enabled or not */
    for (paged = false; paged <= true; paged++) {

        /* Temporary: skip testing for multi/split drivers:
             fail file create when persisting free-space or using paged aggregation strategy */
        if (!contig_addr_vfd && paged)
            continue;

        for (early = false; early <= true; early++) {

            for (filtered = false; filtered <= true; filtered++) {

                for (low = H5F_LIBVER_EARLIEST; low < H5F_LIBVER_NBOUNDS; low++) {
                    if ((libver_fapl = H5Pcopy(fapl)) < 0)
                        TEST_ERROR;

                    for (high = H5F_LIBVER_EARLIEST; high < H5F_LIBVER_NBOUNDS; high++) {

                        hid_t       my_fcpl = H5I_INVALID_HID;
                        herr_t      ret;
                        const char *low_string;  /* Message for library version low bound */
                        const char *high_string; /* Message for library version high bound */

                        /* Set version bounds */
                        H5E_BEGIN_TRY
                        {
                            ret = H5Pset_libver_bounds(libver_fapl, low, high);
                        }
                        H5E_END_TRY

                        if (ret < 0) /* Invalid low/high combinations */
                            continue;

                        /* Paged aggregation needs high bound to be at least H5F_LIBVER_V110 */
                        if (paged && high < H5F_LIBVER_V110)
                            continue;

                        low_string  = h5_get_version_string(low);
                        high_string = h5_get_version_string(high);

                        if (paged) {
                            my_fcpl = page_fcpl;

                            if (early) {
                                if (filtered)
                                    printf("\nTesting with paged aggregation, early alloc, filtered and "
                                           "libver (%s, %s)\n",
                                           low_string, high_string);
                                else
                                    printf("\nTesting with paged aggregation, early alloc, non-filtered and "
                                           "libver (%s, %s)\n",
                                           low_string, high_string);
                            }
                            else {

                                if (filtered)
                                    printf("\nTesting with paged aggregation, default alloc, filtered and "
                                           "libver (%s, %s)\n",
                                           low_string, high_string);
                                else
                                    printf("\nTesting with paged aggregation, default alloc, non-filtered "
                                           "and libver (%s, %s)\n",
                                           low_string, high_string);
                            }
                        }
                        else {
                            my_fcpl = fcpl;

                            if (early) {
                                if (filtered)
                                    printf("\nTesting with non-paged aggregation, early alloc, filtered and "
                                           "libver (%s, %s)\n",
                                           low_string, high_string);
                                else
                                    printf("\nTesting with non-paged aggregation, early alloc, non-filtered "
                                           "and libver (%s, %s)\n",
                                           low_string, high_string);
                            }
                            else {
                                if (filtered)
                                    printf("\nTesting with non-paged aggregation, default alloc, filtered "
                                           "and libver (%s, %s)\n",
                                           low_string, high_string);
                                else
                                    printf("\nTesting with non-paged aggregation, default alloc, "
                                           "non-filtered and libver (%s, %s)\n",
                                           low_string, high_string);
                            }
                        }

                        nerrors +=
                            (test_struct_chunk_info_1d(my_fcpl, libver_fapl, filtered, early, CHK_SINGLE) < 0
                                 ? 1
                                 : 0);
                        nerrors +=
                            (test_struct_chunk_info_1d(my_fcpl, libver_fapl, filtered, early, CHK_FA) < 0
                                 ? 1
                                 : 0);
                        nerrors +=
                            (test_struct_chunk_info_1d(my_fcpl, libver_fapl, filtered, early, CHK_EA) < 0
                                 ? 1
                                 : 0);
                        nerrors +=
                            (test_struct_chunk_info_2d_bt2(my_fcpl, libver_fapl, filtered, early) < 0 ? 1
                                                                                                      : 0);
                        nerrors +=
                            (test_struct_chunk_extent_1d(my_fcpl, libver_fapl, filtered, early) < 0 ? 1 : 0);
                        nerrors +=
                            (test_struct_chunk_extent_2d(my_fcpl, libver_fapl, filtered, early, true) < 0
                                 ? 1
                                 : 0);
                        nerrors +=
                            (test_struct_chunk_extent_2d(my_fcpl, libver_fapl, filtered, early, false) < 0
                                 ? 1
                                 : 0);

                        nerrors += (test_struct_chunk_api(my_fcpl, libver_fapl) < 0 ? 1 : 0);
                        nerrors +=
                            (test_struct_chunk_1d_single(my_fcpl, libver_fapl, filtered, early) < 0 ? 1 : 0);
                        nerrors +=
                            (test_struct_chunk_2d_bt2(my_fcpl, libver_fapl, filtered, early) < 0 ? 1 : 0);
                        nerrors +=
                            (test_struct_chunk_1d_fa(my_fcpl, libver_fapl, filtered, early) < 0 ? 1 : 0);
                        nerrors +=
                            (test_struct_chunk_2d_ea(my_fcpl, libver_fapl, filtered, early) < 0 ? 1 : 0);

                        nerrors += (test_struct_chunk_filter_register(my_fcpl, libver_fapl) < 0 ? 1 : 0);

                        /* Tests to be worked on when APIs are implemented */
#ifdef TBD
                        nerrors += (test_struct_chunk_api_defined_erase(my_fapl) < 0 ? 1 : 0);
                        nerrors += (test_sparse_direct_chunk(my_fapl) < 0 ? 1 : 0);
                        nerrors += (test_sparse_direct_chunk_query(my_fapl) < 0 ? 1 : 0);
                        nerrors += (test_dense_chunk_api_on_sparse(my_fapl) < 0 ? 1 : 0);
#endif
                    } /* end for high */

                    h5_delete_all_test_files(FILENAME, libver_fapl);
                    if (H5Pclose(libver_fapl) < 0)
                        TEST_ERROR;

                } /* end for low */

            } /* end filtered */

        } /* end early */

    } /* end paged */

    /*
     * Run the VL-specific tests separately from the matrix above.
     *
     * These cases use the latest file format because structured chunks with
     * a VL section require the newer on-disk layout. The general structured
     * chunk tests above retain their own file-format combinations.
     */
    {
        hid_t    vl_fapl = H5I_INVALID_HID; /* File access settings for VL tests */
        unsigned type;                      /* Chunk index case being tested */
        unsigned use_filter;                /* 0: no filter; 1: filter enabled */

        size_t      stress_iterations = VL_STRESS_DEFAULT_ITERATIONS;
        uint64_t    stress_seed       = UINT64_C(0x6a09e667f3bcc909);
        const char *env_value;

        if ((env_value = getenv("HDF5_STRUCT_CHUNK_VL_STRESS_ITERS")) && *env_value) {
            char              *end    = NULL;
            unsigned long long parsed = strtoull(env_value, &end, 10);

            if (!end || *end != '\0' || parsed > (unsigned long long)SIZE_MAX)
                TEST_ERROR;

            stress_iterations = (size_t)parsed;
        }

        if ((env_value = getenv("HDF5_STRUCT_CHUNK_VL_STRESS_SEED")) && *env_value) {
            char              *end    = NULL;
            unsigned long long parsed = strtoull(env_value, &end, 0);

            if (!end || *end != '\0')
                TEST_ERROR;

            stress_seed = (uint64_t)parsed;
        }

        /* Copy the base FAPL so changing its format bounds affects only VL tests. */
        if ((vl_fapl = H5Pcopy(fapl)) < 0)
            TEST_ERROR;

        /* Read and write these test files using the latest format bounds. */
        if (H5Pset_libver_bounds(vl_fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST) < 0)
            TEST_ERROR;

        /*
         * Check basic VL write, replacement, and readback with each explicitly
         * selected index: single chunk, fixed array, and extensible array.
         *
         * Run each index case once without a filter and once with a filter.
         * A negative test return contributes one failure to nerrors.
         */
        for (type = CHK_SINGLE; type <= CHK_EA; type++) {

            for (use_filter = 0; use_filter < 2; use_filter++) {

                nerrors += (test_struct_chunk_vlen(fcpl, vl_fapl, type, (bool)use_filter) < 0);
            }
        }
        /*
         * Type 0 takes the test function's default branch, which configures
         * the two-dimensional, unlimited case and expects a B-tree 2 index.
         */
        for (use_filter = 0; use_filter < 2; use_filter++) {
            nerrors += (test_struct_chunk_vlen(fcpl, vl_fapl, 0, (bool)use_filter) < 0);
        }

        for (type = CHK_SINGLE; type <= CHK_EA; type++) {
            for (use_filter = 0; use_filter < 2; use_filter++)
                nerrors +=
                    (test_struct_chunk_vlen_dense_emulation(fcpl, vl_fapl, type, (bool)use_filter) < 0);
        }

        for (use_filter = 0; use_filter < 2; use_filter++) {
            nerrors += (test_struct_chunk_vlen_dense_emulation(fcpl, vl_fapl, 0, (bool)use_filter) < 0);
        }

        /*
         * Repeatedly change VL values to exercise removal, reinsertion,
         * and reuse of chunk-local heap storage.
         */
        nerrors += (test_struct_chunk_vlen_churn(fcpl, vl_fapl, false) < 0);
        nerrors += (test_struct_chunk_vlen_churn(fcpl, vl_fapl, true) < 0);

        /*
         * Exercise writes that update only part of the dataset. Run both
         * filter configurations so either section-processing path is checked.
         */
        nerrors += (test_struct_chunk_vlen_partial(fcpl, vl_fapl, false) < 0);
        nerrors += (test_struct_chunk_vlen_partial(fcpl, vl_fapl, true) < 0);

        /*
         * Check large VL payloads and replacement across the same three
         * explicitly selected index types, with and without filtering.
         */
        for (type = CHK_SINGLE; type <= CHK_EA; type++) {

            for (use_filter = 0; use_filter < 2; use_filter++) {

                nerrors += (test_struct_chunk_vlen_large(fcpl, vl_fapl, type, (bool)use_filter) < 0);
            }
        }

        /* Also check large payloads in the B-tree 2 index case. */
        for (use_filter = 0; use_filter < 2; use_filter++) {

            nerrors += (test_struct_chunk_vlen_large(fcpl, vl_fapl, 0, (bool)use_filter) < 0);
        }

        /*
         * Check VL data inside a compound datatype. Each call supplies the
         * filter choice to the same test rather than changing the base FAPL.
         */
        nerrors += (test_struct_chunk_vlen_compound(fcpl, vl_fapl, false) < 0);
        nerrors += (test_struct_chunk_vlen_compound(fcpl, vl_fapl, true) < 0);

        /* Check a datatype containing two separate VL members. */
        nerrors += (test_struct_chunk_vlen_two_members(fcpl, vl_fapl, false) < 0);
        nerrors += (test_struct_chunk_vlen_two_members(fcpl, vl_fapl, true) < 0);

        /*
         * Check removal of VL values and a chunk whose VL section becomes
         * empty. Run each scenario with both filter configurations.
         */
        for (use_filter = 0; use_filter < 2; use_filter++) {
            nerrors += (test_struct_chunk_vlen_erase(fcpl, vl_fapl, (bool)use_filter) < 0);
            nerrors += (test_struct_chunk_vlen_empty_section(fcpl, vl_fapl, (bool)use_filter) < 0);
        }

        /* Check VL values when the memory and file datatypes require conversion. */
        nerrors += (test_struct_chunk_vlen_type_conversion(fcpl, vl_fapl, false) < 0);
        nerrors += (test_struct_chunk_vlen_type_conversion(fcpl, vl_fapl, true) < 0);

        /*
         * Exercise recursive datatype handling and lifecycle transitions
         * independently of the numeric VL stress test.
         */
        nerrors += (test_struct_chunk_vlen_edges(fcpl, vl_fapl, false) < 0);

        nerrors += (test_struct_chunk_vlen_edges(fcpl, vl_fapl, true) < 0);

        /*
         * Test the local heap-set implementation directly. In particular,
         * check stable slot numbers and cached allocation-size accounting.
         * This test does not take a filter argument.
         */
        nerrors += (test_local_heapset_stable_slots(fcpl, vl_fapl) < 0);
        nerrors += (test_local_heapset_growth(fcpl, vl_fapl) < 0);
        nerrors += (test_local_heapset_slot_growth(fcpl, vl_fapl) < 0);

        /* Follow complete growth/erase/empty/reuse lifecycles independently
         * of the optional random-stress iteration count. Each case uses real
         * dataset APIs and verifies every mutation plus each persisted phase.
         */
        for (type = CHK_SINGLE; type <= CHK_EA; type++) {
            for (use_filter = 0; use_filter < 2; use_filter++) {
                nerrors += (test_struct_chunk_vlen_lifecycle(fcpl, vl_fapl, type, (bool)use_filter) < 0);
            }
        }
        for (use_filter = 0; use_filter < 2; use_filter++) {
            nerrors += (test_struct_chunk_vlen_lifecycle(fcpl, vl_fapl, 0, (bool)use_filter) < 0);
        }

        nerrors += (test_struct_chunk_vlen_eviction(fcpl, vl_fapl, false) < 0);
        nerrors += (test_struct_chunk_vlen_eviction(fcpl, vl_fapl, true) < 0);

        /*
         * Run a deterministic state-machine test across every index and both
         * filter modes.  It continuously compares HDF5 with an independent
         * model and periodically forces complete file-close/reopen cycles.
         */
        if (stress_iterations > 0) {

            printf("\nVL stress configuration: %zu iterations per case, seed=%llu\n", stress_iterations,
                   (unsigned long long)stress_seed);

            for (type = CHK_SINGLE; type <= CHK_EA; type++)
                for (use_filter = 0; use_filter < 2; use_filter++)
                    nerrors += (test_struct_chunk_vlen_stress(fcpl, vl_fapl, type, (bool)use_filter,
                                                              stress_iterations, stress_seed) < 0);

            for (use_filter = 0; use_filter < 2; use_filter++)
                nerrors += (test_struct_chunk_vlen_stress(fcpl, vl_fapl, 0, (bool)use_filter,
                                                          stress_iterations, stress_seed) < 0);
        }

        /* Remove files produced by this test group using its VL-specific FAPL. */
        h5_delete_all_test_files(FILENAME, vl_fapl);

        /* Release the copied FAPL */
        if (H5Pclose(vl_fapl) < 0)
            TEST_ERROR;
    } /* end vlen tests */

    if (H5Pclose(fcpl) < 0)
        TEST_ERROR;

    if (H5Pclose(page_fcpl) < 0)
        TEST_ERROR;

    if (H5Pclose(fapl) < 0)
        TEST_ERROR;

    if (nerrors)
        goto error;
    printf("All structured chunk storage tests passed.\n");

    exit(EXIT_SUCCESS);

error:
    nerrors = MAX(1, nerrors);
    printf("***** %d STRUCTURED CHUNK STORAGE TEST%s FAILED! *****\n", nerrors, 1 == nerrors ? "" : "S");
    exit(EXIT_FAILURE);
} /* end main() */
