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
 * Purpose:     Tests the global heap.  The global heap is the set of all
 *              collections but the collections are not related to one
 *              another by anything that appears in the file format.
 */
#include "h5test.h"
#include "H5ACprivate.h"
#include "H5CXprivate.h" /* API Contexts                         */
#include "H5Eprivate.h"
#include "H5Fprivate.h"
#include "H5Gprivate.h"
#include "H5HGprivate.h"
#include "H5Iprivate.h"
#include "H5Pprivate.h"
#include "H5VLprivate.h"

#define H5HG_FRIEND /* Inspect resident heap state in the compact-image regression test */
#include "H5HGpkg.h"

/* Macros for printing error messages in loops.  These print up to
 * GHEAP_REPEATED_ERR_LIM errors, and suppress the rest */
#define GHEAP_REPEATED_ERR_LIM 20

/* Number of heap objects to test */
#define GHEAP_TEST_NOBJS 1024

/*
 * Normal allocation size for one chunk-local H5HG heap.
 * This is an in-memory allocation policy, not a persistent file-format field.
 */
#define H5HG_LOCAL_NORMAL_HEAP_SIZE ((size_t)(512 * 1024))

#define GHEAP_REPEATED_ERR(MSG)                                                                              \
    do {                                                                                                     \
        nerrors++;                                                                                           \
        if (nerrors <= GHEAP_REPEATED_ERR_LIM) {                                                             \
            H5_FAILED();                                                                                     \
            puts(MSG);                                                                                       \
            if (nerrors == GHEAP_REPEATED_ERR_LIM)                                                           \
                puts("    Suppressing further errors...");                                                   \
        }       /* end if */                                                                                 \
    } while (0) /* end GHEAP_REPEATED_ERR */

static const char *FILENAME[] = {"gheap1", "gheap2", "gheap3", "gheap4",   "gheapooo",    "lheap1",
                                 "lheap2", "lheap3", "lheap4", "lheapooo", "lheapencdec", NULL};

/*-------------------------------------------------------------------------
 * Function:    test_1
 *
 * Purpose:     Writes a sequence of objects to the global heap where each
 *              object is larger than the one before.
 *
 * Return:      Success:    0
 *
 *              Failure:    number of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_1(hid_t fapl)
{
    hid_t   file = H5I_INVALID_HID;
    H5F_t  *f    = NULL;
    H5HG_t *obj  = NULL;
    uint8_t out[GHEAP_TEST_NOBJS];
    uint8_t in[GHEAP_TEST_NOBJS];
    size_t  u;
    size_t  size;
    herr_t  status;
    int     nerrors = 0;
    char    filename[1024];

    TESTING("monotonically increasing lengths");

    /* Allocate buffer for H5HG_t */
    if (NULL == (obj = (H5HG_t *)malloc(sizeof(H5HG_t) * GHEAP_TEST_NOBJS)))
        goto error;

    /* Open a clean file */
    h5_fixname(FILENAME[0], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to create file");
        goto error;
    }

    /*
     * Write the objects, monotonically increasing in length.  Since this is
     * a clean file, the addresses allocated for the collections should also
     * be monotonically increasing.
     */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u + 1;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        status = H5HG_insert(f, size, out, obj + u);
        if (status < 0) {
            H5_FAILED();
            puts("    Unable to insert object into global heap");
            nerrors++;
        }
        else if (u && H5_addr_gt(obj[u - 1].addr, obj[u].addr)) {
            H5_FAILED();
            puts("    Collection addresses are not monotonically increasing");
            nerrors++;
        }
    }

    /*
     * Now try to read each object back.
     */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u + 1;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        if (NULL == H5HG_read(f, obj + u, in, NULL)) {
            H5_FAILED();
            puts("    Unable to read object");
            nerrors++;
        }
        else if (memcmp(in, out, size) != 0) {
            H5_FAILED();
            puts("    Value read doesn't match value written");
            nerrors++;
        }
    }

    /* Release buffer */
    free(obj);
    obj = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    if (obj)
        free(obj);
    return MAX(1, nerrors);
}

/*-------------------------------------------------------------------------
 * Function:    test_2
 *
 * Purpose:     Writes a sequence of objects to the global heap where each
 *              object is smaller than the one before.
 *
 * Return:      Success:    0
 *
 *              Failure:     number of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_2(hid_t fapl)
{
    hid_t   file = H5I_INVALID_HID;
    H5F_t  *f    = NULL;
    H5HG_t *obj  = NULL;
    uint8_t out[GHEAP_TEST_NOBJS];
    uint8_t in[GHEAP_TEST_NOBJS];
    size_t  u;
    size_t  size;
    int     nerrors = 0;
    char    filename[1024];

    TESTING("monotonically decreasing lengths");

    /* Allocate buffer for H5HG_t */
    if (NULL == (obj = (H5HG_t *)malloc(sizeof(H5HG_t) * GHEAP_TEST_NOBJS)))
        goto error;

    /* Open a clean file */
    h5_fixname(FILENAME[1], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to create file");
        goto error;
    }

    /*
     * Write the objects, monotonically decreasing in length.
     */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = GHEAP_TEST_NOBJS - u;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        if (H5HG_insert(f, size, out, obj + u) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into global heap");
            nerrors++;
        }
    }

    /*
     * Now try to read each object back.
     */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = GHEAP_TEST_NOBJS - u;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        if (NULL == H5HG_read(f, obj + u, in, NULL)) {
            H5_FAILED();
            puts("    Unable to read object");
            nerrors++;
        }
        else if (memcmp(in, out, size) != 0) {
            H5_FAILED();
            puts("    Value read doesn't match value written");
            nerrors++;
        }
    }

    /* Release buffer */
    free(obj);
    obj = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    if (obj)
        free(obj);
    return MAX(1, nerrors);
}

/*-------------------------------------------------------------------------
 * Function:    test_3
 *
 * Purpose:     Creates a few global heap objects and then removes them all.
 *              The collection should also be removed.
 *
 * Return:      Success:    0
 *
 *              Failure:    number of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_3(hid_t fapl)
{
    hid_t   file = H5I_INVALID_HID;
    H5F_t  *f    = NULL;
    H5HG_t *obj  = NULL;
    uint8_t out[GHEAP_TEST_NOBJS];
    size_t  u;
    size_t  size;
    herr_t  status;
    int     nerrors = 0;
    char    filename[1024];

    TESTING("complete object removal");

    /* Allocate buffer for H5HG_t */
    if (NULL == (obj = (H5HG_t *)malloc(sizeof(H5HG_t) * GHEAP_TEST_NOBJS)))
        goto error;

    /* Open a clean file */
    h5_fixname(FILENAME[2], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to create file");
        goto error;
    }

    /* Create some stuff */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u % 30 + 100;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        status = H5HG_insert(f, size, out, obj + u);
        if (status < 0) {
            H5_FAILED();
            puts("    Unable to insert object into global heap");
            nerrors++;
        }
    }

    /* Remove everything */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        status = H5HG_remove(f, obj + u);
        if (status < 0) {
            H5_FAILED();
            puts("    Unable to remove object");
            nerrors++;
        }
    }

    /* Release buffer */
    free(obj);
    obj = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    if (obj)
        free(obj);
    return MAX(1, nerrors);
}

/*-------------------------------------------------------------------------
 * Function:    test_4
 *
 * Purpose:     Tests the H5HG_remove() feature by writing lots of objects
 *              and occasionally removing some.  When we're done they're all
 *              removed.
 *
 * Return:      Success:    0
 *
 *              Failure:    number of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_4(hid_t fapl)
{
    hid_t   file = H5I_INVALID_HID;
    H5F_t  *f    = NULL;
    H5HG_t *obj  = NULL;
    uint8_t out[GHEAP_TEST_NOBJS];
    size_t  u;
    size_t  size;
    herr_t  status;
    int     nerrors = 0;
    char    filename[1024];

    TESTING("partial object removal");

    /* Allocate buffer for H5HG_t */
    if (NULL == (obj = (H5HG_t *)malloc(sizeof(H5HG_t) * GHEAP_TEST_NOBJS)))
        goto error;

    /* Open a clean file */
    h5_fixname(FILENAME[3], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to create file");
        goto error;
    }

    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        /* Insert */
        size = u % 30 + 100;
        memset(out, (int)('A' + u % 26), size);
        H5Eclear2(H5E_DEFAULT);
        status = H5HG_insert(f, size, out, obj + u);
        if (status < 0) {
            H5_FAILED();
            puts("    Unable to insert object into global heap");
            nerrors++;
        }

        /* Remove every third one beginning with the second, but after the
         * next one has already been inserted.  That is, insert A, B, C;
         * remove B, insert D, E, F; remove E; etc.
         */
        if (1 == (u % 3)) {
            H5Eclear2(H5E_DEFAULT);
            status = H5HG_remove(f, obj + u - 1);
            if (status < 0) {
                H5_FAILED();
                puts("    Unable to remove object");
                nerrors++;
            }
            memset(obj + u - 1, 0, sizeof *obj);
        }
    }

    /* Release buffer */
    free(obj);
    obj = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    if (obj)
        free(obj);
    return MAX(1, nerrors);
}

/*-------------------------------------------------------------------------
 * Function:    test_ooo_indices
 *
 * Purpose:     Tests that indices can be stored out of order.  This can
 *              happen when the indices "wrap around" due to many
 *              insertions and deletions (for example, from rewriting a
 *              VL dataset).
 *
 * Return:      Success:    0
 *
 *              Failure:    number of errors
 *
 *-------------------------------------------------------------------------
 */
static int
test_ooo_indices(hid_t fapl)
{
    hid_t    file = H5I_INVALID_HID;
    H5F_t   *f    = NULL;
    unsigned i, j;
    H5HG_t  *obj = NULL;
    herr_t   status;
    int      nerrors = 0;
    char     filename[1024];

    TESTING("out of order indices");

    if (NULL == (obj = (H5HG_t *)malloc(2000 * sizeof(*obj))))
        goto error;

    /* Open a clean file */
    h5_fixname(FILENAME[4], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to create file");
        goto error;
    }

    /* Alternately insert 1000 entries and remove the previous group of 1000
     * entries, until the indices wrap around.
     */
    for (i = 0; i < 66; i++) {
        /* Insert 1000 entries.  The index into the obj array will alternate up
         * and down by 1000 so the previous set of insertions is preserved and
         * can be deleted.
         */
        for (j = 1000 * ((~i & 1)); j < 1000 * ((~i & 1) + 1); j++) {
            H5Eclear2(H5E_DEFAULT);
            status = H5HG_insert(f, sizeof(j), &j, &obj[j]);
            if (status < 0)
                GHEAP_REPEATED_ERR("    Unable to insert object into global heap");

            /* Check that the index is as expected */
            if (obj[j].idx != ((1000 * i) + j - (1000 * ((~i & 1)))) % ((1U << 16) - 1) + 1)
                GHEAP_REPEATED_ERR("    Unexpected global heap index");
        }

        /* Remove the previous 1000 entries */
        if (i > 0)
            for (j = 1000 * (i & 1); j < 1000 * ((i & 1) + 1); j++) {
                H5Eclear2(H5E_DEFAULT);
                status = H5HG_remove(f, &obj[j]);
                if (status < 0)
                    GHEAP_REPEATED_ERR("    Unable to remove object from global heap");
            }
    }

    /* The indices should have "wrapped around" on the last iteration */
    assert(obj[534].idx == 65535);
    assert(obj[535].idx == 1);

    /* Reopen the file */
    if (H5Fclose(file) < 0)
        goto error;
    if ((file = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to open file");
        goto error;
    } /* end if */

    /* Read the objects to make sure the heap is still readable */
    for (i = 0; i < 1000; i++) {
        if (NULL == H5HG_read(f, &obj[i], &j, NULL))
            goto error;
        if (i != j) {
            H5_FAILED();
            puts("    Incorrect read value");
            goto error;
        }
    }

    if (H5Fclose(file) < 0)
        goto error;
    if (nerrors)
        goto error;
    free(obj);
    obj = NULL;

    PASSED();
    return 0;

error:
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    if (obj)
        free(obj);
    return MAX(1, nerrors);
} /* end test_ooo_indices */

/* Chunk-local H5HG tests adapted from the existing global heap tests.
 */

/*-------------------------------------------------------------------------
 * Function:    test_1_local
 *
 * Purpose:     Inserts objects whose sizes increase monotonically into one
 *              chunk-local heap and verifies that all payloads remain intact
 *              after repeated local-heap extensions.
 *
 *                                           -- AZO   7/09/26
 *-------------------------------------------------------------------------
 */
static int
test_1_local(hid_t fapl)
{
    hid_t        file    = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f       = NULL;            /* Internal file object */
    H5HG_heap_t *heap    = NULL;            /* Chunk-local heap under test */
    size_t      *obj_idx = NULL;            /* Local object indices */
    uint8_t      out[GHEAP_TEST_NOBJS];     /* Expected object contents */
    uint8_t      in[GHEAP_TEST_NOBJS];      /* Read-back buffer */
    size_t       u;                         /* Object index */
    size_t       size;                      /* Current payload size */
    int          nerrors = 0;               /* Number of failures */
    char         filename[1024];            /* Test file name */

    TESTING("local heap with monotonically increasing lengths");

    if (NULL == (obj_idx = (size_t *)calloc(GHEAP_TEST_NOBJS, sizeof(*obj_idx))))
        goto error;

    h5_fixname(FILENAME[0], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    /*
     * H5HG__insert_local() now inserts only into an existing heap.
     * Heap selection/creation belongs to the heap-set layer.
     */
    if (NULL == (heap = H5HG__create_local(f, 2 * H5HG_LOCAL_NORMAL_HEAP_SIZE))) {
        H5_FAILED();
        puts("    Unable to create chunk-local heap");
        goto error;
    }

    /* Insert progressively larger objects into the same local heap. */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u + 1;
        memset(out, (int)('A' + u % 26), size);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__insert_local(f, heap, size, out, &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into chunk-local heap");
            nerrors++;
        }
        else if (0 == obj_idx[u]) {
            H5_FAILED();
            puts("    Chunk-local heap returned reserved index zero");
            nerrors++;
        }
    }

    /* Verify all inserted objects. */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size_t buf_size;

        size = u + 1;
        memset(out, (int)('A' + u % 26), size);
        memset(in, 0, sizeof(in));

        buf_size = sizeof(in);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__read_local(f, heap, obj_idx[u], in, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to read object from chunk-local heap");
            nerrors++;
        }
        else if (memcmp(in, out, size) != 0) {
            H5_FAILED();
            puts("    Local heap value read does not match value written");
            nerrors++;
        }
    }

    if (H5HG__free_local(heap) < 0)
        goto error;
    heap = NULL;

    free(obj_idx);
    obj_idx = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heap)
        H5HG__free_local(heap);
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    free(obj_idx);
    return MAX(1, nerrors);
} /* end test_1_local() */

/*-------------------------------------------------------------------------
 * Function:    test_2_local
 *
 * Purpose:     Inserts objects whose sizes decrease monotonically into one
 *              chunk-local heap and verifies every payload.
 *
 *                                        -- AZO   7/09/26
 *-------------------------------------------------------------------------
 */
static int
test_2_local(hid_t fapl)
{
    hid_t        file    = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f       = NULL;            /* Internal file object */
    H5HG_heap_t *heap    = NULL;            /* Chunk-local heap under test */
    size_t      *obj_idx = NULL;            /* Local object indices */
    uint8_t      out[GHEAP_TEST_NOBJS];     /* Expected object contents */
    uint8_t      in[GHEAP_TEST_NOBJS];      /* Read-back buffer */
    size_t       u;                         /* Object index */
    size_t       size;                      /* Current payload size */
    int          nerrors = 0;               /* Number of failures */
    char         filename[1024];            /* Test file name */

    TESTING("local heap with monotonically decreasing lengths");

    if (NULL == (obj_idx = (size_t *)calloc(GHEAP_TEST_NOBJS, sizeof(*obj_idx))))
        goto error;

    h5_fixname(FILENAME[1], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    if (NULL == (heap = H5HG__create_local(f, 2 * H5HG_LOCAL_NORMAL_HEAP_SIZE))) {
        H5_FAILED();
        puts("    Unable to create chunk-local heap");
        goto error;
    }

    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = GHEAP_TEST_NOBJS - u;
        memset(out, (int)('A' + u % 26), size);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__insert_local(f, heap, size, out, &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into chunk-local heap");
            nerrors++;
        }
    }

    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size_t buf_size;

        size = GHEAP_TEST_NOBJS - u;
        memset(out, (int)('A' + u % 26), size);
        memset(in, 0, sizeof(in));

        buf_size = sizeof(in);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__read_local(f, heap, obj_idx[u], in, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to read object from chunk-local heap");
            nerrors++;
        }
        else if (memcmp(in, out, size) != 0) {
            H5_FAILED();
            puts("    Local heap value read does not match value written");
            nerrors++;
        }
    }

    if (H5HG__free_local(heap) < 0)
        goto error;
    heap = NULL;
    free(obj_idx);
    obj_idx = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heap)
        H5HG__free_local(heap);
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    free(obj_idx);
    return MAX(1, nerrors);
} /* end test_2_local() */

/*-------------------------------------------------------------------------
 * Function:    test_3_local
 *
 * Purpose:     Removes every payload from a chunk-local heap and confirms
 *              that no live object remains. The heap itself is explicitly
 *              freed because its lifetime belongs to the structured chunk.
 *
 *                                          -- AZO   7/10/26
 *-------------------------------------------------------------------------
 */
static int
test_3_local(hid_t fapl)
{
    hid_t        file    = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f       = NULL;            /* Internal file object */
    H5HG_heap_t *heap    = NULL;            /* Chunk-local heap under test */
    size_t      *obj_idx = NULL;            /* Local object indices */
    uint8_t      out[GHEAP_TEST_NOBJS];     /* Object contents */
    size_t       u;                         /* Object index */
    size_t       size;                      /* Current payload size */
    hbool_t      heap_empty = false;        /* Removal empty-state result */
    htri_t       is_empty;                  /* Independent empty check */
    int          nerrors = 0;               /* Number of failures */
    char         filename[1024];            /* Test file name */

    TESTING("complete chunk-local heap object removal");

    if (NULL == (obj_idx = (size_t *)calloc(GHEAP_TEST_NOBJS, sizeof(*obj_idx))))
        goto error;

    h5_fixname(FILENAME[2], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    if (NULL == (heap = H5HG__create_local(f, H5HG_LOCAL_NORMAL_HEAP_SIZE))) {
        H5_FAILED();
        puts("    Unable to create chunk-local heap");
        goto error;
    }

    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u % 30 + 100;
        memset(out, (int)('A' + u % 26), size);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__insert_local(f, heap, size, out, &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into chunk-local heap");
            nerrors++;
        }
    }

    /* The heap must become empty only after the last live object is removed. */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        heap_empty = false;

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__remove_local(f, heap, obj_idx[u], &heap_empty) < 0) {
            H5_FAILED();
            puts("    Unable to remove object from chunk-local heap");
            nerrors++;
        }
        else if (u + 1 < GHEAP_TEST_NOBJS && heap_empty) {
            H5_FAILED();
            puts("    Chunk-local heap reported empty before final removal");
            nerrors++;
        }
    }

    is_empty = H5HG__is_empty_local(heap);
    if (is_empty < 0) {
        H5_FAILED();
        puts("    Unable to determine whether chunk-local heap is empty");
        nerrors++;
    }
    else if (!heap_empty || !is_empty) {
        H5_FAILED();
        puts("    Chunk-local heap is not empty after all removals");
        nerrors++;
    }

    if (H5HG__free_local(heap) < 0)
        goto error;
    heap = NULL;
    free(obj_idx);
    obj_idx = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heap)
        H5HG__free_local(heap);
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    free(obj_idx);
    return MAX(1, nerrors);
} /* end test_3_local() */

/*-------------------------------------------------------------------------
 * Function:    test_4_local
 *
 * Purpose:     Interleaves insertion and removal to exercise repeated local
 *              heap compaction. Surviving objects are verified before they
 *              are removed, and the heap must be empty at the end.
 *
 *
 *                                                -- AZO   7/14/26
 *-------------------------------------------------------------------------
 */
static int
test_4_local(hid_t fapl)
{
    hid_t        file    = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f       = NULL;            /* Internal file object */
    H5HG_heap_t *heap    = NULL;            /* Chunk-local heap under test */
    size_t      *obj_idx = NULL;            /* Local object indices */
    bool        *live    = NULL;            /* Whether each object is live */
    uint8_t      out[GHEAP_TEST_NOBJS];     /* Expected object contents */
    uint8_t      in[GHEAP_TEST_NOBJS];      /* Read-back buffer */
    size_t       u;                         /* Object index */
    size_t       size;                      /* Current payload size */
    hbool_t      heap_empty = false;        /* Final removal result */
    htri_t       is_empty;                  /* Independent empty check */
    int          nerrors = 0;               /* Number of failures */
    char         filename[1024];            /* Test file name */

    TESTING("partial chunk-local heap object removal");

    if (NULL == (obj_idx = (size_t *)calloc(GHEAP_TEST_NOBJS, sizeof(*obj_idx))))
        goto error;
    if (NULL == (live = (bool *)calloc(GHEAP_TEST_NOBJS, sizeof(*live))))
        goto error;

    h5_fixname(FILENAME[3], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    if (NULL == (heap = H5HG__create_local(f, H5HG_LOCAL_NORMAL_HEAP_SIZE))) {
        H5_FAILED();
        puts("    Unable to create chunk-local heap");
        goto error;
    }

    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size = u % 30 + 100;
        memset(out, (int)('A' + u % 26), size);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__insert_local(f, heap, size, out, &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into chunk-local heap");
            nerrors++;
        }
        else
            live[u] = true;

        /* Match the original test's interleaved removal pattern. */
        if (1 == (u % 3)) {
            H5Eclear2(H5E_DEFAULT);
            if (H5HG__remove_local(f, heap, obj_idx[u - 1], NULL) < 0) {
                H5_FAILED();
                puts("    Unable to remove object from chunk-local heap");
                nerrors++;
            }
            else
                live[u - 1] = false;
        }
    }

    /* Compaction must not alter any object that remains live. */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        size_t buf_size;

        if (!live[u])
            continue;

        size = u % 30 + 100;
        memset(out, (int)('A' + u % 26), size);
        memset(in, 0, sizeof(in));

        buf_size = sizeof(in);

        H5Eclear2(H5E_DEFAULT);
        if (H5HG__read_local(f, heap, obj_idx[u], in, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to read surviving chunk-local heap object");
            nerrors++;
        }
        else if (memcmp(in, out, size) != 0) {
            H5_FAILED();
            puts("    Surviving chunk-local heap object was corrupted");
            nerrors++;
        }
    }

    /* Remove all surviving objects and verify the final empty state. */
    for (u = 0; u < GHEAP_TEST_NOBJS; u++) {
        if (!live[u])
            continue;

        heap_empty = false;
        H5Eclear2(H5E_DEFAULT);
        if (H5HG__remove_local(f, heap, obj_idx[u], &heap_empty) < 0) {
            H5_FAILED();
            puts("    Unable to remove surviving chunk-local heap object");
            nerrors++;
        }
        else
            live[u] = false;
    }

    is_empty = H5HG__is_empty_local(heap);
    if (is_empty < 0) {
        H5_FAILED();
        puts("    Unable to determine whether chunk-local heap is empty");
        nerrors++;
    }
    else if (!heap_empty || !is_empty) {
        H5_FAILED();
        puts("    Chunk-local heap is not empty after final removals");
        nerrors++;
    }

    if (H5HG__free_local(heap) < 0)
        goto error;
    heap = NULL;
    free(obj_idx);
    obj_idx = NULL;
    free(live);
    live = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heap)
        H5HG__free_local(heap);
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    free(obj_idx);
    free(live);
    return MAX(1, nerrors);
} /* end test_4_local() */

/*-------------------------------------------------------------------------
 * Function:    test_ooo_indices_local
 *
 * Purpose:     Forces the 16-bit local object index to wrap and reuse cleared
 *              entries. The heap is then encoded and decoded to verify that
 *              out-of-order indices survive the image round trip.
 *
 *                                          -- AZO   7/14/26
 *-------------------------------------------------------------------------
 */
static int
test_ooo_indices_local(hid_t fapl)
{
    hid_t        file      = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f         = NULL;            /* Internal file object */
    H5HG_heap_t *heap      = NULL;            /* Original local heap */
    H5HG_heap_t *decoded   = NULL;            /* Decoded local heap */
    size_t      *obj_idx   = NULL;            /* Current object indices */
    uint8_t     *image     = NULL;            /* Encoded heap image */
    size_t       image_len = 0;               /* Encoded image size */
    unsigned     i;                           /* Iteration group */
    unsigned     j;                           /* Object value/index */
    unsigned     value;                       /* Read-back value */
    size_t       expected;                    /* Expected local index */
    int          nerrors = 0;                 /* Number of failures */
    char         filename[1024];              /* Test file name */

    TESTING("chunk-local heap out-of-order indices");

    if (NULL == (obj_idx = (size_t *)calloc(2000, sizeof(*obj_idx))))
        goto error;

    h5_fixname(FILENAME[4], fapl, filename, sizeof filename);
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    if (NULL == (heap = H5HG__create_local(f, H5HG_LOCAL_NORMAL_HEAP_SIZE))) {
        H5_FAILED();
        puts("    Unable to create chunk-local heap");
        goto error;
    }

    /* Alternate two groups until never-used 16-bit indices are exhausted. */
    for (i = 0; i < 66; i++) {
        for (j = 1000 * ((~i & 1)); j < 1000 * ((~i & 1) + 1); j++) {
            H5Eclear2(H5E_DEFAULT);
            if (H5HG__insert_local(f, heap, sizeof(j), &j, &obj_idx[j]) < 0)
                GHEAP_REPEATED_ERR("    Unable to insert object into chunk-local heap");

            expected = ((1000 * i) + j - (1000 * ((~i & 1)))) % ((1U << 16) - 1) + 1;
            if (obj_idx[j] != expected)
                GHEAP_REPEATED_ERR("    Unexpected chunk-local heap object index");
        }

        if (i > 0)
            for (j = 1000 * (i & 1); j < 1000 * ((i & 1) + 1); j++) {
                H5Eclear2(H5E_DEFAULT);
                if (H5HG__remove_local(f, heap, obj_idx[j], NULL) < 0)
                    GHEAP_REPEATED_ERR("    Unable to remove object from chunk-local heap");
            }
    }

    assert(obj_idx[534] == 65535);
    assert(obj_idx[535] == 1);

    /*
     * Export the complete local heap image without accessing the opaque
     * H5HG_heap_t representation directly.
     */
    if (H5HG__encode_local(f, heap, &image, &image_len) < 0)
        goto error;

    if (H5HG__free_local(heap) < 0)
        goto error;
    heap = NULL;

    if (NULL == (decoded = H5HG__decode_local(f, image, image_len)))
        goto error;

    /* The final live group occupies obj_idx[0..999]. */
    for (i = 0; i < 1000; i++) {
        size_t buf_size = sizeof(value);

        value = 0;
        H5Eclear2(H5E_DEFAULT);

        if (H5HG__read_local(f, decoded, obj_idx[i], &value, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to read decoded chunk-local heap object");
            nerrors++;
            break;
        }

        if (i != value) {
            H5_FAILED();
            puts("    Incorrect value read from decoded chunk-local heap");
            nerrors++;
            break;
        }
    }

    image = H5MM_xfree(image);
    if (H5HG__free_local(decoded) < 0)
        goto error;
    decoded = NULL;
    free(obj_idx);
    obj_idx = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heap)
        H5HG__free_local(heap);
    if (decoded)
        H5HG__free_local(decoded);
    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY
    image = H5MM_xfree(image);
    free(obj_idx);
    return MAX(1, nerrors);
} /* end test_OOO_indices_local() */

/*-------------------------------------------------------------------------
 * Function:    test_encode_decode_local
 *
 * Purpose:     Verifies that a chunk-local heap can be converted to its
 *              serialized H5HG image and reconstructed as a fully usable
 *              in-memory heap.
 *
 *              The test performs the following sequence:
 *
 *                  1. Create a local heap containing four payload objects.
 *
 *                  2. Remove one object before encoding. This leaves a
 *                     cleared object-table entry and forces the encoded
 *                     image to represent a heap whose indices are not all
 *                     live.
 *
 *                  3. Encode the heap into an independent byte buffer and
 *                     destroy the original H5HG_heap_t. This ensures the
 *                     decoder cannot rely on pointers or memory owned by
 *                     the original heap.
 *
 *                  4. Decode a new heap from the byte buffer and verify that
 *                     every object that was live at encode time can still be
 *                     found by its original local index and has unchanged
 *                     contents.
 *
 *                  5. Verify that the removed object remains absent after
 *                     decoding.
 *
 *                  6. Insert a new object into the decoded heap and read it
 *                     back. This verifies that decode reconstructed mutable
 *                     heap state, including the free-space record, object
 *                     table, object pointers, nused, and nalloc.
 *
 *                  7. Encode and decode the modified heap a second time and
 *                     verify that the newly inserted object survives that
 *                     round trip.
 *
 *              The second round trip is important because it verifies that
 *              a heap produced by H5HG__decode_local() can subsequently be
 *              modified and re-serialized, rather than being usable only
 *              for read-only access.
 *
 * Return:      Success:    0
 *              Failure:    number of errors
 *
 *                                          -- AZO   7/14/26
 *-------------------------------------------------------------------------
 */
static int
test_encode_decode_local(hid_t fapl)
{
    /*
     * Initial payloads stored in the local heap. The second payload is
     * deliberately removed before encoding
     */
    static const char *values[] = {"first local payload", "payload removed before encoding",
                                   "third payload is deliberately longer than the first", "fourth"};

    hid_t        file        = H5I_INVALID_HID; /* Test file identifier */
    H5F_t       *f           = NULL;            /* Internal file object */
    H5HG_heap_t *heap        = NULL;            /* Original local heap */
    H5HG_heap_t *decoded     = NULL;            /* First decoded heap */
    H5HG_heap_t *decoded_2   = NULL;            /* Second decoded heap */
    size_t       obj_idx[4]  = {0, 0, 0, 0};    /* Original object indices */
    size_t       new_idx     = 0;               /* Post-decode object index */
    uint8_t     *image       = NULL;            /* First encoded image */
    uint8_t     *image_2     = NULL;            /* Second encoded image */
    size_t       image_len   = 0;               /* First image size */
    size_t       image_len_2 = 0;               /* Second image size */
    char         read_buf[128];                 /* Payload read-back buffer */
    const char  *new_value = "inserted after decode";
    size_t       u;                          /* Value index */
    bool         removed_reappeared = false; /* Removed-object check */
    int          nerrors            = 0;     /* Number of failures */
    char         filename[1024];             /* Test file name */

    TESTING("chunk-local heap encode/decode round trip");

    /*
     * Create a file so the local H5HG routines can use the file's configured
     * encoded length width. The local heap itself is not written as a normal
     * file-backed global heap.
     */
    h5_fixname(FILENAME[0], fapl, filename, sizeof filename);

    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;

    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    /*
     * H5HG__insert_local() now operates on an already-created member heap.
     */
    if (NULL == (heap = H5HG__create_local(f, H5HG_LOCAL_NORMAL_HEAP_SIZE))) {
        H5_FAILED();
        puts("    Unable to create chunk-local heap");
        goto error;
    }

    /*
     * Build the original local heap. Each returned index is retained because the same
     * index must identify the same surviving payload after decode.
     */
    for (u = 0; u < 4; u++) {
        H5Eclear2(H5E_DEFAULT);

        if (H5HG__insert_local(f, heap, (strlen(values[u]) + 1), values[u], &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object before local heap encoding");
            nerrors++;
        }
    }

    /*
     * Remove the second object before encoding. This creates a cleared
     * object-table slot and tests whether the serialized image preserves
     * the distinction between live and removed indices.
     */
    if (H5HG__remove_local(f, heap, obj_idx[1], NULL) < 0) {
        H5_FAILED();
        puts("    Unable to remove object before local heap encoding");
        nerrors++;
        goto error;
    }

    /*
     * Export the complete serialized H5HG collection image. The encoder
     * allocates IMAGE and retuns its exact IMAGE_LEN.
     */
    if (H5HG__encode_local(f, heap, &image, &image_len) < 0) {
        H5_FAILED();
        puts("    Unable to encode chunk-local heap");
        nerrors++;
        goto error;
    }

    /* Destroy the source heap so decode cannot rely on its storage. */
    if (H5HG__free_local(heap) < 0)
        goto error;

    heap = NULL;

    /* Reconstruct a new in-memory heap from the serialized image */
    if (NULL == (decoded = H5HG__decode_local(f, image, image_len))) {
        H5_FAILED();
        puts("    Unable to decode chunk-local heap");
        nerrors++;
        goto error;
    }

    /*
     * Verify the new NULL-destination size-query behavior.
     */
    {
        size_t buf_size = 0;

        H5Eclear2(H5E_DEFAULT);

        if (H5HG__read_local(f, decoded, obj_idx[0], NULL, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to query chunk-local heap object size");
            nerrors++;
        }
        else if (buf_size != strlen(values[0]) + 1) {
            H5_FAILED();
            puts("    Incorrect chunk-local heap object size returned");
            nerrors++;
        }
    }

    /*
     * Verify that an undersized destination fails and reports the required
     * buffer size without copying past the caller's capacity.
     */
    {
        char   small_buf[1];
        size_t buf_size = sizeof(small_buf);

        H5E_BEGIN_TRY
        {
            if (H5HG__read_local(f, decoded, obj_idx[0], small_buf, &buf_size) >= 0) {
                H5_FAILED();
                puts("    Undersized chunk-local heap read unexpectedly succeeded");
                nerrors++;
            }
        }
        H5E_END_TRY

        if (buf_size != strlen(values[0]) + 1) {
            H5_FAILED();
            puts("    Chunk-local heap read did not report required buffer size");
            nerrors++;
        }
    }

    /*
     * Verify all objects that were live when the heap was encoded.
     * Their original local indices and logical payload contents
     * must be preserved.
     */
    for (u = 0; u < 4; u++) {
        size_t buf_size;

        if (1 == u)
            continue;

        memset(read_buf, 0, sizeof(read_buf));
        buf_size = sizeof(read_buf);

        H5Eclear2(H5E_DEFAULT);

        if (H5HG__read_local(f, decoded, obj_idx[u], read_buf, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to read surviving object after local heap decode");
            nerrors++;
        }
        else if (strcmp(read_buf, values[u]) != 0) {
            H5_FAILED();
            puts("    Decoded payload does not match encoded payload");
            nerrors++;
        }
    }

    /*
     * Reading the removed index should fail. Supress the expected HDF5 error
     * stack while checking that the decoder did not recreate a cleared
     * object-table entry.
     */
    H5E_BEGIN_TRY
    {
        size_t buf_size = sizeof(read_buf);

        memset(read_buf, 0, sizeof(read_buf));
        removed_reappeared = (H5HG__read_local(f, decoded, obj_idx[1], read_buf, &buf_size) >= 0);
    }
    H5E_END_TRY

    if (removed_reappeared) {
        H5_FAILED();
        puts("    Removed local heap object reappeared after decode");
        nerrors++;
    }

    /* Direct per-heap insertion does not reserve or grow capacity.
     * Remove a record large enough to hold the new payload.
     */
    if (H5HG__remove_local(f, decoded, obj_idx[2], NULL) < 0) {
        H5_FAILED();
        puts("    Unable to make room in decoded chunk-local heap");
        nerrors++;
        goto error;
    }
    /*
     * A correctly decoded heap must support normal mutation. Insert a new
     * payload to verify that the decoder restored the free-space record,
     * object-table allocation, next-index state, and internal pointers.
     */
    if (H5HG__insert_local(f, decoded, strlen(new_value) + 1, new_value, &new_idx) < 0) {
        H5_FAILED();
        puts("    Unable to insert object into decoded chunk-local heap");
        nerrors++;
        goto error;
    }

    /* Verify the newly inserted payload immediately */
    {
        size_t buf_size = sizeof(read_buf);

        memset(read_buf, 0, sizeof(read_buf));

        if (H5HG__read_local(f, decoded, new_idx, read_buf, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to read object inserted after decode");
            nerrors++;
        }
        else if (strcmp(read_buf, new_value) != 0) {
            H5_FAILED();
            puts("    Object inserted after decode has incorrect contents");
            nerrors++;
        }
    }

    /*
     * Serialize the modified decoded heap. This verifies that a heap
     * created by H5HG__decode_local() remains compatible with the
     * normal local encode path after subsequent mutation.
     */
    if (H5HG__encode_local(f, decoded, &image_2, &image_len_2) < 0) {
        H5_FAILED();
        puts("    Unable to re-encode modified local heap");
        nerrors++;
        goto error;
    }

    /*
     * Destroy the first decoded heap so the second decode must once again
     * reconstruct all in-memory state solely from the serialized image.
     */
    if (H5HG__free_local(decoded) < 0)
        goto error;

    decoded = NULL;

    if (NULL == (decoded_2 = H5HG__decode_local(f, image_2, image_len_2))) {
        H5_FAILED();
        puts("    Unable to decode modified local heap image");
        nerrors++;
        goto error;
    }

    /*
     * Confirm that the payload inserted after the first decode survived the second
     * encode/decode cycle at the same local index.
     */
    {
        size_t buf_size = sizeof(read_buf);

        memset(read_buf, 0, sizeof(read_buf));

        if (H5HG__read_local(f, decoded_2, new_idx, read_buf, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to read post-decode insertion after second round trip");
            nerrors++;
        }
        else if (strcmp(read_buf, new_value) != 0) {
            H5_FAILED();
            puts("    Post-decode insertion did not survive second round trip");
            nerrors++;
        }
    }

    /* Release both serialized images returned by H5HG__encode_local() */
    image   = H5MM_xfree(image);
    image_2 = H5MM_xfree(image_2);

    if (H5HG__free_local(decoded_2) < 0)
        goto error;

    decoded_2 = NULL;

    if (H5Fclose(file) < 0)
        goto error;

    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    /*
     * Release whichever heap representations were successfully created
     * before the failure.
     */
    if (heap)
        H5HG__free_local(heap);

    if (decoded)
        H5HG__free_local(decoded);

    if (decoded_2)
        H5HG__free_local(decoded_2);

    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY

    /* H5HG__encode_local() allocates these images with H5MM_malloc() */
    image   = H5MM_xfree(image);
    image_2 = H5MM_xfree(image_2);

    return MAX(1, nerrors);
}

/*-------------------------------------------------------------------------
 * Function:    test_local_compact_serialization
 *
 * Purpose:     Regression test for omitting trailing unused heap capacity
 *              from the serialized image without shrinking the resident heap.
 *              Delete an interior object, check the returned image length
 *              against both its header and an independently calculated live
 *              record size, and verify that encoding changes neither resident
 *              storage nor heap-set allocation accounting. Decode both the
 *              member and the complete heap set, preserving surviving indices.
 *              Insert through the heap-set manager to force member growth,
 *              then check all surviving and newly inserted payloads.
 *
 *              Direct member insertion does not grow storage; growth must be
 *              exercised through H5HG__insert_local_heapset().
 *
 * Return:      Success:    0
 *              Failure:    1
 *
 *              Added for the compact-image length regression, 10/01/26.
 *-------------------------------------------------------------------------
 */
static int
test_local_compact_serialization(hid_t fapl)
{
    hid_t                 file    = H5I_INVALID_HID;
    H5F_t                *f       = NULL;
    H5HG_local_heapset_t *heapset = NULL, *decoded = NULL;
    H5HG_heap_t          *heap = NULL, *member = NULL;
    uint16_t              slot[3] = {0}, idx[3] = {0}, new_slot = 0, new_idx = 0;
    uint8_t               payload[512], read_buf[512];
    uint8_t              *image = NULL, *set_image = NULL, *snapshot = NULL;
    uint8_t              *large = NULL, *large_read = NULL;
    H5HG_obj_t           *table_snapshot = NULL;
    const uint8_t        *p;
    size_t                image_len = 0, set_len = 0, expected, capacity, free_size;
    size_t                alloc_size, nalloc, nused, nlive, buf_size, large_size, old_capacity;
    hsize_t               header_len = 0;
    bool                  removed_present;
    size_t                u;
    char                  filename[1024];

    TESTING("compact local heap serialization, resident preservation and reload growth");

    h5_fixname(FILENAME[0], fapl, filename, sizeof(filename));
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;
    if (NULL == (f = (H5F_t *)H5VL_object(file)))
        goto error;

    /* Three distinguishable objects must share one normal member heap. */
    for (u = 0; u < 3; u++) {
        memset(payload, (int)(0x31 + u), sizeof(payload));
        if (H5HG__insert_local_heapset(f, &heapset, sizeof(payload), payload, &slot[u], &idx[u]) < 0)
            goto error;
    }
    if (slot[0] != slot[1] || slot[0] != slot[2]) {
        puts("    Test setup did not place the objects in one member heap");
        goto error;
    }
    if (H5HG__remove_local_heapset(f, heapset, slot[1], idx[1]) < 0)
        goto error;

    heap       = heapset->heaps[slot[0]];
    capacity   = heap->size;
    free_size  = heap->obj[0].size;
    alloc_size = heapset->alloc_size;
    nalloc     = heap->nalloc;
    nused      = heap->nused;
    nlive      = heap->nlive;

    /* Include collection/object headers and required alignment, not spare capacity. */
    expected = H5HG_SIZEOF_HDR(f) + 2 * (H5HG_SIZEOF_OBJHDR(f) + H5HG_ALIGN(sizeof(payload)));
    if (free_size == 0 || free_size <= capacity / 2 || expected != capacity - free_size ||
        heap->obj[0].begin != heap->chunk + expected) {
        puts("    Test setup lacks substantial trailing free space after interior deletion");
        goto error;
    }

    if (NULL == (snapshot = (uint8_t *)malloc(capacity)) ||
        NULL == (table_snapshot = (H5HG_obj_t *)malloc(nalloc * sizeof(*table_snapshot))))
        goto error;
    memcpy(snapshot, heap->chunk, capacity);
    memcpy(table_snapshot, heap->obj, nalloc * sizeof(*table_snapshot));

    if (H5HG__encode_local(f, heap, &image, &image_len) < 0)
        goto error;
    if (!image || image_len < H5HG_SIZEOF_HDR(f))
        goto error;
    p = image + H5_SIZEOF_MAGIC + 4;
    H5F_DECODE_LENGTH(f, p, header_len);
    if (image_len != expected || header_len != (hsize_t)expected || image_len >= capacity) {
        puts("    Compact image length, header length and live record size disagree");
        goto error;
    }

    /* Exercise the enclosing encoder too: its compact-member check caught the bug. */
    if (H5HG__encode_local_heapset(f, heapset, &set_image, &set_len) < 0)
        goto error;
    if (heap->size != capacity || heap->nalloc != nalloc || heap->nused != nused || heap->nlive != nlive ||
        heap->obj[0].size != free_size || heapset->alloc_size != alloc_size ||
        memcmp(snapshot, heap->chunk, capacity) ||
        memcmp(table_snapshot, heap->obj, nalloc * sizeof(*table_snapshot))) {
        puts("    Encoding modified resident heap storage or allocation accounting");
        goto error;
    }

    if (NULL == (member = H5HG__decode_local(f, image, image_len)) ||
        NULL == (decoded = H5HG__decode_local_heapset(f, set_image, set_len)))
        goto error;

    /* Verify both decode entry points at the original object indices. */
    for (u = 0; u < 3; u++) {
        if (u == 1)
            continue;
        memset(payload, (int)(0x31 + u), sizeof(payload));
        buf_size = sizeof(read_buf);
        if (H5HG__read_local(f, member, idx[u], read_buf, &buf_size) < 0 || buf_size != sizeof(payload) ||
            memcmp(read_buf, payload, sizeof(payload)))
            goto error;
        buf_size = sizeof(read_buf);
        if (H5HG__read_local_heapset(f, decoded, slot[u], idx[u], read_buf, &buf_size) < 0 ||
            buf_size != sizeof(payload) || memcmp(read_buf, payload, sizeof(payload)))
            goto error;
    }
    H5E_BEGIN_TRY
    {
        buf_size        = sizeof(read_buf);
        removed_present = H5HG__read_local(f, member, idx[1], read_buf, &buf_size) >= 0;
        buf_size        = sizeof(read_buf);
        removed_present |= H5HG__read_local_heapset(f, decoded, slot[1], idx[1], read_buf, &buf_size) >= 0;
    }
    H5E_END_TRY
    if (removed_present) {
        puts("    Removed interior object reappeared after compact decode");
        goto error;
    }

    /* Choose a payload larger than the decoded capacity, forcing actual growth.
     * Keep it small enough that the manager can extend this normal member.
     */
    old_capacity = decoded->heaps[slot[0]]->size;
    if (old_capacity >= H5HG_LOCAL_NORMAL_HEAP_SIZE / 4)
        goto error;
    large_size = old_capacity + 1024;
    if (NULL == (large = (uint8_t *)malloc(large_size)) ||
        NULL == (large_read = (uint8_t *)malloc(large_size)))
        goto error;
    for (u = 0; u < large_size; u++)
        large[u] = (uint8_t)(u % 251);
    if (H5HG__insert_local_heapset(f, &decoded, large_size, large, &new_slot, &new_idx) < 0)
        goto error;
    if (new_slot != slot[0] || decoded->heaps[new_slot]->size <= old_capacity) {
        puts("    Post-decode insertion did not grow the existing member heap");
        goto error;
    }
    buf_size = large_size;
    if (H5HG__read_local_heapset(f, decoded, new_slot, new_idx, large_read, &buf_size) < 0 ||
        buf_size != large_size || memcmp(large_read, large, large_size))
        goto error;
    for (u = 0; u < 3; u++) {
        if (u == 1)
            continue;
        memset(payload, (int)(0x31 + u), sizeof(payload));
        buf_size = sizeof(read_buf);
        if (H5HG__read_local_heapset(f, decoded, slot[u], idx[u], read_buf, &buf_size) < 0 ||
            buf_size != sizeof(payload) || memcmp(read_buf, payload, sizeof(payload))) {
            puts("    Growing a decoded member changed a surviving payload");
            goto error;
        }
    }

    if (H5HG__free_local(member) < 0)
        goto error;
    member = NULL;
    if (H5HG__free_local_heapset(decoded) < 0)
        goto error;
    decoded = NULL;
    if (H5HG__free_local_heapset(heapset) < 0)
        goto error;
    heapset = NULL;
    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;
    free(snapshot);
    free(table_snapshot);
    free(large);
    free(large_read);
    image     = H5MM_xfree(image);
    set_image = H5MM_xfree(set_image);
    PASSED();
    return 0;

error:
    H5_FAILED();
    H5E_BEGIN_TRY
    {
        if (member)
            H5HG__free_local(member);
        if (decoded)
            H5HG__free_local_heapset(decoded);
        if (heapset)
            H5HG__free_local_heapset(heapset);
        if (file >= 0)
            H5Fclose(file);
    }
    H5E_END_TRY
    free(snapshot);
    free(table_snapshot);
    free(large);
    free(large_read);
    image     = H5MM_xfree(image);
    set_image = H5MM_xfree(set_image);
    return 1;
} /* end test_local_compact_serialization() */

/*-------------------------------------------------------------------------
 * Function:    test_heapset_basic
 *
 * Purpose:     Tests basic chunk-local heap-set insertion, lookup, removal,
 *              and empty-state behavior.
 *
 *              The test begins with a NULL heap-set pointer and verifies
 *              that the first insertion creates the heap set lazily.
 *              Several small objects are inserted and should share the
 *              same member heap, demonstrating that heap slot zero is a
 *              valid descriptor-visible slot.
 *
 *              The objects are read back through their returned
 *              {heap slot, object index} references. The test also inserts
 *              a zero-length payload and verifies that it receives a real,
 *              nonzero object index and that a NULL destination can be used
 *              to query its logical size.
 *
 *              Finally, all objects are removed and the heap set must report
 *              empty. Encoding the empty heap set must produce no logical
 *              H5_SECTION_VL image.
 *
 * Return:      Success:    0
 *              Failure:    number of errors
 *
 *                                              -- AZO   8/26/26
 *-------------------------------------------------------------------------
 */
static int
test_heapset_basic(hid_t fapl)
{
    static const char *values[] = {"first heap-set payload", "second heap-set payload",
                                   "third heap-set payload"};

    hid_t                 file         = H5I_INVALID_HID;
    H5F_t                *f            = NULL;
    H5HG_local_heapset_t *heapset      = NULL;
    uint16_t              heap_slot[4] = {0, 0, 0, 0};
    uint16_t              obj_idx[4]   = {0, 0, 0, 0};
    uint8_t              *image        = NULL;
    size_t                image_len    = 0;
    char                  read_buf[128];
    size_t                buf_size;
    htri_t                is_empty;
    size_t                u;
    int                   nerrors = 0;
    char                  filename[1024];

    TESTING("basic chunk-local heap-set operations");

    h5_fixname(FILENAME[0], fapl, filename, sizeof filename);

    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;

    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    /* Small objects should all fit in the first normal member heap. */
    for (u = 0; u < 3; u++) {
        if (H5HG__insert_local_heapset(f, &heapset, strlen(values[u]) + 1, values[u], &heap_slot[u],
                                       &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into chunk-local heap set");
            nerrors++;
            goto error;
        }

        if (0 != heap_slot[u] || 0 == obj_idx[u]) {
            H5_FAILED();
            puts("    Unexpected heap-set reference for small object");
            nerrors++;
        }
    }

    if (NULL == heapset) {
        H5_FAILED();
        puts("    Heap set was not created by insertion");
        nerrors++;
        goto error;
    }

    /* Read the normal payloads back through their composite references. */
    for (u = 0; u < 3; u++) {
        memset(read_buf, 0, sizeof(read_buf));
        buf_size = sizeof(read_buf);

        if (H5HG__read_local_heapset(f, heapset, heap_slot[u], obj_idx[u], read_buf, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to read object from chunk-local heap set");
            nerrors++;
        }
        else if (strcmp(read_buf, values[u]) != 0) {
            H5_FAILED();
            puts("    Heap-set value read does not match value written");
            nerrors++;
        }
    }

    /*
     * A zero-length non-null VL value still owns a real heap object and
     * therefore receives a nonzero object index.
     */
    if (H5HG__insert_local_heapset(f, &heapset, 0, NULL, &heap_slot[3], &obj_idx[3]) < 0) {
        H5_FAILED();
        puts("    Unable to insert zero-length heap-set object");
        nerrors++;
        goto error;
    }

    if (0 != heap_slot[3] || 0 == obj_idx[3]) {
        H5_FAILED();
        puts("    Invalid reference returned for zero-length object");
        nerrors++;
    }

    /* NULL destination queries the logical payload size. */
    buf_size = 1;

    if (H5HG__read_local_heapset(f, heapset, heap_slot[3], obj_idx[3], NULL, &buf_size) < 0) {
        H5_FAILED();
        puts("    Unable to query zero-length heap-set object");
        nerrors++;
    }
    else if (0 != buf_size) {
        H5_FAILED();
        puts("    Zero-length heap-set object reported nonzero size");
        nerrors++;
    }

    is_empty = H5HG__is_empty_local_heapset(heapset);
    if (is_empty < 0 || is_empty) {
        H5_FAILED();
        puts("    Non-empty heap set reported empty");
        nerrors++;
    }

    /* Remove every live object. */
    for (u = 0; u < 4; u++) {
        if (H5HG__remove_local_heapset(f, heapset, heap_slot[u], obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to remove object from chunk-local heap set");
            nerrors++;
            goto error;
        }
    }

    is_empty = H5HG__is_empty_local_heapset(heapset);
    if (is_empty <= 0) {
        H5_FAILED();
        puts("    Heap set did not become empty after all removals");
        nerrors++;
    }

    /* An empty heap set has no logical VL heap image. */
    if (H5HG__encode_local_heapset(f, heapset, &image, &image_len) < 0) {
        H5_FAILED();
        puts("    Unable to encode empty chunk-local heap set");
        nerrors++;
    }
    else if (NULL != image || 0 != image_len) {
        H5_FAILED();
        puts("    Empty heap set produced a non-empty image");
        nerrors++;
    }

    if (H5HG__free_local_heapset(heapset) < 0)
        goto error;
    heapset = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    image = H5MM_xfree(image);

    if (heapset)
        H5HG__free_local_heapset(heapset);

    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY

    return MAX(1, nerrors);
} /* end test_heapset_basic() */

/*-------------------------------------------------------------------------
 * Function:    test_heapset_multiple_heaps
 *
 * Purpose:     Tests creation of multiple bounded member heaps and the
 *              stable-slot rules used by chunk-local VL references.
 *
 *              Three payloads are chosen so that each object fits in one
 *              normal member heap, but two objects cannot coexist in the
 *              same normal heap. The three insertions must therefore return
 *              heap slots 0, 1, and 2.
 *
 *              The object in slot 1 is then removed. Slot 2 must remain
 *              valid at its original number because persistent VL
 *              references cannot be renumbered. A subsequent insertion
 *              must reuse the now-empty slot 1 instead of creating a new
 *              slot after slot 2.
 *
 *              The test therefore exercises bounded heap creation, growth
 *              of heap-set pointer capacity, preservation of interior holes,
 *              reuse of stable slots, and trailing-slot trimming entirely
 *              through the public heap-set behavior.
 *
 * Return:      Success:    0
 *              Failure:    number of errors
 *
 *                                              -- AZO   8/26/26
 *-------------------------------------------------------------------------
 */
static int
test_heapset_multiple_heaps(hid_t fapl)
{
    hid_t                 file             = H5I_INVALID_HID;
    H5F_t                *f                = NULL;
    H5HG_local_heapset_t *heapset          = NULL;
    uint8_t              *out              = NULL;
    uint8_t              *in               = NULL;
    size_t                payload_size     = H5HG_LOCAL_NORMAL_HEAP_SIZE / 2;
    uint16_t              heap_slot[3]     = {0, 0, 0};
    uint16_t              obj_idx[3]       = {0, 0, 0};
    uint16_t              replacement_slot = 0;
    uint16_t              replacement_idx  = 0;
    size_t                buf_size;
    size_t                u;
    htri_t                is_empty;
    int                   nerrors = 0;
    char                  filename[1024];

    TESTING("chunk-local heap-set stable slot reuse");

    if (NULL == (out = (uint8_t *)malloc(payload_size)))
        goto error;

    if (NULL == (in = (uint8_t *)malloc(payload_size)))
        goto error;

    h5_fixname(FILENAME[1], fapl, filename, sizeof filename);

    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;

    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    /*
     * Each payload fits by itself, but two cannot fit in one normal heap.
     * This forces creation of three member heaps.
     */
    for (u = 0; u < 3; u++) {
        memset(out, (int)('A' + u), payload_size);

        if (H5HG__insert_local_heapset(f, &heapset, payload_size, out, &heap_slot[u], &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert object into chunk-local heap set");
            nerrors++;
            goto error;
        }

        if (heap_slot[u] != (uint16_t)u || 0 == obj_idx[u]) {
            H5_FAILED();
            puts("    Unexpected stable heap slot");
            nerrors++;
            goto error;
        }
    }

    /*
     * Remove slot 1. Slot 2 must remain valid at the same stable number.
     */
    if (H5HG__remove_local_heapset(f, heapset, heap_slot[1], obj_idx[1]) < 0) {
        H5_FAILED();
        puts("    Unable to remove middle heap-set object");
        nerrors++;
        goto error;
    }

    memset(in, 0, payload_size);
    buf_size = payload_size;

    if (H5HG__read_local_heapset(f, heapset, heap_slot[2], obj_idx[2], in, &buf_size) < 0) {
        H5_FAILED();
        puts("    Stable slot changed after middle heap removal");
        nerrors++;
    }
    else {
        memset(out, 'C', payload_size);

        if (memcmp(in, out, payload_size) != 0) {
            H5_FAILED();
            puts("    Object in later stable slot was corrupted");
            nerrors++;
        }
    }

    /*
     * No active member has space for this object, so the manager should
     * reuse the empty stable slot 1.
     */
    memset(out, 'D', payload_size);

    if (H5HG__insert_local_heapset(f, &heapset, payload_size, out, &replacement_slot, &replacement_idx) < 0) {
        H5_FAILED();
        puts("    Unable to insert replacement heap-set object");
        nerrors++;
        goto error;
    }

    if (1 != replacement_slot || 0 == replacement_idx) {
        H5_FAILED();
        puts("    Empty stable heap slot was not reused");
        nerrors++;
    }

    memset(in, 0, payload_size);
    buf_size = payload_size;

    if (H5HG__read_local_heapset(f, heapset, replacement_slot, replacement_idx, in, &buf_size) < 0) {
        H5_FAILED();
        puts("    Unable to read replacement heap-set object");
        nerrors++;
    }
    else if (memcmp(in, out, payload_size) != 0) {
        H5_FAILED();
        puts("    Replacement heap-set payload is incorrect");
        nerrors++;
    }

    /*
     * Remove the trailing slot first, followed by slot 1 and slot 0.
     * This indirectly exercises trimming of unused trailing stable slots.
     */
    if (H5HG__remove_local_heapset(f, heapset, heap_slot[2], obj_idx[2]) < 0 ||
        H5HG__remove_local_heapset(f, heapset, replacement_slot, replacement_idx) < 0 ||
        H5HG__remove_local_heapset(f, heapset, heap_slot[0], obj_idx[0]) < 0) {
        H5_FAILED();
        puts("    Unable to remove remaining heap-set objects");
        nerrors++;
        goto error;
    }

    is_empty = H5HG__is_empty_local_heapset(heapset);
    if (is_empty <= 0) {
        H5_FAILED();
        puts("    Heap set did not become empty");
        nerrors++;
    }

    if (H5HG__free_local_heapset(heapset) < 0)
        goto error;
    heapset = NULL;

    free(out);
    out = NULL;
    free(in);
    in = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heapset)
        H5HG__free_local_heapset(heapset);

    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY

    free(out);
    free(in);

    return MAX(1, nerrors);
} /* end test_heapset_multiple_heaps() */

/*-------------------------------------------------------------------------
 * Function:    test_heapset_oversized
 *
 * Purpose:     Tests heap-set handling of a VL payload that is too large to
 *              fit in a normal bounded member heap.
 *
 *              The oversized payload must remain unsplit and be placed in a
 *              dedicated larger H5HG member heap. A later small payload must
 *              not reuse that dedicated oversized heap; it should create a
 *              normal member heap in the next stable slot. A second small
 *              payload should then reuse that normal member heap.
 *
 *              The test verifies this policy only through the returned
 *              stable heap-slot values and payload reads, without inspecting
 *              the internal heap-set or member-heap structures.
 *
 *              The oversized object is then removed while the normal heap
 *              remains live. The normal objects must continue to be
 *              accessible through their original references, demonstrating
 *              that stable slot numbering is preserved.
 *
 * Return:      Success:    0
 *              Failure:    number of errors
 *
 *                                              -- AZO   8/26/26
 *-------------------------------------------------------------------------
 */
static int
test_heapset_oversized(hid_t fapl)
{
    static const char *small_values[] = {"small payload one", "small payload two"};

    hid_t                 file           = H5I_INVALID_HID;
    H5F_t                *f              = NULL;
    H5HG_local_heapset_t *heapset        = NULL;
    uint8_t              *out            = NULL;
    uint8_t              *in             = NULL;
    size_t                oversized_size = H5HG_LOCAL_NORMAL_HEAP_SIZE;
    uint16_t              oversized_slot = 0;
    uint16_t              oversized_idx  = 0;
    uint16_t              small_slot[2]  = {0, 0};
    uint16_t              small_idx[2]   = {0, 0};
    size_t                buf_size;
    size_t                u;
    htri_t                is_empty;
    int                   nerrors = 0;
    char                  filename[1024];

    TESTING("chunk-local heap-set oversized object");

    if (NULL == (out = (uint8_t *)malloc(oversized_size)))
        goto error;

    if (NULL == (in = (uint8_t *)malloc(oversized_size)))
        goto error;

    h5_fixname(FILENAME[2], fapl, filename, sizeof filename);

    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;

    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    /*
     * Once H5HG headers are included, this object is larger than a normal
     * member heap and therefore requires a dedicated heap.
     */
    memset(out, 'X', oversized_size);

    if (H5HG__insert_local_heapset(f, &heapset, oversized_size, out, &oversized_slot, &oversized_idx) < 0) {
        H5_FAILED();
        puts("    Unable to insert oversized heap-set object");
        nerrors++;
        goto error;
    }

    if (0 != oversized_slot || 0 == oversized_idx) {
        H5_FAILED();
        puts("    Unexpected reference for oversized heap-set object");
        nerrors++;
    }

    /*
     * Normal payloads should use a separate normal heap. Both small
     * payloads should therefore return slot 1.
     */
    for (u = 0; u < 2; u++) {
        if (H5HG__insert_local_heapset(f, &heapset, strlen(small_values[u]) + 1, small_values[u],
                                       &small_slot[u], &small_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to insert normal heap-set object");
            nerrors++;
            goto error;
        }
    }

    if (1 != small_slot[0] || 1 != small_slot[1] || 0 == small_idx[0] || 0 == small_idx[1]) {
        H5_FAILED();
        puts("    Normal object incorrectly used oversized heap");
        nerrors++;
    }

    /* Verify the oversized payload. */
    memset(in, 0, oversized_size);
    buf_size = oversized_size;

    if (H5HG__read_local_heapset(f, heapset, oversized_slot, oversized_idx, in, &buf_size) < 0) {
        H5_FAILED();
        puts("    Unable to read oversized heap-set object");
        nerrors++;
    }
    else if (memcmp(in, out, oversized_size) != 0) {
        H5_FAILED();
        puts("    Oversized heap-set payload is incorrect");
        nerrors++;
    }

    /*
     * Removing slot zero must not change the references to the normal heap
     * in slot one.
     */
    if (H5HG__remove_local_heapset(f, heapset, oversized_slot, oversized_idx) < 0) {
        H5_FAILED();
        puts("    Unable to remove oversized heap-set object");
        nerrors++;
        goto error;
    }

    for (u = 0; u < 2; u++) {
        char small_buf[64];

        memset(small_buf, 0, sizeof(small_buf));
        buf_size = sizeof(small_buf);

        if (H5HG__read_local_heapset(f, heapset, small_slot[u], small_idx[u], small_buf, &buf_size) < 0) {
            H5_FAILED();
            puts("    Normal heap-set reference changed after removal");
            nerrors++;
        }
        else if (strcmp(small_buf, small_values[u]) != 0) {
            H5_FAILED();
            puts("    Normal heap-set payload is incorrect");
            nerrors++;
        }
    }

    for (u = 0; u < 2; u++) {
        if (H5HG__remove_local_heapset(f, heapset, small_slot[u], small_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to remove normal heap-set object");
            nerrors++;
            goto error;
        }
    }

    is_empty = H5HG__is_empty_local_heapset(heapset);
    if (is_empty <= 0) {
        H5_FAILED();
        puts("    Heap set did not become empty");
        nerrors++;
    }

    if (H5HG__free_local_heapset(heapset) < 0)
        goto error;
    heapset = NULL;

    free(out);
    out = NULL;
    free(in);
    in = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heapset)
        H5HG__free_local_heapset(heapset);

    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY

    free(out);
    free(in);

    return MAX(1, nerrors);
} /* end test_heapset_oversized() */

/*-------------------------------------------------------------------------
 * Function:    test_heapset_encode_decode
 *
 * Purpose:     Tests serialization and reconstruction of a chunk-local heap
 *              set containing multiple member heaps and an interior stable
 *              slot hole.
 *
 *              Three large objects are first inserted so that they occupy
 *              stable heap slots 0, 1, and 2. The object in slot 1 is then
 *              removed, leaving slot 2 active at its original number.
 *
 *              The complete heap set is encoded, the original in-memory
 *              heap set is destroyed, and a new heap set is decoded from
 *              the serialized image. Objects in slots 0 and 2 must remain
 *              readable using exactly the same composite references they
 *              had before encoding.
 *
 *              A new large object is then inserted into the decoded heap
 *              set. It must reuse the restored interior hole at slot 1.
 *              The modified heap set is encoded and decoded a second time
 *              to verify that a reconstructed heap set remains mutable and
 *              serializable and that all three stable references survive
 *              the second round trip.
 *
 *              This directly tests the outer heap-set directory, including
 *              preservation of stable slot numbers and NULL directory
 *              entries, while leaving individual H5HG image validation to
 *              the existing per-heap encode/decode tests.
 *
 * Return:      Success:    0
 *              Failure:    number of errors
 *
 *                                              -- AZO   8/26/26
 *-------------------------------------------------------------------------
 */
static int
test_heapset_encode_decode(hid_t fapl)
{
    hid_t                 file             = H5I_INVALID_HID;
    H5F_t                *f                = NULL;
    H5HG_local_heapset_t *heapset          = NULL;
    H5HG_local_heapset_t *decoded          = NULL;
    H5HG_local_heapset_t *decoded_2        = NULL;
    uint8_t              *out              = NULL;
    uint8_t              *in               = NULL;
    uint8_t              *image            = NULL;
    uint8_t              *image_2          = NULL;
    size_t                image_len        = 0;
    size_t                image_len_2      = 0;
    size_t                payload_size     = H5HG_LOCAL_NORMAL_HEAP_SIZE / 2;
    uint16_t              heap_slot[3]     = {0, 0, 0};
    uint16_t              obj_idx[3]       = {0, 0, 0};
    uint16_t              replacement_slot = 0;
    uint16_t              replacement_idx  = 0;
    size_t                buf_size;
    size_t                u;
    int                   nerrors = 0;
    char                  filename[1024];

    TESTING("chunk-local heap-set encode/decode round trip");

    if (NULL == (out = (uint8_t *)malloc(payload_size)))
        goto error;

    if (NULL == (in = (uint8_t *)malloc(payload_size)))
        goto error;

    h5_fixname(FILENAME[3], fapl, filename, sizeof filename);

    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        goto error;

    if (NULL == (f = (H5F_t *)H5VL_object(file))) {
        H5_FAILED();
        puts("    Unable to obtain internal file object");
        goto error;
    }

    /* Force three separate bounded member heaps. */
    for (u = 0; u < 3; u++) {
        memset(out, (int)('A' + u), payload_size);

        if (H5HG__insert_local_heapset(f, &heapset, payload_size, out, &heap_slot[u], &obj_idx[u]) < 0) {
            H5_FAILED();
            puts("    Unable to build heap set before encoding");
            nerrors++;
            goto error;
        }
    }

    if (0 != heap_slot[0] || 1 != heap_slot[1] || 2 != heap_slot[2]) {
        H5_FAILED();
        puts("    Unexpected heap-set slot layout before encoding");
        nerrors++;
        goto error;
    }

    /* Create an interior stable-slot hole before serialization. */
    if (H5HG__remove_local_heapset(f, heapset, heap_slot[1], obj_idx[1]) < 0) {
        H5_FAILED();
        puts("    Unable to remove middle heap-set object");
        nerrors++;
        goto error;
    }

    if (H5HG__encode_local_heapset(f, heapset, &image, &image_len) < 0) {
        H5_FAILED();
        puts("    Unable to encode chunk-local heap set");
        nerrors++;
        goto error;
    }

    if (NULL == image || 0 == image_len) {
        H5_FAILED();
        puts("    Non-empty heap set produced an empty image");
        nerrors++;
        goto error;
    }

    if (H5HG__free_local_heapset(heapset) < 0)
        goto error;
    heapset = NULL;

    if (NULL == (decoded = H5HG__decode_local_heapset(f, image, image_len))) {
        H5_FAILED();
        puts("    Unable to decode chunk-local heap set");
        nerrors++;
        goto error;
    }

    /*
     * Slots zero and two must still resolve using their original references.
     */
    for (u = 0; u < 3; u += 2) {
        memset(in, 0, payload_size);
        buf_size = payload_size;

        if (H5HG__read_local_heapset(f, decoded, heap_slot[u], obj_idx[u], in, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to read object after heap-set decode");
            nerrors++;
        }
        else {
            memset(out, (int)('A' + u), payload_size);

            if (memcmp(in, out, payload_size) != 0) {
                H5_FAILED();
                puts("    Decoded heap-set payload is incorrect");
                nerrors++;
            }
        }
    }

    /*
     * The restored hole at slot one should be reused by the next large
     * insertion.
     */
    memset(out, 'D', payload_size);

    if (H5HG__insert_local_heapset(f, &decoded, payload_size, out, &replacement_slot, &replacement_idx) < 0) {
        H5_FAILED();
        puts("    Unable to insert into decoded chunk-local heap set");
        nerrors++;
        goto error;
    }

    if (1 != replacement_slot || 0 == replacement_idx) {
        H5_FAILED();
        puts("    Decoded heap set did not restore the stable-slot hole");
        nerrors++;
    }

    /* Re-encode the modified decoded heap set. */
    if (H5HG__encode_local_heapset(f, decoded, &image_2, &image_len_2) < 0) {
        H5_FAILED();
        puts("    Unable to re-encode decoded chunk-local heap set");
        nerrors++;
        goto error;
    }

    if (H5HG__free_local_heapset(decoded) < 0)
        goto error;
    decoded = NULL;

    if (NULL == (decoded_2 = H5HG__decode_local_heapset(f, image_2, image_len_2))) {
        H5_FAILED();
        puts("    Unable to decode modified chunk-local heap set");
        nerrors++;
        goto error;
    }

    /*
     * Verify all three live references after the second round trip.
     */
    for (u = 0; u < 3; u++) {
        uint16_t slot;
        uint16_t idx;
        int      fill;

        if (0 == u) {
            slot = heap_slot[0];
            idx  = obj_idx[0];
            fill = 'A';
        }
        else if (1 == u) {
            slot = replacement_slot;
            idx  = replacement_idx;
            fill = 'D';
        }
        else {
            slot = heap_slot[2];
            idx  = obj_idx[2];
            fill = 'C';
        }

        memset(in, 0, payload_size);
        buf_size = payload_size;

        if (H5HG__read_local_heapset(f, decoded_2, slot, idx, in, &buf_size) < 0) {
            H5_FAILED();
            puts("    Unable to read object after second heap-set decode");
            nerrors++;
        }
        else {
            memset(out, fill, payload_size);

            if (memcmp(in, out, payload_size) != 0) {
                H5_FAILED();
                puts("    Heap-set payload did not survive second round trip");
                nerrors++;
            }
        }
    }

    image   = H5MM_xfree(image);
    image_2 = H5MM_xfree(image_2);

    if (H5HG__free_local_heapset(decoded_2) < 0)
        goto error;
    decoded_2 = NULL;

    free(out);
    out = NULL;
    free(in);
    in = NULL;

    if (H5Fclose(file) < 0)
        goto error;
    file = H5I_INVALID_HID;

    if (nerrors)
        goto error;

    PASSED();
    return 0;

error:
    if (heapset)
        H5HG__free_local_heapset(heapset);

    if (decoded)
        H5HG__free_local_heapset(decoded);

    if (decoded_2)
        H5HG__free_local_heapset(decoded_2);

    image   = H5MM_xfree(image);
    image_2 = H5MM_xfree(image_2);

    H5E_BEGIN_TRY
    {
        H5Fclose(file);
    }
    H5E_END_TRY

    free(out);
    free(in);

    return MAX(1, nerrors);
} /* end test_heapset_encode_decode() */

/*-------------------------------------------------------------------------
 * Function:	main
 *
 * Purpose:     Tests global heap.
 *
 * Return:      EXIT_SUCCESS/EXIT_FAILURE
 *
 *-------------------------------------------------------------------------
 */
int
main(void)
{
    int         nerrors        = 0;
    hid_t       fapl_id        = H5I_INVALID_HID;
    H5CX_node_t api_ctx        = {{0}, NULL}; /* API context node to push */
    bool        api_ctx_pushed = false;       /* Whether API context pushed */

    h5_test_init();
    if ((fapl_id = h5_fileaccess()) < 0)
        goto error;

    /* Push API context */
    if (H5CX_push(&api_ctx) < 0)
        FAIL_STACK_ERROR;
    api_ctx_pushed = true;

    nerrors += test_1(fapl_id);
    nerrors += test_2(fapl_id);
    nerrors += test_3(fapl_id);
    nerrors += test_4(fapl_id);
    nerrors += test_ooo_indices(fapl_id);

    if (nerrors)
        goto error;

    puts("All global heap tests passed.");

    /* Tests for the chunk-local heap */
    nerrors += test_1_local(fapl_id);
    nerrors += test_2_local(fapl_id);
    nerrors += test_3_local(fapl_id);
    nerrors += test_4_local(fapl_id);
    nerrors += test_ooo_indices_local(fapl_id);
    nerrors += test_encode_decode_local(fapl_id);
    nerrors += test_local_compact_serialization(fapl_id);

    nerrors += test_heapset_basic(fapl_id);
    nerrors += test_heapset_multiple_heaps(fapl_id);
    nerrors += test_heapset_oversized(fapl_id);
    nerrors += test_heapset_encode_decode(fapl_id);

    /* Verify symbol table messages are cached */
    nerrors += (h5_verify_cached_stabs(FILENAME, fapl_id) < 0 ? 1 : 0);

    if (nerrors)
        goto error;

    puts("All chunk-local heap tests passed.");

    /* Pop API context */
    if (api_ctx_pushed && H5CX_pop(false) < 0)
        FAIL_STACK_ERROR;
    api_ctx_pushed = false;

    h5_cleanup(FILENAME, fapl_id);
    exit(EXIT_SUCCESS);

error:
    H5E_BEGIN_TRY
    {
        H5Pclose(fapl_id);
    }
    H5E_END_TRY

    if (api_ctx_pushed)
        H5CX_pop(false);

    puts("*** TESTS FAILED ***");
    exit(EXIT_FAILURE);
} /* end main() */
