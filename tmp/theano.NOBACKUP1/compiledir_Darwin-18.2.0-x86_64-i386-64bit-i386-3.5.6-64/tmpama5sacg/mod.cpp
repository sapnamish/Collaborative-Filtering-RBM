#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <vector>
#include <algorithm>
//////////////////////
////  Support Code
//////////////////////

    namespace {
    struct __struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V5;
PyObject* storage_V7;
PyObject* storage_V1;
        

        __struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e() {
            // This is only somewhat safe because we:
            //  1) Are not a virtual class
            //  2) Do not use any virtual classes in the members
            //  3) Deal with mostly POD and pointers

            // If this changes, we would have to revise this, but for
            // now I am tired of chasing segfaults because
            // initialization code had an error and some pointer has
            // a junk value.
            #ifndef THEANO_DONT_MEMSET_STRUCT
            memset(this, 0, sizeof(*this));
            #endif
        }
        ~__struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V3, PyObject* storage_V5, PyObject* storage_V7, PyObject* storage_V1) {
            Py_XINCREF(storage_V3);
Py_XINCREF(storage_V5);
Py_XINCREF(storage_V7);
Py_XINCREF(storage_V1);
            this->storage_V3 = storage_V3;
this->storage_V5 = storage_V5;
this->storage_V7 = storage_V7;
this->storage_V1 = storage_V1;
            





            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_3:

double __DUMMY_3;
__label_5:

double __DUMMY_5;
__label_7:

double __DUMMY_7;
__label_10:

double __DUMMY_10;

            Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V5);
Py_XDECREF(this->storage_V7);
Py_XDECREF(this->storage_V1);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
    
        PyArrayObject* V1;
        
            typedef npy_float32 dtype_V1;
            
    PyObject* py_V3;
    
        PyArrayObject* V3;
        
            typedef npy_float32 dtype_V3;
            
    PyObject* py_V5;
    
        PyArrayObject* V5;
        
            typedef npy_float32 dtype_V5;
            
    PyObject* py_V7;
    
        PyArrayObject* V7;
        
            typedef npy_float32 dtype_V7;
            
{

    py_V1 = PyList_GET_ITEM(storage_V1, 0);
    {Py_XINCREF(py_V1);}
    
        if (py_V1 == Py_None)
        {
            
        V1 = NULL;
        
        }
        else
        {
            
            V1 = NULL;
            if (py_V1 == Py_None) {
                // We can either fail here or set V1 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            if (!PyArray_Check(py_V1)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            // We expect NPY_FLOAT32
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V1)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V1;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT32), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT32,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V1),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V1) != NPY_FLOAT32) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT32) got %d",
                             NPY_FLOAT32, PyArray_TYPE((PyArrayObject*) py_V1));
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            
        V1 = (PyArrayObject*)(py_V1);
        Py_XINCREF(V1);
        
        }
        
{

    py_V3 = PyList_GET_ITEM(storage_V3, 0);
    {Py_XINCREF(py_V3);}
    
            V3 = NULL;
            if (py_V3 == Py_None) {
                // We can either fail here or set V3 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            if (!PyArray_Check(py_V3)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            // We expect NPY_FLOAT32
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V3)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V3;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT32), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT32,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V3),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V3) != NPY_FLOAT32) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT32) got %d",
                             NPY_FLOAT32, PyArray_TYPE((PyArrayObject*) py_V3));
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            
        V3 = (PyArrayObject*)(py_V3);
        Py_XINCREF(V3);
        
{

    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
            V5 = NULL;
            if (py_V5 == Py_None) {
                // We can either fail here or set V5 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            if (!PyArray_Check(py_V5)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            // We expect NPY_FLOAT32
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V5)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V5;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT32), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT32,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V5),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V5) != NPY_FLOAT32) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT32) got %d",
                             NPY_FLOAT32, PyArray_TYPE((PyArrayObject*) py_V5));
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            
        V5 = (PyArrayObject*)(py_V5);
        Py_XINCREF(V5);
        
{

    py_V7 = PyList_GET_ITEM(storage_V7, 0);
    {Py_XINCREF(py_V7);}
    
            V7 = NULL;
            if (py_V7 == Py_None) {
                // We can either fail here or set V7 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_8;}
            }
            if (!PyArray_Check(py_V7)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_8;}
            }
            // We expect NPY_FLOAT32
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V7)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V7;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT32), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT32,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V7),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_8;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V7) != NPY_FLOAT32) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT32) got %d",
                             NPY_FLOAT32, PyArray_TYPE((PyArrayObject*) py_V7));
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_8;}
            }
            
        V7 = (PyArrayObject*)(py_V7);
        Py_XINCREF(V7);
        
{
// Op class Elemwise

        npy_float32* V3_iter;
        
                npy_intp V3_n0;
                ssize_t V3_stride0;
                int V3_jump0_0;
                
                npy_intp V3_n1;
                ssize_t V3_stride1;
                int V3_jump1_1;
                
        npy_float32* V5_iter;
        
                npy_intp V5_n0;
                ssize_t V5_stride0;
                int V5_jump0_0;
                
                npy_intp V5_n1;
                ssize_t V5_stride1;
                int V5_jump1_1;
                
        npy_float32* V7_iter;
        
                npy_intp V7_n0;
                ssize_t V7_stride0;
                int V7_jump0_0;
                
                npy_intp V7_n1;
                ssize_t V7_stride1;
                int V7_jump1_1;
                

            if (PyArray_NDIM(V3) < 2) {
                PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_9;}
            }
            
                V3_n1 = PyArray_DIMS(V3)[1];
                V3_stride1 = PyArray_STRIDES(V3)[1] / sizeof(npy_float32);
                V3_jump1_1 = (V3_stride1) - (0);
                
                V3_n0 = PyArray_DIMS(V3)[0];
                V3_stride0 = PyArray_STRIDES(V3)[0] / sizeof(npy_float32);
                V3_jump0_0 = (V3_stride0) - (V3_n1*V3_stride1);
                
            if (PyArray_NDIM(V5) < 2) {
                PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_9;}
            }
            
                V5_n1 = PyArray_DIMS(V5)[1];
                V5_stride1 = PyArray_STRIDES(V5)[1] / sizeof(npy_float32);
                V5_jump1_1 = (V5_stride1) - (0);
                
                V5_n0 = PyArray_DIMS(V5)[0];
                V5_stride0 = PyArray_STRIDES(V5)[0] / sizeof(npy_float32);
                V5_jump0_0 = (V5_stride0) - (V5_n1*V5_stride1);
                
            if (PyArray_NDIM(V7) < 2) {
                PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_9;}
            }
            
                V7_n1 = PyArray_DIMS(V7)[1];
                V7_stride1 = PyArray_STRIDES(V7)[1] / sizeof(npy_float32);
                V7_jump1_1 = (V7_stride1) - (0);
                
                V7_n0 = PyArray_DIMS(V7)[0];
                V7_stride0 = PyArray_STRIDES(V7)[0] / sizeof(npy_float32);
                V7_jump0_0 = (V7_stride0) - (V7_n1*V7_stride1);
                
            if (V3_n0 != V5_n0)
            {
                PyErr_Format(PyExc_ValueError, "Input dimension mis-match. (input[%i].shape[%i] = %lld, input[%i].shape[%i] = %lld)",
                   0,
                   0,
                   (long long int) V3_n0,
                   1,
                   0,
                   (long long int) V5_n0
                );
                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_9;}
            }
            
            if (V3_n0 != V7_n0)
            {
                PyErr_Format(PyExc_ValueError, "Input dimension mis-match. (input[%i].shape[%i] = %lld, input[%i].shape[%i] = %lld)",
                   0,
                   0,
                   (long long int) V3_n0,
                   2,
                   0,
                   (long long int) V7_n0
                );
                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_9;}
            }
            
            if (V3_n1 != V5_n1)
            {
                PyErr_Format(PyExc_ValueError, "Input dimension mis-match. (input[%i].shape[%i] = %lld, input[%i].shape[%i] = %lld)",
                   0,
                   1,
                   (long long int) V3_n1,
                   1,
                   1,
                   (long long int) V5_n1
                );
                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_9;}
            }
            
            if (V3_n1 != V7_n1)
            {
                PyErr_Format(PyExc_ValueError, "Input dimension mis-match. (input[%i].shape[%i] = %lld, input[%i].shape[%i] = %lld)",
                   0,
                   1,
                   (long long int) V3_n1,
                   2,
                   1,
                   (long long int) V7_n1
                );
                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_9;}
            }
            

            if (V1) {
                Py_XDECREF(V1);
            }
            V1 = V7;
            Py_XINCREF(V1);
            

            if((PyArray_ISCONTIGUOUS(V3) && PyArray_ISCONTIGUOUS(V5) && PyArray_ISCONTIGUOUS(V7) && PyArray_ISCONTIGUOUS(V1)) || (PyArray_ISFORTRAN(V3) && PyArray_ISFORTRAN(V5) && PyArray_ISFORTRAN(V7) && PyArray_ISFORTRAN(V1))){
                
                    // All output have the same size
                    npy_intp n = PyArray_SIZE(V1);
                    
            dtype_V3 * V3_ptr = (dtype_V3*) PyArray_DATA(V3);
                            
            dtype_V5 * V5_ptr = (dtype_V5*) PyArray_DATA(V5);
                            
            dtype_V7 * V7_ptr = (dtype_V7*) PyArray_DATA(V7);
                            
            dtype_V1 * V1_ptr = (dtype_V1*) PyArray_DATA(V1);
                            
                    for(int i=0; i<n; i++){
                        
            dtype_V3& V3_i = V3_ptr[i];
                            
            dtype_V5& V5_i = V5_ptr[i];
                            
            dtype_V7& V7_i = V7_ptr[i];
                            
            dtype_V1& V1_i = V1_ptr[i];
                            
                        {
npy_float32 V9_tmp1;
V9_tmp1 = V7_i * V5_i;
npy_float32 V9_tmp2;
V9_tmp2 = V3_i * V5_i;
V1_i = V9_tmp2 - V9_tmp1;
}
;
                    }
                    
            }else{
                {

    std::vector< std::pair<int, int> > V7_loops(2);
    std::vector< std::pair<int, int> >::iterator V7_loops_it = V7_loops.begin();
    
            V7_loops_it->first = abs(PyArray_STRIDES(V7)[0]);
            
        V7_loops_it->second = 0;
        ++V7_loops_it;
        
            V7_loops_it->first = abs(PyArray_STRIDES(V7)[1]);
            
        V7_loops_it->second = 1;
        ++V7_loops_it;
        
    // rbegin and rend are reversed iterators, so this sorts in decreasing order
    std::sort(V7_loops.rbegin(), V7_loops.rend());
    

    int init_totals[2] = {V3_n0, V3_n1};
    
    V7_loops_it = V7_loops.begin();
    
        int TOTAL_0 = init_totals[V7_loops_it->second];
        ++V7_loops_it;
        
        int TOTAL_1 = init_totals[V7_loops_it->second];
        ++V7_loops_it;
        

    int init_strides[3][2] = {
        V3_stride0, V3_stride1, 
V5_stride0, V5_stride1, 
V7_stride0, V7_stride1
    };
    std::vector< std::pair<int, int> >::reverse_iterator V7_loops_rit;
    
        V7_loops_rit = V7_loops.rbegin();
            int V3_stride_l1 = init_strides[0][V7_loops_rit->second];
            ++V7_loops_rit;
            
            int V3_stride_l0 = init_strides[0][V7_loops_rit->second];
            ++V7_loops_rit;
            
        V7_loops_rit = V7_loops.rbegin();
            int V5_stride_l1 = init_strides[1][V7_loops_rit->second];
            ++V7_loops_rit;
            
            int V5_stride_l0 = init_strides[1][V7_loops_rit->second];
            ++V7_loops_rit;
            
        V7_loops_rit = V7_loops.rbegin();
            int V7_stride_l1 = init_strides[2][V7_loops_rit->second];
            ++V7_loops_rit;
            
            int V7_stride_l0 = init_strides[2][V7_loops_rit->second];
            ++V7_loops_rit;
            
V3_iter = (npy_float32*)(PyArray_DATA(V3));
V5_iter = (npy_float32*)(PyArray_DATA(V5));
V7_iter = (npy_float32*)(PyArray_DATA(V7));


        for(int ITER_0 = 0; ITER_0<TOTAL_0; ITER_0++)
        { // begin loop 0
            
            
        for(int ITER_1 = 0; ITER_1<TOTAL_1; ITER_1++)
        { // begin loop 1
            npy_float32 &V3_i = * ( V3_iter+V3_stride_l1*ITER_1+V3_stride_l0*ITER_0);
npy_float32 &V5_i = * ( V5_iter+V5_stride_l1*ITER_1+V5_stride_l0*ITER_0);
npy_float32 &V7_i = * ( V7_iter+V7_stride_l1*ITER_1+V7_stride_l0*ITER_0);

            
        {
            #define V1_i V7_i

            {
npy_float32 V9_tmp1;
V9_tmp1 = V7_i * V5_i;
npy_float32 V9_tmp2;
V9_tmp2 = V3_i * V5_i;
V1_i = V9_tmp2 - V9_tmp1;
}

            #undef V1_i

        }
        
        } // end loop 1
        
        } // end loop 0
        
}

            }
            __label_9:

double __DUMMY_9;

}
__label_8:

        if (V7) {
            Py_XDECREF(V7);
        }
        
    {Py_XDECREF(py_V7);}
    
double __DUMMY_8;

}
__label_6:

        if (V5) {
            Py_XDECREF(V5);
        }
        
    {Py_XDECREF(py_V5);}
    
double __DUMMY_6;

}
__label_4:

        if (V3) {
            Py_XDECREF(V3);
        }
        
    {Py_XDECREF(py_V3);}
    
double __DUMMY_4;

}
__label_2:

    if (!__failure) {
      
        {Py_XDECREF(py_V1);}
        if (!V1) {
            Py_INCREF(Py_None);
            py_V1 = Py_None;
        }
        else if ((void*)py_V1 != (void*)V1) {
            py_V1 = (PyObject*)V1;
        }

        {Py_XINCREF(py_V1);}

        if (V1 && !PyArray_ISALIGNED((PyArrayObject*) py_V1)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V1),
                         (long int) PyArray_NDIM(V1),
                         (long int) (PyArray_NDIM(V1) >= 3 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-3] : -1),
                         (long int) (PyArray_NDIM(V1) >= 2 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-2] : -1),
                         (long int) (PyArray_NDIM(V1) >= 1 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-1] : -1),
                         (long int) (PyArray_NDIM(V1) >= 3 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-3] : -1),
                         (long int) (PyArray_NDIM(V1) >= 2 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-2] : -1),
                         (long int) (PyArray_NDIM(V1) >= 1 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-1] : -1)
        );
            {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V1, 0);
      {Py_XINCREF(py_V1);}
      PyList_SET_ITEM(storage_V1, 0, py_V1);
      {Py_XDECREF(old);}
    }
    
        if (V1) {
            Py_XDECREF(V1);
        }
        
    {Py_XDECREF(py_V1);}
    
double __DUMMY_2;

}

            
        if (__failure) {
            // When there is a failure, this code puts the exception
            // in __ERROR.
            PyObject* err_type = NULL;
            PyObject* err_msg = NULL;
            PyObject* err_traceback = NULL;
            PyErr_Fetch(&err_type, &err_msg, &err_traceback);
            if (!err_type) {err_type = Py_None;Py_INCREF(Py_None);}
            if (!err_msg) {err_msg = Py_None; Py_INCREF(Py_None);}
            if (!err_traceback) {err_traceback = Py_None; Py_INCREF(Py_None);}
            PyObject* old_err_type = PyList_GET_ITEM(__ERROR, 0);
            PyObject* old_err_msg = PyList_GET_ITEM(__ERROR, 1);
            PyObject* old_err_traceback = PyList_GET_ITEM(__ERROR, 2);
            PyList_SET_ITEM(__ERROR, 0, err_type);
            PyList_SET_ITEM(__ERROR, 1, err_msg);
            PyList_SET_ITEM(__ERROR, 2, err_traceback);
            {Py_XDECREF(old_err_type);}
            {Py_XDECREF(old_err_msg);}
            {Py_XDECREF(old_err_traceback);}
        }
        // The failure code is returned to index what code block failed.
        return __failure;
        
        }
    };
    }
    

        static int __struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e_executor(__struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e *self) {
            return self->run();
        }

        static void __struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e_destructor(PyObject *capsule) {
            __struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e *self = (__struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e *)PyCapsule_GetContext(capsule);
            delete self;
        }
        
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (5 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 5, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e* struct_ptr = new __struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3),PyTuple_GET_ITEM(argtuple, 4) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
    PyObject* thunk = PyCapsule_New((void*)(&__struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e_executor), NULL, __struct_compiled_op_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e_destructor);
    if (thunk != NULL && PyCapsule_SetContext(thunk, struct_ptr) != 0) {
        PyErr_Clear();
        Py_DECREF(thunk);
        thunk = NULL;
    }

  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e",
      NULL,
      -1,
      MyMethods,
};

PyMODINIT_FUNC PyInit_m0ef49a1024e490887b27a2af8b03c24b027e06a5fc008a0d4e13ba4bcc676b1e(void) {
   import_array();
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
