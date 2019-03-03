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
    struct __struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318 {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V1;
        

        __struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318() {
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
        ~__struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V3, PyObject* storage_V1) {
            Py_XINCREF(storage_V3);
Py_XINCREF(storage_V1);
            this->storage_V3 = storage_V3;
this->storage_V1 = storage_V1;
            



            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_3:

double __DUMMY_3;
__label_6:

double __DUMMY_6;

            Py_XDECREF(this->storage_V3);
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
// Op class Sum

        PyArrayObject* acc;
        
            typedef npy_float64 dtype_acc;
            
        acc = NULL;
        
        npy_float32* V3_iter;
        
                npy_intp V3_n1;
                ssize_t V3_stride1;
                int V3_jump1_0;
                
                npy_intp V3_n2;
                ssize_t V3_stride2;
                int V3_jump2_1;
                
                npy_intp V3_n0;
                ssize_t V3_stride0;
                int V3_jump0_2;
                

            if (PyArray_NDIM(V3) < 3) {
                PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_5;}
            }
            
                V3_n0 = PyArray_DIMS(V3)[0];
                V3_stride0 = PyArray_STRIDES(V3)[0] / sizeof(npy_float32);
                V3_jump0_2 = (V3_stride0) - (0);
                
                V3_n2 = PyArray_DIMS(V3)[2];
                V3_stride2 = PyArray_STRIDES(V3)[2] / sizeof(npy_float32);
                V3_jump2_1 = (V3_stride2) - (V3_n0*V3_stride0);
                
                V3_n1 = PyArray_DIMS(V3)[1];
                V3_stride1 = PyArray_STRIDES(V3)[1] / sizeof(npy_float32);
                V3_jump1_0 = (V3_stride1) - (V3_n2*V3_stride2);
                

        npy_float32* V1_iter;
        
                npy_intp V1_n0;
                ssize_t V1_stride0;
                int V1_jump0_0;
                
                npy_intp V1_n1;
                ssize_t V1_stride1;
                int V1_jump1_1;
                
                int V1_jumpx_2;
                
    {
        npy_intp dims[2];
        //npy_intp* dims = (npy_intp*)malloc(2 * sizeof(npy_intp));
        dims[0] = V3_n1;
dims[1] = V3_n2;

        if (!V1) {
            V1 = (PyArrayObject*)PyArray_EMPTY(2, dims,
                                                    NPY_FLOAT32,
                                                    0);
        }
        else {
            PyArray_Dims new_dims;
            new_dims.len = 2;
            new_dims.ptr = dims;
            PyObject* success = PyArray_Resize(V1, &new_dims, 0, NPY_CORDER);
            if (!success) {
                // If we can't resize the ndarray we have we can allocate a new one.
                PyErr_Clear();
                Py_XDECREF(V1);
                V1 = (PyArrayObject*)PyArray_EMPTY(2, dims, NPY_FLOAT32, 0);
            } else {
                Py_DECREF(success);
            }
        }
        if (!V1) {
            {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_5;}
        }
    }
    
            if (PyArray_NDIM(V1) < 2) {
                PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_5;}
            }
            
                V1_jumpx_2 = -(0);
                
                V1_n1 = PyArray_DIMS(V1)[1];
                V1_stride1 = PyArray_STRIDES(V1)[1] / sizeof(npy_float32);
                V1_jump1_1 = (V1_stride1) - (0);
                
                V1_n0 = PyArray_DIMS(V1)[0];
                V1_stride0 = PyArray_STRIDES(V1)[0] / sizeof(npy_float32);
                V1_jump0_0 = (V1_stride0) - (V1_n1*V1_stride1);
                
        npy_float64* acc_iter;
        
                npy_intp acc_n0;
                ssize_t acc_stride0;
                int acc_jump0_0;
                
                npy_intp acc_n1;
                ssize_t acc_stride1;
                int acc_jump1_1;
                
                int acc_jumpx_2;
                
    {
        npy_intp dims[2];
        //npy_intp* dims = (npy_intp*)malloc(2 * sizeof(npy_intp));
        dims[0] = V3_n1;
dims[1] = V3_n2;

        if (!acc) {
            acc = (PyArrayObject*)PyArray_EMPTY(2, dims,
                                                    NPY_FLOAT64,
                                                    0);
        }
        else {
            PyArray_Dims new_dims;
            new_dims.len = 2;
            new_dims.ptr = dims;
            PyObject* success = PyArray_Resize(acc, &new_dims, 0, NPY_CORDER);
            if (!success) {
                // If we can't resize the ndarray we have we can allocate a new one.
                PyErr_Clear();
                Py_XDECREF(acc);
                acc = (PyArrayObject*)PyArray_EMPTY(2, dims, NPY_FLOAT64, 0);
            } else {
                Py_DECREF(success);
            }
        }
        if (!acc) {
            {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_5;}
        }
    }
    
            if (PyArray_NDIM(acc) < 2) {
                PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_5;}
            }
            
                acc_jumpx_2 = -(0);
                
                acc_n1 = PyArray_DIMS(acc)[1];
                acc_stride1 = PyArray_STRIDES(acc)[1] / sizeof(npy_float64);
                acc_jump1_1 = (acc_stride1) - (0);
                
                acc_n0 = PyArray_DIMS(acc)[0];
                acc_stride0 = PyArray_STRIDES(acc)[0] / sizeof(npy_float64);
                acc_jump0_0 = (acc_stride0) - (acc_n1*acc_stride1);
                
{
        V3_iter = (npy_float32*)(PyArray_DATA(V3));
acc_iter = (npy_float64*)(PyArray_DATA(acc));

        for (int ITER_0 = acc_n0; ITER_0; ITER_0--) {
            
        
        for (int ITER_1 = acc_n1; ITER_1; ITER_1--) {
            
        npy_float64& acc_i = *acc_iter;
acc_i = 0;
        for (int ITER_2 = V3_n0; ITER_2; ITER_2--) {
            
        {
            npy_float32& V3_i = *V3_iter;

            acc_i = acc_i + V3_i;
        }
        
            V3_iter += V3_jump0_2;
acc_iter += acc_jumpx_2;

        }
        
            V3_iter += V3_jump2_1;
acc_iter += acc_jump1_1;

        }
        
            V3_iter += V3_jump1_0;
acc_iter += acc_jump0_0;

        }
        }

            PyArray_CopyInto(V1, acc);
            
        if (acc) {
            Py_XDECREF(acc);
        }
        __label_5:

double __DUMMY_5;

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
    

        static int __struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318_executor(__struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318 *self) {
            return self->run();
        }

        static void __struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318_destructor(PyObject *capsule) {
            __struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318 *self = (__struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318 *)PyCapsule_GetContext(capsule);
            delete self;
        }
        
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (3 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 3, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318* struct_ptr = new __struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
    PyObject* thunk = PyCapsule_New((void*)(&__struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318_executor), NULL, __struct_compiled_op_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318_destructor);
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
      "m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318",
      NULL,
      -1,
      MyMethods,
};

PyMODINIT_FUNC PyInit_m67ac91a458bd67bc5ba0486bfbc47c5f7b5b9af3840dd61bf6448a0ffab8d318(void) {
   import_array();
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
