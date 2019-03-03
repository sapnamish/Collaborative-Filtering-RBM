#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
//////////////////////
////  Support Code
//////////////////////

        /** ParamsType _Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57 **/
        #ifndef _PARAMS_DE7D1B721A1E0632B7CF04EDF5032C8ECFFA9F9A08492152B926F1A5A7E765D7_BC2229666B96007E875C5F62897EE5B7648DB2BAA5FABF3E771AFAC323AFBD57
        #define _PARAMS_DE7D1B721A1E0632B7CF04EDF5032C8ECFFA9F9A08492152B926F1A5A7E765D7_BC2229666B96007E875C5F62897EE5B7648DB2BAA5FABF3E771AFAC323AFBD57
        struct _Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57 {
            /* Attributes, */
            int _Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57_error;
            
                typedef npy_int64 dtype_i;
            
        npy_int64 i;
        

            /* Constructor. */
            _Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57() {
                _Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57_error = 0;
                
        i = 0;
        
            }

            /* Destructor. */
            ~_Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57() {
                // cleanup() is defined below.
                cleanup();
            }

            /* Cleanup method. */
            void cleanup() {
                
            }

            /* Extraction methods. */
            
            void extract_i(PyObject* py_i) {
                
            if (!PyObject_TypeCheck(py_i, &PyInt64ArrType_Type))
            {
                PyErr_Format(PyExc_ValueError,
                    "Scalar check failed (npy_int64)");
                {this->setErrorOccurred(); return;}
            }
            
        PyArray_ScalarAsCtype(py_i, &i);
        
            }
            

            /* Extract method. */
            
        void extract(PyObject* object, int field_pos) {
            switch(field_pos) {
                // Extraction cases.
                case 0: extract_i(object); break;
                // Default case.
                default:
                    PyErr_Format(PyExc_TypeError, "ParamsType: no extraction defined for a field %d.", field_pos);
                    this->setErrorOccurred();
                    break;
            }
        }
        

            /* Other methods. */
            void setErrorOccurred() {
                ++_Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57_error;
            }
            int errorOccurred() {
                return _Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57_error;
            }
        };
        #endif
        /** End ParamsType _Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57 **/
        

    namespace {
    struct __struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V1;
PyObject* storage_V5;
        
    PyObject* py_V5;
    
        _Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57* V5;
        

        __struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d() {
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
        ~__struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V3, PyObject* storage_V1, PyObject* storage_V5) {
            Py_XINCREF(storage_V3);
Py_XINCREF(storage_V1);
Py_XINCREF(storage_V5);
            this->storage_V3 = storage_V3;
this->storage_V1 = storage_V1;
this->storage_V5 = storage_V5;
            



    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
        /* Seems c_init() is not called for a op param. So I call `new` here. */
        V5 = new _Params_de7d1b721a1e0632b7cf04edf5032c8ecffa9f9a08492152b926f1a5a7e765d7_bc2229666b96007e875c5f62897ee5b7648db2baa5fabf3e771afac323afbd57;

        { // This need a separate namespace for Clinker
        const char* fields[] = {"i"};
        if (py_V5 == Py_None) {
            PyErr_SetString(PyExc_ValueError, "ParamsType: expected an object, not None.");
            {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        return 5;
}
        }
        for (int i = 0; i < 1; ++i) {
            PyObject* o = PyDict_GetItemString(py_V5, fields[i]);
            if (o == NULL) {
                PyErr_Format(PyExc_TypeError, "ParamsType: missing expected attribute \"%s\" in object.", fields[i]);
                {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        return 5;
}
            }
            V5->extract(o, i);
            if (V5->errorOccurred()) {
                /* The extract code from attribute type should have already raised a Python exception,
                 * so we just print the attribute name in stderr. */
                fprintf(stderr, "\nParamsType: error when extracting value for attribute \"%s\".\n", fields[i]);
                {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        return 5;
}
            }
        }
        }
        

            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_3:

double __DUMMY_3;
__label_5:

        delete V5;
        V5 = NULL;
        
    {Py_XDECREF(py_V5);}
    
double __DUMMY_5;
__label_8:

double __DUMMY_8;

            Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V1);
Py_XDECREF(this->storage_V5);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
    
        PyArrayObject* V1;
        
    PyObject* py_V3;
    
        PyArrayObject* V3;
        
{

    py_V1 = PyList_GET_ITEM(storage_V1, 0);
    {Py_XINCREF(py_V1);}
    
        if (py_V1 == Py_None)
        {
            
        V1 = NULL;
        
        }
        else
        {
            
        V1 = (PyArrayObject*)(py_V1);
        Py_XINCREF(V1);
        
        }
        
{

    py_V3 = PyList_GET_ITEM(storage_V3, 0);
    {Py_XINCREF(py_V3);}
    
        V3 = (PyArrayObject*)(py_V3);
        Py_XINCREF(V3);
        
{

{
// Op class Shape_i

    if (V5->i>=PyArray_NDIM(V3)){
        PyErr_SetString(PyExc_TypeError,
            "Number of dimensions lower than expected");
        {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;}
    }
    
    if(!V1)
        V1=(PyArrayObject*)PyArray_EMPTY(0, NULL, NPY_INT64, 0);
    ((npy_int64*)PyArray_DATA(V1))[0]=PyArray_DIMS(V3)[V5->i];
    __label_7:

double __DUMMY_7;

}
__label_6:

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
    

        static int __struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d_executor(__struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d *self) {
            return self->run();
        }

        static void __struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d_destructor(PyObject *capsule) {
            __struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d *self = (__struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d *)PyCapsule_GetContext(capsule);
            delete self;
        }
        
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (4 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 4, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d* struct_ptr = new __struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
    PyObject* thunk = PyCapsule_New((void*)(&__struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d_executor), NULL, __struct_compiled_op_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d_destructor);
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
      "m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d",
      NULL,
      -1,
      MyMethods,
};

PyMODINIT_FUNC PyInit_m60a068f5e4bfe48344c4e636f49e249da0f0a2df673bf7ffb57b86690cf56e0d(void) {
   import_array();
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
