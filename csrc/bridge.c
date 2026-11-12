/* The CPython calls.  See bridge.h. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bridge.h"

int sl_bridge_python_running(void) { return Py_IsInitialized(); }

static PyObject *bridge_module(void) { return PyImport_ImportModule("socklight._bridge"); }

/* "<what>: <ExcType>: <message>" from the pending exception, which is cleared. */
static void exc_message(const char *what, char *err, size_t errlen) {
#if PY_VERSION_HEX >= 0x030C0000
    PyObject *exc = PyErr_GetRaisedException();
#else
    PyObject *type, *exc, *tb;
    PyErr_Fetch(&type, &exc, &tb);
    PyErr_NormalizeException(&type, &exc, &tb);
    Py_XDECREF(type);
    Py_XDECREF(tb);
#endif
    PyObject *s = exc ? PyObject_Str(exc) : NULL;
    const char *msg = s ? PyUnicode_AsUTF8(s) : NULL;
    snprintf(err, errlen, "%s: %s: %s", what, exc ? Py_TYPE(exc)->tp_name : "error", msg ? msg : "?");
    Py_XDECREF(s);
    Py_XDECREF(exc);
    PyErr_Clear();
}

static char *dup_str(PyObject *o) {
    const char *s = PyUnicode_Check(o) ? PyUnicode_AsUTF8(o) : NULL;
    if (!s) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "expected str");
        }
        return NULL;
    }
    char *d = strdup(s);
    if (!d) {
        PyErr_NoMemory();
    }
    return d;
}

static int parse_kind(PyObject *o, uint8_t *kind) {
    static const char *const names[] = {"real", "int", "str", "opt"};
    const char *s = PyUnicode_Check(o) ? PyUnicode_AsUTF8(o) : NULL;
    for (uint8_t k = 0; s && k < 4; k++) {
        if (!strcmp(s, names[k])) {
            *kind = k;
            return 0;
        }
    }
    if (!PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError, "unknown parameter kind %R", o);
    }
    return -1;
}

/* One snapshot entry: (name, (port, ...), ((pname, kind, default), ...)). */
static int parse_model(PyObject *entry, SlModel *out) {
    PyObject *name, *ports, *params;
    if (!PyArg_ParseTuple(entry, "UO!O!", &name, &PyTuple_Type, &ports, &PyTuple_Type, &params)) {
        return -1;
    }
    out->name = dup_str(name);
    out->n = (uint32_t)PyTuple_GET_SIZE(ports);
    out->m = (uint32_t)PyTuple_GET_SIZE(params);
    out->ports = calloc(out->n ? out->n : 1, sizeof(char *));
    out->params = calloc(out->m ? out->m : 1, sizeof(SlParam));
    if (!out->name || !out->ports || !out->params) {
        if (!PyErr_Occurred()) {
            PyErr_NoMemory();
        }
        return -1;
    }
    for (uint32_t i = 0; i < out->n; i++) {
        if (!(out->ports[i] = dup_str(PyTuple_GET_ITEM(ports, i)))) {
            return -1;
        }
    }
    for (uint32_t j = 0; j < out->m; j++) {
        PyObject *pname, *kind, *dflt;
        if (!PyArg_ParseTuple(PyTuple_GET_ITEM(params, j), "UUO", &pname, &kind, &dflt)) {
            return -1;
        }
        if (!(out->params[j].name = dup_str(pname)) || parse_kind(kind, &out->params[j].kind)) {
            return -1;
        }
    }
    return 0;
}

int sl_bridge_snapshot(SlModel **models, uint32_t *count, char *err, size_t errlen) {
    PyGILState_STATE gil = PyGILState_Ensure();
    int rc = -1;
    PyObject *mod = NULL, *res = NULL, *seq = NULL;
    *models = NULL;
    *count = 0;
    if (!(mod = bridge_module()) || !(res = PyObject_CallMethod(mod, "snapshot", NULL)) ||
        !(seq = PySequence_Fast(res, "snapshot() must return a list"))) {
        goto done;
    }
    Py_ssize_t k = PySequence_Fast_GET_SIZE(seq);
    *models = calloc(k ? (size_t)k : 1, sizeof(SlModel));
    if (!*models) {
        PyErr_NoMemory();
        goto done;
    }
    *count = (uint32_t)k;
    for (Py_ssize_t i = 0; i < k; i++) {
        if (parse_model(PySequence_Fast_GET_ITEM(seq, i), &(*models)[i])) {
            goto done;
        }
    }
    rc = 0;
done:
    if (rc) {
        exc_message("snapshot of the socklight registry failed", err, errlen);
    }
    Py_XDECREF(seq);
    Py_XDECREF(res);
    Py_XDECREF(mod);
    PyGILState_Release(gil);
    return rc;
}

int sl_bridge_evaluate(const char *model, const SlValue *vals, uint32_t nvals, double *out, size_t nout) {
    PyGILState_STATE gil = PyGILState_Ensure();
    int rc = -1;
    PyObject *mod = NULL, *values = NULL, *res = NULL, *seq = NULL;
    if (!(mod = bridge_module()) || !(values = PyDict_New())) {
        goto done;
    }
    for (uint32_t j = 0; j < nvals; j++) {
        PyObject *v = vals[j].kind == SL_STR ? PyUnicode_FromString(vals[j].str ? vals[j].str : "")
                                             : PyFloat_FromDouble(vals[j].real);
        if (!v || PyDict_SetItemString(values, vals[j].name, v)) {
            Py_XDECREF(v);
            goto done;
        }
        Py_DECREF(v);
    }
    if (!(res = PyObject_CallMethod(mod, "evaluate", "sO", model, values)) ||
        !(seq = PySequence_Fast(res, "evaluate() must return a list"))) {
        goto done;
    }
    if ((size_t)PySequence_Fast_GET_SIZE(seq) != nout) {
        PyErr_Format(PyExc_ValueError, "%s: evaluate() returned %zd values, expected %zu", model,
                     PySequence_Fast_GET_SIZE(seq), nout);
        goto done;
    }
    for (size_t i = 0; i < nout; i++) {
        out[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, (Py_ssize_t)i));
        if (out[i] == -1.0 && PyErr_Occurred()) {
            goto done;
        }
    }
    rc = 0;
done:
    /* The bridge recorded socklight.last_error() before re-raising. */
    PyErr_Clear();
    Py_XDECREF(seq);
    Py_XDECREF(res);
    Py_XDECREF(values);
    Py_XDECREF(mod);
    PyGILState_Release(gil);
    return rc;
}

void sl_bridge_set_load_error(const char *msg) {
    PyGILState_STATE gil = PyGILState_Ensure();
    PyObject *mod = bridge_module();
    PyObject *res = mod ? PyObject_CallMethod(mod, "set_load_error", "s", msg) : NULL;
    Py_XDECREF(res);
    Py_XDECREF(mod);
    PyErr_Clear();
    PyGILState_Release(gil);
}
