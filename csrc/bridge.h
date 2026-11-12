/* The CPython calls: socklight._bridge.snapshot / evaluate / set_load_error.
 *
 * socklight.osdi is loaded by VACASK's Python extension inside a running
 * interpreter, so these attach to it (PyGILState_Ensure) and never initialise
 * one.  Every function takes and releases the GIL itself, and leaves no Python
 * exception set.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

enum { SL_REAL = 0, SL_INT = 1, SL_STR = 2, SL_OPT = 3 };

typedef struct {
    char *name;     /* original case, for calling back into Python */
    uint8_t kind;   /* SL_REAL, SL_INT, SL_STR or SL_OPT */
    double rdefault; /* signature default; NaN for an opt parameter (None) */
    char *sdefault; /* SL_STR: signature default, kept for the life of the process */
} SlParam;

typedef struct {
    char *name;   /* resource path, original case */
    uint32_t n;   /* ports */
    char **ports; /* sorted, as the registry snapshot gives them */
    uint32_t m;   /* parameters */
    SlParam *params;
} SlModel;

/* One given parameter value for evaluate(): `str` for SL_STR, else `real`. */
typedef struct {
    const char *name;
    uint8_t kind;
    double real;
    const char *str;
} SlValue;

/* Returns 1 if this process has a Python interpreter running. */
int sl_bridge_python_running(void);

/* snapshot(): fills a malloc'd array of `*count` models (strings strdup'd).
 * Returns 0, or -1 with a one-line message in `err`. */
int sl_bridge_snapshot(SlModel **models, uint32_t *count, char *err, size_t errlen);

/* evaluate(model, {name: value}) into `out` (`nout` doubles: interleaved
 * re/im of S, row-major over the snapshot's port order).  Returns 0, or -1
 * with a one-line message in `err`; socklight.last_error() then describes the
 * failure (recorded by the bridge for a Python exception, by us otherwise). */
int sl_bridge_evaluate(const char *model, const SlValue *vals, uint32_t nvals, double *out, size_t nout,
                       char *err, size_t errlen);

/* Records a C-side setup failure of `model` so socklight.last_error() returns it. */
void sl_bridge_record_error(const char *model, const char *msg);

/* set_load_error(msg), so socklight.load_error() returns it. */
void sl_bridge_set_load_error(const char *msg);
