/* The generic device.  See device.h. */
#include <complex.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "device.h"

#define AT(base, off, T) ((T)((char *)(base) + (off)))

/* ── layout ─────────────────────────────────────────────────────────────────── */

static uint64_t place(uint64_t *end, uint64_t align, uint64_t size) {
    uint64_t at = (*end + align - 1) / align * align;
    *end = at + size;
    return at;
}

static uint64_t round_size(uint64_t end) {
    /* never 0: VACASK aligned-allocates the size and treats NULL as out of memory */
    return end ? (end + 7) / 8 * 8 : 8;
}

int sl_layout(Layout *l) {
    const uint64_t nodes = 2 * (uint64_t)l->n, jac = nodes * nodes, m = l->m, words = (m + 31) / 32;
    uint64_t end = 0, off[10];
    off[0] = place(&end, alignof(uint32_t), nodes * sizeof(uint32_t));
    off[1] = place(&end, alignof(double *), jac * sizeof(double *));
    off[2] = place(&end, alignof(double *), jac * sizeof(double *));
    off[3] = place(&end, alignof(bool), sizeof(bool));
    off[4] = place(&end, alignof(double), nodes * sizeof(double));
    off[5] = place(&end, alignof(double), nodes * sizeof(double));
    off[6] = place(&end, alignof(double), jac * sizeof(double));
    off[7] = place(&end, alignof(double), m * sizeof(double));
    off[8] = place(&end, alignof(char *), m * sizeof(char *));
    off[9] = place(&end, alignof(uint32_t), words * sizeof(uint32_t));
    const uint64_t instance_size = round_size(end);

    uint64_t mend = 0;
    const uint64_t m_rvals = place(&mend, alignof(double), m * sizeof(double));
    const uint64_t m_svals = place(&mend, alignof(char *), m * sizeof(char *));
    const uint64_t m_given = place(&mend, alignof(uint32_t), words * sizeof(uint32_t));
    const uint64_t model_size = round_size(mend);

    if (instance_size > UINT32_MAX || model_size > UINT32_MAX) {
        return -1;
    }
    l->i_node_mapping = (uint32_t)off[0];
    l->i_jac_resist = (uint32_t)off[1];
    l->i_jac_react = (uint32_t)off[2];
    l->i_collapsed = (uint32_t)off[3];
    l->i_resid_resist = (uint32_t)off[4];
    l->i_resid_react = (uint32_t)off[5];
    l->i_g = (uint32_t)off[6];
    l->i_rvals = (uint32_t)off[7];
    l->i_svals = (uint32_t)off[8];
    l->i_given = (uint32_t)off[9];
    l->instance_size = (uint32_t)instance_size;
    l->m_rvals = (uint32_t)m_rvals;
    l->m_svals = (uint32_t)m_svals;
    l->m_given = (uint32_t)m_given;
    l->model_size = (uint32_t)model_size;
    return 0;
}

/* ── parameters ─────────────────────────────────────────────────────────────── */

/* Where a struct (instance or model) keeps its parameter values. */
typedef struct {
    char *base;
    uint32_t rvals, svals, given;
} Params;

static Params inst_params(const Layout *L, void *inst) {
    return (Params){(char *)inst, L->i_rvals, L->i_svals, L->i_given};
}

static Params model_params(const Layout *L, void *model) {
    return (Params){(char *)model, L->m_rvals, L->m_svals, L->m_given};
}

static bool is_given(Params p, uint32_t id) {
    return p.base && (AT(p.base, p.given, uint32_t *)[id / 32] >> (id % 32)) & 1u;
}

static void *param_ptr(const Layout *L, Params p, uint32_t id) {
    if (L->params[id].kind == SL_STR) {
        return AT(p.base, p.svals, char **) + id;
    }
    return AT(p.base, p.rvals, double *) + id;
}

void *sl_access(uint32_t slot, void *inst, void *model, uint32_t id, uint32_t flags) {
    const Layout *L = &sl_layouts[slot];
    Params p = (flags & ACCESS_FLAG_INSTANCE) ? inst_params(L, inst) : model_params(L, model);
    if (!p.base || id >= L->m) {
        return NULL;
    }
    if (flags & ACCESS_FLAG_SET) {
        AT(p.base, p.given, uint32_t *)[id / 32] |= 1u << (id % 32);
    }
    return param_ptr(L, p, id);
}

uint32_t sl_given_flag_model(uint32_t slot, void *model, uint32_t id) {
    const Layout *L = &sl_layouts[slot];
    return id < L->m && is_given(model_params(L, model), id);
}

uint32_t sl_given_flag_instance(uint32_t slot, void *inst, uint32_t id) {
    const Layout *L = &sl_layouts[slot];
    return id < L->m && is_given(inst_params(L, inst), id);
}

/* ── S -> Y ─────────────────────────────────────────────────────────────────── */

int sl_s_to_g(uint32_t n, const double *s, double *g) {
    /* Kurokawa power waves with an imaginary nudge on the reference impedance,
     * z0 = 1 + 1e-12j, conj(z0) on the diagonal: solve (z0 S + z0* I) Y = I - S. */
    const double complex z0 = CMPLX(1.0, 1e-12);
    const size_t nn = (size_t)n * n;
    double complex *a = malloc(nn * sizeof *a), *y = malloc(nn * sizeof *y);
    int rc = -1;
    if (!a || !y) {
        goto done;
    }
    for (uint32_t r = 0; r < n; r++) {
        for (uint32_t c = 0; c < n; c++) {
            const double complex sv = CMPLX(s[2 * (r * n + c)], s[2 * (r * n + c) + 1]);
            a[r * n + c] = z0 * sv + (r == c ? conj(z0) : 0.0);
            y[r * n + c] = (r == c ? 1.0 : 0.0) - sv;
        }
    }
    /* Gaussian elimination with partial pivoting, then back-substitution in place. */
    for (uint32_t k = 0; k < n; k++) {
        uint32_t piv = k;
        for (uint32_t r = k + 1; r < n; r++) {
            if (cabs(a[r * n + k]) > cabs(a[piv * n + k])) {
                piv = r;
            }
        }
        if (cabs(a[piv * n + k]) < 1e-150) {
            goto done;
        }
        if (piv != k) {
            for (uint32_t c = 0; c < n; c++) {
                double complex t = a[k * n + c];
                a[k * n + c] = a[piv * n + c];
                a[piv * n + c] = t;
                t = y[k * n + c];
                y[k * n + c] = y[piv * n + c];
                y[piv * n + c] = t;
            }
        }
        for (uint32_t r = k + 1; r < n; r++) {
            const double complex f = a[r * n + k] / a[k * n + k];
            for (uint32_t c = k; c < n; c++) {
                a[r * n + c] -= f * a[k * n + c];
            }
            for (uint32_t c = 0; c < n; c++) {
                y[r * n + c] -= f * y[k * n + c];
            }
        }
    }
    for (uint32_t r = n; r-- > 0;) {
        for (uint32_t c = 0; c < n; c++) {
            double complex acc = y[r * n + c];
            for (uint32_t k = r + 1; k < n; k++) {
                acc -= a[r * n + k] * y[k * n + c];
            }
            y[r * n + c] = acc / a[r * n + r];
        }
    }
    /* The real block of Y over (I, Q) node pairs. */
    const size_t nodes = 2 * (size_t)n;
    for (uint32_t r = 0; r < n; r++) {
        for (uint32_t c = 0; c < n; c++) {
            const double re = creal(y[r * n + c]), im = cimag(y[r * n + c]);
            g[(2 * r) * nodes + 2 * c] = re;
            g[(2 * r) * nodes + 2 * c + 1] = -im;
            g[(2 * r + 1) * nodes + 2 * c] = im;
            g[(2 * r + 1) * nodes + 2 * c + 1] = re;
        }
    }
    rc = 0;
done:
    free(a);
    free(y);
    return rc;
}

/* ── setup ──────────────────────────────────────────────────────────────────── */

static void init_ok(OsdiInitInfo *res) {
    res->flags = 0;
    res->num_errors = 0;
    res->errors = NULL;
}

void sl_setup_model(void *handle, void *model, OsdiSimParas *sim, OsdiInitInfo *res) {
    (void)handle;
    (void)model;
    (void)sim;
    init_ok(res);
}

void sl_setup_instance(uint32_t slot, void *handle, void *inst, void *model, double temperature,
                       uint32_t num_terminals, OsdiSimParas *sim, OsdiInitInfo *res) {
    (void)handle;
    (void)temperature;
    (void)num_terminals;
    (void)sim;
    init_ok(res);
    const Layout *L = &sl_layouts[slot];
    const Params ip = inst_params(L, inst), mp = model_params(L, model);
    const size_t nout = 2 * (size_t)L->n * L->n;
    SlValue *vals = calloc(L->m ? L->m : 1, sizeof *vals);
    double *s = malloc(nout * sizeof *s);
    const char *why = NULL;
    uint32_t nvals = 0;
    if (!vals || !s) {
        why = "out of memory";
        goto done;
    }
    /* Each parameter: instance given, else model given, else omitted (the
     * model's own signature default applies). */
    for (uint32_t j = 0; j < L->m; j++) {
        const Params *src = is_given(ip, j) ? &ip : is_given(mp, j) ? &mp : NULL;
        if (!src) {
            continue;
        }
        SlValue *v = &vals[nvals++];
        v->name = L->params[j].name;
        v->kind = L->params[j].kind;
        if (v->kind == SL_STR) {
            v->str = *(char **)param_ptr(L, *src, j);
        } else {
            v->real = *(double *)param_ptr(L, *src, j);
        }
    }
    if (sl_bridge_evaluate(L->name, vals, nvals, s, nout)) {
        why = "the model raised; socklight.last_error() has the traceback";
        goto done;
    }
    if (sl_s_to_g(L->n, s, AT(inst, L->i_g, double *))) {
        why = "I + S is singular, so the model has no admittance";
    }
done:
    if (why) {
        fprintf(stderr, "socklight.osdi: setup of %s failed: %s\n", L->name, why);
        res->flags = EVAL_RET_FLAG_FATAL;
    }
    free(vals);
    free(s);
}

/* ── eval and load: constant G, no Python ───────────────────────────────────── */

uint32_t sl_eval(uint32_t slot, void *handle, void *inst, void *model, OsdiSimInfo *info) {
    (void)handle;
    (void)model;
    const Layout *L = &sl_layouts[slot];
    const uint32_t nodes = 2 * L->n;
    const uint32_t *map = AT(inst, L->i_node_mapping, const uint32_t *);
    const double *g = AT(inst, L->i_g, const double *);
    double *resid = AT(inst, L->i_resid_resist, double *);
    for (uint32_t r = 0; r < nodes; r++) {
        double acc = 0.0;
        for (uint32_t c = 0; c < nodes; c++) {
            acc += g[r * nodes + c] * info->prev_solve[map[c]];
        }
        resid[r] = acc;
    }
    return 0;
}

void sl_load_residual_resist(uint32_t slot, void *inst, void *model, double *dst) {
    (void)model;
    const Layout *L = &sl_layouts[slot];
    const uint32_t *map = AT(inst, L->i_node_mapping, const uint32_t *);
    const double *resid = AT(inst, L->i_resid_resist, const double *);
    for (uint32_t k = 0; k < 2 * L->n; k++) {
        dst[map[k]] += resid[k];
    }
}

void sl_load_jacobian_resist(uint32_t slot, void *inst, void *model) {
    (void)model;
    const Layout *L = &sl_layouts[slot];
    const uint32_t entries = 4 * L->n * L->n;
    double *const *jac = AT(inst, L->i_jac_resist, double *const *);
    const double *g = AT(inst, L->i_g, const double *);
    for (uint32_t k = 0; k < entries; k++) {
        if (jac[k]) {
            *jac[k] += g[k];
        }
    }
}

void sl_load_residual_react(void *inst, void *model, double *dst) {
    (void)inst;
    (void)model;
    (void)dst;
}

void sl_load_jacobian_react(void *inst, void *model, double alpha) {
    (void)inst;
    (void)model;
    (void)alpha;
}
