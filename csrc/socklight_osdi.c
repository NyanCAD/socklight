/* socklight.osdi: one OSDI device per model in socklight's Python registry.
 *
 * VACASK's Python extension dlopens this file inside a running interpreter.
 * The constructor below snapshots the registry (socklight._bridge.snapshot)
 * and fills one descriptor per model.  VACASK reads the *address* of
 * OSDI_DESCRIPTORS as the array base right after dlopen, so the table is a
 * static array of SOCKLIGHT_MAX_MODELS slots and OSDI_NUM_DESCRIPTORS says how
 * many are filled.
 *
 * OSDI callbacks do not receive their descriptor, so every slot gets its own
 * set of trampolines (generated from sl_slots.inc) that forward the slot index
 * to the generic device in device.c.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "bridge.h"
#include "device.h"
#include "osdi.h"

#ifndef SOCKLIGHT_MAX_MODELS
#error "SOCKLIGHT_MAX_MODELS must be defined (CMake cache variable)"
#endif

#define SL_EXPORT __attribute__((visibility("default")))

SL_EXPORT OsdiDescriptor04 OSDI_DESCRIPTORS[SOCKLIGHT_MAX_MODELS];
SL_EXPORT uint32_t OSDI_NUM_DESCRIPTORS = 0;
SL_EXPORT uint32_t OSDI_DESCRIPTOR_SIZE = (uint32_t)sizeof(OsdiDescriptor04);
SL_EXPORT uint32_t OSDI_VERSION_MAJOR = 0;
SL_EXPORT uint32_t OSDI_VERSION_MINOR = 4;

Layout sl_layouts[SOCKLIGHT_MAX_MODELS];

/* ── per-slot trampolines ───────────────────────────────────────────────────── */

#define SL_SLOT(k)                                                                                   \
    static void *access_##k(void *i, void *m, uint32_t id, uint32_t f) {                             \
        return sl_access(k, i, m, id, f);                                                            \
    }                                                                                                \
    static void setup_model_##k(void *h, void *m, OsdiSimParas *sp, OsdiInitInfo *r) {                \
        sl_setup_model(k, h, m, sp, r);                                                              \
    }                                                                                                \
    static uint32_t given_model_##k(void *m, uint32_t id) { return sl_given_flag_model(k, m, id); }  \
    static uint32_t given_inst_##k(void *i, uint32_t id) { return sl_given_flag_instance(k, i, id); } \
    static void setup_##k(void *h, void *i, void *m, double t, uint32_t nt, OsdiSimParas *sp,        \
                          OsdiInitInfo *r) {                                                         \
        sl_setup_instance(k, h, i, m, t, nt, sp, r);                                                 \
    }                                                                                                \
    static uint32_t eval_##k(void *h, void *i, void *m, OsdiSimInfo *info) {                         \
        return sl_eval(k, h, i, m, info);                                                            \
    }                                                                                                \
    static void resid_##k(void *i, void *m, double *dst) { sl_load_residual_resist(k, i, m, dst); }  \
    static void jac_##k(void *i, void *m) { sl_load_jacobian_resist(k, i, m); }
#include "sl_slots.inc"
#undef SL_SLOT

typedef struct {
    void *(*access)(void *, void *, uint32_t, uint32_t);
    void (*setup_model)(void *, void *, OsdiSimParas *, OsdiInitInfo *);
    uint32_t (*given_model)(void *, uint32_t);
    uint32_t (*given_inst)(void *, uint32_t);
    void (*setup)(void *, void *, void *, double, uint32_t, OsdiSimParas *, OsdiInitInfo *);
    uint32_t (*eval)(void *, void *, void *, OsdiSimInfo *);
    void (*resid)(void *, void *, double *);
    void (*jac)(void *, void *);
} SlotFns;

#define SL_SLOT(k) {access_##k, setup_model_##k, given_model_##k, given_inst_##k, setup_##k, eval_##k, resid_##k, jac_##k},
static const SlotFns SLOT_FNS[SOCKLIGHT_MAX_MODELS] = {
#include "sl_slots.inc"
};
#undef SL_SLOT

/* ── descriptors ────────────────────────────────────────────────────────────── */

static char *suffixed(const char *port, const char *sfx) {
    size_t len = strlen(port) + strlen(sfx) + 1;
    char *s = malloc(len);
    if (s) {
        snprintf(s, len, "%s%s", port, sfx);
    }
    return s;
}

/* Fills slot `k` from `sm`, taking ownership of its strings. */
static int fill(uint32_t k, const SlModel *sm, char *err, size_t errlen) {
    Layout *L = &sl_layouts[k];
    L->name = sm->name;
    L->n = sm->n;
    L->m = sm->m;
    L->params = sm->params;
    if (sm->n == 0) {
        snprintf(err, errlen, "model %s has no ports", sm->name);
        return -1;
    }
    if (sl_layout(L)) {
        snprintf(err, errlen, "model %s is too large (%u ports, %u parameters)", sm->name, sm->n, sm->m);
        return -1;
    }
    const uint32_t nodes = 2 * sm->n, entries = nodes * nodes;
    OsdiNode *node = calloc(nodes, sizeof *node);
    OsdiJacobianEntry *jac = calloc(entries, sizeof *jac);
    OsdiParamOpvar *par = calloc(sm->m ? sm->m : 1, sizeof *par);
    char **pnames = calloc(sm->m ? sm->m : 1, sizeof *pnames);
    if (!node || !jac || !par || !pnames) {
        goto oom;
    }
    for (uint32_t t = 0; t < nodes; t++) {
        if (!(node[t].name = suffixed(sm->ports[t / 2], t % 2 ? "_q" : "_i"))) {
            goto oom;
        }
        node[t].units = (char *)"V";
        node[t].residual_units = (char *)"A";
        node[t].resist_residual_off = L->i_resid_resist + t * (uint32_t)sizeof(double);
        node[t].react_residual_off = L->i_resid_react + t * (uint32_t)sizeof(double);
        node[t].resist_limit_rhs_off = UINT32_MAX;
        node[t].react_limit_rhs_off = UINT32_MAX;
        node[t].is_flow = false;
    }
    for (uint32_t e = 0; e < entries; e++) {
        jac[e].nodes.node_1 = e / nodes;
        jac[e].nodes.node_2 = e % nodes;
        jac[e].react_ptr_off = L->i_jac_react + e * (uint32_t)sizeof(double *);
        jac[e].flags = JACOBIAN_ENTRY_RESIST;
    }
    for (uint32_t j = 0; j < sm->m; j++) {
        pnames[j] = sm->params[j].name;
        par[j].name = &pnames[j];
        par[j].num_alias = 0;
        par[j].description = (char *)"";
        par[j].units = (char *)"";
        par[j].flags = PARA_KIND_INST | (sm->params[j].kind == SL_STR ? PARA_TY_STR : PARA_TY_REAL);
        par[j].len = 0;
    }
    const SlotFns *f = &SLOT_FNS[k];
    OSDI_DESCRIPTORS[k] = (OsdiDescriptor04){
        .name = sm->name,
        .num_nodes = nodes,
        .num_terminals = nodes,
        .nodes = node,
        .num_jacobian_entries = entries,
        .jacobian_entries = jac,
        .num_collapsible = 0,
        .collapsible = NULL,
        .collapsed_offset = L->i_collapsed,
        .noise_sources = NULL,
        .num_noise_src = 0,
        .num_params = sm->m,
        .num_instance_params = sm->m,
        .num_opvars = 0,
        .param_opvar = par,
        .node_mapping_offset = L->i_node_mapping,
        .jacobian_ptr_resist_offset = L->i_jac_resist,
        .num_states = 0,
        .state_idx_off = 0,
        .bound_step_offset = UINT32_MAX,
        .instance_size = L->instance_size,
        .model_size = L->model_size,
        .access = f->access,
        .setup_model = f->setup_model,
        .setup_instance = f->setup,
        .eval = f->eval,
        .load_residual_resist = f->resid,
        .load_residual_react = sl_load_residual_react,
        .load_jacobian_resist = f->jac,
        .load_jacobian_react = sl_load_jacobian_react,
        .given_flag_model = f->given_model,
        .given_flag_instance = f->given_inst,
        .num_resistive_jacobian_entries = entries,
        .num_reactive_jacobian_entries = 0,
    };
    return 0;
oom:
    snprintf(err, errlen, "out of memory describing model %s", sm->name);
    return -1;
}

/* VACASK lowercases device and parameter names; refuse what would collide. */
static int check_collisions(const SlModel *models, uint32_t count, char *err, size_t errlen) {
    for (uint32_t a = 0; a < count; a++) {
        for (uint32_t b = a + 1; b < count; b++) {
            if (!strcasecmp(models[a].name, models[b].name)) {
                snprintf(err, errlen, "models %s and %s collide case-insensitively", models[a].name,
                         models[b].name);
                return -1;
            }
        }
        const SlModel *sm = &models[a];
        for (uint32_t i = 0; i < sm->m; i++) {
            for (uint32_t j = i + 1; j < sm->m; j++) {
                if (!strcasecmp(sm->params[i].name, sm->params[j].name)) {
                    snprintf(err, errlen, "model %s: parameters %s and %s collide case-insensitively",
                             sm->name, sm->params[i].name, sm->params[j].name);
                    return -1;
                }
            }
        }
    }
    return 0;
}

__attribute__((constructor)) static void socklight_load(void) {
    char err[1024];
    SlModel *models = NULL;
    uint32_t count = 0;
    OSDI_NUM_DESCRIPTORS = 0;
    if (!sl_bridge_python_running()) {
        fprintf(stderr, "socklight.osdi: no Python interpreter; load it from VACASK's Python extension\n");
        return;
    }
    if (sl_bridge_snapshot(&models, &count, err, sizeof err)) {
        goto fail;
    }
    if (count > SOCKLIGHT_MAX_MODELS) {
        snprintf(err, sizeof err,
                 "%u models registered, socklight.osdi holds at most %d "
                 "(rebuild with -DSOCKLIGHT_MAX_MODELS=<n>)",
                 count, SOCKLIGHT_MAX_MODELS);
        goto fail;
    }
    if (check_collisions(models, count, err, sizeof err)) {
        goto fail;
    }
    for (uint32_t k = 0; k < count; k++) {
        if (fill(k, &models[k], err, sizeof err)) {
            goto fail;
        }
    }
    OSDI_NUM_DESCRIPTORS = count;
    /* The descriptors keep the names for the life of the process; the port
     * names live on in the node names. */
    for (uint32_t k = 0; k < count; k++) {
        for (uint32_t i = 0; i < models[k].n; i++) {
            free(models[k].ports[i]);
        }
        free(models[k].ports);
    }
    free(models);
    return;
fail:
    fprintf(stderr, "socklight.osdi: %s\n", err);
    sl_bridge_set_load_error(err);
}
