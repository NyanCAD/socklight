/* The generic device: one implementation of every OSDI callback, told which
 * model it serves by its slot index (the per-slot trampolines in
 * socklight_osdi.c supply it, since OSDI callbacks do not get their descriptor).
 *
 * Storage is laid out per model from its port count n and parameter count m:
 *
 *   Instance: node_mapping[2n] u32, jac_resist[4n^2] double*,
 *             jac_react[4n^2] double*, collapsed[1] bool,
 *             resid_resist[2n], resid_react[2n] double, G[4n^2] double,
 *             rvals[m] double, svals[m] char*, given[ceil(m/32)] u32
 *   Model:    rvals[m] double, svals[m] char*, given[ceil(m/32)] u32
 *
 * Parameter j lives in rvals[j] (real, int and opt parameters, all declared
 * PARA_TY_REAL) or svals[j] (str parameters, PARA_TY_STR).  Every slot holds a
 * real value whether given or not (VACASK reads values back through access()
 * to save and restore them around sweeps): setup_model fills ungiven model
 * parameters with the signature default (NaN for an opt parameter that is
 * None), setup_instance fills ungiven instance parameters from the model.
 */
#pragma once

#include <stdint.h>

#include "bridge.h"
#include "osdi.h"

typedef struct {
    char *name;    /* resource path, original case */
    uint32_t n, m; /* ports, parameters */
    SlParam *params;

    /* instance offsets */
    uint32_t i_node_mapping, i_jac_resist, i_jac_react, i_collapsed;
    uint32_t i_resid_resist, i_resid_react, i_g, i_rvals, i_svals, i_given;
    uint32_t instance_size;
    /* model offsets */
    uint32_t m_rvals, m_svals, m_given;
    uint32_t model_size;
} Layout;

/* Computes the offsets of `l` from l->n and l->m.  Returns -1 if a structure
 * would not fit the 32-bit sizes OSDI uses. */
int sl_layout(Layout *l);

/* One Layout per descriptor slot; defined in socklight_osdi.c. */
extern Layout sl_layouts[];

void *sl_access(uint32_t slot, void *inst, void *model, uint32_t id, uint32_t flags);
uint32_t sl_given_flag_model(uint32_t slot, void *model, uint32_t id);
uint32_t sl_given_flag_instance(uint32_t slot, void *inst, uint32_t id);
void sl_setup_model(uint32_t slot, void *handle, void *model, OsdiSimParas *sim, OsdiInitInfo *res);
void sl_setup_instance(uint32_t slot, void *handle, void *inst, void *model, double temperature,
                       uint32_t num_terminals, OsdiSimParas *sim, OsdiInitInfo *res);
uint32_t sl_eval(uint32_t slot, void *handle, void *inst, void *model, OsdiSimInfo *info);
void sl_load_residual_resist(uint32_t slot, void *inst, void *model, double *dst);
void sl_load_jacobian_resist(uint32_t slot, void *inst, void *model);

/* Slot-independent callbacks. */
void sl_load_residual_react(void *inst, void *model, double *dst);
void sl_load_jacobian_react(void *inst, void *model, double alpha);

/* Y = (I - S)(I + S)^-1 of the n x n S (interleaved re/im, row-major), stamped
 * as the real 2n x 2n block [[Re Y, -Im Y], [Im Y, Re Y]] into g.  Returns -1
 * if I + S is singular or memory runs out. */
int sl_s_to_g(uint32_t n, const double *s, double *g);
