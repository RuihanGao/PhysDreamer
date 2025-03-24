import warp as wp
from mpm_data_structure_double import *
import numpy as np
import math
import pdb

# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33d, U: wp.mat33d, V: wp.mat33d, J: wp.float64, mu: wp.float64, lam: wp.float64
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id =wp.mat33d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    return wp.float64(2.0) * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - wp.float64(0.0))


@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33d, U: wp.mat33d, V: wp.mat33d, J: wp.float64, sig: wp.vec3d, mu: wp.float64, lam: wp.float64
):
    """
    B = F * wp.transpose(F)
    dev(B) = B - (1/3) * tr(B) * I

    For a compressible Rivlin neo-Hookean materia, the cauchy stress is given by:
    mu * J^(-2/3) * dev(B) + lam * J (J - 1) * I
    see: https://en.wikipedia.org/wiki/Neo-Hookean_solid
    """

    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3d(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3d(
        (b[0] + b[1] + b[2]) / wp.float64(3.0),
        (b[0] + b[1] + b[2]) / wp.float64(3.0),
        (b[0] + b[1] + b[2]) / wp.float64(3.0),
    )
    tau = mu * J ** wp.float64(-2.0 / 3.0) * b_hat + lam / wp.float64(2.0) * (J * J - wp.float64(1.0)) * wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))

    return (
        U
        * wp.mat33d(tau[0], wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), tau[1], wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33d, U: wp.mat33d, V: wp.mat33d, sig: wp.vec3d, mu: wp.float64, lam: wp.float64
):
    sig = wp.vec3d(
        wp.max(sig[0], wp.float64(0.01)), wp.max(sig[1], wp.float64(0.01)), wp.max(sig[2], wp.float64(0.01))
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3d(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))
    tau = wp.float64(2.0) * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33d(tau[0], wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), tau[1], wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33d, U: wp.mat33d, V: wp.mat33d, sig: wp.vec3d, mu: wp.float64, lam: wp.float64
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = wp.float64(2.0) * mu * wp.log(sig[0]) * (wp.float64(1.0) / sig[0]) + lam * log_sig_sum * (
        wp.float64(2.0) / sig[0]
    )
    center11 = wp.float64(2.0) * mu * wp.log(sig[1]) * (wp.float64(2.0) / sig[1]) + lam * log_sig_sum * (
        wp.float64(2.0) / sig[1]
    )
    center22 = wp.float64(2.0) * mu * wp.log(sig[2]) * (wp.float64(1.0) / sig[2]) + lam * log_sig_sum * (
        wp.float64(1.0) / sig[2]
    )
    center = wp.mat33d(center00, wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), center11, wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), center22)

    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def von_mises_return_mapping(F_trial: wp.mat33d, model: MPMModelStruct, p: int):
    U = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    V = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    sig_old = wp.vec3d(wp.float64(0.0))
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3d(
        wp.max(sig_old[0], wp.float64(0.01)), wp.max(sig_old[1], wp.float64(0.01)), wp.max(sig_old[2], wp.float64(0.01))
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3d(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / wp.float64(3.0)
    tau = wp.float64(2.0) * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3d(
        tau[0] - sum_tau / wp.float64(3.0), tau[1] - sum_tau / wp.float64(3.0), tau[2] - sum_tau / wp.float64(3.0)
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3d(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + wp.float64(1e-6)
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (wp.float64(2.0) * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33d(
            wp.exp(epsilon[0]),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.exp(epsilon[1]),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + wp.float64(2.0) * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


# @wp.func
# def von_mises_return_mapping_with_damage(
#     F_trial: wp.mat33d, model: MPMModelStruct, p: int
# ):
#     U = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
#     V = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
#     sig_old = wp.vec3d(wp.float64(0.0))
#     wp.svd3(F_trial, U, sig_old, V)

#     sig = wp.vec3d(
#         wp.max(sig_old[0], wp.float64(0.01)), wp.max(sig_old[1], wp.float64(0.01)), wp.max(sig_old[2], wp.float64(0.01))
#     )  # add this to prevent NaN in extrem cases
#     epsilon = wp.vec3d(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
#     temp = (epsilon[0] + epsilon[1] + epsilon[2]) / wp.float64(3.0)

#     tau = wp.float64(2.0) * model.mu[p] * epsilon + model.lam[p] * (
#         epsilon[0] + epsilon[1] + epsilon[2]
#     ) * wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))
#     sum_tau = tau[0] + tau[1] + tau[2]
#     cond = wp.vec3d(
#         tau[0] - sum_tau / wp.float64(3.0), tau[1] - sum_tau / wp.float64(3.0), tau[2] - sum_tau / wp.float64(3.0)
#     )
#     if wp.length(cond) > model.yield_stress[p]:
#         if model.yield_stress[p] <= 0:
#             return F_trial
#         epsilon_hat = epsilon - wp.vec3d(temp, temp, temp)
#         epsilon_hat_norm = wp.length(epsilon_hat) + wp.float64(1e-6)
#         delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (wp.float64(2.0) * model.mu[p])
#         epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
#         model.yield_stress[p] = model.yield_stress[p] - model.softening * wp.length(
#             (delta_gamma / epsilon_hat_norm) * epsilon_hat
#         )
#         if model.yield_stress[p] <= 0:
#             model.mu[p] = wp.float64(0.0)
#             model.lam[p] = wp.float64(0.0)
#         sig_elastic = wp.mat33d(
#             wp.exp(epsilon[0]),
#             wp.float64(0.0),
#             wp.float64(0.0),
#             wp.float64(0.0),
#             wp.exp(epsilon[1]),
#             wp.float64(0.0),
#             wp.float64(0.0),
#             wp.float64(0.0),
#             wp.exp(epsilon[2]),
#         )
#         F_elastic = U * sig_elastic * wp.transpose(V)
#         if model.hardening == 1:
#             model.yield_stress[p] = (
#                 model.yield_stress[p] + wp.float64(2.0) * model.mu[p] * model.xi * delta_gamma
#             )
#         return F_elastic
#     else:
#         return F_trial


# # for toothpaste
# @wp.func
# def viscoplasticity_return_mapping_with_StVK(
#     F_trial: wp.mat33d, model: MPMModelStruct, p: int, dt: wp.float64
# ):
#     U = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
#     V = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
#     sig_old = wp.vec3d(wp.float64(0.0))
#     wp.svd3(F_trial, U, sig_old, V)

#     sig = wp.vec3d(
#         wp.max(sig_old[0], wp.float64(0.0)1), wp.max(sig_old[1], wp.float64(0.0)1), wp.max(sig_old[2], wp.float64(0.0)1)
#     )  # add this to prevent NaN in extrem cases
#     b_trial = wp.vec3d(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
#     epsilon = wp.vec3d(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
#     trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
#     epsilon_hat = epsilon - wp.vec3d(
#         trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
#     )
#     s_trial = wp.float64(2.0) * model.mu[p] * epsilon_hat
#     s_trial_norm = wp.length(s_trial)
#     y = s_trial_norm - wp.sqrt(2.0 / 3.0) * model.yield_stress[p]
#     if y > 0:
#         mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
#         s_new_norm = s_trial_norm - y / (
#             wp.float64(1.0) + model.plastic_viscosity / (2.0 * mu_hat * dt)
#         )
#         s_new = (s_new_norm / s_trial_norm) * s_trial
#         epsilon_new = wp.float64(1.0) / (2.0 * model.mu[p]) * s_new + wp.vec3d(
#             trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
#         )
#         sig_elastic = wp.mat33d(
#             wp.exp(epsilon_new[0]),
#             wp.float64(0.0),
#             wp.float64(0.0),
#             wp.float64(0.0),
#             wp.exp(epsilon_new[1]),
#             wp.float64(0.0),
#             wp.float64(0.0),
#             wp.float64(0.0),
#             wp.exp(epsilon_new[2]),
#         )
#         F_elastic = U * sig_elastic * wp.transpose(V)
#         return F_elastic
#     else:
#         return F_trial


# @wp.func
# def sand_return_mapping(
#     F_trial: wp.mat33d, state: MPMStateStruct, model: MPMModelStruct, p: int
# ):
#     U = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
#     V = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
#     sig = wp.vec3d(wp.float64(0.0))
#     wp.svd3(F_trial, U, sig, V)

#     epsilon = wp.vec3d(
#         wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
#         wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
#         wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
#     )
#     sigma_out = wp.mat33d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
#     tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
#     epsilon_hat = epsilon - wp.vec3d(tr / 3.0, tr / 3.0, tr / 3.0)
#     epsilon_hat_norm = wp.length(epsilon_hat)
#     delta_gamma = (
#         epsilon_hat_norm
#         + (3.0 * model.lam[p] + wp.float64(2.0) * model.mu[p])
#         / (2.0 * model.mu[p])
#         * tr
#         * model.alpha
#     )

#     if delta_gamma <= 0:
#         F_elastic = F_trial

#     if delta_gamma > 0 and tr > 0:
#         F_elastic = U * wp.transpose(V)

#     if delta_gamma > 0 and tr <= 0:
#         H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
#         s_new = wp.vec3d(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

#         F_elastic = U * wp.diag(s_new) * wp.transpose(V)
#     return F_elastic


@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    model.mu[p] = model.E[p] / (wp.float64(2.0) * (wp.float64(1.0) + model.nu[p]))
    model.lam[p] = (
        model.E[p] * model.nu[p] / ((wp.float64(1.0) + model.nu[p]) * (wp.float64(1.0) - wp.float64(2.0) * model.nu[p]))
    )


@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = wp.float64(0.0)
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33d, dw: wp.mat33d, i: int, j: int, k: int
):
    dweight = wp.vec3d(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx

# 2025-03-14
@wp.func
def compute_dweight_mat(
    model: MPMModelStruct, w: wp.mat33d, dw: wp.mat33d, i: int, j: int, k: int
):
    return wp.mat33d(
        dw[0, i] * w[1, j] * w[2, k], dw[0, i] * dw[1, j] * w[2, k], dw[0, i] * w[1, j] * dw[2, k],
        w[0, i] * dw[1, j] * w[2, k], w[0, i] * w[1, j] * dw[2, k], w[0, i] * w[1, j] * dw[2, k],
        w[0, i] * w[1, j] * dw[2, k], w[0, i] * w[1, j] * dw[2, k], w[0, i] * w[1, j] * dw[2, k]
    ) * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33d, dt: wp.float64):
    cov_n = wp.mat33d(wp.float64(0.0))
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.func
def update_cov_differentiable(
    state: MPMStateStruct,
    next_state: MPMStateStruct,
    p: int,
    grad_v: wp.mat33d,
    dt: wp.float64,
):
    cov_n = wp.mat33d(wp.float64(0.0))
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    next_state.particle_cov[p * 6] = cov_np1[0, 0]
    next_state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    next_state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    next_state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    next_state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    next_state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: wp.float64):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    # output:               grid_v_in, grid_m
    p = wp.tid()
    if state.particle_selection[p] == 0:
        stress = state.particle_stress[p]

        # convert world-space position to grid-space
        grid_pos = state.particle_x[p] * model.inv_dx
        # find the base grid note (bottom left in 3D)
        base_pos_x = wp.int(grid_pos[0] - wp.float64(0.5))
        base_pos_y = wp.int(grid_pos[1] - wp.float64(0.5))
        base_pos_z = wp.int(grid_pos[2] - wp.float64(0.5))
        # fractional offset of the particle within the grid cell
        fx = grid_pos - wp.vec3d(
            wp.float64(base_pos_x), wp.float64(base_pos_y), wp.float64(base_pos_z)
        )

        # basis function for tricubic interpolation
        wa = wp.vec3d(wp.float64(1.5)) - fx
        wb = fx - wp.vec3d(wp.float64(1.0))
        wc = fx - wp.vec3d(wp.float64(0.5))
        w = wp.mat33d(
            wp.cw_mul(wa, wa) * wp.float64(0.5),
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)) - wp.cw_mul(wb, wb) + wp.vec3d(wp.float64(0.75)),
            wp.cw_mul(wc, wc) * wp.float64(0.5),
        )
        # gradient of the weight functions
        dw = wp.mat33d(fx - wp.vec3d(wp.float64(1.5)), -wp.float64(2.0) * (fx - wp.vec3d(wp.float64(1.0))), fx - wp.vec3d(wp.float64(0.5)))

        # Loop over the 3x3x3 grid node neighborhood
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    
                    dpos = (
                        wp.vec3d(wp.float64(i), wp.float64(j), wp.float64(k)) - fx
                    ) * model.dx
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    # if weight < 0:
                    #     print(weight)
                    dweight = compute_dweight(model, w, dw, i, j, k)

                    C = state.particle_C[p]
                    # if model.rpic = 0, standard apic
                    C = (wp.float64(1.0) - model.rpic_damping) * C + model.rpic_damping / wp.float64(2.0) * (
                        C - wp.transpose(C)
                    )

                    # C = (wp.float64(1.0) - model.rpic_damping) * state.particle_C[
                    #     p
                    # ] + model.rpic_damping / wp.float64(2.0) * (
                    #     state.particle_C[p] - wp.transpose(state.particle_C[p])
                    # )

                    if model.rpic_damping < -0.001:
                        # standard pic
                        C = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

                    elastic_force = -state.particle_vol[p] * stress * dweight
                    v_in_add = (
                        weight
                        * state.particle_mass[p]
                        * (state.particle_v[p] + C * dpos)
                        + dt * elastic_force
                    )
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                    )


@wp.kernel
def p2g_apic_with_stress_simplified(state: MPMStateStruct, model: MPMModelStruct, dt: wp.float64):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        stress = state.particle_stress[p]

        # convert world-space position to grid-space
        grid_pos = state.particle_x[p] * model.inv_dx
        # find the base grid note (bottom left in 3D)
        base_pos_x = wp.int(grid_pos[0] - wp.float64(0.5))
        base_pos_y = wp.int(grid_pos[1] - wp.float64(0.5))
        base_pos_z = wp.int(grid_pos[2] - wp.float64(0.5))
        # fractional offset of the particle within the grid cell
        fx = grid_pos - wp.vec3d(
            wp.float64(base_pos_x), wp.float64(base_pos_y), wp.float64(base_pos_z)
        )

        # basis function for tricubic interpolation
        wa = wp.vec3d(wp.float64(1.5)) - fx
        wb = fx - wp.vec3d(wp.float64(1.0))
        wc = fx - wp.vec3d(wp.float64(0.5))
        w = wp.mat33d(
            wp.cw_mul(wa, wa) * wp.float64(0.5),
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)) - wp.cw_mul(wb, wb) + wp.vec3d(wp.float64(0.75)),
            wp.cw_mul(wc, wc) * wp.float64(0.5),
        )
        # gradient of the weight functions
        dw = wp.mat33d(fx - wp.vec3d(wp.float64(1.5)), -wp.float64(2.0) * (fx - wp.vec3d(wp.float64(1.0))), fx - wp.vec3d(wp.float64(0.5)))

        # Loop over the 3x3x3 grid node neighborhood
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dweight = compute_dweight(model, w, dw, i, j, k)

                    # print(dweight)

                    elastic_force = -state.particle_vol[p] * stress * dweight
                    # print(elastic_force)

                    v_in_add = dt * elastic_force
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)


@wp.kernel
def p2g_apply_impulse(time: wp.float64, dt: wp.float64, state: MPMStateStruct, model: MPMModelStruct, param: Impulse_modifier):

    p = wp.tid()
    if time >= param.start_time and time < param.end_time:
        # print(time)
        if state.particle_selection[p] == 0 and param.mask[p] >= 1:
            grid_pos = state.particle_x[p] * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - wp.float64(0.5))
            base_pos_y = wp.int(grid_pos[1] - wp.float64(0.5))
            base_pos_z = wp.int(grid_pos[2] - wp.float64(0.5))

            fx = grid_pos - wp.vec3d(
                wp.float64(base_pos_x), wp.float64(base_pos_y), wp.float64(base_pos_z)
            )
            wa = wp.vec3d(wp.float64(1.5)) - fx
            wb = fx - wp.vec3d(wp.float64(1.0))
            wc = fx - wp.vec3d(wp.float64(0.5))
            w = wp.mat33d(
                wp.cw_mul(wa, wa) * wp.float64(0.5),
                wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)) - wp.cw_mul(wb, wb) + wp.vec3d(wp.float64(0.75)),
                wp.cw_mul(wc, wc) * wp.float64(0.5),
            )

            # particle impulse
            impulse = wp.vec3d(
                param.force[0],
                param.force[1],
                param.force[2],
            ) # actually force, so we multiply by dt later to get impulse

            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                        # if weight < wp.float64(0.0):
                        #     print(weight)

                        v_in_add = weight * impulse * dt
                        
                        wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                        

# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: wp.float64
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (
            wp.float64(1.0) / state.grid_m[grid_x, grid_y, grid_z]
        )
        # add gravity
        v_out = v_out + dt * model.gravitational_accelaration
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out


@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: wp.float64):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - wp.float64(0.5))
        base_pos_y = wp.int(grid_pos[1] - wp.float64(0.5))
        base_pos_z = wp.int(grid_pos[2] - wp.float64(0.5))
        fx = grid_pos - wp.vec3d(
            wp.float64(base_pos_x), wp.float64(base_pos_y), wp.float64(base_pos_z)
        )
        wa = wp.vec3d(wp.float64(1.5)) - fx
        wb = fx - wp.vec3d(wp.float64(1.0))
        wc = fx - wp.vec3d(wp.float64(0.5))
        w = wp.mat33d(
            wp.cw_mul(wa, wa) * wp.float64(0.5),
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)) - wp.cw_mul(wb, wb) + wp.vec3d(wp.float64(0.75)),
            wp.cw_mul(wc, wc) * wp.float64(0.5),
        )
        dw = wp.mat33d(fx - wp.vec3d(wp.float64(1.5)), -wp.float64(2.0) * (fx - wp.vec3d(wp.float64(1.0))), fx - wp.vec3d(wp.float64(0.5)))
        new_v = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        new_C = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        new_F = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3d(wp.float64(i), wp.float64(j), wp.float64(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * wp.float64(4.0)
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        state.particle_v[p] = new_v
        # state.particle_x[p] = state.particle_x[p] + dt * new_v
        # state.particle_x[p] = state.particle_x[p] + dt * state.particle_v[p]

        # wp.atomic_add(state.particle_x, p, dt * state.particle_v[p]) # old one is this..
        wp.atomic_add(state.particle_x, p, dt * new_v)  # debug
        # new_x = state.particle_x[p] + dt * state.particle_v[p]
        # state.particle_x[p] = new_x

        state.particle_C[p] = new_C

        I33 = wp.mat33d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        state.particle_F_trial[p] = F_tmp
        # debug for jelly
        # wp.atomic_add(state.particle_F_trial, p, new_F * dt * state.particle_F[p])

        if model.update_cov_with_F:
            update_cov(state, p, new_F, dt)


@wp.kernel
def g2p_differentiable(
    state: MPMStateStruct, next_state: MPMStateStruct, model: MPMModelStruct, dt: wp.float64
):
    """
    Compute:
        next_state.particle_v, next_state.particle_x, next_state.particle_C, next_state.particle_F_trial
    """
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - wp.float64(0.5))
        base_pos_y = wp.int(grid_pos[1] - wp.float64(0.5))
        base_pos_z = wp.int(grid_pos[2] - wp.float64(0.5))
        fx = grid_pos - wp.vec3d(
            wp.float64(base_pos_x), wp.float64(base_pos_y), wp.float64(base_pos_z)
        )
        wa = wp.vec3d(wp.float64(1.5)) - fx
        wb = fx - wp.vec3d(wp.float64(1.0))
        wc = fx - wp.vec3d(wp.float64(0.5))
        w = wp.mat33d(
            wp.cw_mul(wa, wa) * wp.float64(0.5),
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)) - wp.cw_mul(wb, wb) + wp.vec3d(wp.float64(0.75)),
            wp.cw_mul(wc, wc) * wp.float64(0.5),
        )
        dw = wp.mat33d(fx - wp.vec3d(wp.float64(1.5)), -wp.float64(2.0) * (fx - wp.vec3d(wp.float64(1.0))), fx - wp.vec3d(wp.float64(0.5)))
        new_v = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        # new_C = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        new_C = wp.mat33d(new_v, new_v, new_v)
        
        new_F = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3d(wp.float64(i), wp.float64(j), wp.float64(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = (
                        new_v + grid_v * weight
                    )  # TODO, check gradient from static loop
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * wp.float64(4.0)
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        next_state.particle_v[p] = new_v

        # add clip here:
        new_x = state.particle_x[p] + dt * new_v
        dx = wp.float64(1.0) / model.inv_dx
        a_min = dx * wp.float64(2.0)
        a_max = model.grid_lim - dx * wp.float64(2.0)

        new_x_clamped = wp.vec3d(
            wp.clamp(new_x[0], a_min, a_max),
            wp.clamp(new_x[1], a_min, a_max),
            wp.clamp(new_x[2], a_min, a_max),
        )
        next_state.particle_x[p] = new_x_clamped

        # next_state.particle_x[p] = new_x

        next_state.particle_C[p] = new_C

        I33_1 = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
        I33_2 = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
        I33_3 = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        I33 = wp.mat33d(I33_1, I33_2, I33_3)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        next_state.particle_F_trial[p] = F_tmp

        if 0:
            update_cov_differentiable(state, next_state, p, new_F, dt)


@wp.kernel
def clip_particle_x(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    posx = state.particle_x[p]
    if state.particle_selection[p] == 0:
        dx = wp.float64(1.0) / model.inv_dx
        a_min = dx * wp.float64(2.0)
        a_max = model.grid_lim - dx * wp.float64(2.0)
        new_x = wp.vec3d(
            wp.clamp(posx[0], a_min, a_max),
            wp.clamp(posx[1], a_min, a_max),
            wp.clamp(posx[2], a_min, a_max),
        )

        state.particle_x[
            p
        ] = new_x  # Warn: this gives wrong gradient, don't use this for backward


# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: wp.float64
):
    """
    state.particle_F_trial => state.particle_F   # return mapping
    state.particle_F => state.particle_stress    # stress-strain

    TODO: check the gradient of SVD!  is wp.svd3 differentiable? I guess so
    """
    p = wp.tid()
    if state.particle_selection[p] == 0:
    #     # apply return mapping
    #     if model.material == 1:  # metal
    #         state.particle_F[p] = von_mises_return_mapping(
    #             state.particle_F_trial[p], model, p
    #         )
    #     elif model.material == 2:  # sand
    #         state.particle_F[p] = sand_return_mapping(
    #             state.particle_F_trial[p], state, model, p
    #         )
    #     elif model.material == 3:  # visplas, with StVk+VM, no thickening
    #         state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
    #             state.particle_F_trial[p], model, p, dt
    #         )
    #     elif model.material == 5:
    #         state.particle_F[p] = von_mises_return_mapping_with_damage(
    #             state.particle_F_trial[p], model, p
    #         )
        # 20250319: Simplify the requuired kernels to test the double precision data type
        if model.material == 0 or model.material == 4 or model.material == 6: # elastic, jelly, or neo-hookean
            state.particle_F[p] = state.particle_F_trial[p]
        else:
            print(model.material)
            # TODO: add back other material types. May need further modification of the kernel functions to support double precision
            state.particle_F[p] = state.particle_F_trial[p]
            
            

        # also compute stress here
        J = wp.determinant(state.particle_F[p])
        U = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        V = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        sig = wp.vec3d(wp.float64(0.0))
        stress = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        wp.svd3(state.particle_F[p], U, sig, V)
        if model.material == 0 or model.material == 5:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
        if model.material == 1:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 2:
            stress = kirchoff_stress_drucker_prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 3:
            # temporarily use stvk, subject to change
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 6:
            stress = kirchoff_stress_neoHookean(
                state.particle_F[p], U, V, J, sig, model.mu[p], model.lam[p]
            )
        # stress = (stress + wp.transpose(stress)) / wp.float64(2.0)  # enfore symmetry
        state.particle_stress[p] = (stress + wp.transpose(stress)) / wp.float64(2.0)


# @wp.kernel
# def compute_cov_from_F(state: MPMStateStruct, model: MPMModelStruct):
#     p = wp.tid()

#     F = state.particle_F_trial[p]

#     init_cov = wp.mat33d(wp.float64(0.0))
#     init_cov[0, 0] = state.particle_init_cov[p * 6]
#     init_cov[0, 1] = state.particle_init_cov[p * 6 + 1]
#     init_cov[0, 2] = state.particle_init_cov[p * 6 + 2]
#     init_cov[1, 0] = state.particle_init_cov[p * 6 + 1]
#     init_cov[1, 1] = state.particle_init_cov[p * 6 + 3]
#     init_cov[1, 2] = state.particle_init_cov[p * 6 + 4]
#     init_cov[2, 0] = state.particle_init_cov[p * 6 + 2]
#     init_cov[2, 1] = state.particle_init_cov[p * 6 + 4]
#     init_cov[2, 2] = state.particle_init_cov[p * 6 + 5]

#     cov = F * init_cov * wp.transpose(F)

#     state.particle_cov[p * 6] = cov[0, 0]
#     state.particle_cov[p * 6 + 1] = cov[0, 1]
#     state.particle_cov[p * 6 + 2] = cov[0, 2]
#     state.particle_cov[p * 6 + 3] = cov[1, 1]
#     state.particle_cov[p * 6 + 4] = cov[1, 2]
#     state.particle_cov[p * 6 + 5] = cov[2, 2]


# @wp.kernel
# def compute_R_from_F(state: MPMStateStruct, model: MPMModelStruct):
#     p = wp.tid()

#     F = state.particle_F_trial[p]

#     # polar svd decomposition
#     U = wp.mat33d(wp.float64(0.0))
#     V = wp.mat33d(wp.float64(0.0))
#     sig = wp.vec3d(wp.float64(0.0))
#     wp.svd3(F, U, sig, V)

#     if wp.determinant(U) < wp.float64(0.0):
#         U[0, 2] = -U[0, 2]
#         U[1, 2] = -U[1, 2]
#         U[2, 2] = -U[2, 2]

#     if wp.determinant(V) < wp.float64(0.0):
#         V[0, 2] = -V[0, 2]
#         V[1, 2] = -V[1, 2]
#         V[2, 2] = -V[2, 2]

#     # compute rotation matrix
#     R = U * wp.transpose(V)
#     state.particle_R[p] = wp.transpose(R) # particle R is removed


@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: wp.float64):
    grid_x, grid_y, grid_z = wp.tid()
    # state.grid_v_out[grid_x, grid_y, grid_z] = (
    #     state.grid_v_out[grid_x, grid_y, grid_z] * scale
    # )
    wp.atomic_sub(
        state.grid_v_out,
        grid_x,
        grid_y,
        grid_z,
        (wp.float64(1.0) - scale) * state.grid_v_out[grid_x, grid_y, grid_z],
    )


@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        model.E[p] = params_modifier.E
        model.nu[p] = params_modifier.nu
        state.particle_density[p] = params_modifier.density


@wp.kernel
def selection_add_impulse_on_particles(
    state: MPMStateStruct, impulse_modifier: Impulse_modifier
):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
    ):
        impulse_modifier.mask[p] = 1
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_cylinder(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(
        offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal
    )
    if (
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def compute_position_l2_loss(
    mpm_state: MPMStateStruct,
    gt_pos: wp.array(dtype=wp.vec3d),
    loss: wp.array(dtype=wp.float64),
):
    tid = wp.tid()

    pos = mpm_state.particle_x[tid]
    pos_gt = gt_pos[tid]

    # l1_diff = wp.abs(pos - pos_gt)
    l2 = wp.length(pos - pos_gt)

    wp.atomic_add(loss, 0, l2)


@wp.kernel
def aggregate_grad(x: wp.array(dtype=wp.float64), grad: wp.array(dtype=wp.float64)):
    tid = wp.tid()

    # gradient descent step
    wp.atomic_add(x, 0, grad[tid])


@wp.kernel
def set_F_C_p2g(
    state: MPMStateStruct, model: MPMModelStruct, target_pos: wp.array(dtype=wp.vec3d)
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - wp.float64(0.5))
        base_pos_y = wp.int(grid_pos[1] - wp.float64(0.5))
        base_pos_z = wp.int(grid_pos[2] - wp.float64(0.5))
        fx = grid_pos - wp.vec3d(
            wp.float64(base_pos_x), wp.float64(base_pos_y), wp.float64(base_pos_z)
        )
        wa = wp.vec3d(wp.float64(1.5)) - fx
        wb = fx - wp.vec3d(wp.float64(1.0))
        wc = fx - wp.vec3d(wp.float64(0.5))
        w = wp.mat33d(
            wp.cw_mul(wa, wa) * wp.float64(0.5),
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)) - wp.cw_mul(wb, wb) + wp.vec3d(wp.float64(0.75)),
            wp.cw_mul(wc, wc) * wp.float64(0.5),
        )
        # p2g for displacement
        particle_disp = target_pos[p] - state.particle_x[p]
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    v_in_add = weight * state.particle_mass[p] * particle_disp
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                    )


@wp.kernel
def set_F_C_g2p(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - wp.float64(0.5))
        base_pos_y = wp.int(grid_pos[1] - wp.float64(0.5))
        base_pos_z = wp.int(grid_pos[2] - wp.float64(0.5))
        fx = grid_pos - wp.vec3d(
            wp.float64(base_pos_x), wp.float64(base_pos_y), wp.float64(base_pos_z)
        )
        wa = wp.vec3d(wp.float64(1.5)) - fx
        wb = fx - wp.vec3d(wp.float64(1.0))
        wc = fx - wp.vec3d(wp.float64(0.5))
        w = wp.mat33d(
            wp.cw_mul(wa, wa) * wp.float64(0.5),
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)) - wp.cw_mul(wb, wb) + wp.vec3d(wp.float64(0.75)),
            wp.cw_mul(wc, wc) * wp.float64(0.5),
        )
        dw = wp.mat33d(fx - wp.vec3d(wp.float64(1.5)), -wp.float64(2.0) * (fx - wp.vec3d(wp.float64(1.0))), fx - wp.vec3d(wp.float64(0.5)))
        new_C = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        new_F = wp.mat33d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

        # g2p for C and F
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3d(wp.float64(i), wp.float64(j), wp.float64(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * wp.float64(4.0)
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        # C should still be zero..
        # state.particle_C[p] = new_C
        I33 = wp.mat33d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        F_tmp = I33 + new_F
        state.particle_F_trial[p] = F_tmp

        if model.update_cov_with_F:
            update_cov(state, p, new_F, wp.float64(1.0))


@wp.kernel
def compute_posloss_with_grad(
    mpm_state: MPMStateStruct,
    gt_pos: wp.array(dtype=wp.vec3d),
    grad: wp.array(dtype=wp.vec3d),
    dt: wp.float64,
    loss: wp.array(dtype=wp.float64),
):
    tid = wp.tid()

    pos = mpm_state.particle_x[tid]
    pos_gt = gt_pos[tid]

    # l1_diff = wp.abs(pos - pos_gt)
    # l2 = wp.length(pos - (pos_gt - grad[tid] * dt))
    diff = pos - (pos_gt - grad[tid] * dt)
    l2 = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, l2)



@wp.kernel
def compute_stressloss_with_grad(
    mpm_state: MPMStateStruct,
    gt_stress: wp.array(dtype=wp.mat33d),
    grad: wp.array(dtype=wp.mat33d),
    dt: wp.float64,
    loss: wp.array(dtype=wp.float64),
):
    tid = wp.tid()

    stress = mpm_state.particle_stress[tid]
    stress_gt = gt_stress[tid]

    stress_diff = stress - (stress_gt - grad[tid] * dt)
    l2 = wp.ddot(stress_diff, stress_diff)

    wp.atomic_add(loss, 0, l2)



@wp.kernel
def compute_veloloss_with_grad(
    mpm_state: MPMStateStruct,
    gt_pos: wp.array(dtype=wp.vec3d),
    grad: wp.array(dtype=wp.vec3d),
    dt: wp.float64,
    loss: wp.array(dtype=wp.float64),
):
    tid = wp.tid()

    pos = mpm_state.particle_v[tid]
    pos_gt = gt_pos[tid]

    # l1_diff = wp.abs(pos - pos_gt)
    # l2 = wp.length(pos - (pos_gt - grad[tid] * dt))

    diff = pos - (pos_gt - grad[tid] * dt)
    l2 = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, l2)


@wp.kernel
def compute_Floss_with_grad(
    mpm_state: MPMStateStruct,
    gt_mat: wp.array(dtype=wp.mat33d),
    grad: wp.array(dtype=wp.mat33d),
    dt: wp.float64,
    loss: wp.array(dtype=wp.float64),
):
    tid = wp.tid()

    mat_ = mpm_state.particle_F_trial[tid]
    mat_gt = gt_mat[tid]

    mat_gt = mat_gt - grad[tid] * dt
    # l1_diff = wp.abs(pos - pos_gt)
    mat_diff = mat_ - mat_gt

    l2 = wp.ddot(mat_diff, mat_diff)
    # l2 = wp.sqrt(
    #     mat_diff[0, 0] ** wp.float64(2.0)
    #     + mat_diff[0, 1] ** wp.float64(2.0)
    #     + mat_diff[0, 2] ** wp.float64(2.0)
    #     + mat_diff[1, 0] ** wp.float64(2.0)
    #     + mat_diff[1, 1] ** wp.float64(2.0)
    #     + mat_diff[1, 2] ** wp.float64(2.0)
    #     + mat_diff[2, 0] ** wp.float64(2.0)
    #     + mat_diff[2, 1] ** wp.float64(2.0)
    #     + mat_diff[2, 2] ** wp.float64(2.0)
    # )

    wp.atomic_add(loss, 0, l2)


### 2025-03-13 Create loss kernel for unit test ###
@wp.kernel
def compute_stress_loss_with_grad(
    mpm_state: MPMStateStruct,  # Current MPM state
    grad_stress_wp: wp.array(dtype=wp.mat33d),  # Incoming gradient from loss w.r.t. stress
    scale: wp.float64,  # Scaling factor (default 1.0)
    loss: wp.array(dtype=wp.float64)  # Output loss value (scalar)
):
    """
    Compute loss gradient w.r.t. stress and accumulate it in the Warp computation graph.
    """

    tid = wp.tid()  # Thread index for particles

    # Fetch the current stress tensor
    stress_tensor = mpm_state.particle_stress[tid]

    # Compute dot product between stress and its gradient
    stress_loss = wp.ddot(stress_tensor, grad_stress_wp[tid]) * scale

    # Accumulate loss contribution
    wp.atomic_add(loss, 0, stress_loss)



@wp.kernel
def compute_grid_vin_loss_with_grad(
    mpm_state: MPMStateStruct,  
    grad_grid_v_wp: wp.array3d(dtype=wp.vec3d),  
    scale: wp.float64,  
    loss: wp.array(dtype=wp.float64)  
):
    """
    Compute how grid velocity `grid_v_in` contributes to the loss gradient.
    """

    i, j, k = wp.tid()  # Thread index for grid

    # Fetch the current grid velocity
    grid_v_tensor = mpm_state.grid_v_in[i, j, k]

    # Compute how grid velocity affects loss
    grid_v_loss = wp.dot(grid_v_tensor, grad_grid_v_wp[i, j, k]) * scale

    # Accumulate the loss gradient contribution
    wp.atomic_add(loss, 0, grid_v_loss)

# create another kernel just to sum up all elememts. check if that matches with loss computation with gradient
@wp.kernel
def sum_grid_vin(
    mpm_state: MPMStateStruct,
    loss: wp.array(dtype=wp.float64)  
):
    """
    Compute how grid velocity `grid_v_in` contributes to the loss gradient.
    """

    i, j, k = wp.tid()  # Thread index for grid

    # Fetch the current grid velocity
    grid_v_tensor = mpm_state.grid_v_in[i, j, k]

    # Accumulate the loss gradient contribution
    wp.atomic_add(loss, 0, grid_v_tensor[0])
    wp.atomic_add(loss, 0, grid_v_tensor[1])
    wp.atomic_add(loss, 0, grid_v_tensor[2])


@wp.kernel
def compute_Closs_with_grad(
    mpm_state: MPMStateStruct,
    gt_mat: wp.array(dtype=wp.mat33d),
    grad: wp.array(dtype=wp.mat33d),
    dt: wp.float64,
    loss: wp.array(dtype=wp.float64),
):
    tid = wp.tid()

    mat_ = mpm_state.particle_C[tid]
    mat_gt = gt_mat[tid]

    mat_gt = mat_gt - grad[tid] * dt
    # l1_diff = wp.abs(pos - pos_gt)

    mat_diff = mat_ - mat_gt
    l2 = wp.ddot(mat_diff, mat_diff)

    wp.atomic_add(loss, 0, l2)
