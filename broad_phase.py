import numpy as np
import transform_util as tutil

def SE3_interpolation_coeffs(control_pqc):
    '''
    control_pnts: [..., NT, 7]
    eval_t: [..., NE] - 0~1
    return eval_pqc: [..., NE, 7]
    
    also support linear part only -> end size should be 3
    '''

    NT = control_pqc.shape[-2]
    dt = 1.0/(NT-1)

    if control_pqc.shape[-1] == 7:
        SE3_traj = True
    else:
        SE3_traj = False
    

    twist = tutil.pqc_minus(control_pqc[...,2:,:], control_pqc[...,:-2,:])/(2*dt)
    twist1 = tutil.pqc_minus(control_pqc[...,2:,:], control_pqc[...,1:-1,:])/dt
    twist2 = tutil.pqc_minus(control_pqc[...,:-2,:], control_pqc[...,1:-1,:])/dt
    acc = (twist2 + twist1)/dt

    if SE3_traj:
        # conver to the global frame
        twist = tutil.se3_rot(twist, control_pqc[...,:-2,3:])
        acc = tutil.se3_rot(acc, control_pqc[...,1:-1,3:])

    twist = np.concatenate([np.zeros_like(twist[...,:1,:]), twist, np.zeros_like(twist[...,:1,:])], axis=-2) # velocity in se3 (NT, 6)
    acc = np.concatenate([np.zeros_like(acc[...,:1,:]), acc, np.zeros_like(acc[...,:1,:])], axis=-2) # acceleration in se3 (NT, 6)

    # position, vel, acc constraint for each segment (NT-1)
    # position - 3*NT-3, vel - 3*NT-3, acc - 3*NT-3
    # coefficient matrix A: [3*(NT-1), 6*NT]
    # p(t)=a0+a1t+a2t**2+a4t**4+a5t**5
    # v(t)=a1+2*a2t+4*a4t**3+5*a5t**4
    # a(t)=2*a2+12*a4t+20*a5t**3
    # b=A@x - x = NT X [a0, a1, a2, a3, a4, a5]
    # A = (NT, 6, 6)
    # b = (NT, 6)
    # x = A^-1@b
    # p(0) = a0
    a0 = control_pqc[...,:-1,:]
    a1 = twist[..., :-1, :]
    a2 = acc[..., :-1, :]*0.5

    T_poly = np.array([dt, dt**2, dt**3, dt**4, dt**5]) # (NT, 5)
    A_mat = np.stack([
        T_poly[2:],
        np.array([3, 4, 5]) * T_poly[1:-1],
        np.array([6, 12, 20]) * T_poly[:-2],
    ], axis=-2) # (5, 5)

    A_mat_inv = np.linalg.inv(A_mat)

    if SE3_traj:
        pos_tmp = tutil.se3_rot(tutil.pqc_minus(control_pqc[...,1:,:], a0), a0[...,3:])
    else:
        pos_tmp = control_pqc[...,1:,:] - a0

    b_mat = np.stack([
        pos_tmp-a1*dt-a2*dt**2,
        twist[..., 1:, :]-a1-2*a2*dt,
        acc[..., 1:, :]-2*a2,
    ], axis=-1) # (..., NT-1, 6, 5)

    coeffs = np.einsum('...ij,...tqj->...tqi', A_mat_inv, b_mat)
    coeffs = np.concatenate([a1[...,None], a2[...,None], coeffs], axis=-1)
    return a0, coeffs


def SE3_interpolation_eval(a0, coeffs, eval_t):
    '''
    a0: [..., NT-1, 7]
    coeffs: [..., NT-1, 6, 5]
    eval_t: [..., NE] - 0~1
    return eval_pqc: [..., NE, 7]
    '''

    if a0.shape[-1] == 7:
        SE3_traj = True
    else:
        SE3_traj = False

    NT = a0.shape[-2]+1

    # identify index of segment
    bc_outer_shape = np.broadcast_shapes(coeffs.shape[:-3], eval_t.shape[:-1])
    coeffs = np.broadcast_to(coeffs, bc_outer_shape + coeffs.shape[-3:])
    eval_t = np.broadcast_to(eval_t, bc_outer_shape + eval_t.shape[-1:])

    seg_idx = np.floor(eval_t*(NT-1)).astype(np.int32) # (..., NE)
    seg_idx = np.clip(seg_idx, 0, NT-2)
    coeffs_eval = np.take_along_axis(coeffs, seg_idx[...,None,None], axis=-3) # (..., NE, 6, 5)
    a0_eval = np.take_along_axis(a0, seg_idx[...,None], axis=-2) # (..., NE, 6)
    eval_t = eval_t - seg_idx.astype(np.float32)/(NT-1) # (..., NE)
    eval_T_poly = np.stack([np.ones_like(eval_t), np.ones_like(eval_t), np.ones_like(eval_t), eval_t, eval_t**2, eval_t**3, eval_t**4, eval_t**5], axis=-1) # (..., NE, 5)
    eval_p = np.sum(coeffs_eval * eval_T_poly[..., None, 3:], axis=-1) # (..., NE, 6)

    if SE3_traj:
        eval_p = tutil.se3_rot(eval_p, tutil.qinv(a0_eval[...,3:]))
        eval_pqc = tutil.pq_multi(a0_eval, tutil.pqc_Exp(eval_p))
    else:
        eval_pqc = a0_eval + eval_p

    vel_eval = np.sum(coeffs_eval * np.array([1, 2, 3, 4, 5]) * eval_T_poly[...,None,2:-1], axis=-1)
    acc_eval = np.sum(coeffs_eval[...,1:] * np.array([2, 6, 12, 20]) * eval_T_poly[...,None,2:-2], axis=-1)
    jerk_eval = np.sum(coeffs_eval[...,2:] * np.array([6, 24, 60]) * eval_T_poly[...,None,2:-3], axis=-1)

    return eval_pqc, vel_eval, acc_eval, jerk_eval

