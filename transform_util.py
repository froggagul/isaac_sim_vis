'''
author: Dongwon Son
2023-10-17
'''
import numpy as np
import einops

def rand_sphere(outer_shape):
    ext = np.random.normal(size=outer_shape + (5,))
    return (ext / np.linalg.norm(ext, axis=-1, keepdims=True))[...,-3:]


def safe_norm(x, axis, keepdims=False, eps=0.0):
    is_zero = np.all(np.isclose(x,0.), axis=axis, keepdims=True)
    # temporarily swap x with ones if is_zero, then swap back
    x = np.where(is_zero, np.ones_like(x), x)
    n = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    n = np.where(is_zero if keepdims else np.squeeze(is_zero, -1), 0., n)
    return n.clip(eps)

# quaternion operations
def normalize(vec, eps=1e-8):
    # return vec/(safe_norm(vec, axis=-1, keepdims=True, eps=eps) + 1e-8)
    return vec/safe_norm(vec, axis=-1, keepdims=True, eps=eps)

def quw2wu(quw):
    return np.concatenate([quw[...,-1:], quw[...,:3]], axis=-1)

def qrand(outer_shape, jkey=None):
    if jkey is None:
        return qrand_np(outer_shape)
    else:
        return normalize(np.random.normal(jkey, outer_shape + (4,)))

def qrand_np(outer_shape):
    q = np.random.normal(size=outer_shape+(4,))
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q

def line2q(zaxis, yaxis=np.array([1,0,0])):
    Rm = line2Rm(zaxis, yaxis)
    return Rm2q(Rm)

def qmulti(q1, q2):
    b,c,d,a = np.split(q1, 4, axis=-1)
    f,g,h,e = np.split(q2, 4, axis=-1)
    w,x,y,z = a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g, a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e
    return np.concatenate([x,y,z,w], axis=-1)

def qmulti_np(q1, q2):
    b,c,d,a = np.split(q1, 4, axis=-1)
    f,g,h,e = np.split(q2, 4, axis=-1)
    w,x,y,z = a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g, a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e
    return np.concatenate([x,y,z,w], axis=-1)

def qinv(q):
    x,y,z,w = np.split(q, 4, axis=-1)
    return np.concatenate([-x,-y,-z,w], axis=-1)

def qinv_np(q):
    x,y,z,w = np.split(q, 4, axis=-1)
    return np.concatenate([-x,-y,-z,w], axis=-1)

def q2aa(q):
    return 2*qlog(q)[...,:3]

def qlog(q):
    # Clamp to avoid domain errors in arccos due to floating-point inaccuracies
    q_w = np.clip(q[..., 3:], -1 + 1e-7, 1 - 1e-7)
    
    # Compute alpha with clamped w-component
    alpha = np.arccos(q_w)
    sinalpha = np.sin(alpha)
    
    # Ensure stable division by using a safe minimum threshold for sinalpha
    safe_sinalpha = np.where(np.abs(sinalpha) < 1e-6, 1e-6, sinalpha)
    n = q[..., :3] / (safe_sinalpha * np.sign(sinalpha))
    
    # Use a threshold to check for small values of alpha
    res = np.where(np.abs(q_w) < 1 - 1e-6, n * alpha, np.zeros_like(n))
    
    # Concatenate result with an additional zero for the w-component
    return np.concatenate([res, np.zeros_like(res[..., :1])], axis=-1)

def qLog(q):
    return qvee(qlog(q))

def qvee(phi):
    return 2*phi[...,:-1]

def qhat(w):
    return np.concatenate([w*0.5, np.zeros_like(w[...,0:1])], axis=-1)

def aa2q(aa):
    return qexp(aa*0.5)

def q2R(q):
    i,j,k,r = np.split(q, 4, axis=-1)
    R1 = np.concatenate([1-2*(j**2+k**2), 2*(i*j-k*r), 2*(i*k+j*r)], axis=-1)
    R2 = np.concatenate([2*(i*j+k*r), 1-2*(i**2+k**2), 2*(j*k-i*r)], axis=-1)
    R3 = np.concatenate([2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i**2+j**2)], axis=-1)
    return np.stack([R1,R2,R3], axis=-2)

def qexp(logq):
    if isinstance(logq, np.ndarray):
        alpha = np.linalg.norm(logq[...,:3], axis=-1, keepdims=True)
        alpha = np.maximum(alpha, 1e-6)
        return np.concatenate([logq[...,:3]/alpha*np.sin(alpha), np.cos(alpha)], axis=-1)
    else:
        alpha = safe_norm(logq[...,:3], axis=-1, keepdims=True)
        alpha = np.maximum(alpha, 1e-6)
        return np.concatenate([logq[...,:3]/alpha*np.sin(alpha), np.cos(alpha)], axis=-1)

def pq_quatnormalize(pqc):
    return np.concatenate([pqc[...,:3], normalize(pqc[...,3:])], axis=-1)

def qExp(w):
    return qexp(qhat(w))

def qaction(quat, pos):
    return qmulti(qmulti(quat, np.concatenate([pos, np.zeros_like(pos[...,:1])], axis=-1)), qinv(quat))[...,:3]

def qaction_np(quat, pos):
    return qmulti_np(qmulti_np(quat, np.concatenate([pos, np.zeros_like(pos[...,:1])], axis=-1)), qinv_np(quat))[...,:3]

def qnoise(quat, scale=np.pi*10/180):
    lq = np.random.normal(scale=scale, size=quat[...,:3].shape)
    return qmulti(quat, qexp(lq))

def qzero(outer_shape):
    return np.concatenate([np.zeros(outer_shape + (3,)), np.ones(outer_shape + (1,))], axis=-1)

# posquat operations
def pq_inv(pos, quat=None):
    is_pqc = False
    if pos.shape[-1] == 7:
        is_pqc = True
        assert quat is None
        quat = pos[...,3:]
        pos = pos[...,:3]
    quat_inv = qinv(quat)
    if is_pqc:
        return np.concat([-qaction(quat_inv, pos), quat_inv], axis=-1)
    else:
        return -qaction(quat_inv, pos), quat_inv

def pq_action(translate, rotate, pnt=None):
    if translate.shape[-1] == 7:
        assert pnt is None
        assert rotate.shape[-1] == 3
        pnt = rotate
        pos = translate[...,:3]
        quat = translate[...,3:]
        return qaction(quat, pnt) + pos
    return qaction(rotate, pnt) + translate

def pq_multi(pos1, quat1, pos2=None, quat2=None):
    if pos1.shape[-1] == 7:
        assert quat1.shape[-1] == 7
        assert pos2 is None
        assert quat2 is None
        pos2 = quat1[...,:3]
        quat2 = quat1[...,3:]
        quat1 = pos1[...,3:]
        pos1 = pos1[...,:3]
        return np.concat([qaction(quat1, pos2)+pos1, qmulti(quat1, quat2)], axis=-1)
    else:
        assert pos2 is not None
        assert quat2 is not None
        return qaction(quat1, pos2)+pos1, qmulti(quat1, quat2)

def pqc_Exp(twist):
    return np.concat([twist[...,:3], qExp(twist[...,3:])], axis=-1)

def pqc_Log(pqc):
    return np.concat([pqc[...,:3], qLog(pqc[...,3:])], axis=-1)

def pqc_minus(pqc1, pqc2):
    '''
    pqc1 - pqc2
    '''
    if pqc1.shape[-1] != 7:
        # only position
        return pqc1 - pqc2
    pqc_exp = pq_multi(pq_inv(pqc2), pqc1)
    pqc_exp = pq_quatnormalize(pqc_exp)
    return pqc_Log(pqc_exp)


def pq2H(pos, quat=None):
    if pos.shape[-1] == 7:
        assert quat is None
        quat = pos[...,-4:]
        pos = pos[...,:3]
    else:
        assert quat is not None

    R = q2R(quat)
    return H_from_Rpos(R, pos)

# homogineous transforms
def H_from_Rpos(R, pos):
    H = np.zeros(pos.shape[:-1] + (4,4))
    H = H.at[...,-1,-1].set(1)
    H = H.at[...,:3,:3].set(R)
    H = H.at[...,:3,3].set(pos)
    return H

def H_inv(H):
    R = H[...,:3,:3]
    p = H[...,:3, 3:]
    return H_from_Rpos(T(R), (-T(R)@p)[...,0])

def H2pq(H, concat=False):
    R = H[...,:3,:3]
    p = H[...,:3, 3]
    if concat:
        return np.concatenate([p, Rm2q(R)], axis=-1)
    else:
        return p, Rm2q(R)

# Rm util
def Rm_inv(Rm):
    return T(Rm)

def line2Rm(zaxis, yaxis=np.array([1,0,0])):
    zaxis = normalize(zaxis + np.array([0,1e-6,0]))
    xaxis = np.cross(yaxis, zaxis)
    xaxis = normalize(xaxis)
    yaxis = np.cross(zaxis, xaxis)
    Rm = np.stack([xaxis, yaxis, zaxis], axis=-1)
    return Rm

def line2Rm_np(zaxis, yaxis=np.array([1,0,0])):
    zaxis = (zaxis + np.array([0,1e-6,0]))
    zaxis = zaxis/np.linalg.norm(zaxis, axis=-1, keepdims=True)
    xaxis = np.cross(yaxis, zaxis)
    xaxis = xaxis/np.linalg.norm(xaxis, axis=-1, keepdims=True)
    yaxis = np.cross(zaxis, xaxis)
    Rm = np.stack([xaxis, yaxis, zaxis], axis=-1)
    return Rm

def Rm2q(Rm):
    Rm = einops.rearrange(Rm, '... i j -> ... j i')
    con1 = (Rm[...,2,2] < 0) & (Rm[...,0,0] > Rm[...,1,1])
    con2 = (Rm[...,2,2] < 0) & (Rm[...,0,0] <= Rm[...,1,1])
    con3 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] < -Rm[...,1,1])
    con4 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] >= -Rm[...,1,1]) 

    t1 = 1 + Rm[...,0,0] - Rm[...,1,1] - Rm[...,2,2]
    t2 = 1 - Rm[...,0,0] + Rm[...,1,1] - Rm[...,2,2]
    t3 = 1 - Rm[...,0,0] - Rm[...,1,1] + Rm[...,2,2]
    t4 = 1 + Rm[...,0,0] + Rm[...,1,1] + Rm[...,2,2]

    q1 = np.stack([t1, Rm[...,0,1]+Rm[...,1,0], Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]-Rm[...,2,1]], axis=-1) / np.sqrt(t1.clip(1e-7))[...,None]
    q2 = np.stack([Rm[...,0,1]+Rm[...,1,0], t2, Rm[...,1,2]+Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2]], axis=-1) / np.sqrt(t2.clip(1e-7))[...,None]
    q3 = np.stack([Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]+Rm[...,2,1], t3, Rm[...,0,1]-Rm[...,1,0]], axis=-1) / np.sqrt(t3.clip(1e-7))[...,None]
    q4 = np.stack([Rm[...,1,2]-Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2], Rm[...,0,1]-Rm[...,1,0], t4], axis=-1) / np.sqrt(t4.clip(1e-7))[...,None]
 
    q = np.zeros(Rm.shape[:-2]+(4,))
    q = np.where(con1[...,None], q1, q)
    q = np.where(con2[...,None], q2, q)
    q = np.where(con3[...,None], q3, q)
    q = np.where(con4[...,None], q4, q)
    q *= 0.5

    return q

def pRm_inv(pos, Rm):
    return (-T(Rm)@pos[...,None,:])[...,0], T(Rm)

def pRm_action(pos, Rm, x):
    return (Rm @ x[...,None,:])[...,0] + pos

def se3_rot(se3, quat):
    if se3.shape[-1] == 3:
        # only position
        return se3
    return np.concat([qaction(quat, se3[...,:3]), qaction(quat, se3[...,3:])], axis=-1)


# 6d utils
def R6d2Rm(x, gram_schmidt=False):
    xv, yv = x[...,:3], x[...,3:]
    xv = normalize(xv)
    if gram_schmidt:
        yv = normalize(yv - np.einsum('...i,...i',yv,xv)[...,None]*xv)
        zv = np.cross(xv, yv)
    else:
        zv = np.cross(xv, yv)
        zv = normalize(zv)
        yv = np.cross(zv, xv)
    return np.stack([xv,yv,zv], -1)

# 9d utils
def R9d2Rm(x):
    xm = einops.rearrange(x, '... (t i) -> ... t i', t=3)
    u, s, vt = np.linalg.svd(xm)
    # vt = einops.rearrange(v, '... i j -> ... j i')
    det = np.linalg.det(np.matmul(u,vt))
    vtn = np.concatenate([vt[...,:2,:], vt[...,2:,:]*det[...,None,None]], axis=-2)
    return np.matmul(u,vtn)


# general
def T(mat):
    return einops.rearrange(mat, '... i j -> ... j i')

def pq2SE2h(pos, quat=None):
    if pos.shape[-1] == 7:
        assert quat is None
        quat = pos[...,-4:]
        pos = pos[...,:3]
    z_angle = q2aa(quat)[...,2]
    SE2 = np.concat([pos[...,:2], z_angle[...,None]], axis=-1)
    height = pos[...,2]
    return SE2, height

def SE2h2pq(SE2, height, concat=False):
    height = np.array(height)
    pos = np.concatenate([SE2[...,:2], height[...,None]], axis=-1)
    quat = aa2q(np.concatenate([np.zeros_like(SE2[...,:2]), SE2[...,2:]], axis=-1))
    if concat:
        return np.concat([pos, quat], axis=-1)
    else:
        return pos, quat

# euler angle
def Rm2ZYZeuler(Rm):
    sy = np.sqrt(Rm[...,0,2]**2+Rm[...,1,2]**2)
    v1 = np.arctan2(Rm[...,1,2], Rm[...,0,2])
    v2 = np.arctan2(sy, Rm[...,2,2])
    v3 = np.arctan2(Rm[...,2,1], -Rm[...,2,0])

    v1n = np.arctan2(-Rm[...,0,1], Rm[...,1,1])
    v1 = np.where(sy < 1e-6, v1n, v1)
    v3 = np.where(sy < 1e-6, np.zeros_like(v1), v3)

    return np.stack([v1,v2,v3],-1)

def Rm2YXYeuler(Rm):
    sy = np.sqrt(np.sqrt(Rm[...,0,1]**2+Rm[...,2,1]**2))
    v1 = np.arctan2(Rm[...,0,1], Rm[...,2,1])
    v2 = np.arctan2(sy, Rm[...,1,1])
    v3 = np.arctan2(Rm[...,1,0], -Rm[...,1,2])

    v1n = np.arctan2(-Rm[...,2,0], Rm[...,0,0])
    v1 = np.where(sy < 1e-6, v1n, v1)
    v3 = np.where(sy < 1e-6, np.zeros_like(v1), v3)

    return np.stack([v1,v2,v3],-1)

def YXYeuler2Rm(YXYeuler):
    c1,c2,c3 = np.split(np.cos(YXYeuler), 3, -1)
    s1,s2,s3 = np.split(np.sin(YXYeuler), 3, -1)
    return np.stack([np.concatenate([c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],-1),
            np.concatenate([s2*s3, c2, -c3*s2],-1),
            np.concatenate([-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3],-1)], -2)

def wigner_D_order1_from_Rm(Rm):
    r1,r2,r3 = np.split(Rm,3,-2)
    r11,r12,r13 = np.split(r1,3,-1)
    r21,r22,r23 = np.split(r2,3,-1)
    r31,r32,r33 = np.split(r3,3,-1)

    return np.concatenate([np.c_[r22, r23, r21],
                np.c_[r32, r33, r31],
                np.c_[r12, r13, r11]], axis=-2)

def q2ZYZeuler(q):
    return Rm2ZYZeuler(q2R(q))

def q2XYZeuler(q):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = np.split(q, 4, -1)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return np.concatenate([roll_x, pitch_y, yaw_z], -1) # in radians

def XYZeuler2q(euler):
    """
    Convert euler angles (roll, pitch, yaw) to quaternion
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    roll_x, pitch_y, yaw_z = np.split(euler, 3, -1)
    cy = np.cos(yaw_z * 0.5)
    sy = np.sin(yaw_z * 0.5)
    cp = np.cos(pitch_y * 0.5)
    sp = np.sin(pitch_y * 0.5)
    cr = np.cos(roll_x * 0.5)
    sr = np.sin(roll_x * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.concatenate([x, y, z, w], -1)

# # widger D matrix
# Jd = [None,None,None,None,None]
# Jd[1] = np.array([[ 0., -1.,  0.],
#         [-1.,  0.,  0.],
#         [ 0.,  0.,  1.]])
# Jd[2] = np.array([[ 0.       ,  0.       ,  0.       , -1.       ,  0.       ],
#        [ 0.       ,  1.       ,  0.       ,  0.       ,  0.       ],
#        [ 0.       ,  0.       , -0.5      ,  0.       , -0.8660254],
#        [-1.       ,  0.       ,  0.       ,  0.       ,  0.       ],
#        [ 0.       ,  0.       , -0.8660254,  0.       ,  0.5      ]])
# Jd[3] = np.array([[ 0.        ,  0.        ,  0.        ,  0.79056942,  0.        , -0.61237244,  0.        ],
#        [ 0.        ,  1.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  0.61237244,  0.        , 0.79056942,  0.        ],
#        [ 0.79056942,  0.        ,  0.61237244,  0.        ,  0.        , 0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  0.        , -0.25      , 0.        , -0.96824584],
#        [-0.61237244,  0.        ,  0.79056942,  0.        ,  0.        , 0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  0.        , -0.96824584, 0.        ,  0.25      ]])
   
# def wigner_d_matrix(degree, ZYZeuler):
#     '''
#     here, alpha, beta, gamma: alpha, beta, gamma = sciR.from_quat(quat).as_euler('ZYZ')
#     ZYZ euler with relative Rz@Ry@Rz
#     Note that when degree is 1 wigner_d matrix is not equal to rotation matrix
#     The equality comes from sciR.from_quat(q).as_matrix() = wigner_d_matrix(1, *sciR.from_quat(q).as_euler('YXY'))
#     '''
#     """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
#     if degree==0:
#         return np.array([[1.0]])
#     if degree==1:
#         return YXYeuler2Rm(ZYZeuler)
#     origin_outer_shape = ZYZeuler.shape[:-1]
#     ZYZeuler = ZYZeuler.reshape((-1,3))
#     alpha, beta, gamma = np.split(ZYZeuler,3,-1)
#     J = Jd[degree]
#     x_a = z_rot_mat(alpha, degree)
#     x_b = z_rot_mat(beta, degree)
#     x_c = z_rot_mat(gamma, degree)
#     res = x_a @ J @ x_b @ J @ x_c
#     return res.reshape(origin_outer_shape+res.shape[-2:])

# def wigner_d_from_quat(degree, quat):
#     return wigner_d_from_RmV2(degree, q2R(quat))
#     # if degree==1:
#     #     return wigner_D_order1_from_Rm(q2R(quat))
#     # return wigner_d_matrix(degree, q2ZYZeuler(quat))

# def wigner_d_from_Rm(degree, Rm):
#     if degree==1:
#         return wigner_D_order1_from_Rm(Rm)
#     return wigner_d_matrix(degree, Rm2ZYZeuler(Rm))

# def wigner_d_from_RmV2(degree, Rm):
#     # assert degree <= 3
#     if degree==0:
#         return 1
#     if degree==1:
#         return wigner_D_order1_from_Rm(Rm)
#     if degree > 3:
#         return wigner_d_from_Rm(degree, Rm)
#     Rm_flat = einops.rearrange(Rm, '... i j -> ... (i j)')
#     Rm_concat = einops.rearrange((Rm_flat[...,None]*Rm_flat[...,None,:]), '... i j -> ... (i j)')
#     if degree==3:
#         Rm_concat = einops.rearrange((Rm_concat[...,None]*Rm_flat[...,None,:]), '... i j -> ... (i j)')
    
#     return einops.rearrange(np.einsum('...i,...ik', Rm_concat, WDCOEF[degree]), '... (r i)-> ... r i', r=2*degree+1)



# def z_rot_mat(angle, l):
#     '''
#     angle : (... 1)
#     '''
#     outer_shape = angle.shape[:-1]
#     order = 2*l+1
#     m = np.zeros(outer_shape + (order, order))
#     inds = np.arange(0, order)
#     reversed_inds = np.arange(2*l, -1, -1)
#     frequencies = np.arange(l, -l -1, -1)

#     m = m.at[..., inds, reversed_inds].set(np.sin(frequencies * angle))
#     m = m.at[..., inds, inds].set(np.cos(frequencies * angle))
#     return m


# def x_to_alpha_beta(x):
#     '''
#     Convert point (x, y, z) on the sphere into (alpha, beta)
#     '''
#     # x = x / np.linalg.norm(x, axis=-1, keepdims=True)
#     x = normalize(x)
#     beta = np.arccos(x[...,2])
#     alpha = np.arctan2(x[...,1], x[...,0])
#     return alpha, beta

# def sh_via_wigner_d(l, pnt):
#     a, b = x_to_alpha_beta(pnt)
#     return wigner_d_matrix(l, np.stack([a, b, np.zeros_like(a)], -1))[...,:,l]


# import os
# import pickle
# if not os.path.exists('Wigner_D_coef.pkl'):
#     WDCOEF = [None,None,None,None]
#     ns_ = 100000
#     Rmin = q2R(qrand((ns_,)))
#     Rmin = np.array(Rmin).astype(np.float64)
#     Rmin_flat = Rmin.reshape(-1,9)

#     # order 2
#     y_ = np.array(wigner_d_from_Rm(2,Rmin).reshape((ns_,-1))).astype(np.float64)
#     Rmin_concat = (Rmin_flat[...,None]*Rmin_flat[...,None,:]).reshape((ns_,-1))
#     WDCOEF[2] = np.linalg.pinv(Rmin_concat)@y_
#     WDCOEF[2] = np.where(np.abs(WDCOEF[2])<1e-5, 0, WDCOEF[2])
#     print(np.max(np.abs(Rmin_concat@WDCOEF[2]-y_)))

#     #order 3
#     Rmin_concat = (Rmin_concat[...,None]*Rmin_flat[...,None,:]).reshape((ns_,-1)).astype(np.float64)
#     y_ = np.array(wigner_d_from_Rm(3,Rmin).reshape((ns_,-1))).astype(np.float64)
#     WDCOEF[3] = np.linalg.pinv(Rmin_concat)@y_
#     WDCOEF[3] = np.where(np.abs(WDCOEF[3])<1e-5, 0, WDCOEF[3])

#     print(np.max(np.abs(Rmin_concat@WDCOEF[3]-y_)))
#     with open('Wigner_D_coef.pkl', 'wb') as f:
#         pickle.dump(WDCOEF, f)
#     del Rmin, y_, Rmin_concat
# else:
#     with open('Wigner_D_coef.pkl', 'rb') as f:
#         WDCOEF = pickle.load(f)
