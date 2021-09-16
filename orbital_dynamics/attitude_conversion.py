"""
    satvis: On-orbit visualisation of satellite proximity operations
    Copyright (C) 2021  Ben Guthrie

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np


def relative_state_from_absolutes(chaser_q, target_q):
    chaser_q_conj = quaternion_conjugate(chaser_q)
    rel_q = quaternion_xproduct(target_q, chaser_q_conj)
    return rel_q


def absolute_state_from_relatives(chaser_q, target_q):
    abs_q = quaternion_xproduct(target_q, chaser_q)
    return abs_q


def quaternion_dotproduct(qc, qd):
    """Get the result of qc.qd."""
    qc_matrix = np.array([[qc[3], -qc[2], qc[1], qc[0]],
                          [qc[2], qc[3], -qc[0], qc[1]],
                          [-qc[1], qc[0], qc[3], qc[2]],
                          [-qc[0], -qc[1], -qc[2], qc[3]]])
    q = np.dot(qc_matrix, qd)
    return q

def quaternion_xproduct(qc, qd):
    """Get the result of qcxqd, ie. rotation qd followed by qc."""
    qc_matrix = np.array([[qc[3], qc[2], -qc[1], qc[0]],
                          [-qc[2], qc[3], qc[0], qc[1]],
                          [qc[1], -qc[0], qc[3], qc[2]],
                          [-qc[0], -qc[1], -qc[2], qc[3]]])
    q = np.dot(qc_matrix, qd)
    return q

def rotate_by_quaternion(vec, q):
    """ Rotate a unit vector by a quaternion. """
    q_vec = np.append(vec, 0)
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    q_tmp = quaternion_xproduct(q, q_vec)
    return quaternion_xproduct(q_tmp, q_conj)[:3]

def quaternion_conjugate(q):
    return q * [-1, -1, -1, 1]

def normalise_sign(q):
    if q[3] < 0:
        q = -q
    return q

def angle_between_quaternions(q0, q1):
    return np.arccos(2*np.dot(q0, q1)**2 - 1)


def get_rotation_arb_frame(target_q_current, target_q_next, chaser_q_current, chaser_q_next):
    chrdq = quaternion_xproduct(chaser_q_next, quaternion_conjugate(chaser_q_current)) # q21
    tgtrelq0 = relative_state_from_absolutes(chaser_q_current, target_q_current) # q31
    tgtrelq1 = relative_state_from_absolutes(chaser_q_next, target_q_next) # q42
    dqrel = quaternion_xproduct(tgtrelq1, quaternion_conjugate(tgtrelq0)) # q31->42
    dqrel_rot = quaternion_xproduct(quaternion_conjugate(tgtrelq0), quaternion_xproduct(dqrel, tgtrelq0))
    dqrel2abs = quaternion_xproduct(dqrel_rot, chrdq)
    rotated_abs_dq = normalise_sign(dqrel2abs)

    chrdq = quaternion_xproduct(chaser_q_next, quaternion_conjugate(chaser_q_current)) # q21
    q_true = quaternion_xproduct(target_q_next, quaternion_conjugate(target_q_current)) # q43
    q_to_arb_frame = relative_state_from_absolutes(chaser_q_current, target_q_current) # q35
    q_current = quaternion_xproduct(quaternion_conjugate(q_to_arb_frame), quaternion_xproduct(q_true, q_to_arb_frame)) # q65
    q_seen = quaternion_xproduct(quaternion_conjugate(chrdq), q_current) # q75
    q_seen = normalise_sign(q_seen)

    if not np.allclose(q_seen, rotated_abs_dq):
        print(q_seen)
        print(rotated_abs_dq)
        raise ValueError()

    return q_seen


def get_true_rotation(target_q, chaser_q, rotation):
    frame_rotation = quaternion_xproduct(target_q, quaternion_conjugate(chaser_q))
    true_rotation = quaternion_xproduct(frame_rotation, quaternion_xproduct(rotation, quaternion_conjugate(frame_rotation)))
    return true_rotation


def get_arb_rotation(target_q, chaser_q, true_rotation):
    frame_rotation = quaternion_xproduct(target_q, quaternion_conjugate(chaser_q))
    rotation = quaternion_xproduct(quaternion_conjugate(frame_rotation), quaternion_xproduct(true_rotation, frame_rotation))
    return rotation



def get_new_frame(target_q, chaser_q, rotation):
    true_rotation = get_true_rotation(target_q, chaser_q, rotation)
    new_frame = quaternion_xproduct(true_rotation, target_q)

    # Ensure the quaternion has unit norm
    new_frame = new_frame / np.linalg.norm(new_frame)
    return new_frame


def get_full_quaternion(q):
    """ Convert the 3-element quaternion to a 4-element quaternion. """
    q = np.atleast_2d(q)
    q4 = np.sqrt(1 - np.sum(q**2, axis=1))[:, None]
    return np.concatenate((q, q4), axis=1)


def get_e_and_phi_from_quaternion(quaternion):
    """Get the unit vector e and rotation angle phi from the quaternion."""
    if not quaternion[3] == 1.:
        unit_vector = np.array(quaternion[:3] / np.linalg.norm(quaternion[:3]))
        phi = 2 * np.arccos(quaternion[3])
    else:
        unit_vector = np.array([1., 0., 0.])
        phi = 0.
    return unit_vector, phi


def vec_from_quat(q):
    """ Get the instantaneous angular velocity from the quaternion and time step."""
    axis, angle = get_e_and_phi_from_quaternion(q)
    vec = np.array(angle * axis)
    return vec

def quat_from_vec(vec):
    angle = np.linalg.norm(vec)
    quat = np.zeros(4)
    quat[3] = np.cos(angle/2.)
    if angle != 0:
        axis = vec / np.linalg.norm(vec)
        quat[:3] = axis * np.sin(angle/2.)
    return quat


def quat_avg(q_set, qt):
    n = q_set.shape[0]

    epsilon = 1E-3
    max_iter = 1000
    for t in range(max_iter):
        err_vec = np.zeros((n, 3))
        for i in range(n):
            # calc error quaternion and transform to error vector
            qi_err = quaternion_xproduct(q_set[i], quaternion_conjugate(qt))
            qi_err /= np.linalg.norm(qi_err)
            vi_err = vec_from_quat(qi_err)

            # restrict vector angles to (-pi, pi]
            vi_norm = np.linalg.norm(vi_err)
            if vi_norm == 0:
                err_vec[i,:] = np.zeros(3)
            else:
                err_vec[i,:] = (-np.pi + np.mod(vi_norm + np.pi, 2 * np.pi)) / vi_norm * vi_err

        # measure derivation between estimate and real, then update estimate
        err = np.mean(err_vec, axis=0)
        qt = quaternion_xproduct(quat_from_vec(err), qt)
        qt = qt / np.linalg.norm(qt)

        if np.linalg.norm(err) < epsilon:
            break

    return qt, err_vec
