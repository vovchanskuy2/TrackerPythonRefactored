# processor.py
import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def make_basis(forward, up_hint):
    forward = normalize(forward)

    right = normalize(np.cross(up_hint, forward))
    up = normalize(np.cross(forward, right))

    return right, up, forward


def basis_to_quat(right, up, forward):
    # 3x3 rotation matrix
    m = np.array([
        [right[0], up[0], forward[0]],
        [right[1], up[1], forward[1]],
        [right[2], up[2], forward[2]],
    ])

    t = np.trace(m)

    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    else:
        # fallback (упрощённый)
        w, x, y, z = 1, 0, 0, 0

    return [x, y, z, w]


def matrix_to_quat(rot_mat):
    # Assuming columns are right, up, forward
    right = rot_mat[:, 0]
    up = rot_mat[:, 1]
    forward = rot_mat[:, 2]
    return basis_to_quat(right, up, forward)


def vec(a, b):
    return b - a


def compute_bone_rotation(a, b, c):
    """
    a → b → c (например плечо → локоть → кисть)
    """
    ab = vec(a, b)
    bc = vec(b, c)

    forward = normalize(bc)
    up_hint = normalize(ab)

    right, up, forward = make_basis(forward, up_hint)

    return basis_to_quat(right, up, forward)


def process_pose(pose_lms):
    if not pose_lms:
        return {}

    pose = np.array([np.array([lm.x, lm.y, lm.z]) for lm in pose_lms[0]])

    bones = {}

    # Pelvis
    mid_hip = (pose[23] + pose[24]) / 2
    mid_shoulder = (pose[11] + pose[12]) / 2
    up = normalize(mid_shoulder - mid_hip)
    left_to_right_hip = normalize(pose[24] - pose[23])
    forward = normalize(np.cross(up, left_to_right_hip))
    right = normalize(np.cross(forward, up))
    bones["pelvis"] = basis_to_quat(right, up, forward)

    # Spine (approximating one spine bone)
    mid_head_approx = pose[0]  # nose
    bones["spine_01"] = compute_bone_rotation(mid_hip, mid_shoulder, mid_head_approx)

    # Clavicles
    bones["clavicle_l"] = compute_bone_rotation(mid_shoulder, pose[11], pose[13])
    bones["clavicle_r"] = compute_bone_rotation(mid_shoulder, pose[12], pose[14])

    # Upper arms
    bones["upperarm_l"] = compute_bone_rotation(pose[11], pose[13], pose[15])
    bones["upperarm_r"] = compute_bone_rotation(pose[12], pose[14], pose[16])

    # Lower arms
    bones["lowerarm_l"] = compute_bone_rotation(pose[13], pose[15], pose[19])
    bones["lowerarm_r"] = compute_bone_rotation(pose[14], pose[16], pose[20])

    # Thighs
    bones["thigh_l"] = compute_bone_rotation(pose[23], pose[25], pose[27])
    bones["thigh_r"] = compute_bone_rotation(pose[24], pose[26], pose[28])

    # Calves
    bones["calf_l"] = compute_bone_rotation(pose[25], pose[27], pose[31])
    bones["calf_r"] = compute_bone_rotation(pose[26], pose[28], pose[32])

    # Feet
    bones["foot_l"] = compute_bone_rotation(pose[27], pose[31], pose[29])  # ankle to foot_index, hint heel (swapped for better?)
    bones["foot_r"] = compute_bone_rotation(pose[28], pose[32], pose[30])

    # Neck
    mid_ear = (pose[7] + pose[8]) / 2
    bones["neck_01"] = compute_bone_rotation(mid_shoulder, mid_ear, pose[0])

    return bones


def process_hands(hands_lms, handedness):
    bones = {}
    if not hands_lms or not handedness:
        return bones

    for i, hand_lms in enumerate(hands_lms):
        if len(hand_lms) != 21:
            continue

        hand_arr = np.array([np.array([lm.x, lm.y, lm.z]) for lm in hand_lms])

        h = handedness[i][0].category_name.lower()  # Assuming handedness is list of list
        side = '_l' if 'left' in h else '_r' if 'right' in h else ''
        if not side:
            continue

        # Hand (wrist/palm) rotation
        mcp_points = [hand_arr[j] for j in [5, 9, 13, 17]]  # index, middle, ring, pinky mcp
        mid_palm = np.mean(mcp_points, axis=0)
        forward = normalize(mid_palm - hand_arr[0])
        thumb_dir = normalize(hand_arr[2] - hand_arr[0])  # thumb mcp
        if side == '_r':  # Adjust for handedness
            thumb_dir = -thumb_dir
        right = normalize(np.cross(forward, thumb_dir))
        up = normalize(np.cross(right, forward))
        bones[f"hand{side}"] = basis_to_quat(right, up, forward)

        # Fingers
        fingers = {
            "thumb": [0, 1, 2, 3, 4],
            "index": [0, 5, 6, 7, 8],
            "middle": [0, 9, 10, 11, 12],
            "ring": [0, 13, 14, 15, 16],
            "pinky": [0, 17, 18, 19, 20]
        }
        for f, chain in fingers.items():
            for j in range(len(chain) - 2):
                a = hand_arr[chain[j]]
                b = hand_arr[chain[j + 1]]
                c = hand_arr[chain[j + 2]]
                q = compute_bone_rotation(a, b, c)
                bone_idx = f"{j + 1:02d}"
                bones[f"{f}{bone_idx}{side}"] = q

    return bones


def process_face(face_lms, face_blendshapes, facial_transformation_matrixes):
    bones = {}
    curves = {}

    if facial_transformation_matrixes and len(facial_transformation_matrixes) > 0:
        mat = np.array(facial_transformation_matrixes[0]).reshape(4, 4, order='F')
        rot_mat = mat[:3, :3]
        q = matrix_to_quat(rot_mat)
        bones["head"] = q

    if face_blendshapes and len(face_blendshapes) > 0:
        curves = {cat.category_name: cat.score for cat in face_blendshapes[0]}

    return bones, curves


def process_frame(pose_lms, hands_lms, handedness, face_lms, face_blendshapes, facial_transformation_matrixes):
    data = {
        "bones": {},
        "curves": {}
    }

    data["bones"].update(process_pose(pose_lms))

    data["bones"].update(process_hands(hands_lms, handedness))

    face_bones, face_curves = process_face(face_lms, face_blendshapes, facial_transformation_matrixes)
    data["bones"].update(face_bones)
    data["curves"].update(face_curves)

    return data