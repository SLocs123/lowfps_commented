import numpy as np

def low_fps_distance(atracks, btracks):
    """Build the cost matrix used to match old tracks to new detections."""
    num_atracks = len(atracks)
    num_btracks = len(btracks)
    costs = np.zeros((num_atracks, num_btracks), dtype=float)
    
    
    for i, atrack in enumerate(atracks):
        for j, btrack in enumerate(btracks):
            cost = get_average_distance(atrack, btrack)
            costs[i, j] = cost
    
    return costs

def get_average_distance(atrack, btrack, max_disp=300.0):
    """Combine appearance, geometry, class, and lane cues into one cost."""
    gate = 0.2

    vect_cost = _vector_difference(atrack.tlbr, btrack.tlbr, atrack.history, vector=atrack.direction)
    appearance_cost = _appearance_cos_cost(atrack.embed_history, btrack.embed_history)
    displacement_cost = _centre_euclidean(atrack.tlbr, btrack.tlbr, max_disp)
    class_cost = _class_mismatch_penalty(atrack.cls, btrack.cls, btrack.score)
    lane_cost = _lane_assignement_cost(atrack.ltr_dist, atrack.rtl_dist, btrack.ltr_dist, btrack.rtl_dist, atrack.direction_line, btrack.direction_line)

    # Hard reject obviously impossible matches before combining softer cues.
    if displacement_cost is not None and displacement_cost == 1.0:
        return 1.0
    
    if vect_cost is not None and vect_cost > gate:
        return 1.0
        
    if appearance_cost is None or displacement_cost is None:
        return 1.0
    
    if lane_cost is not None and lane_cost == 1.0:
        return 1.0

    # Appearance is the strongest cue, position is secondary, class is weakest.
    combined_cost = (0.7 * appearance_cost) + (0.1 * displacement_cost) + (0.2 * class_cost)
    if lane_cost is not None and lane_cost > 0.0:
        combined_cost += 0.4 * lane_cost
    return min(combined_cost, 1.0)


def _lane_assignement_cost(altr, artl, bltr, brtl, a_assigned, b_assigned):
    """Penalty for assigning the track and detection to different expected lanes."""
    
    if altr is None or artl is None or bltr is None or brtl is None:
        return 0.0
    
    high_conf_distance = 20
    
    assigned_dist_a = altr if altr < artl else artl
    assigned_dist_b = bltr if bltr < brtl else brtl
    
    if a_assigned == b_assigned:
        return 0.0
    
    
    if assigned_dist_a < high_conf_distance:
        return 0.5
    else:
        return 1.0
    


def _class_mismatch_penalty(acls, bcls, bscore, min_score=0.6, max_score=0.85):
    """
    Return a class mismatch penalty in [0, 1] that scales with detection confidence.
    Low-confidence detections are treated as less trustworthy for class comparisons.
    """
    if acls == bcls:
        return 0.0

    confidence = float(np.clip(bscore, 0.0, 1.0))
    if min_score >= 1.0:
        return 0.0
    
    if confidence > max_score:
        return 1.0

    trust = np.clip((confidence - min_score) / (max_score - min_score), 0.0, 1.0)
    return float(trust)

def _appearance_cos_cost(aembed_history, bembed_history):
    """Compare average appearance embeddings using cosine distance."""
    if not aembed_history or not bembed_history:
        return None

    # Average over history to reduce noise from any single cropped frame.
    a_embed = np.mean(aembed_history, axis=0)
    b_embed = np.mean(bembed_history, axis=0)
    
    sim = _cosine_similarity(a_embed, b_embed)
    if sim is None:
        return None

    return (1.0 - (sim + 1.0) / 2.0)

def _centre_euclidean(atlbr, btlbr, max_disp):
    """Normalize centre-point displacement into a `[0, 1]` distance."""
    if max_disp <= 0:
        raise ValueError("max_disp must be > 0")
    a = _get_centre(atlbr)
    b = _get_centre(btlbr)
    return np.clip(np.linalg.norm(a - b) / max_disp, 0.0, 1.0)


def _mean_manhatten(atlbr, btlbr, max_disp):
    """Alternative box distance metric kept from earlier experiments."""
    if max_disp <= 0:
        raise ValueError("max_disp must be > 0")

    a = np.array([[atlbr[0], atlbr[1]],
                  [atlbr[2], atlbr[1]],
                  [atlbr[0], atlbr[3]],
                  [atlbr[2], atlbr[3]]], dtype=float)

    b = np.array([[btlbr[0], btlbr[1]],
                  [btlbr[2], btlbr[1]],
                  [btlbr[0], btlbr[3]],
                  [btlbr[2], btlbr[3]]], dtype=float)

    mean_disp = np.abs(a - b).sum(axis=1).mean()
    return np.clip(mean_disp / max_disp, 0.0, 1.0)


def _vector_difference(atlbr, btlbr, history, vector=None):
    """Compare the candidate motion against the recent or expected direction."""
    a_last = history[-1]
    b = _get_centre(btlbr)

    last_5 = history[-5:]
    avg_vector = _get_avg_vector(last_5) if vector is None else vector
    new_vector = b - a_last

    if avg_vector is None:
        return None

    sim = _cosine_similarity(new_vector, avg_vector)
    if sim is None:
        return None

    return 1.0 - (sim + 1.0) / 2.0


def _cosine_similarity(v1, v2, eps=1e-8):
    """Safe cosine similarity that returns `None` for near-zero vectors."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return None

    sim = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.clip(sim, -1.0, 1.0))


def _shape_difference(atlbr, btlbr, eps=1e-9):
    """Alternative overlap-based shape penalty kept for experimentation."""
    ax1, ay1, ax2, ay2 = map(float, atlbr)
    bx1, by1, bx2, by2 = map(float, btlbr)

    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 1.0

    inter = min(aw, bw) * min(ah, bh)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    iou = inter / (union + eps)
    return 1.0 - np.clip(iou, 0.0, 1.0)
    
    
def _get_centre(tlbr):
    """Return the centre point of a `[x1, y1, x2, y2]` box."""
    c = np.array([
        (tlbr[0] + tlbr[2]) / 2.0,
        (tlbr[1] + tlbr[3]) / 2.0
    ], dtype=float)
    return c

def _get_avg_vector(points):
    """Average the step vectors across a short point history."""
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return None

    if len(points) == 2:
        return points[1] - points[0]

    vectors = points[1:] - points[:-1]
    return vectors.mean(axis=0)