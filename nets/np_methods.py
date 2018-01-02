
import numpy as np


def ssd_bboxes_decode(rlocalisations,
                      ssd_anchors,
                      #prior_scaling=[.1, .1, .2, .2]):
                      prior_scaling=[0.10000000149, 0.10000000149, 0.20000000298, 0.20000000298]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Return:
      numpy array Nx4: ymin, xmin, ymax, xmax
    """
    # Reshape for easier broadcasting.
    l_shape = rlocalisations.shape
    print("l_shape[", l_shape, "]l_shape")

    rlocalisations = np.reshape(rlocalisations,
                                    (-1, l_shape[-2], l_shape[-1]))
    print("rlocalisations new shape:", rlocalisations.shape)
    #print("ssd_anchors", ssd_anchors)


    yref, xref, href, wref = ssd_anchors
    xref = np.reshape(xref, [-1, 1])
    yref = np.reshape(yref, [-1, 1])

    # Compute center, height and width
    cx = rlocalisations[:, :, 0] * wref * prior_scaling[0] + xref
    cy = rlocalisations[:, :, 1] * href * prior_scaling[1] + yref
    w = wref * np.exp(rlocalisations[:, :, 2] * prior_scaling[2])
    h = href * np.exp(rlocalisations[:, :, 3] * prior_scaling[3])
    print("cx[", cx.shape) #, cx, "]")
    print("cy[", cy.shape) #, cy, "]")
    print("w[", w.shape) #, w, "]")
    print("h[", h.shape) #, h, "]")
    # bboxes: ymin, xmin, xmax, ymax.
    bboxes = np.zeros_like(rlocalisations)
    bboxes[:, :, 0] = (cy - h / 2.) 
    bboxes[:, :, 1] = (cx - w / 2.)
    bboxes[:, :, 2] = (cy + h / 2.)
    bboxes[:, :, 3] = (cx + w / 2.)
    # Back to original shape.
    bboxes = np.reshape(bboxes, l_shape)
    return bboxes


def TreateBoxesCore(rpredictions,
                            rlocalisations,
                            ssd_anchors,
                            decode):
    """Extract classes, scores and bounding boxes from features in one layer.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    # First decode localizations features if necessary.
    if decode:
        rlocalisations = ssd_bboxes_decode(rlocalisations, ssd_anchors)

    ps = rpredictions.shape
    print("rpredictions first shape:", ps)
    print("rlocalisations first shape:", rlocalisations.shape)

    # Reshape features to: Batches x N x N_labels | 4.
    p_shape = rpredictions.shape
    batch_size = p_shape[0] if len(p_shape) == 5 else 1

    l_shape = rlocalisations.shape
    print("l_shape:", l_shape)
    rpredictions = np.reshape(rpredictions, 
                                   (batch_size, ps[1]*ps[2], -1)) 

    rlocalisations = np.reshape(rlocalisations,
                                     (batch_size, -1, l_shape[-1])) 

    print("rpredictions second shape:", rpredictions.shape)
    print("rlocalisations second shape:", rlocalisations.shape)


    # Boxes selection: use threshold or score > no-label criteria.
    # Class prediction and scores: assign 0. to 0-class
    classes = np.argmax(rpredictions, axis=2)
    scores = np.amax(rpredictions, axis=2)
    #mask = (classes > 0)
    mask = (classes == 1) #xxxxxxxxxxxxxxxxxxxxxxxxxxyy
    classes = classes[mask]
    scores = scores[mask]
    bboxes = rlocalisations[mask]

    return classes, scores, bboxes


def TreateBoxes(predictions_net,
                      localizations_net,
                      anchors_net,
                      decode):
    """Extract classes, scores and bounding boxes from network output layers.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    print("predictions_net len:", len(predictions_net))
    for i in range(len(predictions_net)):
        classes, scores, bboxes = TreateBoxesCore(
                        predictions_net[i], 
                        localizations_net[i], 
                        anchors_net[i],
                        decode)
        if (i==0 or i==1 or i==2):
            continue
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)
        # Debug information.
        # l_layers.append(i)
        # l_idxes.append((i, idxes))

    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes


# =========================================================================== #
# Common functions for bboxes handling and selection.
# =========================================================================== #
def bboxes_sort(classes, scores, bboxes, top_k):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes


def bboxes_resize(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform.
    """
    bboxes = np.copy(bboxes)
    # Translate.
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    # Resize.
    resize = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= resize[0]
    bboxes[:, 1] /= resize[1]
    bboxes[:, 2] /= resize[0]
    bboxes[:, 3] /= resize[1]
    return bboxes


def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard


def bboxes_intersection(bboxes_ref, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes_ref = np.transpose(bboxes_ref)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes_ref[0], bboxes2[0])
    int_xmin = np.maximum(bboxes_ref[1], bboxes2[1])
    int_ymax = np.minimum(bboxes_ref[2], bboxes2[2])
    int_xmax = np.minimum(bboxes_ref[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol = (bboxes_ref[2] - bboxes_ref[0]) * (bboxes_ref[3] - bboxes_ref[1])
    score = int_vol / vol
    return score


def bboxes_nms(classes, scores, bboxes, nms_threshold):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            #overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            overlap = bboxes_intersection(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]






