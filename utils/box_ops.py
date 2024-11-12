# utils/box_ops.py
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from center-x, center-y, width, height to x1, y1, x2, y2 format.

    Args:
        x (torch.Tensor): Bounding boxes in (cx, cy, w, h) format.
            Shape: (..., 4) where ... indicates any number of dimensions

    Returns:
        torch.Tensor: Bounding boxes in (x1, y1, x2, y2) format
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding boxes from x1, y1, x2, y2 to center-x, center-y, width, height format.

    Args:
        x (torch.Tensor): Bounding boxes in (x1, y1, x2, y2) format
            Shape: (..., 4) where ... indicates any number of dimensions

    Returns:
        torch.Tensor: Bounding boxes in (cx, cy, w, h) format
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) between two sets of boxes.

    Args:
        boxes1 (torch.Tensor): First set of boxes in (x1, y1, x2, y2) format
            Shape: (N, 4)
        boxes2 (torch.Tensor): Second set of boxes in (x1, y1, x2, y2) format
            Shape: (M, 4)

    Returns:
        tuple: A tuple containing:
            - iou (torch.Tensor): IoU values, shape (N, M)
            - union (torch.Tensor): Union areas, shape (N, M)
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    # Get intersections
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)  # (N,M,2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N,M)

    # Calculate union
    union = area1[:, None] + area2 - inter

    # Calculate IoU
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Compute Generalized Intersection over Union (GIoU) between two sets of boxes.

    Args:
        boxes1 (torch.Tensor): First set of boxes in (x1, y1, x2, y2) format
            Shape: (N, 4)
        boxes2 (torch.Tensor): Second set of boxes in (x1, y1, x2, y2) format
            Shape: (M, 4)

    Returns:
        torch.Tensor: GIoU values, shape (N, M)
    """
    # Calculate IoU
    iou, union = box_iou(boxes1, boxes2)

    # Find the enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # (N,M,2)

    # Calculate area of enclosing box
    area = wh[:, :, 0] * wh[:, :, 1]

    # Calculate GIoU
    giou = iou - (area - union) / area
    return giou