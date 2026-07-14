from logic.zones import SAFE_ZONE, HIGH_RISK_ZONE, point_in_zone

def is_person_at_height(person_box, image_height, threshold=0.8):
    """
    Determines if a person is working at height
    based on bounding box vertical position
    """
    if person_box is None:
        return False
    y_bottom = person_box[3]
    if y_bottom / image_height < threshold:
        return True
    else:
        return False


def get_person_zone(person,w):
    x1, y1, x2, y2 = person["bbox"]
    cx = int((x1 + x2) / 2)
    cy = int(y2)

    if cx<0.6*w:
        return "SAFE"
    else:
        return "HIGH_RISK"