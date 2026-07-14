def _to_bbox(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return (x1, y1, x2, y2)


def _box_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_in_box(point, bbox):
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def _intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return float((x_right - x_left) * (y_bottom - y_top))


def _normalize_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _class_groups(model):
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    normalized = {idx: _normalize_name(label) for idx, label in names.items()}

    person_keys = {"person", "worker", "man", "woman"}
    helmet_keys = {"helmet", "hardhat", "hard_hat", "safety_helmet", "safetyhelmet"}
    harness_keys = {"harness", "safety_harness", "body_harness", "belt"}

    person_ids = {i for i, n in normalized.items() if n in person_keys}
    helmet_ids = {i for i, n in normalized.items() if n in helmet_keys}
    harness_ids = {i for i, n in normalized.items() if n in harness_keys}

    # Backward-compatibility fallback for the original class index convention.
    if not person_ids and not helmet_ids and not harness_ids and len(names) >= 3:
        harness_ids = {0}
        helmet_ids = {1}
        person_ids = {2}

    return person_ids, helmet_ids, harness_ids


def _has_helmet(person_bbox, helmet_bboxes):
    px1, py1, px2, py2 = person_bbox
    head_region = (px1, py1, px2, py1 + int(0.40 * (py2 - py1)))
    for helmet_bbox in helmet_bboxes:
        c = _box_center(helmet_bbox)
        if _point_in_box(c, head_region):
            return True
    return False


def _has_harness(person_bbox, harness_bboxes):
    px1, py1, px2, py2 = person_bbox
    torso_region = (
        px1,
        py1 + int(0.25 * (py2 - py1)),
        px2,
        py1 + int(0.85 * (py2 - py1)),
    )
    for harness_bbox in harness_bboxes:
        c = _box_center(harness_bbox)
        if _point_in_box(c, torso_region):
            return True
        if _intersection_area(harness_bbox, torso_region) > 0:
            return True
    return False


def detect_ppe(frame, model, conf=0.25):
    results = model(frame, conf=conf)
    boxes = results[0].boxes

    person_ids, helmet_ids, harness_ids = _class_groups(model)

    person_bboxes = []
    helmet_bboxes = []
    harness_bboxes = []

    for box in boxes:
        cls = int(box.cls[0])
        bbox = _to_bbox(box.xyxy[0])

        if cls in person_ids:
            person_bboxes.append(bbox)
        elif cls in helmet_ids:
            helmet_bboxes.append(bbox)
        elif cls in harness_ids:
            harness_bboxes.append(bbox)

    persons = []
    for i, person_bbox in enumerate(person_bboxes):
        persons.append(
            {
                "person_id": i,
                "helmet": _has_helmet(person_bbox, helmet_bboxes),
                "harness": _has_harness(person_bbox, harness_bboxes),
                "bbox": person_bbox,
            }
        )

    return persons
