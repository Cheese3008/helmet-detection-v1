import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
from ultralytics import YOLO


@dataclass
class Det:
    cls: int
    conf: float
    xyxy: Tuple[int, int, int, int]  # x1,y1,x2,y2
    track_id: Optional[int] = None


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def draw_label(img, text, x1, y1, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y1b = max(0, y1 - th - 10)
    cv2.rectangle(img, (x1, y1b), (x1 + tw + 10, y1), color, -1)
    cv2.putText(img, text, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_violation(frame, out_dir="outputs/violations"):
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"violation_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def extract_dets_from_result(r, w, h) -> List[Det]:
    dets: List[Det] = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets

    boxes = r.boxes.xyxy.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    ids = None
    if hasattr(r.boxes, "id") and r.boxes.id is not None:
        ids = r.boxes.id.cpu().numpy().astype(int)

    for i, ((x1, y1, x2, y2), c, cf) in enumerate(zip(boxes, clss, confs)):
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        tid = int(ids[i]) if ids is not None else None
        dets.append(Det(cls=c, conf=float(cf), xyxy=(x1, y1, x2, y2), track_id=tid))
    return dets


def main():
    # ===== CONFIG =====
    helmet_model_path = "models/best.pt"     # Helmet / No-Helmet / person (your trained model)
    coco_model_path = "yolov8n.pt"           # COCO pretrained: person + motorcycle + ...
    source = "D:\Hoc-May\Project\helmet_detection\Test.mp4" # webcam; or r"D:\...\Test.mp4"

    conf_helmet = 0.25
    conf_coco = 0.30
    iou_thres = 0.45

    head_ratio = 0.35            # head region = top 35% of rider bbox (used if you check Helmet-in-head)
    use_nohelmet_class = True     # if your model has "No-Helmet" class -> easiest
    save_each_violation_sec = 2.0
    show_fps = True
    draw_motorcycle_boxes = False
    # ==================

    helmet_model = YOLO(helmet_model_path)
    coco_model = YOLO(coco_model_path)

    # --- class ids ---
    # Helmet model classes:
    hm_names = helmet_model.names
    hm_name_to_id = {v.lower(): k for k, v in hm_names.items()}
    hm_person_id = hm_name_to_id.get("person", None)
    hm_helmet_id = hm_name_to_id.get("helmet", None)
    hm_nohelmet_id = hm_name_to_id.get("no-helmet", None) or hm_name_to_id.get("no_helmet", None)

    print("[Helmet model classes]", hm_names)
    print("hm_person_id:", hm_person_id, "hm_helmet_id:", hm_helmet_id, "hm_nohelmet_id:", hm_nohelmet_id)

    # COCO model classes:
    coco_names = coco_model.names
    coco_name_to_id = {v.lower(): k for k, v in coco_names.items()}
    coco_person_id = coco_name_to_id.get("person", None)
    coco_moto_id = coco_name_to_id.get("motorcycle", None)

    print("[COCO model classes]", "person:", coco_person_id, "motorcycle:", coco_moto_id)

    if coco_person_id is None or coco_moto_id is None:
        raise RuntimeError("COCO model must contain person & motorcycle (should be true for yolov8n.pt).")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    last_save_t = 0.0
    prev_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # ===== 1) COCO detect: motorcycles + persons =====
        coco_res = coco_model.predict(frame, conf=conf_coco, iou=iou_thres, verbose=False)[0]
        coco_dets = extract_dets_from_result(coco_res, w, h)
        coco_persons = [d for d in coco_dets if d.cls == coco_person_id]
        coco_motos = [d for d in coco_dets if d.cls == coco_moto_id]

        # Rider filtering: person center must lie inside a motorcycle box
        riders: List[Det] = []
        for p in coco_persons:
            px1, py1, px2, py2 = p.xyxy
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            for m in coco_motos:
                if point_in_box(pcx, pcy, m.xyxy):
                    riders.append(p)
                    break

        # Optional draw motos
        if draw_motorcycle_boxes:
            for m in coco_motos:
                x1, y1, x2, y2 = m.xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                draw_label(frame, f"MOTO {m.conf:.2f}", x1, y1, (255, 255, 0))

        # ===== 2) Helmet model detect: helmet/no-helmet (and maybe person) =====
        hm_res = helmet_model.predict(frame, conf=conf_helmet, iou=iou_thres, verbose=False)[0]
        hm_dets = extract_dets_from_result(hm_res, w, h)

        hm_helmets = [d for d in hm_dets if (hm_helmet_id is not None and d.cls == hm_helmet_id)]
        hm_nohelmets = [d for d in hm_dets if (hm_nohelmet_id is not None and d.cls == hm_nohelmet_id)]

        any_violation = False

        # ===== 3) For each rider only -> decide helmet/no-helmet =====
        for rider in riders:
            rx1, ry1, rx2, ry2 = rider.xyxy

            has_helmet = False
            is_nohelmet = False

            if use_nohelmet_class and hm_nohelmet_id is not None:
                # Rule A: if any no-helmet detection center lies inside rider bbox -> violation
                for nh in hm_nohelmets:
                    nx1, ny1, nx2, ny2 = nh.xyxy
                    ncx, ncy = (nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0
                    if point_in_box(ncx, ncy, rider.xyxy):
                        is_nohelmet = True
                        break
                # If not no-helmet, try helmet inside head region
                if not is_nohelmet and hm_helmet_id is not None:
                    head_y2 = int(ry1 + head_ratio * (ry2 - ry1))
                    head_box = (rx1, ry1, rx2, head_y2)
                    for hl in hm_helmets:
                        hx1, hy1, hx2, hy2 = hl.xyxy
                        hcx, hcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
                        if point_in_box(hcx, hcy, head_box):
                            has_helmet = True
                            break
            else:
                # Rule B (no no-helmet class): helmet must be in head region, else violation
                head_y2 = int(ry1 + head_ratio * (ry2 - ry1))
                head_box = (rx1, ry1, rx2, head_y2)
                for hl in hm_helmets:
                    hx1, hy1, hx2, hy2 = hl.xyxy
                    hcx, hcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
                    if point_in_box(hcx, hcy, head_box):
                        has_helmet = True
                        break
                is_nohelmet = not has_helmet

            if is_nohelmet:
                color = (0, 0, 255)
                label = f"RIDER: NO_HELMET"
                any_violation = True
            else:
                color = (0, 255, 0)
                label = f"RIDER: HELMET"

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
            draw_label(frame, label, rx1, ry1, color)

        # Save violation image with cooldown
        now = time.time()
        if any_violation and (now - last_save_t) > save_each_violation_sec:
            save_path = save_violation(frame)
            print("Saved violation:", save_path)
            last_save_t = now

        # FPS
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        if show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Helmet Detection (Riders Only)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()