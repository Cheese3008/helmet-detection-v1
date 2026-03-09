import os
import csv
import sys
import time
import queue
import threading
import argparse
import statistics
from datetime import datetime
from typing import List, Optional

import cv2

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from detect import (
    YOLO,
    Det,
    ensure_dir,
    extract_dets_from_result,
    point_in_box,
    draw_label,
)

_PROJECT_DIR = os.path.dirname(_SRC_DIR)

#  1. VIOLATION SAVER  –  Lưu ảnh vi phạm & ghi log CSV

class ViolationSaver:

    def __init__(self, cooldown_sec: float = 2.0):
        """
        Parameters
        cooldown_sec : thời gian chờ tối thiểu (giây) giữa 2 lần lưu
                       cùng 1 track_id – tránh lưu trùng ảnh
        """
        self.base_dir     = os.path.join(_PROJECT_DIR, "outputs", "violations")
        self.cooldown_sec = cooldown_sec
        self._last_saved : dict       = {}   # {track_id: timestamp}
        self._log_buffer : List[dict] = []   # buffer trước khi flush CSV
        self._total_saved: int        = 0
        ensure_dir(self.base_dir)

    def save(self, frame, violations: list) -> list:
        """
        Lưu ảnh vi phạm nếu qua cooldown.
        Parameters
        frame      : numpy array BGR
        violations : list dict có key 'track_id' (tuỳ chọn)
        Returns
        -------
        Danh sách đường dẫn ảnh đã lưu trong lần gọi.
        """
        now         = time.time()
        saved       = []
        should_save = False

        for det in violations:
            tid  = det.get("track_id") or "global"
            last = self._last_saved.get(tid, 0)
            if (now - last) >= self.cooldown_sec:
                should_save = True
                self._last_saved[tid] = now

        if not should_save:
            return saved

        # Tạo thư mục theo ngày: violations/YYYY-MM-DD/
        date_str = datetime.now().strftime("%Y-%m-%d")
        day_dir  = os.path.join(self.base_dir, date_str)
        ensure_dir(day_dir)

        # Tên file theo timestamp đến mili-giây
        ts_str   = datetime.now().strftime("%H%M%S_%f")[:-3]
        filepath = os.path.join(day_dir, f"violation_{ts_str}.jpg")

        cv2.imwrite(filepath, frame)
        saved.append(filepath)
        self._total_saved += 1

        # Thêm vào buffer log
        self._log_buffer.append({
            "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file"        : filepath,
            "n_violations": len(violations),
            "track_ids"   : ";".join(str(d.get("track_id", "?")) for d in violations),
        })

        print(f"[SAVE] Vi phạm → {filepath}")
        return saved

    def save_snapshot(self, frame) -> str:
        """Lưu ảnh chụp màn hình thủ công (nhấn phím S)."""
        out_dir = os.path.join(_PROJECT_DIR, "outputs")
        ensure_dir(out_dir)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"snap_{ts}.jpg")
        cv2.imwrite(path, frame)
        return path

    def flush_csv(self) -> Optional[str]:
        """
        Ghi toàn bộ buffer log ra file CSV.
        Tự động append nếu file đã tồn tại – không ghi đè.

        Returns đường dẫn CSV hoặc None nếu buffer rỗng.
        """
        if not self._log_buffer:
            return None

        date_str   = datetime.now().strftime("%Y%m%d")
        csv_path   = os.path.join(self.base_dir, f"violations_log_{date_str}.csv")
        fieldnames = ["timestamp", "file", "n_violations", "track_ids"]

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self._log_buffer)

        n = len(self._log_buffer)
        self._log_buffer.clear()
        print(f"[CSV] Đã ghi {n} bản ghi → {csv_path}")
        return csv_path


#  2. PERFORMANCE ANALYZER  –  Đo FPS & thống kê hiệu năng

class PerformanceAnalyzer:
    """
    Đo lường và phân tích hiệu năng realtime.

    Tính năng:
      - FPS tính bằng Exponential Moving Average (EMA) – hiển thị mượt
      - Lưu toàn bộ lịch sử để tính thống kê cuối phiên
      - Xuất báo cáo: avg / min / max / median / std FPS và inference ms
    """

    def __init__(self, alpha: float = 0.1):
        """
        Parameters
        ----------
        alpha : hệ số EMA (0 < alpha < 1)
                Nhỏ → FPS mượt hơn | Lớn → phản ứng nhanh hơn
        """
        self.alpha             = alpha
        self._fps_ema          = 0.0
        self._prev_time        = time.perf_counter()
        self._fps_history      : List[float] = []
        self._infer_ms_history : List[float] = []
        self._frame_count      : int         = 0

    def update(self, infer_ms: float) -> float:
        """
        Gọi sau mỗi frame.

        Parameters
        infer_ms : thời gian inference của frame (mili-giây)

        Returns
        FPS EMA hiện tại để hiển thị lên màn hình.
        """
        now     = time.perf_counter()
        elapsed = now - self._prev_time
        self._prev_time = now

        if elapsed > 0:
            self._fps_ema = (self.alpha * (1.0 / elapsed)
                             + (1.0 - self.alpha) * self._fps_ema)

        self._fps_history.append(self._fps_ema)
        self._infer_ms_history.append(infer_ms)
        self._frame_count += 1
        return self._fps_ema

    def get_stats(self) -> dict:
        """Trả về dict thống kê đầy đủ."""
        if not self._fps_history:
            return {}
        fps   = self._fps_history
        infer = self._infer_ms_history
        return {
            "total_frames"    : self._frame_count,
            "fps_avg"         : round(statistics.mean(fps),            2),
            "fps_min"         : round(min(fps),                        2),
            "fps_max"         : round(max(fps),                        2),
            "fps_median"      : round(statistics.median(fps),          2),
            "fps_stdev"       : round(statistics.stdev(fps) if len(fps) > 1 else 0.0, 2),
            "infer_avg_ms"    : round(statistics.mean(infer),          2),
            "infer_min_ms"    : round(min(infer),                      2),
            "infer_max_ms"    : round(max(infer),                      2),
            "infer_median_ms" : round(statistics.median(infer),        2),
        }

    def print_report(self):
        """In báo cáo hiệu năng ra console."""
        s = self.get_stats()
        if not s:
            print("[PERF] Chưa có dữ liệu.")
            return
        print(f"\n{'═'*44}")
        print(f"  BÁO CÁO HIỆU NĂNG  ({s['total_frames']} frames)")
        print(f"{'═'*44}")
        print(f"  FPS trung bình    : {s['fps_avg']:>8.1f}")
        print(f"  FPS nhỏ nhất      : {s['fps_min']:>8.1f}")
        print(f"  FPS lớn nhất      : {s['fps_max']:>8.1f}")
        print(f"  FPS trung vị      : {s['fps_median']:>8.1f}")
        print(f"  FPS độ lệch chuẩn : {s['fps_stdev']:>8.2f}")
        print(f"  {'─'*38}")
        print(f"  Infer TB  (ms)    : {s['infer_avg_ms']:>8.1f}")
        print(f"  Infer Min (ms)    : {s['infer_min_ms']:>8.1f}")
        print(f"  Infer Max (ms)    : {s['infer_max_ms']:>8.1f}")
        print(f"{'═'*44}\n")


#  3. ASYNC VIDEO WRITER  –  Ghi video trên thread riêng, không block vòng lặp chính

class AsyncVideoWriter:
    """
    Ghi video trên thread riêng để không làm lag vòng lặp detection.
    Dùng queue để nhận frame từ main thread và ghi xuống file.
    """

    def __init__(self, path: str, fourcc, fps: float, size: tuple):
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        self._queue  = queue.Queue(maxsize=64)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            self._writer.write(frame)

    def write(self, frame):
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Bỏ frame nếu queue đầy, ưu tiên tốc độ

    def release(self):
        self._queue.put(None)   # Gửi tín hiệu dừng
        self._thread.join()     # Chờ ghi hết frame còn lại
        self._writer.release()




def draw_hud(frame, fps: float, total_violations: int, frame_violations: int):
    """Vẽ HUD góc trên-trái: FPS + vi phạm frame + ngày giờ."""
    now_str  = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 88), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"FPS : {fps:.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Vi pham : {frame_violations}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, now_str,
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


#  4. BENCHMARK  –  So sánh FPS theo kích thước ảnh

def run_benchmark(video_path: str,
                  helmet_model, coco_model,
                  hm_helmet_id, hm_nohelmet_id,
                  coco_person_id, coco_moto_id,
                  sizes: List[int] = None,
                  n_frames: int = 200):
    """
    Chạy benchmark: test video với nhiều kích thước ảnh, đo FPS và inference.
    Kết quả xuất ra: outputs/benchmark_results.csv

    Parameters
        sizes    : list chiều rộng cần test, vd [320, 480, 640, 800]
        n_frames : số frame test cho mỗi kích thước
    """
    if sizes is None:
        sizes = [320, 480, 640, 800]

    conf_helmet = 0.25
    conf_coco   = 0.30
    iou_thres   = 0.45
    out_csv     = os.path.join(_PROJECT_DIR, "outputs", "benchmark_results.csv")
    results     = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[BENCHMARK ERROR] Không mở được video: {video_path}")
        return

    print(f"\n{'═'*62}")
    print(f"  BENCHMARK – {n_frames} frames / kích thước  |  {os.path.basename(video_path)}")
    print(f"{'═'*62}")
    print(f"  {'Width':>7}  {'FPS TB':>7}  {'FPS Min':>7}  "
          f"{'FPS Max':>7}  {'Infer TB':>9}  {'Infer Max':>10}")
    print(f"  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}  {'─'*10}")

    for width in sizes:
        perf = PerformanceAnalyzer()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        count = 0

        while count < n_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            oh, ow = frame.shape[:2]
            frame  = cv2.resize(frame, (width, int(oh * width / ow)))
            h, w   = frame.shape[:2]

            t0 = time.perf_counter()
            coco_res  = coco_model.predict(frame, conf=conf_coco,
                                           iou=iou_thres, verbose=False)[0]
            coco_dets = extract_dets_from_result(coco_res, w, h)
            riders    = [d for d in coco_dets if d.cls == coco_person_id]
            if riders:
                hm_res = helmet_model.predict(frame, conf=conf_helmet,
                                              iou=iou_thres, verbose=False)[0]
                extract_dets_from_result(hm_res, w, h)
            infer_ms = (time.perf_counter() - t0) * 1000

            perf.update(infer_ms)
            count += 1

        s = perf.get_stats()
        if not s:
            continue

        results.append({
            "resize_width" : width,
            "fps_avg"      : s["fps_avg"],
            "fps_min"      : s["fps_min"],
            "fps_max"      : s["fps_max"],
            "fps_median"   : s["fps_median"],
            "infer_avg_ms" : s["infer_avg_ms"],
            "infer_min_ms" : s["infer_min_ms"],
            "infer_max_ms" : s["infer_max_ms"],
            "frames_tested": s["total_frames"],
        })
        print(f"  {width:>7}  {s['fps_avg']:>7.1f}  {s['fps_min']:>7.1f}  "
              f"{s['fps_max']:>7.1f}  {s['infer_avg_ms']:>9.1f}  "
              f"{s['infer_max_ms']:>10.1f}")

    cap.release()
    print(f"{'═'*62}\n")

    if results:
        ensure_dir(os.path.join(_PROJECT_DIR, "outputs"))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"[BENCHMARK] Kết quả → {out_csv}")


#  5. MAIN  –  Entry point với argparse CLI

def main():
    parser = argparse.ArgumentParser(
        description="Helmet Detection – Deployment & Performance (Nguyễn Quốc Tường)")
    parser.add_argument("--source", type=str, default="0",
        help="Nguồn đầu vào: 0=webcam | đường dẫn video | đường dẫn ảnh")
    parser.add_argument("--resize", type=int, default=640,
        help="Resize frame (chiều rộng px) để tăng tốc. Mặc định: 640")
    parser.add_argument("--skip-frames", type=int, default=2,
        help="Chỉ inference 1 frame trên N frame, frame còn lại dùng kết quả cũ. Mặc định: 2")
    parser.add_argument("--save-video", action="store_true",
        help="Lưu video kết quả vào outputs/")
    parser.add_argument("--perf-report", action="store_true",
        help="In báo cáo hiệu năng chi tiết sau khi chạy xong")
    parser.add_argument("--benchmark", action="store_true",
        help="Chạy benchmark so sánh FPS theo kích thước ảnh")
    parser.add_argument("--sizes", nargs="+", type=int, default=[320, 480, 640, 800],
        help="Kích thước width để benchmark. Vd: --sizes 320 480 640 800")
    parser.add_argument("--bench-frames", type=int, default=200,
        help="Số frame test mỗi kích thước khi benchmark")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    helmet_model_path = os.path.join(_SRC_DIR, "..", "models", "best.pt")
    coco_model_path   = os.path.join(_SRC_DIR, "..", "models", "yolov8n.pt")

    conf_helmet           = 0.25
    conf_coco             = 0.30
    iou_thres             = 0.45
    head_ratio            = 0.35
    use_nohelmet_class    = True
    draw_motorcycle_boxes = False

    print("[INIT] Đang load model…")
    helmet_model = YOLO(helmet_model_path)
    coco_model   = YOLO(coco_model_path)

    hm_names       = helmet_model.names
    hm_name_to_id  = {v.lower(): k for k, v in hm_names.items()}
    hm_person_id   = hm_name_to_id.get("person")
    hm_helmet_id   = hm_name_to_id.get("helmet")
    hm_nohelmet_id = hm_name_to_id.get("no-helmet") or hm_name_to_id.get("no_helmet")

    coco_names      = coco_model.names
    coco_name_to_id = {v.lower(): k for k, v in coco_names.items()}
    coco_person_id  = coco_name_to_id.get("person")
    coco_moto_id    = coco_name_to_id.get("motorcycle")

    print("[Helmet model classes]", hm_names)
    print("[COCO] person:", coco_person_id, "| motorcycle:", coco_moto_id)

    if coco_person_id is None or coco_moto_id is None:
        raise RuntimeError("COCO model phải có class person & motorcycle.")
    print("[INIT] Sẵn sàng!\n")

    # Chế độ BENCHMARK
    if args.benchmark:
        if args.source.isdigit():
            print("[ERROR] Benchmark cần file video, không dùng webcam.")
            return
        run_benchmark(
            video_path     = args.source,
            helmet_model   = helmet_model,
            coco_model     = coco_model,
            hm_helmet_id   = hm_helmet_id,
            hm_nohelmet_id = hm_nohelmet_id,
            coco_person_id = coco_person_id,
            coco_moto_id   = coco_moto_id,
            sizes          = args.sizes,
            n_frames       = args.bench_frames,
        )
        return

    # Chế độ ẢNH TĨNH 
    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if os.path.splitext(args.source)[1].lower() in IMAGE_EXT:
        print(f"[MODE] Ảnh tĩnh → {args.source}")
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"[ERROR] Không đọc được ảnh: {args.source}")
            return

        if args.resize:
            oh, ow = frame.shape[:2]
            frame  = cv2.resize(frame, (args.resize, int(oh * args.resize / ow)))

        h, w  = frame.shape[:2]
        saver = ViolationSaver()
        perf  = PerformanceAnalyzer()

        t0 = time.perf_counter()
        coco_res     = coco_model.predict(frame, conf=conf_coco, iou=iou_thres, verbose=False)[0]
        coco_dets    = extract_dets_from_result(coco_res, w, h)
        persons      = [d for d in coco_dets if d.cls == coco_person_id]
        motos        = [d for d in coco_dets if d.cls == coco_moto_id]

        riders: List[Det] = []
        for p in persons:
            px1, py1, px2, py2 = p.xyxy
            pcx, pcy = (px1+px2)/2.0, (py1+py2)/2.0
            for m in motos:
                if point_in_box(pcx, pcy, m.xyxy):
                    riders.append(p); break

        hm_res       = helmet_model.predict(frame, conf=conf_helmet, iou=iou_thres, verbose=False)[0]
        hm_dets      = extract_dets_from_result(hm_res, w, h)
        hm_helmets   = [d for d in hm_dets if hm_helmet_id   is not None and d.cls == hm_helmet_id]
        hm_nohelmets = [d for d in hm_dets if hm_nohelmet_id is not None and d.cls == hm_nohelmet_id]
        infer_ms     = (time.perf_counter() - t0) * 1000
        perf.update(infer_ms)

        violations_info = []
        for rider in riders:
            rx1, ry1, rx2, ry2 = rider.xyxy
            is_nohelmet = False
            has_helmet  = False

            if use_nohelmet_class and hm_nohelmet_id is not None:
                for nh in hm_nohelmets:
                    nx1, ny1, nx2, ny2 = nh.xyxy
                    if point_in_box((nx1+nx2)/2, (ny1+ny2)/2, rider.xyxy):
                        is_nohelmet = True; break
                if not is_nohelmet:
                    head_box = (rx1, ry1, rx2, int(ry1+head_ratio*(ry2-ry1)))
                    for hl in hm_helmets:
                        hx1, hy1, hx2, hy2 = hl.xyxy
                        if point_in_box((hx1+hx2)/2, (hy1+hy2)/2, head_box):
                            has_helmet = True; break
            else:
                head_box = (rx1, ry1, rx2, int(ry1+head_ratio*(ry2-ry1)))
                for hl in hm_helmets:
                    hx1, hy1, hx2, hy2 = hl.xyxy
                    if point_in_box((hx1+hx2)/2, (hy1+hy2)/2, head_box):
                        has_helmet = True; break
                is_nohelmet = not has_helmet

            color = (0, 0, 255) if is_nohelmet else (0, 255, 0)
            label = "RIDER: NO_HELMET" if is_nohelmet else "RIDER: HELMET"
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
            draw_label(frame, label, rx1, ry1, color)
            if is_nohelmet:
                violations_info.append({"track_id": rider.track_id})

        if violations_info:
            saver.save(frame, violations_info)
            saver.flush_csv()

        cv2.putText(frame,
                    f"Vi pham: {len(violations_info)}  |  Infer: {infer_ms:.1f}ms",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out_dir = os.path.join(_PROJECT_DIR, "outputs")
        ensure_dir(out_dir)
        out_img = os.path.join(out_dir, "result_" + os.path.basename(args.source))
        cv2.imwrite(out_img, frame)
        print(f"[INFO] Kết quả → {out_img}  |  Vi phạm: {len(violations_info)}  |  {infer_ms:.1f} ms")

        cv2.imshow("Result – nhan phim bat ky de dong", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Chế độ WEBCAM / VIDEO ─────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w   = args.resize if args.resize else orig_w
    out_h   = int(orig_h * out_w / orig_w) if args.resize else orig_h

    saver = ViolationSaver(cooldown_sec=2.0)
    perf  = PerformanceAnalyzer(alpha=0.1)

    # Biến trạng thái cho frame skipping
    frame_idx          = 0
    last_riders        = []
    last_hm_helmets    = []
    last_hm_nohelmets  = []

    writer = None
    if args.save_video:
        out_dir  = os.path.join(_PROJECT_DIR, "outputs")
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"result_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        writer   = AsyncVideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                    src_fps, (out_w, out_h))
        print(f"[INFO] Lưu video → {out_path}")

    total_violations = 0
    mode_str = "Webcam" if str(args.source).isdigit() else os.path.basename(str(args.source))
    print(f"[MODE] {mode_str}  |  Resize: {args.resize or 'không'}px  |  Skip: 1/{args.skip_frames}")
    print("[INFO] Nhấn [Q]/[ESC] để thoát  |  [S] chụp màn hình\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] Hết frames.")
            break

        if args.resize:
            frame = cv2.resize(frame, (out_w, out_h))

        h, w = frame.shape[:2]

        # Inference + đo thời gian
        t0 = time.perf_counter()

        # Frame skipping: chỉ chạy inference trên 1/skip_frames frame
        if frame_idx % args.skip_frames == 0:
            coco_res     = coco_model.predict(frame, conf=conf_coco, iou=iou_thres, verbose=False)[0]
            coco_dets    = extract_dets_from_result(coco_res, w, h)
            coco_persons = [d for d in coco_dets if d.cls == coco_person_id]
            coco_motos   = [d for d in coco_dets if d.cls == coco_moto_id]

            last_riders = []
            for p in coco_persons:
                px1, py1, px2, py2 = p.xyxy
                pcx, pcy = (px1+px2)/2.0, (py1+py2)/2.0
                for m in coco_motos:
                    if point_in_box(pcx, pcy, m.xyxy):
                        last_riders.append(p); break

            if draw_motorcycle_boxes:
                for m in coco_motos:
                    x1, y1, x2, y2 = m.xyxy
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    draw_label(frame, f"MOTO {m.conf:.2f}", x1, y1, (255, 255, 0))

            # Chỉ chạy helmet model khi có rider
            if last_riders:
                hm_res            = helmet_model.predict(frame, conf=conf_helmet, iou=iou_thres, verbose=False)[0]
                hm_dets           = extract_dets_from_result(hm_res, w, h)
                last_hm_helmets   = [d for d in hm_dets if hm_helmet_id   is not None and d.cls == hm_helmet_id]
                last_hm_nohelmets = [d for d in hm_dets if hm_nohelmet_id is not None and d.cls == hm_nohelmet_id]
            else:
                last_hm_helmets   = []
                last_hm_nohelmets = []

        frame_idx += 1

        # Dùng kết quả inference mới nhất (có thể từ frame trước)
        riders       = last_riders
        hm_helmets   = last_hm_helmets
        hm_nohelmets = last_hm_nohelmets

        infer_ms = (time.perf_counter() - t0) * 1000
        fps_cur  = perf.update(infer_ms)

        # ── Logic helmet / no-helmet ──────────────────────────────────────────
        frame_violations = []

        for rider in riders:
            rx1, ry1, rx2, ry2 = rider.xyxy
            has_helmet  = False
            is_nohelmet = False

            if use_nohelmet_class and hm_nohelmet_id is not None:
                for nh in hm_nohelmets:
                    nx1, ny1, nx2, ny2 = nh.xyxy
                    if point_in_box((nx1+nx2)/2, (ny1+ny2)/2, rider.xyxy):
                        is_nohelmet = True; break
                if not is_nohelmet and hm_helmet_id is not None:
                    head_box = (rx1, ry1, rx2, int(ry1+head_ratio*(ry2-ry1)))
                    for hl in hm_helmets:
                        hx1, hy1, hx2, hy2 = hl.xyxy
                        if point_in_box((hx1+hx2)/2, (hy1+hy2)/2, head_box):
                            has_helmet = True; break
            else:
                head_box = (rx1, ry1, rx2, int(ry1+head_ratio*(ry2-ry1)))
                for hl in hm_helmets:
                    hx1, hy1, hx2, hy2 = hl.xyxy
                    if point_in_box((hx1+hx2)/2, (hy1+hy2)/2, head_box):
                        has_helmet = True; break
                is_nohelmet = not has_helmet

            color = (0, 0, 255) if is_nohelmet else (0, 255, 0)
            label = "RIDER: NO_HELMET" if is_nohelmet else "RIDER: HELMET"
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
            draw_label(frame, label, rx1, ry1, color)
            if is_nohelmet:
                frame_violations.append({"track_id": rider.track_id})

        # Lưu vi phạm
        if frame_violations:
            saver.save(frame, frame_violations)
            total_violations += len(frame_violations)

        #  HUD
        frame = draw_hud(frame, fps_cur, total_violations, len(frame_violations))

        if writer:
            writer.write(frame)

        cv2.imshow("Helmet Detection – [Q] thoat | [S] chup", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("s"):
            snap = saver.save_snapshot(frame)
            print(f"[SNAP] {snap}")

    # Dọn dẹp + báo cáo 
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Tổng vi phạm: {total_violations}")

    if args.perf_report:
        perf.print_report()

    csv_out = saver.flush_csv()
    if csv_out:
        print(f"[CSV] Log vi phạm → {csv_out}")

if __name__ == "__main__":
    main()