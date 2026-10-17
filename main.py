"""Запуск трекера на видео.

Первый кадр видео — эталон объекта. На выходе видео с рамкой и подписью
и CSV со статистикой по каждому кадру.

    python3 main.py input.avi output.mp4 --label "Mona Lisa"
"""

import argparse
import csv
from pathlib import Path

import cv2

from tracker import SiftTracker, draw


def track_video(src, dst, label, show_points=True, csv_path=None, use_prev=True,
                verbose=True):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, first = cap.read()
    if not ok:
        raise SystemExit("Видео пустое")

    tracker = SiftTracker(first, use_prev=use_prev)
    h, w = first.shape[:2]
    writer = None
    if dst:
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise SystemExit(f"Не удалось создать {dst}")

    rows = []
    frame, idx = first, 0
    while ok:
        res = tracker.process(frame)
        if writer:
            writer.write(draw(frame, res, label, show_points))
        rows.append(dict(frame=idx, found=int(res.found), source=res.source,
                         keypoints=res.keypoints, matches=res.matches,
                         inliers=res.inliers, ref_inliers=res.ref_inliers,
                         time_ms=round(res.time_ms, 2), reason=res.reason,
                         corners="" if res.corners is None else
                         " ".join(f"{v:.1f}" for v in res.corners.ravel())))
        ok, frame = cap.read()
        idx += 1
    cap.release()
    if writer:
        writer.release()

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=rows[0].keys())
            wr.writeheader()
            wr.writerows(rows)

    found = sum(r["found"] for r in rows)
    mean_ms = sum(r["time_ms"] for r in rows) / len(rows)
    if verbose:
        print(f"{Path(src).name}: кадров {len(rows)}, объект найден на {found} "
              f"({100 * found / len(rows):.1f}%), {mean_ms:.1f} мс/кадр "
              f"(эталон: {len(tracker.ref_pts)} точек)")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Трекинг плоского объекта по SIFT")
    ap.add_argument("input", help="входное видео, первый кадр — эталон объекта")
    ap.add_argument("output", help="выходное видео (.mp4)")
    ap.add_argument("--label", default="Mona Lisa", help="подпись у рамки")
    ap.add_argument("--no-points", action="store_true", help="не рисовать inlier-точки")
    ap.add_argument("--detect-only", action="store_true",
                    help="только детекция по эталону, без связи с прошлым кадром")
    ap.add_argument("--csv", help="сохранить статистику по кадрам в CSV")
    args = ap.parse_args()
    track_video(args.input, args.output, args.label, not args.no_points, args.csv,
                use_prev=not args.detect_only)


if __name__ == "__main__":
    main()
