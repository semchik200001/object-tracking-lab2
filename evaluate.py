"""Прогон трекера на всех тестовых видео и подготовка материалов для отчёта.

Для каждого видео из data/ в results/ сохраняются:
  <имя>.mp4        — видео с рамкой (H.264, нужен ffmpeg)
  <имя>.csv        — статистика по каждому кадру
  <имя>.gif        — короткое превью для README
  <имя>_grid.jpg   — сетка кадров результата
Общие файлы:
  summary.csv      — сводная таблица: трекинг и только детекция
  drift.csv        — дрейф рамки при сцепке гомографий с guided и без
  timeline.png     — inliers и потери объекта по кадрам
  compare_modes.jpg, compare_drift.jpg — режимы трекера на одних и тех же кадрах

    python3 evaluate.py                       # три тестовых видео
    python3 evaluate.py data/my.mp4 --label "Книга"
"""

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main import track_video
from tracker import SiftTracker, draw

TEST_VIDEOS = ["data/mona-lisa.avi", "data/mona-lisa-blur.avi",
               "data/mona-lisa-blur-extra-credit.avi"]
OUT = Path("results")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def make_media(raw, name, n_frames):
    """H.264-видео, GIF-превью и сетка кадров из сырого результата."""
    mp4, gif, grid = OUT / f"{name}.mp4", OUT / f"{name}.gif", OUT / f"{name}_grid.jpg"
    ffmpeg("-i", str(raw), "-c:v", "libx264", "-crf", "28", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4))
    # GIF: каждый 4-й кадр, 288 px, общая палитра на 64 цвета
    ffmpeg("-i", str(mp4), "-vf",
           "select='not(mod(n,4))',setpts=N/8/TB,scale=288:-1:flags=lanczos,"
           "split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none",
           "-r", "8", str(gif))
    # Сетка из 12 равномерно взятых кадров: 4×3 для горизонтального видео,
    # 6×2 для вертикального
    cap = cv2.VideoCapture(str(raw))
    portrait = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    step = n_frames // 12 + 1
    layout = "scale=200:-1,tile=6x2" if portrait else "scale=320:-1,tile=4x3"
    ffmpeg("-i", str(mp4), "-vf", f"select='not(mod(n,{step}))',{layout}",
           "-frames:v", "1", "-q:v", "3", str(grid))
    raw.unlink()


def stats(rows):
    n = len(rows)
    found = [r for r in rows if r["found"]]
    by = {k: sum(r["source"] == k for r in rows) for k in ("ref", "guided", "prev")}
    return dict(frames=n, found=len(found), found_pct=round(100 * len(found) / n, 1),
                by_ref=by["ref"], by_guided=by["guided"], by_prev=by["prev"],
                mean_inliers=round(sum(r["inliers"] for r in found) / max(len(found), 1), 1),
                ms_per_frame=round(sum(r["time_ms"] for r in rows) / n, 1))


def drift_experiment(video, split=200, gt_inliers=30):
    """Насколько уезжает рамка, если объект ведётся только сцепкой гомографий.

    Разметки нет, поэтому опорными считаются кадры, где прямая детекция по
    эталону даёт не меньше gt_inliers inliers: там рамка точная. Трекер
    запускается в режиме «сначала сцепка» (прямая детекция только при потере),
    и его рамка сравнивается с опорной: средняя ошибка по четырём углам.
    Учитываются только кадры, где рамка получена трекингом (сцепка или
    guided), а не повторной детекцией по эталону.
    """
    cap, frames = cv2.VideoCapture(video), []
    while (f := cap.read())[0]:
        frames.append(f[1])
    det = SiftTracker(frames[0], use_prev=False)
    gt = {}
    for i, f in enumerate(frames):
        r = det.process(f)
        if r.found and r.inliers >= gt_inliers:
            gt[i] = r.corners

    rows = []
    for mode, guided in (("сцепка", False), ("сцепка + guided", True)):
        tr = SiftTracker(frames[0], strong_inliers=10 ** 9, guided=guided)
        err = {}
        for i, f in enumerate(frames):
            r = tr.process(f)
            if i in gt and r.found and r.source in ("prev", "guided"):
                err[i] = np.linalg.norm(r.corners - gt[i], axis=1).mean()
        row = dict(video=Path(video).stem, mode=mode)
        for part, keep in ((f"до {split}", lambda i: i < split),
                           (f"после {split}", lambda i: i >= split)):
            e = [v for i, v in err.items() if keep(i)]
            row[f"кадров {part}"] = len(e)
            row[f"ср. ошибка {part}, px"] = round(float(np.mean(e)), 1) if e else ""
            row[f"макс. ошибка {part}, px"] = round(float(np.max(e)), 1) if e else ""
        rows.append(row)
    return rows


def compare_figure(video, modes, frames_to_show, path, label="Mona Lisa", zoom=False):
    """Сетка: строки — режимы трекера, столбцы — выбранные кадры.

    zoom=True — вместо всего кадра показывается участок вокруг объекта
    (по рамке последнего режима), одинаковый для всех строк: так виден
    сдвиг рамки, когда объект далеко и мелкий.
    """
    cap, frames = cv2.VideoCapture(video), []
    while (f := cap.read())[0] and len(frames) <= max(frames_to_show):
        frames.append(f[1])
    cap.release()
    results = []
    for _, params in modes:
        tr = SiftTracker(frames[0], **params)
        results.append({i: r for i, f in enumerate(frames)
                        if (r := tr.process(f)) and i in frames_to_show})

    rows = []
    for (title, _), res_by_frame in zip(modes, results):
        tiles = []
        for i in frames_to_show:
            f, res = frames[i], res_by_frame[i]
            if zoom:
                c = results[-1][i].corners
                cx, cy = c.mean(axis=0)
                half_w = max(np.ptp(c[:, 0]), np.ptp(c[:, 1]) * 16 / 9) * 0.8 + 10
                x0 = int(np.clip(cx - half_w, 0, f.shape[1] - 2 * half_w))
                y0 = int(np.clip(cy - half_w * 9 / 16, 0, f.shape[0] - 2 * half_w * 9 / 16))
                crop = f[y0:y0 + int(2 * half_w * 9 / 16), x0:x0 + int(2 * half_w)]
                tile = cv2.resize(crop, (320, 180), interpolation=cv2.INTER_CUBIC)
                if res.found:
                    k = 320 / crop.shape[1]
                    quad = ((res.corners - [x0, y0]) * k).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(tile, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                tile = cv2.resize(draw(f, res, label, show_points=False), (320, 180))
            text = f"{title}, кадр {i}" + ("" if res.found else ": не найден")
            cv2.rectangle(tile, (0, 0), (320, 22), (255, 255, 255), -1)
            cv2.putText(tile, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)


def plot_timeline(all_rows, path):
    """Малые кратные: по панели на видео, inliers по эталону и по прошлому кадру,
    серые полосы — кадры, где объект не найден."""
    blue, orange = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(len(all_rows), 1, figsize=(10, 2.3 * len(all_rows)),
                             sharex=True, sharey=True)
    if len(all_rows) == 1:
        axes = [axes]
    for ax, (name, rows) in zip(axes, all_rows.items()):
        x = [r["frame"] for r in rows]
        for r in rows:
            if not r["found"]:
                ax.axvspan(r["frame"] - 0.5, r["frame"] + 0.5, color="#d9d8d4", lw=0)
        ax.plot(x, [r["ref_inliers"] for r in rows], color=blue, lw=1.2,
                label="inliers по эталону")
        ax.plot(x, [r["inliers"] if r["source"] in ("prev", "guided") else None
                    for r in rows],
                color=orange, lw=1.2, label="inliers при трекинге (сцепка / guided)")
        ax.axhline(25, color="#52514e", lw=0.8, ls="--")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(bottom=0)
        ax.set_title(name, loc="left", fontsize=10, color="#0b0b0b")
        ax.set_ylabel("пар")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e8e7e3", lw=0.6)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel("номер кадра  (серым — объект не найден, пунктир — порог 25 inliers)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=TEST_VIDEOS)
    ap.add_argument("--label", default="Mona Lisa")
    ap.add_argument("--plot", help="имя файла графика (по умолчанию timeline<суффикс>.png)")
    args = ap.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("Нужен ffmpeg")
    OUT.mkdir(exist_ok=True)

    summary, all_rows = [], {}
    for v in args.videos:
        name = Path(v).stem
        raw = OUT / f"{name}_raw.mp4"
        rows = track_video(v, raw, args.label, csv_path=OUT / f"{name}.csv")
        make_media(raw, name, len(rows))
        all_rows[name] = rows
        summary.append(dict(video=name, mode="трекинг", **stats(rows)))
        det = track_video(v, None, args.label, use_prev=False, verbose=False)
        summary.append(dict(video=name, mode="только детекция", **stats(det)))

    suffix = "" if args.videos == TEST_VIDEOS else f"_{Path(args.videos[0]).stem}"
    plot_timeline(all_rows, OUT / (args.plot or f"timeline{suffix}.png"))
    write_csv(OUT / f"summary{suffix}.csv", summary)
    drift = [row for v in args.videos for row in drift_experiment(v)]
    write_csv(OUT / f"drift{suffix}.csv", drift)
    if args.videos == TEST_VIDEOS:
        compare_figure(TEST_VIDEOS[0], [("только детекция", dict(use_prev=False)),
                                        ("трекинг", dict())],
                       [60, 180, 300, 520], OUT / "compare_modes.jpg")
        compare_figure(TEST_VIDEOS[0],
                       [("сцепка", dict(strong_inliers=10 ** 9, guided=False)),
                        ("сцепка + guided", dict(strong_inliers=10 ** 9))],
                       [100, 250, 400, 435], OUT / "compare_drift.jpg")
    elif Path(args.videos[0]).stem == "kar-karych":
        # Собственное видео: кадры, где одна сцепка уезжает сильнее всего
        compare_figure(args.videos[0],
                       [("сцепка", dict(strong_inliers=10 ** 9, guided=False)),
                        ("сцепка + guided", dict(strong_inliers=10 ** 9))],
                       [300, 347, 482, 505], OUT / "compare_drift_kar-karych.jpg", zoom=True)
    print()
    for row in summary + drift:
        print(row)


if __name__ == "__main__":
    main()
