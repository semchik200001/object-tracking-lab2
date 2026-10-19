"""Трекинг плоского объекта по ключевым точкам SIFT.

Эталон объекта задаётся первым кадром видео. На каждом кадре:

1. Детекция: ключевые точки кадра сопоставляются с эталоном
   (полный перебор + тест Лоу), по парам методом RANSAC оценивается гомография
   «эталон → кадр».
2. Трекинг: если связь с эталоном слабая (объект далеко, размыт),
   кадр сопоставляется с последним кадром, где объект был найден, точнее
   с точками внутри найденной там рамки. Получается гомография
   «прошлый кадр → кадр», и она умножается на гомографию прошлого кадра:
   H_t = H_{t-1→t} · H_{t-1}.
3. Уточнение (guided matching): сцепленная гомография служит прогнозом.
   Точки эталона проецируются на кадр, и пара для каждой ищется только
   рядом с прогнозом. Так находится больше пар с эталоном, и рамка снова
   привязывается к нему, а ошибки сцепки не накапливаются.

Углы эталона переносятся на кадр найденной гомографией и проверяются
на правдоподобность.
"""

from dataclasses import dataclass, field
import time

import cv2
import numpy as np


@dataclass
class TrackResult:
    found: bool
    corners: np.ndarray | None = None   # 4×2, углы объекта на кадре
    H: np.ndarray | None = None         # гомография «эталон → кадр»
    source: str = ""                    # "ref" — по эталону, "guided" — по эталону
                                        # с прогнозом, "prev" — по прошлому кадру
    matches: int = 0                    # пар после теста Лоу
    inliers: int = 0                    # пар, согласных с гомографией
    ref_inliers: int = 0                # inliers при сопоставлении с эталоном
    keypoints: int = 0                  # ключевых точек на кадре
    time_ms: float = 0.0
    inlier_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    reason: str = ""                    # почему объект не найден


@dataclass
class _Estimate:
    H: np.ndarray | None
    matches: int
    inliers: int
    dst_inliers: np.ndarray


class SiftTracker:
    def __init__(self, reference, use_prev=True, ratio=0.75, ransac_thresh=5.0,
                 min_matches=10, min_inliers=10, strong_inliers=25, min_area=0.002,
                 max_lost=15, guided=True, guided_radius=25.0, guided_inliers=15):
        """
        reference      — эталонное изображение объекта (BGR или серое)
        use_prev       — использовать сопоставление с прошлым кадром
                         (False — только детекция по эталону)
        ratio          — порог теста Лоу
        ransac_thresh  — допустимая ошибка репроекции в RANSAC, пиксели
        min_matches    — минимум пар для оценки гомографии
        min_inliers    — минимум inlier-пар, чтобы принять гомографию
        strong_inliers — столько inliers по эталону достаточно, чтобы
                         не обращаться к прошлому кадру
        min_area       — минимальная площадь рамки как доля площади кадра
        max_lost       — сколько кадров подряд без объекта помнить последний
                         удачный кадр и пытаться связаться с ним
        guided         — уточнять сцепленную гомографию по эталону
        guided_radius  — радиус поиска пары вокруг прогноза в пикселях кадра
                         шириной 640; для других размеров масштабируется
                         пропорционально ширине эталона
        guided_inliers — минимум inliers, чтобы принять уточнение
        """
        self.use_prev = use_prev
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.strong_inliers = strong_inliers
        self.min_area = min_area
        self.max_lost = max_lost
        self.guided = guided
        self.guided_inliers = guided_inliers

        self.sift = cv2.SIFT_create()
        # Полный перебор по евклидовой метрике: точек немного (сотни), поэтому
        # он быстр, точен и, в отличие от FLANN, даёт одинаковый результат
        # при каждом запуске
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        ref_gray = _to_gray(reference)
        h, w = ref_gray.shape
        self.guided_radius = guided_radius * w / 640
        self.ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        self.ref_pts, self.ref_des = self._detect(ref_gray)
        if self.ref_des is None or len(self.ref_pts) < min_matches:
            raise ValueError("На эталоне слишком мало ключевых точек")

        # Состояние трекинга: точки последнего удачного кадра внутри рамки,
        # его гомография и число кадров подряд, где объект не найден
        self.prev_pts = None
        self.prev_des = None
        self.prev_H = None
        self.lost = 0

    def process(self, frame):
        t0 = time.perf_counter()
        res = self._process(_to_gray(frame))
        res.time_ms = (time.perf_counter() - t0) * 1000
        return res

    def _detect(self, gray):
        """Ключевые точки SIFT: координаты (N×2) и дескрипторы (N×128)."""
        kp, des = self.sift.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kp]).reshape(-1, 2)
        return pts, des

    def _process(self, gray):
        pts, des = self._detect(gray)
        if des is None or len(pts) < 2:
            self._lose()
            return TrackResult(False, keypoints=len(pts), reason="нет точек")

        # 1. Детекция по эталону
        ref = self._estimate(self.ref_pts, self.ref_des, pts, des)
        res = self._accept(ref, None, gray.shape, "ref", self.strong_inliers)

        # 2. Связь с эталоном слабая — трекинг от последнего удачного кадра
        if res is None and self.use_prev and self.prev_H is not None:
            prev = self._estimate(self.prev_pts, self.prev_des, pts, des)
            res = self._accept(prev, self.prev_H, gray.shape, "prev", self.min_inliers)
            # 3. Уточнение по эталону с прогнозом от сцепки
            if res is not None and self.guided:
                g = self._guided(res.H, pts, des)
                refined = self._accept(g, None, gray.shape, "guided", self.guided_inliers)
                # Пары искались рядом с прогнозом, поэтому честное уточнение
                # не может сдвинуть рамку дальше радиуса поиска
                if refined is not None and np.linalg.norm(
                        refined.corners - res.corners, axis=1).mean() < self.guided_radius:
                    res = refined

        # 4. Прошлого кадра нет или он не помог — слабая детекция по эталону
        if res is None:
            res = self._accept(ref, None, gray.shape, "ref", self.min_inliers)

        if res is None:
            self._lose()
            reason = ("мало пар" if ref.matches < self.min_matches
                      else "мало inliers" if ref.inliers < self.min_inliers
                      else "рамка неправдоподобна")
            return TrackResult(False, matches=ref.matches, inliers=ref.inliers,
                               ref_inliers=ref.inliers, keypoints=len(pts),
                               reason=reason)

        res.ref_inliers = ref.inliers
        res.keypoints = len(pts)
        self.prev_H = res.H
        self.lost = 0
        # Запоминаем точки внутри рамки: по ним будем искать объект на следующем кадре
        inside = _inside(pts, res.corners)
        self.prev_pts, self.prev_des = pts[inside], des[inside]
        if len(self.prev_pts) < self.min_matches:
            self._reset()
        return res

    def _estimate(self, src_pts, src_des, dst_pts, dst_des):
        """Сопоставление двух наборов точек и гомография src → dst."""
        none = _Estimate(None, 0, 0, np.empty((0, 2)))
        if src_des is None or len(src_des) < 2:
            return none
        pairs = self.matcher.knnMatch(src_des, dst_des, k=2)
        good = [p[0] for p in pairs
                if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if len(good) < self.min_matches:
            return _Estimate(None, len(good), 0, none.dst_inliers)
        src = src_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        dst = dst_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return _Estimate(None, len(good), 0, none.dst_inliers)
        mask = mask.ravel().astype(bool)
        return _Estimate(H, len(good), int(mask.sum()), dst[mask].reshape(-1, 2))

    def _guided(self, H_pred, pts, des):
        """Сопоставление с эталоном в окрестности прогноза H_pred."""
        none = _Estimate(None, 0, 0, np.empty((0, 2)))
        pred = cv2.perspectiveTransform(self.ref_pts.reshape(-1, 1, 2), H_pred).reshape(-1, 2)
        # Кандидаты для точки эталона — точки кадра не дальше радиуса от прогноза
        near = ((pred[:, None, :] - pts[None, :, :]) ** 2).sum(-1) < self.guided_radius ** 2
        if near.sum() < self.min_matches:
            return none
        a, b = self.ref_des.astype(np.float32), des.astype(np.float32)
        d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        dist = np.sqrt(np.maximum(d2, 0))
        dist[~near] = np.inf
        order = np.argsort(dist, axis=1)[:, :2]
        rows = np.arange(len(dist))
        best, second = dist[rows, order[:, 0]], dist[rows, order[:, 1]]
        # Тест Лоу среди соседей: нужны хотя бы два кандидата
        ok = np.isfinite(second) & (best < self.ratio * second)
        if ok.sum() < self.min_matches:
            return _Estimate(None, int(ok.sum()), 0, none.dst_inliers)
        src = self.ref_pts[ok].reshape(-1, 1, 2)
        dst = pts[order[ok, 0]].reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return _Estimate(None, int(ok.sum()), 0, none.dst_inliers)
        mask = mask.ravel().astype(bool)
        return _Estimate(H, int(ok.sum()), int(mask.sum()), dst[mask].reshape(-1, 2))

    def _accept(self, est, H_prev, shape, source, need_inliers):
        """Принимает оценку, если inliers хватает и рамка правдоподобна."""
        if est.H is None or est.inliers < need_inliers:
            return None
        H = est.H if H_prev is None else est.H @ H_prev
        corners = cv2.perspectiveTransform(self.ref_corners, H).reshape(-1, 2)
        if not _plausible(corners, shape, self.min_area):
            return None
        return TrackResult(True, corners, H, source, est.matches, est.inliers,
                           inlier_points=est.dst_inliers)

    def _lose(self):
        """Объект не найден: последний удачный кадр помним ещё max_lost кадров."""
        self.lost += 1
        if self.lost > self.max_lost:
            self._reset()

    def _reset(self):
        self.prev_pts = self.prev_des = self.prev_H = None


def _plausible(corners, frame_shape, min_area):
    """Отсекает вырожденные гомографии: самопересечения, «вывернутую» рамку,
    слишком маленькую, огромную или неправдоподобно вытянутую рамку."""
    if not np.all(np.isfinite(corners)):
        return False
    quad = corners.astype(np.float32)
    if not cv2.isContourConvex(quad):
        return False
    # Знак площади: порядок обхода углов должен сохраниться (без отражения)
    signed = 0.5 * np.sum(quad[:, 0] * np.roll(quad[:, 1], -1)
                          - np.roll(quad[:, 0], -1) * quad[:, 1])
    h, w = frame_shape[:2]
    if not min_area * w * h <= signed <= 10 * w * h:
        return False
    sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    return sides.max() <= 10 * sides.min()


def _inside(pts, corners):
    quad = corners.astype(np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(quad, (float(x), float(y)), False) >= 0
                     for x, y in pts], dtype=bool)


def _to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def draw(frame, res, label, show_points=True, color=(0, 255, 0)):
    """Рисует на кадре рамку объекта, подпись и inlier-точки."""
    out = frame.copy()
    if not res.found:
        return out
    if show_points:
        for x, y in res.inlier_points.astype(int):
            cv2.circle(out, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
    quad = res.corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [quad], True, color, 2, cv2.LINE_AA)

    # Подпись по центру рамки: белый текст с тёмной обводкой
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    cx, cy = res.corners.mean(axis=0)
    org = (int(cx - tw / 2), int(cy + th / 2))
    cv2.putText(out, label, org, font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(out, label, org, font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out
