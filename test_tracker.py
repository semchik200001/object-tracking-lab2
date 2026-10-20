"""Проверка трекера на синтетических кадрах с известной гомографией.

Эталон — текстурное изображение. Кадр получается переносом эталона на фон
по заданной гомографии, поэтому истинное положение углов известно точно.
"""

import cv2
import numpy as np
import pytest

from tracker import SiftTracker

W, H = 640, 360


def texture(seed, size=(H, W), contrast=1.0):
    """Сглаженный шум с крупными и мелкими деталями: много точек SIFT."""
    rng = np.random.default_rng(seed)
    img = np.zeros(size, np.float32)
    for k in (3, 9, 27):
        img += cv2.GaussianBlur(rng.random(size, dtype=np.float32), (0, 0), k / 3) * k
    img = (img - img.min()) / (img.max() - img.min())
    img = 128 + (img - 0.5) * 255 * contrast
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


REF = texture(0)
CORNERS = np.float32([[0, 0], [W, 0], [W, H], [0, H]])


def make_frame(dst_corners, background_seed=1):
    """Эталон, перенесённый в четырёхугольник dst_corners на постороннем фоне."""
    Hm = cv2.getPerspectiveTransform(CORNERS, np.float32(dst_corners))
    frame = texture(background_seed, contrast=0.4)
    warped = cv2.warpPerspective(REF, Hm, (W, H))
    mask = cv2.warpPerspective(np.full((H, W), 255, np.uint8), Hm, (W, H))
    frame[mask > 0] = warped[mask > 0]
    return frame


def corner_error(found, expected):
    return np.linalg.norm(found - np.float32(expected), axis=1).mean()


def quad(cx, cy, scale, angle_deg, tilt=0.0):
    """Четырёхугольник: эталон, уменьшенный, повёрнутый и с наклоном."""
    c = (CORNERS - [W / 2, H / 2]) * scale
    c[:2, 0] *= 1 - tilt          # верхняя сторона короче — перспектива
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return c @ rot.T + [cx, cy]


def test_first_frame_is_whole_frame():
    res = SiftTracker(REF).process(REF)
    assert res.found and res.source == "ref"
    assert corner_error(res.corners, CORNERS) < 0.5


@pytest.mark.parametrize("scale, angle, tilt", [
    (0.6, 0, 0.0), (0.5, 30, 0.0), (0.5, -60, 0.2), (0.4, 90, 0.3), (0.7, 180, 0.1),
])
def test_detection_known_homography(scale, angle, tilt):
    expected = quad(W / 2, H / 2, scale, angle, tilt)
    res = SiftTracker(REF).process(make_frame(expected))
    assert res.found
    assert corner_error(res.corners, expected) < 2.0


def test_no_object_not_found():
    tr = SiftTracker(REF)
    for seed in (5, 6, 7):
        assert not tr.process(texture(seed)).found


def sequence(n=120, s_min=0.2):
    """Объект плавно уменьшается до s_min, возвращается, ездит и поворачивается."""
    scales = np.r_[np.geomspace(0.6, s_min, n // 2), np.geomspace(s_min, 0.6, n - n // 2)]
    return [quad(W / 2 + 60 * np.sin(i / 10), H / 2 + 20 * np.cos(i / 13), s,
                 15 * np.sin(i / 15), 0.1 * (1 + np.sin(i / 20)))
            for i, s in enumerate(scales)]


def run(tracker, quads):
    tracker.process(REF)
    out = []
    for q in quads:
        res = tracker.process(make_frame(q))
        out.append((res, None if not res.found else corner_error(res.corners, q)))
    return out


def test_chain_only():
    """Детекция по эталону только на первом кадре, дальше одна сцепка
    гомографий: проверка формулы H_t = H_{t-1→t} · H_{t-1}. Объект ведётся
    на всех кадрах, но ошибка понемногу накапливается (дрейф)."""
    out = run(SiftTracker(REF, strong_inliers=10 ** 9, guided=False), sequence())
    assert all(r.found and r.source == "prev" for r, _ in out)
    err = [e for _, e in out]
    assert max(err) < 10.0
    assert np.mean(err[-20:]) > 3 * np.mean(err[:20])


def test_guided_refinement():
    """С guided matching рамка привязывается к эталону, и ошибка не растёт:
    в конце последовательности она такая же, как в начале."""
    out = run(SiftTracker(REF, strong_inliers=10 ** 9), sequence())
    assert all(r.found for r, _ in out)
    assert sum(r.source == "guided" for r, _ in out) > len(out) // 2
    err = [e for _, e in out]
    assert max(err) < 2.0
    assert np.mean(err[-20:]) < 0.5


def test_recovers_after_lost_frames():
    """Пять кадров без объекта: последний удачный кадр помнится, и после
    возвращения объекта трекинг продолжается от него."""
    tr = SiftTracker(REF, strong_inliers=10 ** 9)
    quads = sequence(40)
    run(tr, quads[:20])
    for seed in range(10, 15):
        assert not tr.process(texture(seed, contrast=0.4)).found
    assert tr.prev_H is not None
    res = tr.process(make_frame(quads[20]))
    assert res.found and res.source in ("prev", "guided")
    assert corner_error(res.corners, quads[20]) < 2.0


def test_forgets_after_max_lost():
    """max_lost=3: после трёх кадров без объекта последний удачный кадр
    ещё помнится, после четвёртого забывается."""
    tr = SiftTracker(REF, max_lost=3)
    tr.process(REF)
    for seed in range(10, 13):
        tr.process(texture(seed, contrast=0.4))
    assert tr.prev_H is not None
    tr.process(texture(13, contrast=0.4))
    assert tr.prev_H is None


def test_guided_rejects_blown_up_frame():
    """Уточнение, у которого один угол улетел или рамка схлопнулась,
    отбрасывается; небольшой сдвиг рамки принимается."""
    tr = SiftTracker(REF)
    r = tr.guided_radius
    pred = quad(W / 2, H / 2, 0.5, 10)
    assert tr._close(pred + [0.5 * r, -0.3 * r], pred)
    flown = pred.copy()
    flown[3] += [-3 * r, 3 * r]
    assert not tr._close(flown, pred)
    collapsed = (pred - pred.mean(axis=0)) * [1.0, 0.6] + pred.mean(axis=0)
    assert not tr._close(collapsed, pred)
