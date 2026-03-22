"""
InvisINK — Air-Drawing Math Solver
===================================
Draw mathematical expressions in the air using hand gestures,
then solve them with a thumb-up. Uses MediaPipe for hand tracking,
a TensorFlow CNN for character recognition, and SymPy for solving.

Gestures
--------
- INDEX finger only   → Draw
- INDEX + MIDDLE      → Erase
- THUMB UP            → Solve
- OPEN HAND           → Clear canvas
- Q key               → Quit

Author  : InvisINK Team
Project : Final Year Engineering Project
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import warnings

from sympy import (
    Eq, Float, Integer, NumberSymbol, Tuple,
    cos, diff, integrate, latex, nsimplify, parse_expr,
    pi, simplify, sin, solve as sym_solve, sqrt,
    symbols, tan,
)
from sympy import rad as sym_rad
from sympy.parsing.sympy_parser import (
    parse_expr as parser_parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

# -- Drawing -------------------------------------------------------
BRUSH_THICKNESS: int = 14
ERASER_RADIUS: int = 28
GLOW_BLUR_KERNEL: tuple[int, int] = (19, 19)
GLOW_BLEND_ALPHA: float = 0.55

# -- Recognition thresholds ----------------------------------------
MIN_CONTOUR_W: int = 10
MIN_CONTOUR_H: int = 8
ROW_GROUP_THRESHOLD: int = 80
EQUALS_MERGE_Y_DIST: int = 55
EQUALS_OVERLAP_RATIO: float = 0.35
EQUALS_ASPECT_RATIO: float = 1.2
CONFIDENCE_THRESHOLD_EQUALS: float = 0.80
CONFIDENCE_THRESHOLD_CORRECTION: float = 0.70

# -- Solve flash animation -----------------------------------------
SOLVE_FLASH_FRAMES: int = 14
SOLVE_FLASH_MAX_ALPHA: float = 0.28
SOLVE_FLASH_BRIGHTNESS: int = 240

# -- UI layout -----------------------------------------------------
TOP_PANEL_HEIGHT: int = 110
BOTTOM_BAR_HEIGHT: int = 54
BADGE_RECT: tuple[int, int, int, int] = (16, 12, 176, 52)
MAX_EXPR_CHARS: int = 72
MAX_SOL_CHARS: int = 68

# -- MediaPipe -----------------------------------------------------
MAX_HANDS: int = 2
DETECTION_CONFIDENCE: float = 0.80
TRACKING_CONFIDENCE: float = 0.70

# -- FPS -----------------------------------------------------------
FPS_SMOOTHING: float = 0.9
FPS_INITIAL: float = 30.0

# -- Design tokens -------------------------------------------------
COLOR = {
    "neon_draw":    (0,   240, 160),
    "neon_glow":    (0,   200, 120),
    "accent_cyan":  (255, 220,   0),
    "accent_gold":  (0,   200, 255),
    "success":      (80,  220, 100),
    "text_dim":     (130, 130, 160),
    "danger":       (80,   80, 240),
    "erase":        (180, 100, 255),
    "overlay_dark": (8,     8,  14),
}

MODE_COLORS = {
    "DRAW":  (0,   240, 160),
    "ERASE": (180, 100, 255),
    "SOLVE": (0,   200, 255),
    "CLEAR": (80,   80, 240),
    "IDLE":  (130, 130, 160),
}

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX

# -- Character corrections for low-confidence predictions ----------
CHAR_CORRECTIONS: dict[str, str] = {
    'z': 'x', 'Z': 'x', 'l': '1', 'O': '0',
    'o': '0', 'I': '1', 'S': '5', 's': '5',
}

# -- Logging --------------------------------------------------------
logger = logging.getLogger("InvisINK")


# ═══════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def prepare_roi(roi: np.ndarray, size: int = 32) -> np.ndarray:
    """Resize and center a character ROI into a square canvas for CNN input.

    Parameters
    ----------
    roi : np.ndarray
        Grayscale image of a single character bounding box.
    size : int
        Target square dimension (default 32×32).

    Returns
    -------
    np.ndarray
        Normalized float array shaped ``(1, size, size, 1)`` ready for prediction.
    """
    h, w = roi.shape
    scale = (size - 10) / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)

    final = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - nh) // 2
    x_off = (size - nw) // 2
    final[y_off:y_off + nh, x_off:x_off + nw] = resized

    return final.reshape(1, size, size, 1) / 255.0


# ═══════════════════════════════════════════════════════════════════
#  EXPRESSION PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

# SymPy transformation pipeline — replaces fragile hand-rolled regexes
SYMPY_TRANSFORMATIONS = (
    standard_transformations
    + (implicit_multiplication_application, convert_xor)
)


def preprocess_expr(raw: str) -> str:
    """Clean and normalize a raw recognized expression string for SymPy parsing.

    Handles caret-to-power conversion, double-dash-to-equals, trig degree
    wrapping, and backslash removal. Heavy lifting (implicit multiplication,
    etc.) is delegated to SymPy's own transformation pipeline.

    Parameters
    ----------
    raw : str
        The raw expression string assembled from recognized characters.

    Returns
    -------
    str
        A cleaned expression string suitable for ``parse_expr``.
    """
    s = raw.strip().lower()
    s = s.replace('^', '**')
    s = s.replace('--', '=')

    # Wrap trig functions in radians (handles simple cases)
    for fn in ('sin', 'cos', 'tan'):
        s = re.sub(rf'{fn}\(([^)]+)\)', rf'{fn}(rad(\1))', s)

    # Remove stray backslashes
    s = s.replace('\\', '')
    return s


# ═══════════════════════════════════════════════════════════════════
#  SOLUTION FORMATTING
# ═══════════════════════════════════════════════════════════════════

def fmt(val: Any) -> str:
    """Format a single SymPy value for display.

    Returns integers when the value is close to a whole number,
    otherwise a compact decimal, or complex notation if needed.
    """
    try:
        n = complex(val.evalf())
        if abs(n.imag) < 1e-9:
            r = n.real
            if abs(r - round(r)) < 1e-9:
                return str(int(round(r)))
            return f"{r:.5g}"
        else:
            return f"{n.real:.4g}+{n.imag:.4g}i"
    except Exception:
        return str(val)


def format_solution(sol: Any, vars_found: set) -> str:
    """Produce a human-readable solution string from SymPy output.

    Handles dicts (multi-variable), lists of tuples (simultaneous),
    plain lists, and single values.
    """
    if sol is None or sol == [] or sol == {}:
        return "No solution found"

    var_names = [str(v) for v in vars_found]

    if isinstance(sol, dict):
        parts = [f"{k} = {fmt(v)}" for k, v in sol.items()]
        return ",  ".join(parts)

    if isinstance(sol, list) and sol and isinstance(sol[0], (tuple, Tuple)):
        results = []
        for tup in sol:
            parts = [f"{var_names[i]} = {fmt(tup[i])}" for i in range(len(tup))]
            results.append("(" + ", ".join(parts) + ")")
        return " or ".join(results)

    if isinstance(sol, list):
        label = var_names[0] if var_names else "x"
        real_sols = [v for v in sol if abs(complex(v.evalf()).imag) < 1e-9]
        display = real_sols if real_sols else sol
        if len(display) == 1:
            return f"{label} = {fmt(display[0])}"
        return label + " = " + ",  ".join(fmt(v) for v in display)

    return str(sol)


# ═══════════════════════════════════════════════════════════════════
#  ADVANCED MATH ENGINE
# ═══════════════════════════════════════════════════════════════════

def advanced_math_solver(final_lines: list[str]) -> str:
    """Parse and solve one or more lines of mathematical expressions.

    Supports:
      - Arithmetic evaluation (``2+3``)
      - Single and simultaneous equations (``x+2=5``)
      - Differentiation (``diff(x**2)``)
      - Integration (``int(x**2)``)

    Parameters
    ----------
    final_lines : list[str]
        Each element is one line of recognized characters.

    Returns
    -------
    str
        Human-readable solution string, or an error message.
    """
    try:
        x, y, z = symbols('x y z')
        local_dict: dict[str, Any] = {
            'sin': sin, 'cos': cos, 'tan': tan,
            'pi': pi, 'rad': sym_rad, 'sqrt': sqrt,
            'x': x, 'y': y, 'z': z,
        }
        all_equations: list[Eq] = []
        vars_found: set = set()

        for raw_line in final_lines:
            line = raw_line.strip()
            if not line:
                continue

            # -- Differentiation -----------------------------------
            if 'diff' in line.lower():
                m = re.search(r'diff\((.+)\)', line, re.IGNORECASE)
                if m:
                    inner = preprocess_expr(m.group(1))
                    expr = parser_parse_expr(
                        inner, local_dict=local_dict,
                        transformations=SYMPY_TRANSFORMATIONS,
                    )
                    return f"d/dx = {diff(expr, x)}"

            # -- Integration ---------------------------------------
            if line.lower().startswith('int(') or re.search(r'\bint\(', line, re.I):
                m = re.search(r'int\((.+)\)', line, re.IGNORECASE)
                if m:
                    inner = preprocess_expr(m.group(1))
                    expr = parser_parse_expr(
                        inner, local_dict=local_dict,
                        transformations=SYMPY_TRANSFORMATIONS,
                    )
                    return f"∫dx = {integrate(expr, x)}"

            # -- Equation ------------------------------------------
            if '=' in line:
                lhs_raw, rhs_raw = line.split('=', 1)
                rhs_raw = rhs_raw.strip() or '0'
                lhs = preprocess_expr(lhs_raw)
                rhs = preprocess_expr(rhs_raw)
                eq = Eq(
                    parser_parse_expr(lhs, local_dict=local_dict,
                                      transformations=SYMPY_TRANSFORMATIONS),
                    parser_parse_expr(rhs, local_dict=local_dict,
                                      transformations=SYMPY_TRANSFORMATIONS),
                )
                all_equations.append(eq)
                vars_found.update(eq.free_symbols)

            # -- Plain expression ----------------------------------
            else:
                clean = preprocess_expr(line)
                res = parser_parse_expr(
                    clean, local_dict=local_dict,
                    transformations=SYMPY_TRANSFORMATIONS,
                ).evalf()
                try:
                    n = complex(res)
                    if abs(n.imag) < 1e-9:
                        r = n.real
                        return f"= {int(round(r)) if abs(r - round(r)) < 1e-9 else f'{r:.6g}'}"
                    return f"= {n.real:.4g} + {n.imag:.4g}i"
                except Exception:
                    return f"= {res}"

        if all_equations:
            sol = sym_solve(all_equations, list(vars_found))
            return format_solution(sol, vars_found)

        return "Nothing to solve"

    except Exception as e:
        logger.exception("Solver error")
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════
#  UI DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════

def draw_rounded_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 12,
    thickness: int = -1,
    alpha: float = 0.6,
) -> None:
    """Draw a rounded rectangle with alpha blending onto *img* in-place."""
    x1, y1 = pt1
    x2, y2 = pt2
    ov = img.copy()
    cv2.rectangle(ov, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(ov, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [
        (x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius), (x2 - radius, y2 - radius),
    ]:
        cv2.circle(ov, (cx, cy), radius, color, thickness)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def draw_circle_glow(
    img: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled circle with a soft glow halo onto *img* in-place."""
    for i in range(3, 0, -1):
        ov = img.copy()
        cv2.circle(ov, center, radius + i * 6, color, 2)
        cv2.addWeighted(ov, 0.12 * i / 3, img, 1 - 0.12 * i / 3, 0, img)
    cv2.circle(img, center, radius, color, -1)


def put_text_shadow(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Put text with a dark drop-shadow for readability on any background."""
    cv2.putText(img, text, (pos[0] + 2, pos[1] + 2), font, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def truncate_text(text: str, max_chars: int = MAX_EXPR_CHARS) -> str:
    """Return *text* truncated with an ellipsis if it exceeds *max_chars*."""
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


# ═══════════════════════════════════════════════════════════════════
#  CONTOUR → CHARACTER PIPELINE
# ═══════════════════════════════════════════════════════════════════

def classify_char(model: tf.keras.Model, label_map: dict, roi: np.ndarray) -> tuple[str, float]:
    """Run the CNN on a single character ROI and return (char, confidence).

    Applies heuristic corrections for commonly confused characters
    when confidence is below threshold.
    """
    pred = model.predict(prepare_roi(roi), verbose=0)
    idx = int(np.argmax(pred))
    conf = float(pred[0][idx])
    char = str(label_map[idx]).replace('\\', '').strip()

    h, w = roi.shape
    aspect = w / max(h, 1)

    # Wide, flat strokes are likely '='
    if aspect > 2.5 and h > 6 and conf < CONFIDENCE_THRESHOLD_EQUALS:
        char = '='

    # Swap commonly misclassified characters
    if conf < CONFIDENCE_THRESHOLD_CORRECTION and char in CHAR_CORRECTIONS:
        char = CHAR_CORRECTIONS[char]

    return char, conf


def merge_equals(rects: list[list[int]]) -> list[list[Any]]:
    """Merge horizontally-aligned, wide bounding boxes into '=' symbols.

    Two horizontally-overlapping, individually wide rectangles that are
    vertically close are treated as the top and bottom bars of an equals sign.

    Returns a list of ``[x, y, w, h, forced_char | None]``.
    """
    merged: list[list[Any]] = []
    skip: set[int] = set()
    rects.sort(key=lambda r: r[0])

    for i in range(len(rects)):
        if i in skip:
            continue
        x1, y1, w1, h1 = rects[i]
        found_pair = False
        for j in range(i + 1, len(rects)):
            if j in skip:
                continue
            x2, y2, w2, h2 = rects[j]
            x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
            both_wide = (w1 > h1 * EQUALS_ASPECT_RATIO) and (w2 > h2 * EQUALS_ASPECT_RATIO)
            if (x_overlap > min(w1, w2) * EQUALS_OVERLAP_RATIO
                    and abs(y1 - y2) < EQUALS_MERGE_Y_DIST
                    and both_wide):
                mx = min(x1, x2)
                my = min(y1, y2)
                merged.append([
                    mx, my,
                    max(x1 + w1, x2 + w2) - mx,
                    max(y1 + h1, y2 + h2) - my,
                    '=',
                ])
                skip.add(i)
                skip.add(j)
                found_pair = True
                break
        if not found_pair and i not in skip:
            merged.append([x1, y1, w1, h1, None])

    return merged


def run_solve(
    model: tf.keras.Model,
    label_map: dict,
    canvas: np.ndarray,
) -> tuple[str, str]:
    """Extract characters from the drawing canvas, assemble lines, and solve.

    Returns
    -------
    tuple[str, str]
        ``(expression_string, solution_string)``
    """
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[list[int]] = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > MIN_CONTOUR_W and bh > MIN_CONTOUR_H:
            rects.append([bx, by, bw, bh])
    if not rects:
        return "", "Draw something first"

    items = merge_equals(rects)
    char_items: list[dict[str, Any]] = []

    for item in items:
        bx, by, bw, bh, forced_char = item
        if forced_char == '=':
            char_items.append({'x': bx, 'y': by, 'w': bw, 'h': bh, 'char': '='})
        else:
            roi = canvas[by:by + bh, bx:bx + bw]
            char, conf = classify_char(model, label_map, roi)
            logger.debug("  [%4d,%3d] %d×%d  →  '%s'  conf=%.2f", bx, by, bw, bh, char, conf)
            char_items.append({'x': bx, 'y': by, 'w': bw, 'h': bh, 'char': char})

    # Group characters into rows by vertical proximity
    char_items.sort(key=lambda c: c['y'])
    rows: list[list[dict]] = []
    curr_row: list[dict] = [char_items[0]]
    for item in char_items[1:]:
        if abs(item['y'] - curr_row[-1]['y']) < ROW_GROUP_THRESHOLD:
            curr_row.append(item)
        else:
            rows.append(curr_row)
            curr_row = [item]
    rows.append(curr_row)

    final_lines = [
        "".join(c['char'] for c in sorted(row, key=lambda c: c['x']))
        for row in rows
    ]

    logger.info("--- SOLVE ---")
    for ln in final_lines:
        logger.info("  Line: %s", ln)

    result = advanced_math_solver(final_lines)
    logger.info("  Result: %s", result)

    return "  |  ".join(final_lines), result or "No result"


# ═══════════════════════════════════════════════════════════════════
#  FINGER STATE DETECTOR
# ═══════════════════════════════════════════════════════════════════

def fingers_up(lm: Any, hand_label: str) -> list[bool]:
    """Return a 5-element list indicating which fingers are raised.

    Index mapping: ``[thumb, index, middle, ring, pinky]``.
    Thumb detection is mirrored based on *hand_label*.
    """
    tips = [4, 8, 12, 16, 20]
    f: list[bool] = []

    # Thumb — compare tip x vs. IP joint x (mirrored for left hand)
    t_tip = lm.landmark[tips[0]].x
    t_ip = lm.landmark[tips[0] - 1].x
    f.append(t_tip < t_ip if hand_label == "Right" else t_tip > t_ip)

    # Other fingers — tip y above PIP joint y
    for i in range(1, 5):
        f.append(lm.landmark[tips[i]].y < lm.landmark[tips[i] - 2].y)

    return f


# ═══════════════════════════════════════════════════════════════════
#  HUD RENDERER
# ═══════════════════════════════════════════════════════════════════

def render_hud(
    display: np.ndarray,
    mode: str,
    last_expression: str,
    last_solution: str,
    handedness_str: str,
) -> None:
    """Draw the full heads-up display: top panel, badges, and bottom hint bar."""
    h, w = display.shape[:2]
    mode_col = MODE_COLORS.get(mode, COLOR["text_dim"])

    # ── Top panel background ──────────────────
    ov = display.copy()
    cv2.rectangle(ov, (0, 0), (w, TOP_PANEL_HEIGHT), COLOR["overlay_dark"], -1)
    cv2.addWeighted(ov, 0.82, display, 0.18, 0, display)

    # Accent separator line + glow
    cv2.line(display, (0, TOP_PANEL_HEIGHT), (w, TOP_PANEL_HEIGHT), mode_col, 2)
    gv = display.copy()
    cv2.line(gv, (0, TOP_PANEL_HEIGHT), (w, TOP_PANEL_HEIGHT), mode_col, 8)
    cv2.addWeighted(gv, 0.20, display, 0.80, 0, display)

    # ── Mode badge (top-left) ─────────────────
    bx1, by1, bx2, by2 = BADGE_RECT
    draw_rounded_rect(display, (bx1, by1), (bx2, by2), mode_col, radius=8, alpha=0.18)
    cv2.rectangle(display, (bx1, by1), (bx2, by2), mode_col, 1)
    put_text_shadow(display, mode, (30, 41), FONT, 0.72, mode_col)

    # ── Handedness (top-right) ────────────────
    hand_text = f"{handedness_str} Hand"
    (htw, _), _ = cv2.getTextSize(hand_text, FONT_MONO, 0.6, 1)
    put_text_shadow(display, hand_text, (w - htw - 20, 38), FONT_MONO, 0.6, COLOR["accent_gold"])

    # ── ROW 2 : Expression — LEFT-aligned ────
    expr_str = truncate_text(last_expression) if last_expression else "—"
    put_text_shadow(display, "expr:", (20, 76), FONT_MONO, 0.48, COLOR["text_dim"])
    put_text_shadow(display, expr_str, (76, 76), FONT_MONO, 0.60, COLOR["accent_cyan"])

    # ── ROW 3 : Solution — LEFT-aligned ──────
    if last_solution:
        sol_str = truncate_text(last_solution, MAX_SOL_CHARS)
        sol_col = COLOR["success"]
    else:
        sol_str = "draw expression and show THUMB UP to solve"
        sol_col = COLOR["text_dim"]

    put_text_shadow(display, "ans:", (20, 100), FONT_MONO, 0.48, COLOR["text_dim"])
    put_text_shadow(display, sol_str, (76, 100), FONT_MONO, 0.60, sol_col)

    # ── Bottom hint bar ───────────────────────
    bov = display.copy()
    cv2.rectangle(bov, (0, h - BOTTOM_BAR_HEIGHT), (w, h), COLOR["overlay_dark"], -1)
    cv2.addWeighted(bov, 0.85, display, 0.15, 0, display)
    cv2.line(display, (0, h - BOTTOM_BAR_HEIGHT), (w, h - BOTTOM_BAR_HEIGHT), (40, 40, 56), 1)

    hints = [
        ("INDEX",        "Draw",  COLOR["neon_draw"]),
        ("INDEX+MIDDLE", "Erase", COLOR["erase"]),
        ("THUMB UP",     "Solve", COLOR["accent_gold"]),
        ("OPEN HAND",    "Clear", COLOR["danger"]),
        ("Q key",        "Quit",  COLOR["text_dim"]),
    ]
    slot_w = w // len(hints)
    for i, (key, label, col) in enumerate(hints):
        cx = i * slot_w + slot_w // 2
        (kw, _), _ = cv2.getTextSize(key, FONT_MONO, 0.48, 1)
        (lw, _), _ = cv2.getTextSize(label, FONT_MONO, 0.40, 1)
        put_text_shadow(display, key,   (cx - kw // 2, h - 33), FONT_MONO, 0.48, col)
        put_text_shadow(display, label, (cx - lw // 2, h - 14), FONT_MONO, 0.40, COLOR["text_dim"])
    for i in range(1, len(hints)):
        cv2.line(display, (i * slot_w, h - BOTTOM_BAR_HEIGHT + 6),
                 (i * slot_w, h - 6), (40, 40, 56), 1)


# ═══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════

class InvisINKApp:
    """Main application encapsulating all mutable state and the run loop.

    Attributes
    ----------
    model : tf.keras.Model
        Pre-trained CNN for character classification.
    label_map : dict
        Index → character label mapping.
    """

    def __init__(self) -> None:
        """Load the model, label map, and initialize application state."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings("ignore")

        try:
            self.model = tf.keras.models.load_model('air_math_model.h5')
            self.label_map = np.load('label_map1.npy', allow_pickle=True).item()
            logger.info("InvisINK — System Online ✦")
        except Exception as e:
            logger.critical("Ensure air_math_model.h5 and label_map1.npy are present.\n%s", e)
            sys.exit(1)

        # -- Drawing state -----------------------------------------
        self.px: int = 0
        self.py: int = 0
        self.submit_lock: bool = False
        self.clear_lock: bool = False
        self.last_expression: str = ""
        self.last_solution: str = ""
        self.mode: str = "IDLE"
        self.solve_flash: int = 0
        self.fps: float = FPS_INITIAL
        self.fps_t: float = time.time()

    # ── gesture processing ────────────────────────────────────────

    def _handle_draw(self, ix: int, iy: int, canvas: np.ndarray) -> None:
        """Handle the DRAW gesture (index finger only)."""
        self.mode = "DRAW"
        if self.px:
            cv2.line(canvas, (self.px, self.py), (ix, iy), 255, BRUSH_THICKNESS)
        self.px, self.py = ix, iy
        self.submit_lock = False
        self.clear_lock = False

    def _handle_erase(self, ix: int, iy: int, canvas: np.ndarray, display: np.ndarray) -> None:
        """Handle the ERASE gesture (index + middle fingers)."""
        self.mode = "ERASE"
        cv2.circle(canvas, (ix, iy), ERASER_RADIUS, 0, -1)
        cv2.circle(display, (ix, iy), ERASER_RADIUS, COLOR["erase"], 2)
        self.px = self.py = 0

    def _handle_solve(self, canvas: np.ndarray) -> None:
        """Handle the SOLVE gesture (thumb up only)."""
        self.mode = "SOLVE"
        self.submit_lock = True
        self.solve_flash = SOLVE_FLASH_FRAMES
        self.last_expression, self.last_solution = run_solve(
            self.model, self.label_map, canvas,
        )

    def _handle_clear(self, canvas: np.ndarray) -> None:
        """Handle the CLEAR gesture (all fingers open)."""
        self.mode = "CLEAR"
        canvas[:] = 0
        self.last_expression = ""
        self.last_solution = ""
        self.clear_lock = True
        self.submit_lock = False
        self.px = self.py = 0

    # ── main loop ─────────────────────────────────────────────────

    def run(self) -> None:
        """Open the camera and enter the main gesture-processing loop."""
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

        hands = mp_hands.Hands(
            max_num_hands=MAX_HANDS,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )

        skel_lm = mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=4)
        skel_conn = mp_draw.DrawingSpec(color=(0, 180, 255), thickness=2)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.critical("Cannot open camera (index 0). Check your webcam connection.")
            sys.exit(1)

        sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        drawing_canvas = np.zeros((sh, sw), dtype=np.uint8)
        black_bg = np.zeros((sh, sw, 3), dtype=np.uint8)

        logger.info("Press Q inside the window to quit.\n")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Failed to read frame from camera.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                display = black_bg.copy()
                detected_handedness = "Right"

                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    hand_info = res.multi_handedness[0]
                    detected_handedness = hand_info.classification[0].label

                    mp_draw.draw_landmarks(display, lm, mp_hands.HAND_CONNECTIONS,
                                           skel_lm, skel_conn)

                    f = fingers_up(lm, detected_handedness)
                    ix = int(lm.landmark[8].x * sw)
                    iy = int(lm.landmark[8].y * sh)

                    draw_circle_glow(display, (ix, iy), 10, COLOR["neon_draw"])

                    # ── Gesture dispatch ──────────────
                    if f == [False, True, False, False, False]:
                        self._handle_draw(ix, iy, drawing_canvas)

                    elif f == [False, True, True, False, False]:
                        self._handle_erase(ix, iy, drawing_canvas, display)

                    elif f[0] and not any(f[1:]) and not self.submit_lock:
                        self._handle_solve(drawing_canvas)

                    elif all(f) and not self.clear_lock:
                        self._handle_clear(drawing_canvas)

                    else:
                        self.px = self.py = 0
                        if not self.submit_lock:
                            self.mode = "IDLE"
                else:
                    self.px = self.py = 0
                    self.mode = "IDLE"

                # ── Neon drawing composite ────────
                glow = np.zeros_like(display)
                glow[drawing_canvas > 0] = COLOR["neon_glow"]
                blurred = cv2.GaussianBlur(glow, GLOW_BLUR_KERNEL, 0)
                display = cv2.addWeighted(display, 1.0, blurred, GLOW_BLEND_ALPHA, 0)
                display[drawing_canvas > 0] = COLOR["neon_draw"]

                # ── Solve flash ───────────────────
                if self.solve_flash > 0:
                    a = self.solve_flash / SOLVE_FLASH_FRAMES * SOLVE_FLASH_MAX_ALPHA
                    flash = np.full_like(display, SOLVE_FLASH_BRIGHTNESS)
                    display = cv2.addWeighted(display, 1 - a, flash, a, 0)
                    self.solve_flash -= 1

                # ── HUD & FPS ─────────────────────
                render_hud(display, self.mode, self.last_expression,
                           self.last_solution, detected_handedness)

                now = time.time()
                self.fps = FPS_SMOOTHING * self.fps + (1 - FPS_SMOOTHING) / max(now - self.fps_t, 1e-5)
                self.fps_t = now
                put_text_shadow(display, f"{self.fps:.0f} fps",
                                (sw - 80, TOP_PANEL_HEIGHT - 14), FONT_MONO, 0.45, COLOR["text_dim"])

                cv2.imshow("InvisINK", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Session ended.")


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    """Configure logging and launch the InvisINK application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = InvisINKApp()
    app.run()


if __name__ == '__main__':
    main()