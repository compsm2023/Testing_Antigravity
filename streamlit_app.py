"""
InvisINK — Streamlit Web Interface
====================================
A lag-free Streamlit web app for the InvisINK air-drawing math solver.
Uses ``streamlit-webrtc`` for real-time camera processing in the browser,
with MediaPipe hand tracking and a TensorFlow CNN for character recognition.

Run with:
    streamlit run streamlit_app.py

Requirements (on top of new.py's dependencies):
    pip install streamlit streamlit-webrtc
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
import warnings

from streamlit_webrtc import (
    VideoProcessorBase,
    webrtc_streamer,
    WebRtcMode,
)

# Import solver logic from new.py (same directory)
from new import (
    BRUSH_THICKNESS,
    ERASER_RADIUS,
    COLOR,
    CONFIDENCE_THRESHOLD_CORRECTION,
    CONFIDENCE_THRESHOLD_EQUALS,
    CHAR_CORRECTIONS,
    EQUALS_ASPECT_RATIO,
    EQUALS_MERGE_Y_DIST,
    EQUALS_OVERLAP_RATIO,
    GLOW_BLEND_ALPHA,
    GLOW_BLUR_KERNEL,
    MIN_CONTOUR_H,
    MIN_CONTOUR_W,
    ROW_GROUP_THRESHOLD,
    advanced_math_solver,
    classify_char,
    fingers_up,
    merge_equals,
    prepare_roi,
    preprocess_expr,
    run_solve,
)

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

logger = logging.getLogger("InvisINK-Streamlit")

# Page config — MUST be first Streamlit command
st.set_page_config(
    page_title="InvisINK — Air Math Solver",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Premium dark theme
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Global dark theme ────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0a0a14 0%, #0d1117 50%, #0a0f1a 100%);
    }

    /* ── Sidebar ──────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
        border-right: 1px solid rgba(0, 240, 160, 0.15);
    }

    /* ── Headers ──────────────────────────────── */
    h1 {
        background: linear-gradient(90deg, #00f0a0, #00c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #e0e0f0 !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ── Cards ────────────────────────────────── */
    .result-card {
        background: linear-gradient(135deg, rgba(0,240,160,0.08), rgba(0,200,255,0.06));
        border: 1px solid rgba(0, 240, 160, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin: 12px 0;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    .result-card:hover {
        border-color: rgba(0, 240, 160, 0.5);
        box-shadow: 0 0 30px rgba(0, 240, 160, 0.1);
    }

    .expression-text {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 1.3rem;
        color: #ffdc00;
        letter-spacing: 0.5px;
    }
    .solution-text {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #50dc64;
        text-shadow: 0 0 20px rgba(80, 220, 100, 0.3);
    }

    /* ── Status badge ─────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .badge-draw  { background: rgba(0,240,160,0.15); color: #00f0a0; border: 1px solid rgba(0,240,160,0.3); }
    .badge-erase { background: rgba(180,100,255,0.15); color: #b464ff; border: 1px solid rgba(180,100,255,0.3); }
    .badge-solve { background: rgba(0,200,255,0.15); color: #00c8ff; border: 1px solid rgba(0,200,255,0.3); }
    .badge-clear { background: rgba(80,80,240,0.15); color: #5050f0; border: 1px solid rgba(80,80,240,0.3); }
    .badge-idle  { background: rgba(130,130,160,0.12); color: #8282a0; border: 1px solid rgba(130,130,160,0.2); }

    /* ── Gesture help ─────────────────────────── */
    .gesture-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 12px 0;
    }
    .gesture-item {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .gesture-item:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(0, 240, 160, 0.2);
    }
    .gesture-icon { font-size: 1.6rem; margin-bottom: 4px; }
    .gesture-label {
        font-size: 0.78rem;
        color: #8282a0;
        font-family: 'Inter', sans-serif;
    }
    .gesture-action {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e0e0f0;
    }

    /* ── Footer ───────────────────────────────── */
    .footer {
        text-align: center;
        color: #4a4a6a;
        font-size: 0.75rem;
        padding: 20px;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 30px;
    }

    /* ── Hide Streamlit defaults ──────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Video container ──────────────────────── */
    .stVideo, [data-testid="stVideo"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 240, 160, 0.15);
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADING (cached so it's loaded only once)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading InvisINK AI model…")
def load_model() -> tuple[tf.keras.Model, dict]:
    """Load the TensorFlow model and label map (cached across sessions)."""
    model = tf.keras.models.load_model('air_math_model.h5')
    label_map = np.load('label_map1.npy', allow_pickle=True).item()
    return model, label_map


# ═══════════════════════════════════════════════════════════════════
#  VIDEO PROCESSOR (runs in a separate thread — zero lag)
# ═══════════════════════════════════════════════════════════════════

class InvisINKProcessor(VideoProcessorBase):
    """Real-time video frame processor for the InvisINK system.

    Runs in a dedicated thread managed by ``streamlit-webrtc``,
    ensuring the Streamlit UI stays responsive and lag-free.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Drawing state
        self.px: int = 0
        self.py: int = 0
        self.submit_lock: bool = False
        self.clear_lock: bool = False
        self.mode: str = "IDLE"

        # Results (read by main thread)
        self.last_expression: str = ""
        self.last_solution: str = ""
        self.solve_requested: bool = False

        # Canvas — initialized lazily on first frame
        self._canvas: np.ndarray | None = None
        self._frame_shape: tuple[int, int] | None = None

        # MediaPipe hands — one instance per processor
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.80,
            min_tracking_confidence=0.70,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_hands = mp.solutions.hands

        # Model
        model, label_map = load_model()
        self._model = model
        self._label_map = label_map

        # Skeleton style
        self._skel_lm = self._mp_draw.DrawingSpec(
            color=(0, 255, 200), thickness=2, circle_radius=3,
        )
        self._skel_conn = self._mp_draw.DrawingSpec(
            color=(0, 180, 255), thickness=2,
        )

    # ── Public methods for main thread ────────────────────────────

    def get_state(self) -> dict[str, str]:
        """Thread-safe read of current mode, expression, and solution."""
        with self._lock:
            return {
                "mode": self.mode,
                "expression": self.last_expression,
                "solution": self.last_solution,
            }

    def request_solve(self) -> None:
        """Request a solve from the main thread (button click)."""
        with self._lock:
            self.solve_requested = True

    def request_clear(self) -> None:
        """Request a canvas clear from the main thread (button click)."""
        with self._lock:
            if self._canvas is not None:
                self._canvas[:] = 0
            self.last_expression = ""
            self.last_solution = ""
            self.mode = "CLEAR"
            self.px = self.py = 0
            self.submit_lock = False

    # ── Frame callback (runs every frame in worker thread) ────────

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process a single video frame. Called ~30fps by streamlit-webrtc."""
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        # Lazy canvas init
        if self._canvas is None or self._frame_shape != (h, w):
            self._canvas = np.zeros((h, w), dtype=np.uint8)
            self._frame_shape = (h, w)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        # Black background display
        display = np.zeros((h, w, 3), dtype=np.uint8)

        with self._lock:
            # Check for button-triggered solve
            if self.solve_requested:
                self.solve_requested = False
                self._do_solve()

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                hand_info = result.multi_handedness[0]
                hand_label = hand_info.classification[0].label

                # Draw skeleton
                self._mp_draw.draw_landmarks(
                    display, lm, self._mp_hands.HAND_CONNECTIONS,
                    self._skel_lm, self._skel_conn,
                )

                f = fingers_up(lm, hand_label)
                ix = int(lm.landmark[8].x * w)
                iy = int(lm.landmark[8].y * h)

                # Cursor glow
                cv2.circle(display, (ix, iy), 10, COLOR["neon_draw"], -1)

                # ── Gesture dispatch ──────────
                if f == [False, True, False, False, False]:
                    self.mode = "DRAW"
                    if self.px:
                        cv2.line(self._canvas, (self.px, self.py),
                                 (ix, iy), 255, BRUSH_THICKNESS)
                    self.px, self.py = ix, iy
                    self.submit_lock = False
                    self.clear_lock = False

                elif f == [False, True, True, False, False]:
                    self.mode = "ERASE"
                    cv2.circle(self._canvas, (ix, iy), ERASER_RADIUS, 0, -1)
                    cv2.circle(display, (ix, iy), ERASER_RADIUS, COLOR["erase"], 2)
                    self.px = self.py = 0

                elif f[0] and not any(f[1:]) and not self.submit_lock:
                    self._do_solve()

                elif all(f) and not self.clear_lock:
                    self.mode = "CLEAR"
                    self._canvas[:] = 0
                    self.last_expression = ""
                    self.last_solution = ""
                    self.clear_lock = True
                    self.submit_lock = False
                    self.px = self.py = 0

                else:
                    self.px = self.py = 0
                    if not self.submit_lock:
                        self.mode = "IDLE"
            else:
                self.px = self.py = 0
                self.mode = "IDLE"

        # ── Neon drawing composite ────────────
        canvas = self._canvas
        glow = np.zeros_like(display)
        glow[canvas > 0] = COLOR["neon_glow"]
        blurred = cv2.GaussianBlur(glow, GLOW_BLUR_KERNEL, 0)
        display = cv2.addWeighted(display, 1.0, blurred, GLOW_BLEND_ALPHA, 0)
        display[canvas > 0] = COLOR["neon_draw"]

        # ── Mode indicator on frame ───────────
        mode_text = self.mode
        cv2.putText(display, mode_text, (20, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, COLOR.get("neon_draw", (255, 255, 255)),
                    1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(display, format="bgr24")

    def _do_solve(self) -> None:
        """Internal solve — runs within the lock."""
        self.mode = "SOLVE"
        self.submit_lock = True
        if self._canvas is not None:
            self.last_expression, self.last_solution = run_solve(
                self._model, self._label_map, self._canvas,
            )


# ═══════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """Render the sidebar with gesture guide and project info."""
    with st.sidebar:
        st.markdown("## ✦ InvisINK")
        st.markdown("##### Air-Drawing Math Solver")
        st.divider()

        st.markdown("### 🖐️ Gesture Guide")
        st.markdown("""
        <div class="gesture-grid">
            <div class="gesture-item">
                <div class="gesture-icon">☝️</div>
                <div class="gesture-action" style="color:#00f0a0">Draw</div>
                <div class="gesture-label">Index finger</div>
            </div>
            <div class="gesture-item">
                <div class="gesture-icon">✌️</div>
                <div class="gesture-action" style="color:#b464ff">Erase</div>
                <div class="gesture-label">Index + Middle</div>
            </div>
            <div class="gesture-item">
                <div class="gesture-icon">👍</div>
                <div class="gesture-action" style="color:#00c8ff">Solve</div>
                <div class="gesture-label">Thumb up</div>
            </div>
            <div class="gesture-item">
                <div class="gesture-icon">🖐️</div>
                <div class="gesture-action" style="color:#5050f0">Clear</div>
                <div class="gesture-label">Open hand</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### ⚡ Supported Math")
        st.markdown("""
        - **Arithmetic** — `2 + 3 × 7`
        - **Algebra** — `x + 2 = 5`
        - **Systems** — multi-line equations
        - **Calculus** — `diff(x²)`, `int(x²)`
        - **Trig** — `sin(45)`, `cos(60)`
        """)

        st.divider()
        st.caption("InvisINK — Final Year Project ✦")


def get_mode_badge_html(mode: str) -> str:
    """Return an HTML badge for the current gesture mode."""
    badge_class = {
        "DRAW": "badge-draw",
        "ERASE": "badge-erase",
        "SOLVE": "badge-solve",
        "CLEAR": "badge-clear",
        "IDLE": "badge-idle",
    }.get(mode, "badge-idle")
    return f'<span class="status-badge {badge_class}">{mode}</span>'


def main() -> None:
    """Entry point for the Streamlit app."""
    render_sidebar()

    # ── Header ────────────────────────────────
    st.markdown("# ✦ InvisINK")
    st.markdown("*Draw math in the air. Solve instantly.*")

    # ── Two-column layout ─────────────────────
    col_video, col_results = st.columns([3, 2], gap="large")

    with col_video:
        st.markdown("### 📹 Live Canvas")

        ctx = webrtc_streamer(
            key="invisink",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=InvisINKProcessor,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

        # ── Action buttons ────────────────────
        btn_col1, btn_col2, _ = st.columns([1, 1, 2])
        with btn_col1:
            if st.button("🧮 Solve", use_container_width=True, type="primary"):
                if ctx.video_processor:
                    ctx.video_processor.request_solve()
        with btn_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                if ctx.video_processor:
                    ctx.video_processor.request_clear()

    with col_results:
        st.markdown("### 📊 Results")

        # Placeholders for live updates
        mode_placeholder = st.empty()
        result_placeholder = st.empty()

        if ctx.state.playing and ctx.video_processor:
            state = ctx.video_processor.get_state()
            mode = state["mode"]
            expression = state["expression"]
            solution = state["solution"]

            mode_placeholder.markdown(
                f"**Status:** {get_mode_badge_html(mode)}",
                unsafe_allow_html=True,
            )

            expr_display = expression if expression else "—"
            sol_display = solution if solution else "Waiting for input…"

            result_placeholder.markdown(f"""
            <div class="result-card">
                <div style="color: #8282a0; font-size: 0.8rem; margin-bottom: 6px;
                            font-family: 'Inter', sans-serif; text-transform: uppercase;
                            letter-spacing: 1px;">Expression</div>
                <div class="expression-text">{expr_display}</div>
                <div style="height: 16px;"></div>
                <div style="color: #8282a0; font-size: 0.8rem; margin-bottom: 6px;
                            font-family: 'Inter', sans-serif; text-transform: uppercase;
                            letter-spacing: 1px;">Answer</div>
                <div class="solution-text">{sol_display}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            mode_placeholder.markdown(
                f"**Status:** {get_mode_badge_html('IDLE')}",
                unsafe_allow_html=True,
            )
            result_placeholder.markdown("""
            <div class="result-card">
                <div style="color: #8282a0; font-size: 0.9rem;
                            font-family: 'Inter', sans-serif;">
                    Start the camera to begin drawing ✦
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────
    st.markdown("""
    <div class="footer">
        InvisINK — Air-Drawing Math Solver &nbsp;|&nbsp; Final Year Engineering Project &nbsp;|&nbsp; ✦
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
