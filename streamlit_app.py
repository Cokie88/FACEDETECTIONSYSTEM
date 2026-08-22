"""
PTU Student Face Recognition System — Streamlit Version
Fixed: use_container_width, image handling, camera capture,
       face encoding consistency, training pipeline, error handling.
"""

import streamlit as st
import sqlite3
import os
import pickle
import base64
import io
import logging
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# ── Logging (visible in Streamlit Cloud logs) ────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("PTU")

# ══════════════════════════════════════════════════════════
# FACE-ENGINE CAPABILITY DETECTION  (computed ONCE at import)
# ──────────────────────────────────────────────────────────
# Both cv2 and face_recognition are OPTIONAL, third-party, compiled
# dependencies. Detect their availability a single time here so the
# rest of the app never has to guess — and a missing dependency is
# always reported as a distinct "engine unavailable" condition, never
# silently mistaken for "no face found in a valid photo".
# ══════════════════════════════════════════════════════════
try:
    import cv2  # noqa: F401
    CV2_AVAILABLE = True
    CV2_IMPORT_ERROR = None
    CV2_VERSION = cv2.__version__
except Exception as _cv2_exc:                       # pragma: no cover
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(_cv2_exc)
    CV2_VERSION = None
    log.warning("cv2 (OpenCV) is not available: %s", CV2_IMPORT_ERROR)

try:
    import face_recognition  # noqa: F401
    FACE_RECOGNITION_AVAILABLE = True
    FACE_RECOGNITION_IMPORT_ERROR = None
except Exception as _fr_exc:                         # pragma: no cover
    FACE_RECOGNITION_AVAILABLE = False
    FACE_RECOGNITION_IMPORT_ERROR = str(_fr_exc)
    log.warning("face_recognition is not available: %s",
                FACE_RECOGNITION_IMPORT_ERROR)

FACE_ENGINE_AVAILABLE = CV2_AVAILABLE or FACE_RECOGNITION_AVAILABLE


def face_engine_status() -> dict:
    """Small diagnostics dict used by the UI / troubleshooting panel."""
    return {
        "cv2_available": CV2_AVAILABLE,
        "cv2_version": CV2_VERSION,
        "cv2_error": CV2_IMPORT_ERROR,
        "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
        "face_recognition_error": FACE_RECOGNITION_IMPORT_ERROR,
        "any_engine_available": FACE_ENGINE_AVAILABLE,
    }

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="PTU Face Recognition",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ────────────────────────────────────────────
ADMIN_USERNAME = "PTUAdmin"
ADMIN_PASSWORD = "PTU2026"
DATABASE       = "database/ptu_students.db"
MATCH_THRESHOLD = 0.6          # Euclidean distance threshold
ALLOWED_TYPES   = ["jpg", "jpeg", "png"]

MAJORS = [
    "Civil Engineering",
    "Electronic Engineering",
    "Mechanical Engineering",
    "Electrical Power Engineering",
    "Computer Engineering",
    "Information Technology",
]
SEMESTERS = [
    "Seminar","I","II","III","IV","V","VI",
    "VII","VIII","IX","X","XI","XII"
]

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#f8f9fa}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#16213e)}
[data-testid="stSidebar"] *{color:#f1f3f9!important}
[data-testid="stSidebar"] .stRadio label{color:#f1f3f9!important}

/* Section labels: "USER PANEL" / "ADMIN PANEL" — high-contrast, clearly visible */
.ptu-sidebar-heading{
  font-size:.72rem;font-weight:800;letter-spacing:1.5px;
  text-transform:uppercase;color:#8fb3ff!important;opacity:1!important;
  padding:.6rem .4rem .3rem;margin:0
}

/* Sidebar buttons: solid, readable in normal / hover / active states */
[data-testid="stSidebar"] .stButton > button{
  background:rgba(255,255,255,.07)!important;
  color:#f1f3f9!important;
  border:1px solid rgba(255,255,255,.16)!important;
  border-radius:8px!important;
  font-weight:600!important;
  text-align:left!important;
  justify-content:flex-start!important;
  padding:.55rem .9rem!important;
  transition:background .15s ease,border-color .15s ease,color .15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover{
  background:#0d6efd!important;
  border-color:#0d6efd!important;
  color:#ffffff!important;
}
[data-testid="stSidebar"] .stButton > button:focus:not(:active){
  box-shadow:0 0 0 2px rgba(13,110,253,.5)!important;
}
[data-testid="stSidebar"] .stButton > button:active,
[data-testid="stSidebar"] .stButton > button[kind="primary"]{
  background:#0b5ed7!important;
  border-color:#0b5ed7!important;
  color:#ffffff!important;
}
/* Sidebar stat mini-cards / body text always stay legible */
[data-testid="stSidebar"] small{color:#c7cee3!important}
.ptu-card{background:#fff;border-radius:12px;padding:1.4rem;
          box-shadow:0 4px 12px rgba(0,0,0,.08);margin-bottom:1rem}
.ptu-card-blue {border-left:4px solid #0d6efd}
.ptu-card-green{border-left:4px solid #28a745}
.ptu-card-warn {border-left:4px solid #ffc107}
.hero{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
      border-radius:14px;padding:2.5rem 2rem;text-align:center;
      color:#fff;margin-bottom:1.5rem}
.hero h1{font-size:2rem;font-weight:700;margin:0}
.hero p{opacity:.8;margin:.5rem 0 0}
.result-card{background:#f0fff4;border:2px solid #28a745;
             border-radius:12px;padding:1.5rem}
.result-card h3{color:#1a7a4a;margin:0 0 .8rem}
.stat-card{border-radius:12px;padding:1.2rem 1.5rem;
           color:#fff;text-align:center}
.stat-num{font-size:2.2rem;font-weight:700}
.stat-lbl{font-size:.9rem;opacity:.85}
.info-row{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-top:1rem}
.info-box{background:#f4f6f7;border-radius:8px;padding:.7rem 1rem;
          border-left:3px solid #0d6efd}
.info-box small{color:#6c757d;font-size:.8rem}
.info-box b{display:block;font-size:1rem;color:#1a1a2e}
.badge-success{background:#28a745;color:#fff;padding:3px 10px;
               border-radius:20px;font-size:.8rem}
.badge-warn{background:#ffc107;color:#333;padding:3px 10px;
            border-radius:20px;font-size:.8rem}
#MainMenu,footer,header{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# DATABASE  (mirrors database/db.py)
# ════════════════════════════════════════════════════════
def get_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT    NOT NULL,
            student_id     TEXT    UNIQUE NOT NULL,
            major          TEXT    NOT NULL,
            semester       TEXT    NOT NULL,
            roll_number    TEXT    NOT NULL,
            face_encodings BLOB,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS face_images (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            image_data TEXT    NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
        );
    """)
    conn.commit()
    return conn


# ════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ════════════════════════════════════════════════════════
def bytes_to_rgb_array(img_bytes: bytes):
    """
    Safely convert raw bytes → PIL RGB Image → numpy uint8 array.
    Returns (array, None) on success, (None, error_str) on failure.
    """
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)  # fix phone-camera rotation
        pil_img = pil_img.convert("RGB")          # handles RGBA / L / P / CMYK
        arr = np.ascontiguousarray(
            np.array(pil_img, dtype=np.uint8))    # shape (H, W, 3)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None, "Image conversion failed: unexpected shape."
        return arr, None
    except UnidentifiedImageError:
        return None, "File is not a valid image."
    except Exception as exc:
        log.error("bytes_to_rgb_array: %s", exc)
        return None, f"Image error: {exc}"


def safe_read_uploaded(uploaded_file):
    """
    Read an st.UploadedFile safely.
    Returns (bytes, None) or (None, error_str).
    """
    try:
        data = uploaded_file.getvalue()          # works after rerun too
        if not data:
            return None, f"{uploaded_file.name}: empty file."
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_TYPES:
            return None, f"{uploaded_file.name}: unsupported type (.{ext})."
        return data, None
    except Exception as exc:
        log.error("safe_read_uploaded: %s", exc)
        return None, f"{uploaded_file.name}: read error — {exc}"


def display_image_from_bytes(img_bytes: bytes, caption: str = ""):
    """
    Show an image from raw bytes without TypeError.
    Uses PIL so the ndarray dtype is guaranteed uint8 RGB.
    """
    try:
        arr, err = bytes_to_rgb_array(img_bytes)
        if err:
            st.warning(f"Cannot display image: {err}")
            return
        st.image(arr, caption=caption, use_container_width=True)
    except Exception as exc:
        log.error("display_image_from_bytes: %s", exc)
        st.warning("Could not display image.")


# ════════════════════════════════════════════════════════
# FACE ENCODING  (mirrors modules/face_utils.py)
# ════════════════════════════════════════════════════════
# We keep ONE encoding path.
# face_recognition  → 128-D float64 vector  (preferred)
# OpenCV fallback   → 128-D float64 vector  (same shape via PCA-like resize)
# Encodings from different methods MUST NOT be mixed in the same DB.
# A "method" flag is stored alongside encodings to detect mismatches.

ENCODING_DIM = 128
ENCODING_METHOD_KEY = "encoding_method"   # stored in session_state

# Sentinel error codes returned by encode_face(). The UI maps these to
# specific, non-misleading messages instead of a generic "invalid image".
ERR_NO_FACE           = "no_face"            # valid image, genuinely no face
ERR_ENGINE_UNAVAILABLE = "engine_unavailable"  # cv2 AND face_recognition missing
ERR_PROCESSING        = "processing_error"    # unexpected runtime error


def _encode_face_recognition(arr: np.ndarray):
    """face_recognition (dlib) path → 128-D vector. Assumes it's available."""
    encs = face_recognition.face_encodings(arr)
    if not encs:
        return None, ERR_NO_FACE
    return encs[0].astype(np.float64), None


def _encode_opencv(arr: np.ndarray):
    """
    OpenCV Haar-cascade fallback path. Assumes cv2 is available.
    Returns a 128-D vector (downsampled+normalized grayscale patch)
    so the shape matches face_recognition output. This is a lower
    accuracy fallback used only when face_recognition/dlib can't run.
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    fc   = cv2.CascadeClassifier(
               cv2.data.haarcascades +
               "haarcascade_frontalface_default.xml")
    faces = fc.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None, ERR_NO_FACE
    # Use largest detected face
    areas = [w*h for (x, y, w, h) in faces]
    x, y, w, h = faces[int(np.argmax(areas))]
    roi  = gray[y:y+h, x:x+w]
    # Resize to 16×8 = 128 pixels (same final dim as face_recognition)
    small = cv2.resize(roi, (16, 8), interpolation=cv2.INTER_AREA)
    flat  = small.flatten().astype(np.float64)
    norm  = np.linalg.norm(flat)
    return (flat / norm if norm > 0 else flat), None


def encode_face(arr: np.ndarray):
    """
    Returns (encoding: np.ndarray[128] | None, method: str, error: str | None).

    error is None on success. Otherwise it is one of:
      ERR_NO_FACE            - a valid, readable image with no detectable face
      ERR_ENGINE_UNAVAILABLE - neither cv2 nor face_recognition could be
                                imported (a dependency/deployment problem,
                                NOT a bad photo — must never be reported to
                                the user as "invalid image")
      ERR_PROCESSING         - an unexpected runtime error while encoding

    Order of preference: face_recognition (dlib, high accuracy) first,
    OpenCV Haar-cascade fallback second. A missing dependency for one
    engine never blocks the other.
    """
    if not FACE_ENGINE_AVAILABLE:
        return None, "none", ERR_ENGINE_UNAVAILABLE

    saw_no_face = False

    if FACE_RECOGNITION_AVAILABLE:
        try:
            enc, err = _encode_face_recognition(arr)
            if err is None:
                return enc, "face_recognition", None
            saw_no_face = True
        except Exception as exc:
            log.error("face_recognition encode error: %s", exc)
            # fall through and still try the OpenCV path below

    if CV2_AVAILABLE:
        try:
            enc, err = _encode_opencv(arr)
            if err is None:
                return enc, "opencv", None
            saw_no_face = True
        except Exception as exc:
            log.error("OpenCV encode error: %s", exc)

    if saw_no_face:
        return None, "none", ERR_NO_FACE
    return None, "none", ERR_PROCESSING


# Human-readable messages shown in the UI for each error code.
ERROR_MESSAGES = {
    ERR_NO_FACE: "No face detected in the image. Please use a clear, "
                 "front-facing photo with good lighting.",
    ERR_ENGINE_UNAVAILABLE: "Face detection engine unavailable on this "
                 "server (missing dependencies: cv2/face_recognition). "
                 "This is a deployment issue, not a problem with your "
                 "photo — please contact the administrator.",
    ERR_PROCESSING: "Could not process this image due to an unexpected "
                 "error. Please try a different photo.",
}


def face_error_message(err_code: str) -> str:
    return ERROR_MESSAGES.get(err_code, f"Error: {err_code}")


def compare_encodings(stored: np.ndarray, query: np.ndarray) -> float:
    """Euclidean distance between two same-shape encodings."""
    if stored.shape != query.shape:
        log.warning("Shape mismatch: %s vs %s", stored.shape, query.shape)
        return 999.0
    return float(np.linalg.norm(stored.astype(np.float64) -
                                query.astype(np.float64)))


# ════════════════════════════════════════════════════════
# RECOGNITION  (mirrors app.py recognize_from_bytes)
# ════════════════════════════════════════════════════════
def recognize_from_bytes(img_bytes: bytes):
    """
    Returns (student_dict | None, error_str | None).
    student_dict contains 'confidence' key added here.
    """
    arr, err = bytes_to_rgb_array(img_bytes)
    if err:
        return None, err

    query_enc, method, err = encode_face(arr)
    if err:
        return None, face_error_message(err)

    db = get_db()
    students = db.execute(
        "SELECT * FROM students WHERE face_encodings IS NOT NULL"
    ).fetchall()

    if not students:
        return None, "No trained students in database yet."

    best_student  = None
    best_distance = MATCH_THRESHOLD   # only accept if strictly below

    for row in students:
        try:
            stored_list = pickle.loads(row["face_encodings"])
            if not isinstance(stored_list, list) or len(stored_list) == 0:
                continue
            for stored_enc in stored_list:
                stored_arr = np.array(stored_enc, dtype=np.float64)
                dist = compare_encodings(stored_arr, query_enc)
                if dist < best_distance:
                    best_distance = dist
                    best_student  = dict(row)
        except Exception as exc:
            log.warning("Skipping student id=%s: %s", row["id"], exc)
            continue

    if best_student:
        best_student["confidence"] = round((1 - best_distance) * 100, 1)
        return best_student, None

    return None, "No matching student found in the database."


# ════════════════════════════════════════════════════════
# TRAINING HELPERS
# ════════════════════════════════════════════════════════
def retrain_student(student_db_id: int, db) -> tuple:
    """
    Re-encode all stored images for a student.
    Returns (success: bool, trained_count: int, skipped_count: int).

    IMPORTANT: if the face-detection engine itself is unavailable
    (dependency/deployment problem), this function does NOT touch the
    student's existing face_encodings — a temporary missing dependency
    must never wipe out previously-trained data.
    """
    if not FACE_ENGINE_AVAILABLE:
        log.error("retrain_student aborted: %s", ERR_ENGINE_UNAVAILABLE)
        return False, 0, 0

    images = db.execute(
        "SELECT id, image_data FROM face_images WHERE student_id=?",
        (student_db_id,)
    ).fetchall()

    if not images:
        db.execute(
            "UPDATE students SET face_encodings=NULL WHERE id=?",
            (student_db_id,))
        db.commit()
        return False, 0, 0

    encodings = []
    skipped   = 0

    for row in images:
        try:
            img_bytes = base64.b64decode(row["image_data"])
            arr, err  = bytes_to_rgb_array(img_bytes)
            if err:
                log.warning("retrain img id=%s: %s", row["id"], err)
                skipped += 1
                continue
            enc, method, err = encode_face(arr)
            if err:
                log.warning("retrain encode img id=%s: %s", row["id"], err)
                skipped += 1
                continue
            encodings.append(enc)
        except Exception as exc:
            log.warning("retrain exception img id=%s: %s", row["id"], exc)
            skipped += 1
            continue

    if encodings:
        db.execute(
            "UPDATE students SET face_encodings=? WHERE id=?",
            (pickle.dumps(encodings), student_db_id))
        db.commit()
        return True, len(encodings), skipped

    # No valid encodings — clear BLOB
    db.execute(
        "UPDATE students SET face_encodings=NULL WHERE id=?",
        (student_db_id,))
    db.commit()
    return False, 0, skipped


# ════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════
for key, val in {
    "admin_logged_in": False,
    "page":            "home",
    "edit_sid":        None,
    "train_sid":       None,
    "train_done":      False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


def go(page: str, **kwargs):
    st.session_state.page = page
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.rerun()


# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
      <div style='font-size:2.5rem'>🎓</div>
      <div style='font-weight:700;font-size:1.05rem'>PTU Face Recognition</div>
      <div style='opacity:.6;font-size:.78rem'>Pyay Technological University</div>
    </div>
    <hr style='border-color:rgba(255,255,255,.15);margin:.4rem 0'>
    """, unsafe_allow_html=True)

    st.markdown("<div class='ptu-sidebar-heading'>👤 User Panel</div>",
                unsafe_allow_html=True)

    if st.button("🏠  Home",           use_container_width=True): go("home")
    if st.button("🔍  Search Student", use_container_width=True): go("search")
    if st.button("📷  Face Detection", use_container_width=True): go("detect")

    st.markdown("<hr style='border-color:rgba(255,255,255,.15);margin:.4rem 0'>",
                unsafe_allow_html=True)
    st.markdown("<div class='ptu-sidebar-heading'>🛡️ Admin Panel</div>",
                unsafe_allow_html=True)

    if st.session_state.admin_logged_in:
        if st.button("📊  Dashboard",  use_container_width=True): go("dashboard")
        if st.button("👥  Students",   use_container_width=True): go("students")
        if st.button("➕  Add Student",use_container_width=True): go("add")
        st.markdown("<hr style='border-color:rgba(255,255,255,.15);margin:.4rem 0'>",
                    unsafe_allow_html=True)
        if st.button("🚪  Logout",     use_container_width=True):
            st.session_state.admin_logged_in = False
            go("home")
    else:
        if st.button("🔐  Admin Login",use_container_width=True): go("login")

    # Live stats
    try:
        _db     = get_db()
        _total  = _db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        _trained= _db.execute(
            "SELECT COUNT(*) FROM students WHERE face_encodings IS NOT NULL"
        ).fetchone()[0]
    except Exception:
        _total = _trained = 0

    st.markdown(f"""
    <hr style='border-color:rgba(255,255,255,.15);margin:.4rem 0'>
    <div style='display:flex;gap:.5rem;padding:.2rem'>
      <div style='flex:1;background:rgba(255,255,255,.1);border-radius:8px;
                  padding:.5rem;text-align:center'>
        <div style='font-size:1.3rem;font-weight:700'>{_total}</div>
        <div style='font-size:.7rem;opacity:.7'>Students</div>
      </div>
      <div style='flex:1;background:rgba(255,255,255,.1);border-radius:8px;
                  padding:.5rem;text-align:center'>
        <div style='font-size:1.3rem;font-weight:700'>{_trained}</div>
        <div style='font-size:.7rem;opacity:.7'>Trained</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Face-engine status badge ──────────────────────────
    if FACE_ENGINE_AVAILABLE and FACE_RECOGNITION_AVAILABLE:
        _eng_badge = "🟢 Engine: face_recognition (dlib)"
    elif FACE_ENGINE_AVAILABLE:
        _eng_badge = "🟡 Engine: OpenCV fallback only"
    else:
        _eng_badge = "🔴 Engine: unavailable"
    st.markdown(
        f"<div style='margin-top:.4rem;font-size:.68rem;text-align:center;"
        f"opacity:.85'>{_eng_badge}</div>", unsafe_allow_html=True)

page = st.session_state.page


# ════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════
if page == "home":
    st.markdown("""
    <div class='hero'>
      <div style='font-size:3rem'>🎓</div>
      <h1>PTU Student Face Recognition System</h1>
      <p>Pyay Technological University — Smart Attendance &amp; Student Management</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='ptu-card ptu-card-blue'>
          <h4>🤖 Face Detection</h4>
          <p style='color:#555;font-size:.9rem'>
          AI-powered recognition using OpenCV and face_recognition library.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='ptu-card ptu-card-green'>
          <h4>💾 Student Database</h4>
          <p style='color:#555;font-size:.9rem'>
          All records stored securely in local SQLite database.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='ptu-card ptu-card-warn'>
          <h4>🧠 AI Training</h4>
          <p style='color:#555;font-size:.9rem'>
          Train face models with minimum 2 images per student.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Search Student by ID",
                     use_container_width=True, type="primary"):
            go("search")
    with col2:
        if st.button("📷 Try Face Detection", use_container_width=True):
            go("detect")


# ════════════════════════════════════════════════════════
# PAGE: SEARCH
# ════════════════════════════════════════════════════════
elif page == "search":
    st.markdown("## 🔍 Search Student by ID")
    st.caption("Enter a Student ID to retrieve the complete student profile")

    with st.form("search_form"):
        sid_input = st.text_input(
            "Student ID", placeholder="e.g. PTU-2024-001",
            label_visibility="collapsed")
        submitted = st.form_submit_button(
            "🔍 Search", type="primary", use_container_width=True)

    if submitted:
        sid_input = sid_input.strip()
        if not sid_input:
            st.error("⚠️ Please enter a Student ID.")
        else:
            try:
                db = get_db()
                row = db.execute(
                    "SELECT * FROM students WHERE student_id=?",
                    (sid_input,)).fetchone()
                if row:
                    s = dict(row)
                    st.success("✅ Student found!")
                    st.markdown(f"""
                    <div class='result-card'>
                      <h3>👤 {s['name']}</h3>
                      <span class='badge-success'>{s['major']}</span>
                      <div class='info-row'>
                        <div class='info-box'>
                          <small>🪪 Student ID</small>
                          <b style='color:#0d6efd'>{s['student_id']}</b>
                        </div>
                        <div class='info-box'>
                          <small>🔢 Roll Number</small><b>{s['roll_number']}</b>
                        </div>
                        <div class='info-box'>
                          <small>🏛️ Major</small><b>{s['major']}</b>
                        </div>
                        <div class='info-box'>
                          <small>📚 Semester</small><b>{s['semester']}</b>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ No student found with ID: **{sid_input}**")
            except Exception as exc:
                log.error("search: %s", exc)
                st.error(f"Database error: {exc}")


# ════════════════════════════════════════════════════════
# PAGE: FACE DETECT
# ════════════════════════════════════════════════════════
elif page == "detect":
    st.markdown("## 📷 Face Detection & Recognition")
    st.caption("Upload a photo or use camera to identify a student")

    if not FACE_ENGINE_AVAILABLE:
        st.error(
            "🚫 **Face detection engine unavailable on this server.** "
            "Neither `cv2` (OpenCV) nor `face_recognition` could be "
            "imported. This is a deployment/dependency problem — "
            "please contact the administrator.")
        with st.expander("🔧 Technical details"):
            st.json(face_engine_status())
    elif not FACE_RECOGNITION_AVAILABLE:
        st.warning(
            "⚠️ Running in **OpenCV-only fallback mode** — recognition "
            "accuracy is reduced because `face_recognition`/`dlib` is "
            "not available.")

    tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Camera Capture"])

    # ── Upload tab ───────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload face image",
            type=ALLOWED_TYPES,
            label_visibility="collapsed",
            key="detect_upload"
        )
        if uploaded is not None:
            img_bytes, err = safe_read_uploaded(uploaded)
            if err:
                st.error(f"❌ {err}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    display_image_from_bytes(img_bytes, "Uploaded Image")
                with col2:
                    with st.spinner("🔍 Detecting and recognizing face…"):
                        result, rec_err = recognize_from_bytes(img_bytes)
                    if result:
                        st.success(
                            f"✅ Student Detected!  "
                            f"Confidence: **{result['confidence']}%**")
                        st.markdown(f"""
                        <div class='result-card'>
                          <h3>Detected Student:</h3>
                          <div class='info-row'>
                            <div class='info-box'>
                              <small>Student Name</small>
                              <b>{result['name']}</b></div>
                            <div class='info-box'>
                              <small>Student ID</small>
                              <b style='color:#0d6efd'>{result['student_id']}</b>
                            </div>
                            <div class='info-box'>
                              <small>Major</small><b>{result['major']}</b></div>
                            <div class='info-box'>
                              <small>Semester</small><b>{result['semester']}</b></div>
                            <div class='info-box'>
                              <small>Roll Number</small>
                              <b>{result['roll_number']}</b></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(result["confidence"] / 100)
                    else:
                        st.error(f"❌ {rec_err}")

    # ── Camera tab ───────────────────────────────────────
    with tab_camera:
        st.info("💡 Allow camera access when prompted by your browser.")
        camera_img = st.camera_input(
            "Take a photo", key="detect_camera")

        if camera_img is not None:
            try:
                img_bytes = camera_img.getvalue()
                if not img_bytes:
                    st.error("❌ Camera returned an empty image. Please try again.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        display_image_from_bytes(img_bytes, "Captured Image")
                    with col2:
                        with st.spinner("🔍 Detecting and recognizing face…"):
                            result, rec_err = recognize_from_bytes(img_bytes)
                        if result:
                            st.success(
                                f"✅ Student Detected!  "
                                f"Confidence: **{result['confidence']}%**")
                            st.markdown(f"""
                            <div class='result-card'>
                              <h3>Detected Student:</h3>
                              <div class='info-row'>
                                <div class='info-box'>
                                  <small>Student Name</small>
                                  <b>{result['name']}</b></div>
                                <div class='info-box'>
                                  <small>Student ID</small>
                                  <b style='color:#0d6efd'>
                                  {result['student_id']}</b></div>
                                <div class='info-box'>
                                  <small>Major</small>
                                  <b>{result['major']}</b></div>
                                <div class='info-box'>
                                  <small>Semester</small>
                                  <b>{result['semester']}</b></div>
                                <div class='info-box'>
                                  <small>Roll Number</small>
                                  <b>{result['roll_number']}</b></div>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(result["confidence"] / 100)
                        else:
                            st.error(f"❌ {rec_err}")
            except Exception as exc:
                log.error("camera detect: %s", exc)
                st.error(f"Camera processing error: {exc}")


# ════════════════════════════════════════════════════════
# PAGE: ADMIN LOGIN
# ════════════════════════════════════════════════════════
elif page == "login":
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div class='ptu-card' style='text-align:center;padding:2rem'>
          <div style='font-size:2.5rem'>🔐</div>
          <h3 style='color:#0f3460'>Administrator Login</h3>
          <p style='color:#666;font-size:.9rem'>PTU Face Recognition System</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username  = st.text_input("Username", placeholder="PTUAdmin")
            password  = st.text_input("Password", type="password",
                                      placeholder="••••••••")
            submitted = st.form_submit_button(
                "🔐 Login", type="primary", use_container_width=True)

        if submitted:
            if (username == ADMIN_USERNAME and
                    password == ADMIN_PASSWORD):
                st.session_state.admin_logged_in = True
                go("dashboard")
            else:
                st.error("❌ Invalid credentials. Please try again.")


# ════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════
elif page == "dashboard":
    if not st.session_state.admin_logged_in:
        go("login")

    st.markdown("## 📊 Admin Dashboard")
    db      = get_db()
    total   = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    trained = db.execute(
        "SELECT COUNT(*) FROM students WHERE face_encodings IS NOT NULL"
    ).fetchone()[0]
    pending = total - trained

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='stat-card' style='background:linear-gradient("
            f"135deg,#667eea,#764ba2)'>"
            f"<div class='stat-num'>{total}</div>"
            f"<div class='stat-lbl'>👥 Total Students</div></div>",
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='stat-card' style='background:linear-gradient("
            f"135deg,#11998e,#38ef7d)'>"
            f"<div class='stat-num'>{trained}</div>"
            f"<div class='stat-lbl'>✅ Trained</div></div>",
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"<div class='stat-card' style='background:linear-gradient("
            f"135deg,#f093fb,#f5576c)'>"
            f"<div class='stat-num'>{pending}</div>"
            f"<div class='stat-lbl'>⏳ Pending</div></div>",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Quick Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("➕ Add Student",
                     use_container_width=True, type="primary"):
            go("add")
    with qa2:
        if st.button("👥 View All", use_container_width=True):
            go("students")
    with qa3:
        if st.button("🔄 Retrain All", use_container_width=True):
            if not FACE_ENGINE_AVAILABLE:
                st.error(face_error_message(ERR_ENGINE_UNAVAILABLE))
            else:
                with st.spinner("Retraining all students…"):
                    rows  = db.execute("SELECT id FROM students").fetchall()
                    count = 0
                    for r in rows:
                        ok, _, _ = retrain_student(r["id"], db)
                        if ok:
                            count += 1
                st.success(f"✅ Retrained {count} / {len(rows)} students.")
    with qa4:
        if st.button("📷 Test Detect", use_container_width=True):
            go("detect")

    st.markdown("#### 🕐 Recently Added Students")
    recent = db.execute(
        "SELECT * FROM students ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    if recent:
        for row in recent:
            s     = dict(row)
            badge = ("✅ Trained" if s["face_encodings"]
                     else "⏳ Pending")
            c1, c2, c3, c4, c5 = st.columns([3,2,3,2,2])
            c1.write(f"**{s['name']}**")
            c2.write(f"`{s['student_id']}`")
            c3.write(s["major"])
            c4.write(s["semester"])
            c5.write(badge)
    else:
        st.info("No students added yet.")

    with st.expander("🔧 System diagnostics (face detection engine)"):
        status = face_engine_status()
        d1, d2 = st.columns(2)
        with d1:
            st.write("**OpenCV (cv2)**")
            if status["cv2_available"]:
                st.success(f"✅ Available — v{status['cv2_version']}")
            else:
                st.error(f"❌ Unavailable — {status['cv2_error']}")
        with d2:
            st.write("**face_recognition (dlib)**")
            if status["face_recognition_available"]:
                st.success("✅ Available")
            else:
                st.error(f"❌ Unavailable — {status['face_recognition_error']}")
        if not status["any_engine_available"]:
            st.error(
                "No face-detection engine is available. Check "
                "`requirements.txt` is named exactly that (Streamlit "
                "Cloud only auto-installs from `requirements.txt`), and "
                "that `packages.txt` includes the system build "
                "dependencies for dlib/OpenCV.")


# ════════════════════════════════════════════════════════
# PAGE: STUDENTS LIST
# ════════════════════════════════════════════════════════
elif page == "students":
    if not st.session_state.admin_logged_in:
        go("login")

    st.markdown("## 👥 All Students")
    db       = get_db()
    students = db.execute(
        "SELECT * FROM students ORDER BY created_at DESC"
    ).fetchall()
    st.caption(f"{len(students)} total records")

    search = st.text_input(
        "Search", placeholder="🔍 Search by name, ID, or major…",
        label_visibility="collapsed")

    for row in students:
        s = dict(row)
        if search and search.lower() not in (
                s["name"] + s["student_id"] + s["major"]).lower():
            continue

        icon = "✅" if s["face_encodings"] else "⏳"
        with st.expander(f"{icon}  {s['name']}  —  `{s['student_id']}`"):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Major**<br>{s['major']}",
                        unsafe_allow_html=True)
            c2.markdown(f"**Semester**<br>{s['semester']}",
                        unsafe_allow_html=True)
            c3.markdown(f"**Roll No.**<br>{s['roll_number']}",
                        unsafe_allow_html=True)
            img_n = db.execute(
                "SELECT COUNT(*) FROM face_images WHERE student_id=?",
                (s["id"],)).fetchone()[0]
            c4.markdown(f"**Images**<br>{img_n} uploaded",
                        unsafe_allow_html=True)

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🧠 Train",
                             key=f"tr_{s['id']}",
                             use_container_width=True):
                    go("train", train_sid=s["id"], train_done=False)
            with b2:
                if st.button("✏️ Edit",
                             key=f"ed_{s['id']}",
                             use_container_width=True):
                    go("edit", edit_sid=s["id"])
            with b3:
                if st.button("🗑️ Delete",
                             key=f"dl_{s['id']}",
                             use_container_width=True):
                    try:
                        db.execute(
                            "DELETE FROM face_images WHERE student_id=?",
                            (s["id"],))
                        db.execute(
                            "DELETE FROM students WHERE id=?",
                            (s["id"],))
                        db.commit()
                        st.success(f"Deleted {s['name']}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete error: {exc}")


# ════════════════════════════════════════════════════════
# PAGE: ADD STUDENT
# ════════════════════════════════════════════════════════
elif page == "add":
    if not st.session_state.admin_logged_in:
        go("login")

    st.markdown("## ➕ Add New Student")

    with st.form("add_form", clear_on_submit=True):
        name    = st.text_input("Student Name *")
        c1, c2  = st.columns(2)
        sid_in  = c1.text_input("Student ID *", placeholder="PTU-2024-001")
        roll    = c2.text_input("Roll Number *")
        major   = st.selectbox("Major *", MAJORS)
        sem     = st.selectbox("Semester *", SEMESTERS)
        ok_btn  = st.form_submit_button(
            "✅ Add Student", type="primary", use_container_width=True)

    if ok_btn:
        if not all([name.strip(), sid_in.strip(), roll.strip()]):
            st.error("❌ All fields are required.")
        else:
            try:
                db = get_db()
                db.execute(
                    "INSERT INTO students "
                    "(name,student_id,major,semester,roll_number) "
                    "VALUES (?,?,?,?,?)",
                    (name.strip(), sid_in.strip(), major, sem, roll.strip()))
                db.commit()
                st.success(f"✅ **{name.strip()}** added successfully!")
                st.info("👉 Go to Students → 🧠 Train to upload face images.")
            except sqlite3.IntegrityError:
                st.error(f"❌ Student ID **{sid_in.strip()}** already exists.")
            except Exception as exc:
                log.error("add student: %s", exc)
                st.error(f"Database error: {exc}")


# ════════════════════════════════════════════════════════
# PAGE: EDIT STUDENT
# ════════════════════════════════════════════════════════
elif page == "edit":
    if not st.session_state.admin_logged_in:
        go("login")

    sid = st.session_state.get("edit_sid")
    if not sid:
        go("students")

    db  = get_db()
    row = db.execute("SELECT * FROM students WHERE id=?", (sid,)).fetchone()
    if not row:
        st.error("Student not found.")
        if st.button("← Back"):
            go("students")
        st.stop()

    s = dict(row)
    st.markdown(f"## ✏️ Edit Student — {s['name']}")

    with st.form("edit_form"):
        name  = st.text_input("Student Name", value=s["name"])
        c1,c2 = st.columns(2)
        sid_v = c1.text_input("Student ID",   value=s["student_id"])
        roll  = c2.text_input("Roll Number",  value=s["roll_number"])
        maj_i = MAJORS.index(s["major"]) if s["major"] in MAJORS else 0
        sem_i = SEMESTERS.index(s["semester"]) if s["semester"] in SEMESTERS else 0
        major = st.selectbox("Major",    MAJORS,    index=maj_i)
        sem   = st.selectbox("Semester", SEMESTERS, index=sem_i)
        b1,b2 = st.columns(2)
        save  = b1.form_submit_button(
            "💾 Save", type="primary", use_container_width=True)
        back  = b2.form_submit_button(
            "Cancel", use_container_width=True)

    if save:
        try:
            db.execute(
                "UPDATE students SET name=?,student_id=?,major=?,"
                "semester=?,roll_number=? WHERE id=?",
                (name.strip(), sid_v.strip(), major, sem,
                 roll.strip(), sid))
            db.commit()
            st.success("✅ Updated successfully!")
            go("students")
        except sqlite3.IntegrityError:
            st.error(f"❌ Student ID **{sid_v.strip()}** already exists.")
        except Exception as exc:
            st.error(f"Update error: {exc}")
    if back:
        go("students")


# ════════════════════════════════════════════════════════
# PAGE: TRAIN FACES
# ════════════════════════════════════════════════════════
elif page == "train":
    if not st.session_state.admin_logged_in:
        go("login")

    sid = st.session_state.get("train_sid")
    if not sid:
        go("students")

    db  = get_db()
    row = db.execute("SELECT * FROM students WHERE id=?", (sid,)).fetchone()
    if not row:
        st.error("Student not found.")
        if st.button("← Back"):
            go("students")
        st.stop()

    s = dict(row)
    st.markdown(f"## 🧠 Train Faces — {s['name']}")

    col_info, col_upload = st.columns([1, 2])

    # ── Student info card ────────────────────────────────
    with col_info:
        img_n = db.execute(
            "SELECT COUNT(*) FROM face_images WHERE student_id=?",
            (sid,)).fetchone()[0]

        st.markdown(f"""
        <div class='ptu-card ptu-card-blue' style='text-align:center'>
          <div style='width:68px;height:68px;border-radius:50%;
               background:linear-gradient(135deg,#667eea,#764ba2);
               display:flex;align-items:center;justify-content:center;
               color:#fff;font-size:1.8rem;margin:0 auto .7rem'>
            {s['name'][0].upper()}
          </div>
          <h4 style='margin:.2rem 0'>{s['name']}</h4>
          <code>{s['student_id']}</code><br><br>
          <span class='badge-success'>{s['major']}</span><br><br>
          <small>Semester: {s['semester']}</small><br>
          <small>Roll No.: {s['roll_number']}</small>
        </div>
        """, unsafe_allow_html=True)

        if s["face_encodings"]:
            st.success("✅ Face Trained")
        else:
            st.warning("⏳ Not Trained Yet")

        st.metric("Training Images", img_n,
                  delta="need ≥ 2" if img_n < 2 else "ready")

        if st.button("← Back to Students",
                     use_container_width=True):
            go("students")

    # ── Upload section ───────────────────────────────────
    with col_upload:
        st.markdown("#### 📤 Upload Face Images")

        if not FACE_ENGINE_AVAILABLE:
            st.error(
                "🚫 **Face detection engine unavailable on this server.** "
                "Neither `cv2` (OpenCV) nor `face_recognition` could be "
                "imported. This is a deployment/dependency problem, not "
                "an issue with your photos — uploads are disabled until "
                "it's fixed. Contact the administrator.")
            with st.expander("🔧 Technical details"):
                st.json(face_engine_status())
        else:
            st.info(
                "Upload **at least 2** clear, front-facing photos. "
                "Multiple images improve recognition accuracy.")
            if not FACE_RECOGNITION_AVAILABLE:
                st.warning(
                    "⚠️ Running in **OpenCV-only fallback mode** — the "
                    "high-accuracy `face_recognition`/`dlib` engine is "
                    "not available, so matches will be less reliable. "
                    "See the technical note in the sidebar/deployment "
                    "docs to enable it.")

        files = st.file_uploader(
            "Choose images",
            type=ALLOWED_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"train_upload_{sid}"      # unique key per student
        )

        # Preview thumbnails
        if files:
            thumb_cols = st.columns(min(len(files), 5))
            for i, f in enumerate(files):
                with thumb_cols[i % 5]:
                    try:
                        thumb_bytes = f.getvalue()
                        arr, err = bytes_to_rgb_array(thumb_bytes)
                        if arr is not None:
                            st.image(arr,
                                     caption=f.name[:12],
                                     use_container_width=True)
                        else:
                            st.caption(f"❌ {f.name[:12]}")
                    except Exception:
                        st.caption(f"❌ {f.name[:12]}")

        train_btn = st.button(
            "🧠 Upload & Train",
            type="primary",
            use_container_width=True,
            disabled=(not files or not FACE_ENGINE_AVAILABLE)
        )

        if train_btn and files and not st.session_state.train_done:
            if not FACE_ENGINE_AVAILABLE:
                # Defensive guard — the button should already be disabled,
                # but never let a dependency outage masquerade as "bad photos".
                st.error(face_error_message(ERR_ENGINE_UNAVAILABLE))
                st.stop()

            progress   = st.progress(0, text="Processing images…")
            total_f    = len(files)
            saved      = 0
            skipped    = 0
            skip_msgs  = []

            for i, f in enumerate(files):
                progress.progress(
                    (i + 1) / total_f,
                    text=f"Processing {f.name} ({i+1}/{total_f})…")

                img_bytes, err = safe_read_uploaded(f)
                if err:
                    skip_msgs.append(f"⚠️ {err}")
                    skipped += 1
                    continue

                arr, err = bytes_to_rgb_array(img_bytes)
                if err:
                    skip_msgs.append(f"⚠️ {f.name}: {err}")
                    skipped += 1
                    continue

                enc, method, err = encode_face(arr)
                if err:
                    skip_msgs.append(f"⚠️ {f.name}: {face_error_message(err)}")
                    skipped += 1
                    continue

                # All good — save to DB
                try:
                    b64 = base64.b64encode(img_bytes).decode()
                    db.execute(
                        "INSERT INTO face_images "
                        "(student_id, image_data) VALUES (?,?)",
                        (sid, b64))
                    saved += 1
                except Exception as exc:
                    skip_msgs.append(f"⚠️ {f.name}: DB error — {exc}")
                    skipped += 1

            db.commit()
            progress.empty()

            # Summary
            st.info(
                f"📊 **Summary:** {total_f} uploaded → "
                f"{saved} valid → {skipped} skipped")
            if skip_msgs:
                with st.expander("⚠️ Skipped files detail"):
                    for m in skip_msgs:
                        st.write(m)

            if saved > 0:
                with st.spinner("Building face encodings…"):
                    ok, n_enc, n_skip = retrain_student(sid, db)
                if ok:
                    st.success(
                        f"✅ Training complete! "
                        f"{n_enc} encoding(s) built, {n_skip} image(s) skipped.")
                    st.session_state.train_done = True
                    st.rerun()
                else:
                    st.error(
                        "❌ Training failed — no valid face encodings "
                        "could be generated. Please use clearer, "
                        "front-facing photos.")
            else:
                st.error(
                    "❌ No valid images were saved. "
                    "Please upload clear front-facing face photos.")

        # ── Existing images gallery ──────────────────────
        st.markdown("#### 🖼️ Existing Training Images")
        existing = db.execute(
            "SELECT id, image_data FROM face_images "
            "WHERE student_id=? ORDER BY id",
            (sid,)).fetchall()

        if existing:
            g_cols = st.columns(min(len(existing), 4))
            for i, img_row in enumerate(existing):
                with g_cols[i % 4]:
                    try:
                        ib  = base64.b64decode(img_row["image_data"])
                        arr, err = bytes_to_rgb_array(ib)
                        if arr is not None:
                            st.image(arr,
                                     caption=f"Image {i+1}",
                                     use_container_width=True)
                        else:
                            st.caption(f"Image {i+1}: corrupted")
                    except Exception as exc:
                        st.caption(f"Image {i+1}: error")
                        log.warning("gallery img %s: %s",
                                    img_row["id"], exc)

                    if st.button("🗑️ Remove",
                                 key=f"rm_{img_row['id']}",
                                 use_container_width=True):
                        db.execute(
                            "DELETE FROM face_images WHERE id=?",
                            (img_row["id"],))
                        db.commit()
                        retrain_student(sid, db)
                        st.session_state.train_done = False
                        st.rerun()
        else:
            st.info("No training images yet. Upload images above.")

        # Reset train_done so next upload session works
        if st.session_state.train_done:
            st.session_state.train_done = False


# ════════════════════════════════════════════════════════
# FALLBACK
# ════════════════════════════════════════════════════════
else:
    go("home")
