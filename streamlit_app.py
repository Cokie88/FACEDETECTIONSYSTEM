"""
PTU Student Face Recognition System — Streamlit Version
Mirrors app.py (Flask) structure exactly:
    - User Panel  : Home, Search by ID, Face Detection
    - Admin Panel : Login, Dashboard, Students List,
    Add Student, Edit Student, Train Faces
"""

import streamlit as st
import sqlite3, os, pickle, base64, io
import numpy as np
from PIL import Image

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="PTU Face Recognition",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants (same as app.py) ───────────────────────────
ADMIN_USERNAME = "PTUAdmin"
ADMIN_PASSWORD = "PTU2026"
DATABASE       = "database/ptu_students.db"

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

# ── Global CSS (matches Bootstrap theme in Flask) ────────
st.markdown("""
<style>
/* ── base ── */
[data-testid="stAppViewContainer"] { background:#f8f9fa; }
[data-testid="stSidebar"]          { background:linear-gradient(180deg,#1a1a2e,#16213e); }
[data-testid="stSidebar"] *        { color:#ffffff !important; }
[data-testid="stSidebar"] .stRadio label { color:#ffffff !important; }

/* ── cards ── */
.ptu-card {
    background:#fff; border-radius:12px; padding:1.5rem;
    box-shadow:0 4px 12px rgba(0,0,0,.08);
    margin-bottom:1rem;
}
.ptu-card-blue  { border-left:4px solid #0d6efd; }
.ptu-card-green { border-left:4px solid #28a745; }
.ptu-card-red   { border-left:4px solid #dc3545; }
.ptu-card-warn  { border-left:4px solid #ffc107; }

/* ── hero ── */
.hero {
    background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
    border-radius:14px; padding:2.5rem 2rem; text-align:center;
    color:#fff; margin-bottom:1.5rem;
}
.hero h1 { font-size:2rem; font-weight:700; margin:0; }
.hero p  { opacity:.8; margin:.5rem 0 0; }

/* ── result card ── */
.result-card {
    background:#f0fff4; border:2px solid #28a745;
    border-radius:12px; padding:1.5rem;
}
.result-card h3 { color:#1a7a4a; margin:0 0 .8rem; }

/* ── stat card ── */
.stat-card {
    border-radius:12px; padding:1.2rem 1.5rem;
    color:#fff; text-align:center;
}
.stat-num  { font-size:2.2rem; font-weight:700; }
.stat-lbl  { font-size:.9rem; opacity:.85; }

/* ── info row ── */
.info-row {
    display:grid; grid-template-columns:1fr 1fr; gap:.8rem;
    margin-top:1rem;
}
.info-box {
    background:#f4f6f7; border-radius:8px; padding:.7rem 1rem;
    border-left:3px solid #0d6efd;
}
.info-box small { color:#6c757d; font-size:.8rem; }
.info-box b     { display:block; font-size:1rem; color:#1a1a2e; }

/* ── badge ── */
.badge-success { background:#28a745; color:#fff; padding:3px 10px;
                 border-radius:20px; font-size:.8rem; }
.badge-warn    { background:#ffc107; color:#333; padding:3px 10px;
                 border-radius:20px; font-size:.8rem; }

/* ── hide streamlit branding ── */
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Database helpers (same as database/db.py) ────────────
def get_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DATABASE)
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    return conn

# ── Face utils (same as modules/face_utils.py) ───────────
def encode_face(img_array):
    try:
        import face_recognition
        encs = face_recognition.face_encodings(img_array)
        return encs[0] if encs else None
    except Exception:
        pass
    try:
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        fc   = cv2.CascadeClassifier(
                   cv2.data.haarcascades +
                   "haarcascade_frontalface_default.xml")
        faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces) == 0:
            return None
        x,y,w,h = faces[0]
        roi  = cv2.resize(gray[y:y+h, x:x+w], (128,128)).flatten().astype(np.float64)
        norm = np.linalg.norm(roi)
        return roi/norm if norm > 0 else roi
    except Exception:
        return None

def recognize_from_bytes(img_bytes):
    try:
        img      = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        encoding = encode_face(np.array(img))
        if encoding is None:
            return None, "No face detected. Please use a clear front-facing photo."
        db       = get_db()
        students = db.execute(
            "SELECT * FROM students WHERE face_encodings IS NOT NULL"
        ).fetchall()
        best, best_dist = None, 0.6
        for s in students:
            try:
                stored = pickle.loads(s["face_encodings"])
                for enc in stored:
                    dist = float(np.linalg.norm(
                        np.array(enc) - np.array(encoding)))
                    if dist < best_dist:
                        best_dist = dist
                        best = dict(s)
            except Exception:
                continue
        if best:
            best["confidence"] = round((1 - best_dist)*100, 1)
            return best, None
        return None, "No matching student found in the database."
    except Exception as e:
        return None, f"Error: {str(e)}"

def retrain_student(sid, db):
    images = db.execute(
        "SELECT image_data FROM face_images WHERE student_id=?", (sid,)
    ).fetchall()
    if not images:
        db.execute("UPDATE students SET face_encodings=NULL WHERE id=?", (sid,))
        db.commit(); return False
    encodings = []
    for row in images:
        try:
            ib  = base64.b64decode(row["image_data"])
            img = Image.open(io.BytesIO(ib)).convert("RGB")
            enc = encode_face(np.array(img))
            if enc is not None:
                encodings.append(enc)
        except Exception:
            continue
    if encodings:
        db.execute("UPDATE students SET face_encodings=? WHERE id=?",
                   (pickle.dumps(encodings), sid))
        db.commit(); return True
    return False

# ── Session state defaults ───────────────────────────────
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "home"
if "edit_student_id" not in st.session_state:
    st.session_state.edit_student_id = None

# ════════════════════════════════════════════════════════
# SIDEBAR  (mirrors Flask navbar + admin sidebar)
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
      <div style='font-size:2.5rem'>🎓</div>
      <div style='font-weight:700;font-size:1.1rem'>PTU Face Recognition</div>
      <div style='opacity:.6;font-size:.8rem'>Pyay Technological University</div>
    </div>
    <hr style='border-color:rgba(255,255,255,.15);margin:.5rem 0'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:.7rem;opacity:.5;text-transform:uppercase;"
                "letter-spacing:1px;padding:.3rem .5rem'>User Panel</div>",
                unsafe_allow_html=True)

    if st.button("🏠  Home",            use_container_width=True):
        st.session_state.page = "home"
    if st.button("🔍  Search Student",  use_container_width=True):
        st.session_state.page = "search"
    if st.button("📷  Face Detection",  use_container_width=True):
        st.session_state.page = "detect"

    st.markdown("<hr style='border-color:rgba(255,255,255,.15);margin:.5rem 0'>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:.7rem;opacity:.5;text-transform:uppercase;"
                "letter-spacing:1px;padding:.3rem .5rem'>Admin Panel</div>",
                unsafe_allow_html=True)

    if st.session_state.admin_logged_in:
        if st.button("📊  Dashboard",       use_container_width=True):
            st.session_state.page = "admin_dashboard"
        if st.button("👥  Students",         use_container_width=True):
            st.session_state.page = "admin_students"
        if st.button("➕  Add Student",      use_container_width=True):
            st.session_state.page = "admin_add"
        st.markdown("<hr style='border-color:rgba(255,255,255,.15);margin:.5rem 0'>",
                    unsafe_allow_html=True)
        if st.button("🚪  Logout",           use_container_width=True):
            st.session_state.admin_logged_in = False
            st.session_state.page = "home"
            st.rerun()
    else:
        if st.button("🔐  Admin Login",      use_container_width=True):
            st.session_state.page = "admin_login"

    # Live stats
    db = get_db()
    total   = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    trained = db.execute(
        "SELECT COUNT(*) FROM students WHERE face_encodings IS NOT NULL"
    ).fetchone()[0]
    st.markdown(f"""
    <hr style='border-color:rgba(255,255,255,.15);margin:.5rem 0'>
    <div style='display:flex;gap:.5rem;padding:.3rem'>
      <div style='flex:1;background:rgba(255,255,255,.1);border-radius:8px;
                  padding:.5rem;text-align:center'>
        <div style='font-size:1.3rem;font-weight:700'>{total}</div>
        <div style='font-size:.7rem;opacity:.7'>Students</div>
      </div>
      <div style='flex:1;background:rgba(255,255,255,.1);border-radius:8px;
                  padding:.5rem;text-align:center'>
        <div style='font-size:1.3rem;font-weight:700'>{trained}</div>
        <div style='font-size:.7rem;opacity:.7'>Trained</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ════════════════════════════════════════════════════════
# USER PANEL — HOME  (mirrors Flask route "/")
# ════════════════════════════════════════════════════════
if page == "home":
    st.markdown("""
    <div class='hero'>
      <div style='font-size:3rem'>🎓</div>
      <h1>PTU Student Face Recognition System</h1>
      <p>Pyay Technological University — Smart Attendance & Student Management</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='ptu-card ptu-card-blue'>
          <h4>🤖 Face Detection</h4>
          <p style='color:#555;font-size:.9rem'>AI-powered face recognition using OpenCV and face_recognition library.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='ptu-card ptu-card-green'>
          <h4>💾 Student Database</h4>
          <p style='color:#555;font-size:.9rem'>All student records stored securely in local SQLite database.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='ptu-card ptu-card-warn'>
          <h4>🧠 AI Training</h4>
          <p style='color:#555;font-size:.9rem'>Train face models with minimum 2 images per student.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍  Search Student by ID",
                     use_container_width=True, type="primary"):
            st.session_state.page = "search"; st.rerun()
    with col2:
        if st.button("📷  Try Face Detection",
                     use_container_width=True):
            st.session_state.page = "detect"; st.rerun()

# ════════════════════════════════════════════════════════
# USER PANEL — SEARCH  (mirrors Flask route "/search")
# ════════════════════════════════════════════════════════
elif page == "search":
    st.markdown("## 🔍 Search Student by ID")
    st.caption("Enter a Student ID to retrieve the complete student profile")

    with st.form("search_form"):
        student_id = st.text_input("Student ID",
                                   placeholder="e.g. PTU-2024-001",
                                   label_visibility="collapsed")
        submitted  = st.form_submit_button("🔍  Search", type="primary",
                                           use_container_width=True)

    if submitted:
        if not student_id.strip():
            st.error("⚠️ Please enter a Student ID.")
        else:
            db = get_db()
            s  = db.execute("SELECT * FROM students WHERE student_id=?",
                            (student_id.strip(),)).fetchone()
            if s:
                s = dict(s)
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
                      <small>🔢 Roll Number</small>
                      <b>{s['roll_number']}</b>
                    </div>
                    <div class='info-box'>
                      <small>🏛️ Major</small>
                      <b>{s['major']}</b>
                    </div>
                    <div class='info-box'>
                      <small>📚 Semester</small>
                      <b>{s['semester']}</b>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ No student found with ID: **{student_id}**")

# ════════════════════════════════════════════════════════
# USER PANEL — FACE DETECT  (mirrors Flask route "/detect")
# ════════════════════════════════════════════════════════
elif page == "detect":
    st.markdown("## 📷 Face Detection & Recognition")
    st.caption("Upload a photo or capture from camera to identify a student")

    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera Capture"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload face image",
            type=["jpg","jpeg","png"],
            label_visibility="collapsed"
        )
        if uploaded:
            col1, col2 = st.columns([1,1])
            with col1:
                st.image(uploaded, caption="Uploaded Image",
                         use_column_width=True)
            with col2:
                with st.spinner("🔍 Detecting face..."):
                    uploaded.seek(0)
                    result, error = recognize_from_bytes(uploaded.read())
                if result:
                    st.success(f"✅ Student Detected!  Confidence: **{result['confidence']}%**")
                    st.markdown(f"""
                    <div class='result-card'>
                      <h3>Detected Student:</h3>
                      <div class='info-row'>
                        <div class='info-box'><small>Student Name</small>
                          <b>{result['name']}</b></div>
                        <div class='info-box'><small>Student ID</small>
                          <b style='color:#0d6efd'>{result['student_id']}</b></div>
                        <div class='info-box'><small>Major</small>
                          <b>{result['major']}</b></div>
                        <div class='info-box'><small>Semester</small>
                          <b>{result['semester']}</b></div>
                        <div class='info-box'><small>Roll Number</small>
                          <b>{result['roll_number']}</b></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(result["confidence"]/100)
                else:
                    st.error(f"❌ {error}")

    with tab2:
        camera = st.camera_input("Take a photo")
        if camera:
            col1, col2 = st.columns([1,1])
            with col1:
                st.image(camera, caption="Captured Image",
                         use_column_width=True)
            with col2:
                with st.spinner("🔍 Detecting face..."):
                    result, error = recognize_from_bytes(camera.getvalue())
                if result:
                    st.success(f"✅ Student Detected!  Confidence: **{result['confidence']}%**")
                    st.markdown(f"""
                    <div class='result-card'>
                      <h3>Detected Student:</h3>
                      <div class='info-row'>
                        <div class='info-box'><small>Student Name</small>
                          <b>{result['name']}</b></div>
                        <div class='info-box'><small>Student ID</small>
                          <b style='color:#0d6efd'>{result['student_id']}</b></div>
                        <div class='info-box'><small>Major</small>
                          <b>{result['major']}</b></div>
                        <div class='info-box'><small>Semester</small>
                          <b>{result['semester']}</b></div>
                        <div class='info-box'><small>Roll Number</small>
                          <b>{result['roll_number']}</b></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(result["confidence"]/100)
                else:
                    st.error(f"❌ {error}")

# ════════════════════════════════════════════════════════
# ADMIN PANEL — LOGIN  (mirrors Flask route "/admin/login")
# ════════════════════════════════════════════════════════
elif page == "admin_login":
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div class='ptu-card' style='text-align:center;padding:2rem'>
          <div style='font-size:2.5rem'>🔐</div>
          <h3 style='color:#0f3460'>Administrator Login</h3>
          <p style='color:#666;font-size:.9rem'>PTU Face Recognition System</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="PTUAdmin")
            password = st.text_input("Password", type="password",
                                     placeholder="••••••••")
            submitted = st.form_submit_button("🔐  Login",
                                              type="primary",
                                              use_container_width=True)
        if submitted:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.session_state.page = "admin_dashboard"
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")

# ════════════════════════════════════════════════════════
# ADMIN PANEL — DASHBOARD  (mirrors Flask route "/admin")
# ════════════════════════════════════════════════════════
elif page == "admin_dashboard":
    if not st.session_state.admin_logged_in:
        st.session_state.page = "admin_login"; st.rerun()

    st.markdown("## 📊 Admin Dashboard")

    db      = get_db()
    total   = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    trained = db.execute(
        "SELECT COUNT(*) FROM students WHERE face_encodings IS NOT NULL"
    ).fetchone()[0]
    pending = total - trained

    # Stat cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class='stat-card'
          style='background:linear-gradient(135deg,#667eea,#764ba2)'>
          <div class='stat-num'>{total}</div>
          <div class='stat-lbl'>👥 Total Students</div></div>""",
          unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='stat-card'
          style='background:linear-gradient(135deg,#11998e,#38ef7d)'>
          <div class='stat-num'>{trained}</div>
          <div class='stat-lbl'>✅ Trained</div></div>""",
          unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='stat-card'
          style='background:linear-gradient(135deg,#f093fb,#f5576c)'>
          <div class='stat-num'>{pending}</div>
          <div class='stat-lbl'>⏳ Pending Training</div></div>""",
          unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick actions
    st.markdown("#### ⚡ Quick Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("➕ Add Student", use_container_width=True, type="primary"):
            st.session_state.page = "admin_add"; st.rerun()
    with qa2:
        if st.button("👥 View All", use_container_width=True):
            st.session_state.page = "admin_students"; st.rerun()
    with qa3:
        if st.button("🔄 Retrain All", use_container_width=True):
            students = db.execute("SELECT id FROM students").fetchall()
            count = sum(1 for s in students if retrain_student(s["id"], db))
            st.success(f"✅ Retrained {count} students!")
    with qa4:
        if st.button("📷 Test Detect", use_container_width=True):
            st.session_state.page = "detect"; st.rerun()

    # Recent students table
    st.markdown("#### 🕐 Recently Added Students")
    recent = db.execute(
        "SELECT * FROM students ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    if recent:
        for s in recent:
            s = dict(s)
            badge = "<span class='badge-success'>✅ Trained</span>" \
                    if s["face_encodings"] else \
                    "<span class='badge-warn'>⏳ Pending</span>"
            col1,col2,col3,col4,col5 = st.columns([3,2,3,2,2])
            col1.write(f"**{s['name']}**")
            col2.write(f"`{s['student_id']}`")
            col3.write(s["major"])
            col4.write(s["semester"])
            col5.markdown(badge, unsafe_allow_html=True)
    else:
        st.info("No students yet.")

# ════════════════════════════════════════════════════════
# ADMIN PANEL — STUDENTS LIST  (mirrors "/admin/students")
# ════════════════════════════════════════════════════════
elif page == "admin_students":
    if not st.session_state.admin_logged_in:
        st.session_state.page = "admin_login"; st.rerun()

    st.markdown("## 👥 All Students")
    db       = get_db()
    students = db.execute(
        "SELECT * FROM students ORDER BY created_at DESC"
    ).fetchall()

    st.caption(f"{len(students)} total records")
    search = st.text_input("🔍 Search by name, ID, or major...",
                           label_visibility="collapsed",
                           placeholder="🔍 Search students...")

    for s in students:
        s = dict(s)
        # filter
        if search and search.lower() not in (
            s["name"]+s["student_id"]+s["major"]).lower():
            continue

        with st.expander(
            f"{'✅' if s['face_encodings'] else '⏳'}  "
            f"{s['name']}  —  `{s['student_id']}`"
        ):
            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f"**Major**<br>{s['major']}", unsafe_allow_html=True)
            c2.markdown(f"**Semester**<br>{s['semester']}", unsafe_allow_html=True)
            c3.markdown(f"**Roll No.**<br>{s['roll_number']}", unsafe_allow_html=True)
            imgs = db.execute(
                "SELECT COUNT(*) FROM face_images WHERE student_id=?",
                (s["id"],)).fetchone()[0]
            c4.markdown(f"**Images**<br>{imgs} uploaded", unsafe_allow_html=True)

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🧠 Train Faces",
                             key=f"train_{s['id']}",
                             use_container_width=True):
                    st.session_state.train_student_id = s["id"]
                    st.session_state.page = "admin_train"
                    st.rerun()
            with b2:
                if st.button("✏️ Edit",
                             key=f"edit_{s['id']}",
                             use_container_width=True):
                    st.session_state.edit_student_id = s["id"]
                    st.session_state.page = "admin_edit"
                    st.rerun()
            with b3:
                if st.button("🗑️ Delete",
                             key=f"del_{s['id']}",
                             use_container_width=True,
                             type="secondary"):
                    db.execute("DELETE FROM students WHERE id=?", (s["id"],))
                    db.execute("DELETE FROM face_images WHERE student_id=?",
                               (s["id"],))
                    db.commit()
                    st.success(f"Deleted {s['name']}"); st.rerun()

# ════════════════════════════════════════════════════════
# ADMIN PANEL — ADD STUDENT  (mirrors "/admin/students/add")
# ════════════════════════════════════════════════════════
elif page == "admin_add":
    if not st.session_state.admin_logged_in:
        st.session_state.page = "admin_login"; st.rerun()

    st.markdown("## ➕ Add New Student")
    st.markdown("<div class='ptu-card ptu-card-blue'>",
                unsafe_allow_html=True)

    with st.form("add_form", clear_on_submit=True):
        name       = st.text_input("Student Name *")
        c1, c2     = st.columns(2)
        student_id = c1.text_input("Student ID *", placeholder="PTU-2024-001")
        roll       = c2.text_input("Roll Number *")
        major      = st.selectbox("Major *", MAJORS)
        semester   = st.selectbox("Semester *", SEMESTERS)
        submitted  = st.form_submit_button("✅ Add Student",
                                           type="primary",
                                           use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if not all([name, student_id, roll]):
            st.error("❌ All fields are required.")
        else:
            db = get_db()
            try:
                db.execute(
                    "INSERT INTO students "
                    "(name,student_id,major,semester,roll_number) "
                    "VALUES (?,?,?,?,?)",
                    (name.strip(), student_id.strip(),
                     major, semester, roll.strip())
                )
                db.commit()
                st.success(f"✅ **{name}** added! Now go to Train Faces.")
                st.info("👉 Go to Students → Train Faces to upload face images.")
            except sqlite3.IntegrityError:
                st.error(f"❌ Student ID **{student_id}** already exists.")

# ════════════════════════════════════════════════════════
# ADMIN PANEL — EDIT STUDENT  (mirrors "/admin/students/edit/<id>")
# ════════════════════════════════════════════════════════
elif page == "admin_edit":
    if not st.session_state.admin_logged_in:
        st.session_state.page = "admin_login"; st.rerun()

    sid = st.session_state.get("edit_student_id")
    db  = get_db()
    s   = db.execute("SELECT * FROM students WHERE id=?", (sid,)).fetchone()

    if not s:
        st.error("Student not found."); st.stop()
    s = dict(s)

    st.markdown(f"## ✏️ Edit Student — {s['name']}")

    with st.form("edit_form"):
        name     = st.text_input("Student Name", value=s["name"])
        c1, c2   = st.columns(2)
        sid_val  = c1.text_input("Student ID", value=s["student_id"])
        roll     = c2.text_input("Roll Number", value=s["roll_number"])
        maj_idx  = MAJORS.index(s["major"]) if s["major"] in MAJORS else 0
        sem_idx  = SEMESTERS.index(s["semester"]) if s["semester"] in SEMESTERS else 0
        major    = st.selectbox("Major", MAJORS, index=maj_idx)
        semester = st.selectbox("Semester", SEMESTERS, index=sem_idx)
        col1, col2 = st.columns(2)
        save   = col1.form_submit_button("💾 Save Changes",
                                         type="primary",
                                         use_container_width=True)
        cancel = col2.form_submit_button("Cancel",
                                         use_container_width=True)

    if save:
        db.execute(
            "UPDATE students SET name=?,student_id=?,major=?,"
            "semester=?,roll_number=? WHERE id=?",
            (name, sid_val, major, semester, roll, sid)
        )
        db.commit()
        st.success("✅ Student updated!")
        st.session_state.page = "admin_students"; st.rerun()
    if cancel:
        st.session_state.page = "admin_students"; st.rerun()

# ════════════════════════════════════════════════════════
# ADMIN PANEL — TRAIN FACES  (mirrors "/admin/train/<id>")
# ════════════════════════════════════════════════════════
elif page == "admin_train":
    if not st.session_state.admin_logged_in:
        st.session_state.page = "admin_login"; st.rerun()

    sid = st.session_state.get("train_student_id")
    db  = get_db()
    s   = db.execute("SELECT * FROM students WHERE id=?", (sid,)).fetchone()

    if not s:
        st.error("Student not found."); st.stop()
    s = dict(s)

    st.markdown(f"## 🧠 Train Faces — {s['name']}")

    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown(f"""
        <div class='ptu-card ptu-card-blue' style='text-align:center'>
          <div style='width:70px;height:70px;border-radius:50%;
               background:linear-gradient(135deg,#667eea,#764ba2);
               display:flex;align-items:center;justify-content:center;
               color:#fff;font-size:1.8rem;margin:0 auto .8rem'>
            {s['name'][0].upper()}
          </div>
          <h4 style='margin:.3rem 0'>{s['name']}</h4>
          <code>{s['student_id']}</code><br>
          <span class='badge-success'>{s['major']}</span>
          <hr>
          <small>Semester: {s['semester']}</small><br>
          <small>Roll No.: {s['roll_number']}</small>
        </div>
        """, unsafe_allow_html=True)

        imgs_count = db.execute(
            "SELECT COUNT(*) FROM face_images WHERE student_id=?",
            (sid,)).fetchone()[0]
        if s["face_encodings"]:
            st.success("✅ Face Trained")
        else:
            st.warning("⏳ Not Trained Yet")
        st.metric("Training Images", imgs_count, delta="min 2 required")

    with col2:
        st.markdown("#### 📤 Upload Face Images")
        st.info("Upload **at least 2** clear front-facing photos for accurate recognition.")

        files = st.file_uploader(
            "Choose face images",
            type=["jpg","jpeg","png"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if files:
            prev_cols = st.columns(min(len(files), 4))
            for i, f in enumerate(files):
                prev_cols[i % 4].image(f, use_column_width=True)

        if st.button("🧠 Upload & Train",
                     type="primary",
                     use_container_width=True,
                     disabled=not files):
            prog = st.progress(0)
            saved = 0
            for i, f in enumerate(files):
                try:
                    img_bytes = f.read()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    enc = encode_face(np.array(img))
                    if enc is None:
                        st.warning(f"⚠️ No face in: {f.name}")
                        continue
                    b64 = base64.b64encode(img_bytes).decode()
                    db.execute(
                        "INSERT INTO face_images (student_id,image_data) VALUES (?,?)",
                        (sid, b64)
                    )
                    saved += 1
                except Exception as e:
                    st.warning(f"Error: {e}")
                prog.progress((i+1)/len(files))
            db.commit()

            if saved:
                if retrain_student(sid, db):
                    st.success(f"✅ {saved} image(s) uploaded and trained!")
                else:
                    st.error("Training failed — no valid face encodings.")
            else:
                st.error("❌ No valid faces detected. Use clear front-facing photos.")
            st.rerun()

        # Show existing images
        st.markdown("#### 🖼️ Existing Training Images")
        images = db.execute(
            "SELECT * FROM face_images WHERE student_id=?", (sid,)
        ).fetchall()

        if images:
            img_cols = st.columns(min(len(images), 4))
            for i, img in enumerate(images):
                with img_cols[i % 4]:
                    try:
                        img_bytes = base64.b64decode(img["image_data"])
                        st.image(img_bytes,
                                 caption=f"Image {i+1}",
                                 use_column_width=True)
                        if st.button("🗑️", key=f"del_img_{img['id']}"):
                            db.execute("DELETE FROM face_images WHERE id=?",
                                       (img["id"],))
                            db.commit()
                            retrain_student(sid, db)
                            st.rerun()
                    except Exception:
                        pass
        else:
            st.info("No training images yet. Upload images above.")

    if st.button("← Back to Students"):
        st.session_state.page = "admin_students"; st.rerun()

# ════════════════════════════════════════════════════════
# FALLBACK
# ════════════════════════════════════════════════════════
else:
    st.session_state.page = "home"; st.rerun()
