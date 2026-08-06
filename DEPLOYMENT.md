# Deployment Guide — GitHub + Streamlit

---

## Step 1: Install Git (if not installed)

- **Windows:** Download from https://git-scm.com/download/win
- **Mac:** `brew install git`
- **Linux:** `sudo apt install git`

---

## Step 2: Create a GitHub Account & Repository

1. Go to https://github.com and sign up / log in
2. Click **"New repository"** (green button)
3. Set:
   - Repository name: `ptu-face-recognition`
   - Visibility: **Public** (required for free Streamlit)
   - ✅ Add README: No (we have our own)
4. Click **Create repository**

---

## Step 3: Push Project to GitHub

Open terminal/command prompt in your project folder:

```bash
# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: PTU Face Recognition System"

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ptu-face-recognition.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 4: Prepare Streamlit Version

**Important:** Streamlit works differently from Flask. You need a Streamlit wrapper.

Create `streamlit_app.py` in your project root:

```python
"""
Streamlit wrapper for PTU Face Recognition System.
Converts Flask routes to Streamlit pages.
"""
import streamlit as st
import sqlite3
import os
import pickle
import base64
import numpy as np
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="PTU Face Recognition System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = "database/ptu_students.db"
ADMIN_USER = "PTUAdmin"
ADMIN_PASS = "PTU2026"

MAJORS = [
    "Civil Engineering", "Electronic Engineering", "Mechanical Engineering",
    "Electrical Power Engineering", "Computer Engineering", "Information Technology"
]
SEMESTERS = ["Seminar", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

def get_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, student_id TEXT UNIQUE NOT NULL,
            major TEXT NOT NULL, semester TEXT NOT NULL,
            roll_number TEXT NOT NULL, face_encodings BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS face_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL, image_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    return conn

def encode_face(img_array):
    try:
        import face_recognition
        encs = face_recognition.face_encodings(img_array)
        return encs[0] if encs else None
    except Exception:
        try:
            import cv2
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            fc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
            if len(faces) == 0: return None
            x,y,w,h = faces[0]
            roi = cv2.resize(gray[y:y+h, x:x+w], (128,128)).flatten().astype(np.float64)
            n = np.linalg.norm(roi)
            return roi/n if n > 0 else roi
        except Exception:
            return None

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🎓 PTU Face Recognition")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 User Panel", "🔍 Search Student", "📷 Face Detection", "🔐 Admin Panel"])
    st.markdown("---")
    st.caption("Pyay Technological University")

# ── USER PANEL ──────────────────────────────────────────
if page == "🏠 User Panel":
    st.title("🎓 PTU Student Face Recognition System")
    st.markdown("##### Pyay Technological University — Smart Attendance & Student Management")
    col1, col2, col3 = st.columns(3)
    col1.metric("🤖 AI Detection", "OpenCV + dlib")
    col2.metric("💾 Database", "SQLite Local")
    col3.metric("🔒 Admin", "Protected")
    st.info("Use the sidebar to navigate between panels.")

# ── SEARCH STUDENT ──────────────────────────────────────
elif page == "🔍 Search Student":
    st.title("🔍 Search Student by ID")
    student_id = st.text_input("Enter Student ID", placeholder="e.g. PTU-2024-001")
    if st.button("Search", type="primary"):
        if student_id.strip():
            db = get_db()
            s = db.execute("SELECT * FROM students WHERE student_id = ?", (student_id.strip(),)).fetchone()
            if s:
                s = dict(s)
                st.success(f"✅ Student Found!")
                col1, col2 = st.columns(2)
                col1.markdown(f"**Student Name:** {s['name']}")
                col1.markdown(f"**Student ID:** `{s['student_id']}`")
                col1.markdown(f"**Major:** {s['major']}")
                col2.markdown(f"**Semester:** {s['semester']}")
                col2.markdown(f"**Roll Number:** {s['roll_number']}")
            else:
                st.error(f"No student found with ID: {student_id}")
        else:
            st.warning("Please enter a Student ID.")

# ── FACE DETECTION ──────────────────────────────────────
elif page == "📷 Face Detection":
    st.title("📷 Face Detection & Recognition")
    uploaded = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)
        img_array = np.array(img)
        encoding = encode_face(img_array)
        if encoding is None:
            st.error("❌ No face detected. Please use a clear front-facing photo.")
        else:
            db = get_db()
            students = db.execute("SELECT * FROM students WHERE face_encodings IS NOT NULL").fetchall()
            best, best_dist = None, 0.6
            for s in students:
                try:
                    stored = pickle.loads(s["face_encodings"])
                    for enc in stored:
                        dist = float(np.linalg.norm(np.array(enc) - np.array(encoding)))
                        if dist < best_dist:
                            best_dist = dist
                            best = dict(s)
                except: continue
            if best:
                conf = round((1 - best_dist) * 100, 1)
                st.success(f"✅ Student Detected! Confidence: {conf}%")
                st.markdown("### Detected Student:")
                col1, col2 = st.columns(2)
                col1.markdown(f"**Student Name:** {best['name']}")
                col1.markdown(f"**Student ID:** `{best['student_id']}`")
                col1.markdown(f"**Major:** {best['major']}")
                col2.markdown(f"**Semester:** {best['semester']}")
                col2.markdown(f"**Roll Number:** {best['roll_number']}")
            else:
                st.warning("No matching student found in the database.")

# ── ADMIN PANEL ─────────────────────────────────────────
elif page == "🔐 Admin Panel":
    if "admin_ok" not in st.session_state:
        st.title("🔐 Admin Login")
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary"):
                if u == ADMIN_USER and p == ADMIN_PASS:
                    st.session_state.admin_ok = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    else:
        st.title("⚙️ Admin Panel")
        admin_page = st.selectbox("Section", ["📊 Dashboard", "👥 Students", "➕ Add Student", "🧠 Train Faces"])
        db = get_db()

        if admin_page == "📊 Dashboard":
            total = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            trained = db.execute("SELECT COUNT(*) FROM students WHERE face_encodings IS NOT NULL").fetchone()[0]
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Students", total)
            c2.metric("Trained", trained)
            c3.metric("Pending", total - trained)
            st.subheader("Recent Students")
            rows = db.execute("SELECT * FROM students ORDER BY created_at DESC LIMIT 10").fetchall()
            for r in rows:
                r = dict(r)
                st.markdown(f"**{r['name']}** | `{r['student_id']}` | {r['major']} | {'✅' if r['face_encodings'] else '⏳'}")

        elif admin_page == "👥 Students":
            students = db.execute("SELECT * FROM students ORDER BY name").fetchall()
            st.write(f"Total: {len(students)} students")
            for s in students:
                s = dict(s)
                with st.expander(f"{s['name']} — {s['student_id']}"):
                    st.write(f"Major: {s['major']} | Semester: {s['semester']} | Roll: {s['roll_number']}")
                    st.write(f"Face Status: {'✅ Trained' if s['face_encodings'] else '⏳ Not trained'}")

        elif admin_page == "➕ Add Student":
            st.subheader("Add New Student")
            with st.form("add"):
                name = st.text_input("Student Name *")
                student_id = st.text_input("Student ID *", placeholder="PTU-2024-001")
                major = st.selectbox("Major *", MAJORS)
                semester = st.selectbox("Semester *", SEMESTERS)
                roll = st.text_input("Roll Number *")
                if st.form_submit_button("Add Student", type="primary"):
                    if all([name, student_id, major, semester, roll]):
                        try:
                            db.execute("INSERT INTO students (name,student_id,major,semester,roll_number) VALUES (?,?,?,?,?)",
                                       (name, student_id, major, semester, roll))
                            db.commit()
                            st.success(f"✅ {name} added successfully!")
                        except sqlite3.IntegrityError:
                            st.error(f"Student ID {student_id} already exists.")
                    else:
                        st.error("All fields are required.")

        elif admin_page == "🧠 Train Faces":
            students = db.execute("SELECT id, name, student_id FROM students ORDER BY name").fetchall()
            if not students:
                st.info("No students yet. Add students first.")
            else:
                opts = {f"{s['name']} ({s['student_id']})": s['id'] for s in students}
                sel = st.selectbox("Select Student", list(opts.keys()))
                sid = opts[sel]
                imgs = db.execute("SELECT * FROM face_images WHERE student_id = ?", (sid,)).fetchall()
                st.write(f"Current images: {len(imgs)} (minimum 2 required)")
                files = st.file_uploader("Upload face images", type=["jpg","jpeg","png"], accept_multiple_files=True)
                if st.button("Upload & Train", type="primary") and files:
                    saved = 0
                    for f in files:
                        try:
                            img_bytes = f.read()
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            enc = encode_face(np.array(img))
                            if enc is None: continue
                            b64 = base64.b64encode(img_bytes).decode()
                            db.execute("INSERT INTO face_images (student_id, image_data) VALUES (?,?)", (sid, b64))
                            saved += 1
                        except: continue
                    db.commit()
                    if saved:
                        all_imgs = db.execute("SELECT image_data FROM face_images WHERE student_id=?", (sid,)).fetchall()
                        encodings = []
                        for ir in all_imgs:
                            try:
                                ib = base64.b64decode(ir["image_data"])
                                img = Image.open(io.BytesIO(ib)).convert("RGB")
                                enc = encode_face(np.array(img))
                                if enc is not None: encodings.append(enc)
                            except: continue
                        if encodings:
                            db.execute("UPDATE students SET face_encodings=? WHERE id=?", (pickle.dumps(encodings), sid))
                            db.commit()
                        st.success(f"✅ {saved} image(s) uploaded and trained!")
                    else:
                        st.error("No valid faces detected in uploaded images.")
        if st.button("🚪 Logout"):
            del st.session_state.admin_ok
            st.rerun()
```

Save this as `streamlit_app.py` in your project root.

---

## Step 5: Add Streamlit Requirements

Create `requirements_streamlit.txt`:

```
streamlit>=1.28.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python-headless>=4.8.0
face-recognition>=1.3.0
```

---

## Step 6: Push Streamlit Version to GitHub

```bash
git add streamlit_app.py requirements_streamlit.txt
git commit -m "Add Streamlit version"
git push
```

---

## Step 7: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - Repository: `YOUR_USERNAME/ptu-face-recognition`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
5. In **Advanced settings** → Requirements file: `requirements_streamlit.txt`
6. Click **Deploy!**
7. Wait 2–3 minutes for deployment

Your app will be live at:
`https://YOUR_USERNAME-ptu-face-recognition-streamlit-app-XXXXX.streamlit.app`

---

## Troubleshooting

### face_recognition install fails
```bash
# Linux (Streamlit Cloud uses Linux):
sudo apt-get install cmake libboost-all-dev libopenblas-dev
pip install dlib face-recognition
```

### SQLite on Streamlit Cloud
Streamlit Cloud has ephemeral storage — data resets on restart.
For persistent data, consider switching to **Supabase** or **Railway** PostgreSQL.

### App crashes on startup
Check the logs in Streamlit Cloud dashboard → Manage app → Logs.

---

## Local Flask vs Streamlit Summary

| Feature | Flask (Local) | Streamlit (Cloud) |
|---------|--------------|-------------------|
| Camera capture | ✅ Full webcam | ⚠️ Upload only |
| UI | Custom Bootstrap | Streamlit components |
| Database | Persistent SQLite | Ephemeral (resets) |
| Setup | pip + python | GitHub + browser |
| URL | localhost:5000 | Public HTTPS link |
