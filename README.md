# PTU Student Face Recognition System

**Pyay Technological University** — Smart Student Management with AI Face Recognition

---

## Features
- 🎓 Student management (add, edit, delete)
- 🔍 Search by Student ID
- 🤖 Face detection & recognition (OpenCV + face_recognition)
- 🧠 Face training with multiple images
- 🔐 Admin panel with secure login
- 💾 SQLite local database (no cloud needed)

## Tech Stack
- **Backend:** Python Flask
- **Database:** SQLite
- **AI/CV:** OpenCV, face_recognition (dlib)
- **Frontend:** HTML, CSS, Bootstrap 5, JavaScript

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on `face_recognition`:** This requires `dlib` which needs CMake and a C++ compiler.
> - **Windows:** Install Visual Studio Build Tools + CMake first
> - **Linux/Mac:** `sudo apt install cmake libboost-all-dev` or `brew install cmake`
> - **Alternative:** If install fails, the system falls back to OpenCV Haar Cascade detection

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
- **User Panel:** http://localhost:5000
- **Admin Panel:** http://localhost:5000/admin/login

---

## Admin Credentials
| Field | Value |
|-------|-------|
| Username | `PTUAdmin` |
| Password | `PTU2026` |

---

## How to Use

### Adding Students & Training Faces
1. Login to Admin Panel → `/admin/login`
2. Click **Add Student** and fill in all fields
3. Click the **Brain (🧠) icon** next to a student → Upload ≥2 face photos
4. System trains automatically on upload

### Face Recognition
1. Go to **User Panel** → `/detect`
2. Upload a photo OR use camera
3. System compares face with trained database
4. Displays matched student information

### Search by Student ID
1. User Panel → enter Student ID
2. Complete student profile is displayed

---

## Project Structure
```
Student-Face-Recognition-System/
├── app.py                    # Main Flask application
├── requirements.txt
├── database/
│   ├── db.py                 # Database setup & connection
│   └── ptu_students.db       # Auto-created SQLite DB
├── modules/
│   └── face_utils.py         # Face encoding & comparison
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/faces/        # Uploaded face images
└── templates/
    ├── base.html
    ├── user/
    │   ├── index.html         # Search page
    │   ├── result.html        # Student profile
    │   └── detect.html        # Face detection
    └── admin/
        ├── login.html
        ├── base_admin.html
        ├── dashboard.html
        ├── students.html
        ├── add_student.html
        ├── edit_student.html
        └── train.html
```

---

## Student Fields
- **Student Name**
- **Student ID** (unique)
- **Major:** Civil / Electronic / Mechanical / Electrical Power / Computer Engineering / IT
- **Semester:** Seminar, I–XII
- **Roll Number**

---

## GitHub + Streamlit Deployment Guide

See **DEPLOYMENT.md** for step-by-step GitHub and Streamlit Cloud deployment instructions.
