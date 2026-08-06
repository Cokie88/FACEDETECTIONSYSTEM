from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import os
import json
import base64
import numpy as np
from database.db import init_db, get_db
from modules.face_utils import encode_face, compare_faces, detect_face_in_image
import io
from PIL import Image
import pickle

app = Flask(__name__)
app.secret_key = 'ptu_face_recognition_secret_2026'
app.config['UPLOAD_FOLDER'] = 'static/uploads/faces'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ADMIN_USERNAME = 'PTUAdmin'
ADMIN_PASSWORD = 'PTU2026'

MAJORS = [
    'Civil Engineering',
    'Electronic Engineering',
    'Mechanical Engineering',
    'Electrical Power Engineering',
    'Computer Engineering and Information Technology',
]

SEMESTERS = ['Semester', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

with app.app_context():
    init_db()

# ─── User Panel ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('user/index.html')

@app.route('/search', methods=['POST'])
def search_student():
    student_id = request.form.get('student_id', '').strip()
    if not student_id:
        return render_template('user/index.html', error='Please enter a Student ID.')
    db = get_db()
    student = db.execute('SELECT * FROM students WHERE student_id = ?', (student_id,)).fetchone()
    if student:
        return render_template('user/result.html', student=dict(student))
    else:
        return render_template('user/index.html', error=f'No student found with ID: {student_id}')

@app.route('/detect', methods=['GET', 'POST'])
def detect_face():
    if request.method == 'POST':
        result = None
        error = None
        image_data = None

        if 'face_image' in request.files and request.files['face_image'].filename:
            file = request.files['face_image']
            img_bytes = file.read()
            image_data = 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode()
            result, error = recognize_from_bytes(img_bytes)
        elif request.form.get('captured_image'):
            b64 = request.form['captured_image'].split(',')[1]
            img_bytes = base64.b64decode(b64)
            image_data = request.form['captured_image']
            result, error = recognize_from_bytes(img_bytes)
        else:
            error = 'No image provided.'

        return render_template('user/detect.html', result=result, error=error, image_data=image_data)
    return render_template('user/detect.html')

def recognize_from_bytes(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)
        encoding = encode_face(img_array)
        if encoding is None:
            return None, 'No face detected in the image. Please try again with a clear face photo.'
        db = get_db()
        students = db.execute('SELECT * FROM students WHERE face_encodings IS NOT NULL').fetchall()
        best_match = None
        best_distance = 0.6
        for student in students:
            try:
                stored = pickle.loads(student['face_encodings'])
                for enc in stored:
                    dist = compare_faces(enc, encoding)
                    if dist < best_distance:
                        best_distance = dist
                        best_match = dict(student)
            except Exception:
                continue
        if best_match:
            best_match['confidence'] = round((1 - best_distance) * 100, 1)
            return best_match, None
        return None, 'No matching student found in the database.'
    except Exception as e:
        return None, f'Error processing image: {str(e)}'

# ─── Admin Panel ──────────────────────────────────────────────────────────────

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        flash('Invalid credentials. Please try again.', 'danger')
    return render_template('admin/login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/admin')
@admin_required
def admin_dashboard():
    db = get_db()
    total = db.execute('SELECT COUNT(*) as c FROM students').fetchone()['c']
    trained = db.execute('SELECT COUNT(*) as c FROM students WHERE face_encodings IS NOT NULL').fetchone()['c']
    recent = db.execute('SELECT * FROM students ORDER BY created_at DESC LIMIT 5').fetchall()
    return render_template('admin/dashboard.html', total=total, trained=trained, recent=[dict(r) for r in recent])

@app.route('/admin/students')
@admin_required
def admin_students():
    db = get_db()
    students = db.execute('SELECT * FROM students ORDER BY created_at DESC').fetchall()
    return render_template('admin/students.html', students=[dict(s) for s in students], majors=MAJORS, semesters=SEMESTERS)

@app.route('/admin/students/add', methods=['GET', 'POST'])
@admin_required
def add_student():
    if request.method == 'POST':
        name = request.form['name'].strip()
        student_id = request.form['student_id'].strip()
        major = request.form['major']
        semester = request.form['semester']
        roll_number = request.form['roll_number'].strip()

        db = get_db()
        existing = db.execute('SELECT id FROM students WHERE student_id = ?', (student_id,)).fetchone()
        if existing:
            flash(f'Student ID {student_id} already exists.', 'danger')
            return render_template('admin/add_student.html', majors=MAJORS, semesters=SEMESTERS)

        db.execute(
            'INSERT INTO students (name, student_id, major, semester, roll_number) VALUES (?, ?, ?, ?, ?)',
            (name, student_id, major, semester, roll_number)
        )
        db.commit()
        flash(f'Student {name} added successfully!', 'success')
        return redirect(url_for('admin_students'))
    return render_template('admin/add_student.html', majors=MAJORS, semesters=SEMESTERS)

@app.route('/admin/students/edit/<int:sid>', methods=['GET', 'POST'])
@admin_required
def edit_student(sid):
    db = get_db()
    student = db.execute('SELECT * FROM students WHERE id = ?', (sid,)).fetchone()
    if not student:
        flash('Student not found.', 'danger')
        return redirect(url_for('admin_students'))
    if request.method == 'POST':
        name = request.form['name'].strip()
        student_id = request.form['student_id'].strip()
        major = request.form['major']
        semester = request.form['semester']
        roll_number = request.form['roll_number'].strip()
        db.execute(
            'UPDATE students SET name=?, student_id=?, major=?, semester=?, roll_number=? WHERE id=?',
            (name, student_id, major, semester, roll_number, sid)
        )
        db.commit()
        flash('Student updated successfully!', 'success')
        return redirect(url_for('admin_students'))
    return render_template('admin/edit_student.html', student=dict(student), majors=MAJORS, semesters=SEMESTERS)

@app.route('/admin/students/delete/<int:sid>', methods=['POST'])
@admin_required
def delete_student(sid):
    db = get_db()
    db.execute('DELETE FROM students WHERE id = ?', (sid,))
    db.execute('DELETE FROM face_images WHERE student_id = ?', (sid,))
    db.commit()
    flash('Student deleted.', 'success')
    return redirect(url_for('admin_students'))

@app.route('/admin/train/<int:sid>', methods=['GET', 'POST'])
@admin_required
def train_student(sid):
    db = get_db()
    student = db.execute('SELECT * FROM students WHERE id = ?', (sid,)).fetchone()
    if not student:
        flash('Student not found.', 'danger')
        return redirect(url_for('admin_students'))
    images = db.execute('SELECT * FROM face_images WHERE student_id = ?', (sid,)).fetchall()
    if request.method == 'POST':
        files = request.files.getlist('face_images')
        saved = 0
        for f in files:
            if f and f.filename:
                img_bytes = f.read()
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    img_array = np.array(img)
                    enc = encode_face(img_array)
                    if enc is None:
                        continue
                    img_b64 = base64.b64encode(img_bytes).decode()
                    db.execute(
                        'INSERT INTO face_images (student_id, image_data) VALUES (?, ?)',
                        (sid, img_b64)
                    )
                    saved += 1
                except Exception:
                    continue
        db.commit()
        if saved > 0:
            retrain_student(sid, db)
            flash(f'{saved} image(s) uploaded and trained successfully!', 'success')
        else:
            flash('No valid face images detected. Please upload clear face photos.', 'danger')
        return redirect(url_for('train_student', sid=sid))
    return render_template('admin/train.html', student=dict(student), images=[dict(i) for i in images])

@app.route('/admin/train/<int:sid>/delete_image/<int:img_id>', methods=['POST'])
@admin_required
def delete_face_image(sid, img_id):
    db = get_db()
    db.execute('DELETE FROM face_images WHERE id = ? AND student_id = ?', (img_id, sid))
    db.commit()
    retrain_student(sid, db)
    flash('Image deleted.', 'success')
    return redirect(url_for('train_student', sid=sid))

@app.route('/admin/retrain_all', methods=['POST'])
@admin_required
def retrain_all():
    db = get_db()
    students = db.execute('SELECT id FROM students').fetchall()
    count = 0
    for s in students:
        if retrain_student(s['id'], db):
            count += 1
    flash(f'Retrained {count} students successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

def retrain_student(sid, db):
    images = db.execute('SELECT image_data FROM face_images WHERE student_id = ?', (sid,)).fetchall()
    if len(images) < 1:
        db.execute('UPDATE students SET face_encodings = NULL WHERE id = ?', (sid,))
        db.commit()
        return False
    encodings = []
    for img_row in images:
        try:
            img_bytes = base64.b64decode(img_row['image_data'])
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img)
            enc = encode_face(img_array)
            if enc is not None:
                encodings.append(enc)
        except Exception:
            continue
    if encodings:
        db.execute('UPDATE students SET face_encodings = ? WHERE id = ?', (pickle.dumps(encodings), sid))
        db.commit()
        return True
    return False

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
