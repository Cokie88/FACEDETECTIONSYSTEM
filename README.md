# PTU Face Recognition — Deployment Notes

## Quick start
- **Local:** `pip install -r requirements.txt && streamlit run streamlit_app.py`
- **Streamlit Cloud:** push this folder to a repo and deploy `streamlit_app.py`.
  `requirements.txt` and `packages.txt` are auto-detected by filename —
  no extra configuration needed.

## Two face-detection engines (auto-detected, no code changes needed)
| Engine | Requires | Accuracy | Install reliability |
|---|---|---|---|
| **OpenCV (Haar cascade)** — default | `opencv-python-headless` only | Good | Installs from a wheel in seconds, always works |
| **face_recognition (dlib)** — optional | `dlib` (compiles from source) | Best | Can take 10-20+ min to build; can fail on constrained hosts |

`requirements.txt` ships with the OpenCV-only path so the app is
**guaranteed to deploy successfully**. If you want the higher-accuracy
dlib engine and can tolerate a slower/riskier build, swap in
`requirements-full.txt` (see comments inside that file). The Python
code in `streamlit_app.py` auto-detects whichever engine(s) are
actually importable at runtime — nothing else needs to change.

## Root causes of the original bugs (see full explanation in chat)
1. **`No module named 'cv2'`** — the requirements file was named
   `requirements_streamlit.txt`. Streamlit Cloud only auto-installs
   from a file named exactly `requirements.txt`, so none of the
   dependencies (including `opencv-python-headless`) were ever
   installed.
2. A second, independent bug: unpinned `opencv-python-headless`
   resolves to `5.0.0`, which **removes/breaks
   `cv2.CascadeClassifier`** — fixed with a `<5.0.0` pin.
3. If `dlib` had been included and its source build failed for any
   reason, pip aborts the *entire* `requirements.txt` install —
   verified empirically — which would silently take `cv2` down with
   it. Fixed by keeping the default `requirements.txt` free of any
   package that requires compiling from source.
4. **"0 valid → skipped" on dependency errors** — the old
   `encode_face()` caught a missing-`cv2` `ImportError` inside the
   "no face detected" fallback path, so a dependency outage was
   mislabeled as "no face in the photo". Now `ERR_ENGINE_UNAVAILABLE`
   is a distinct, clearly-messaged error code, separate from
   `ERR_NO_FACE`.
5. **Sidebar contrast** — "USER PANEL"/"ADMIN PANEL" labels were
   `opacity:.45`; sidebar buttons relied on a blanket
   `color:#fff!important` with no explicit background, so default
   button chrome fought the dark theme. Both now have explicit,
   accessible colors for normal/hover/active states.
