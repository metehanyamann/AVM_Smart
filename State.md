# AVM Smart Track - Exhaustive Project State Document

This document provides a highly granular, module-by-module breakdown of the entire architecture, data flows, business rules, and current state. This ensures any AI assistant stepping into the project has zero context loss and doesn't need to burn tokens scanning files unnecessarily.

---

## 1. Executive Summary
**Project Name:** AVM Smart Track API
**Goal:** Enterprise-grade, real-time multi-camera face tracking, recognition, and authorization system.
**Core Stack:** FastAPI, PostgreSQL (w/ `pgvector`), SQLAlchemy, ONNX Runtime, JWT Authentication, Vanilla JS Frontend.

---

## 2. Directory Structure & Module Breakdown

### `backend/api/v1/` (FastAPI Routers)
This directory exposes the REST API layer.
- `alerts.py` (6.8KB): Handles `WantedFace` triggers. If a recognized face matches an alert threshold, this router triggers security protocols.
- `auth.py` (7.9KB): JWT-based login, registration, and token validation. Enforces RBAC through `Depends(require_role(...))`. 
- `detection.py` (6.5KB): Receives base64/multipart images from cameras and pipes them to the `SCRFDDetector`.
- `recognition.py` (10.1KB): Endpoints for extracting ArcFace 512D embeddings and searching the pgvector database.
- `tracking.py` (8.2KB): Manages active tracking sessions, floor-based analytics, and cross-camera handoffs.
- `face_tokens.py` (5.5KB): Generates secure, verifiable HMAC tokens representing a specific face for external integrations.
- `health.py` (4.1KB): Liveness probes for the database and ONNX models.
- `users.py` (15.2KB): Administrative endpoints for managing users and face embeddings.
- `vectors.py` (9.6KB): Direct operations on the vector database.
- `schemas.py` (13.3KB): Pydantic models validating every API request and response.

### `backend/application/` (Business Logic & Core Services)
The brain of the system.
- `onnx_models.py` (18KB): **CRITICAL.** Directly uses `onnxruntime.InferenceSession`. 
  - Loads `det_10g.onnx` (SCRFD) for detection. Resizes input dynamically to `320x320` for a ~4x speed boost. Extracts 5 facial landmarks.
  - Loads `w600k_r50.onnx` (ArcFace) for recognition. Uses **Umeyama Algorithm** on the 5 landmarks to strictly align and crop the face to `112x112` before extracting the 512D L2-normalized float embedding.
- `tracking.py` (33KB): Implements `RealTimeTracker`. 
  - Uses Cosine Similarity (equivalent to dot product since embeddings are L2 normalized) for re-identification across frames.
  - Implements alpha blending (EMA of 0.15) to update face embeddings dynamically as a person turns their head.
  - Implements a Spatial IoU fallback for tracking when head angles drop similarity.
  - Calculates `FloorVisit` durations and cross-camera transitions.
- `auth_service.py` (7.4KB): Handles bcrypt hashing, JWT minting, and blacklisting.
- `alert_service.py` (10.5KB): Logic for comparing current frames against `WantedFace` lists.
- `user_service.py`, `face_search.py`, `face_detection.py`, `face_token.py`, `feature_extraction.py`: Wrappers bridging the API routers and the underlying infrastructure/ONNX models.

### `backend/infrastructure/` (Data & Config)
- `database.py` (3.4KB): SQLAlchemy ORM setup.
  - Automatically executes `CREATE EXTENSION IF NOT EXISTS vector;`.
  - Defined Tables: `UserDB` (accounts), `FaceEmbedding` (512D pgvector standard faces), `WantedFace` (512D pgvector security watch-list with `wanted_id` and `alert_level`).
  - Creates a default `admin` (pass: `admin123`) on startup if empty.
- `pgvector_client.py` (8.0KB): Replaced the old Milvus integration. Uses `.l2_distance()` natively in Postgres to find matches under a threshold of `< 1.0`.
- `config.py` (1.7KB): Pydantic Settings loading from `.env`.
- `milvus_client.py` (12.3KB): Legacy/Alternative vector DB client.

### `frontend/` (Web UI)
- `index.html` (39KB) & `app.js` (57KB): A massive, monolithic Vanilla JS application handling live webcam capture, canvas bounding box drawing (Green=Known, White=Unknown, Yellow=Selected), API interactions, and JWT token storage in `localStorage`.

---

## 3. Application Workflows

### 3.1 Face Recognition Pipeline
1. `app.js` captures a frame from the webcam every ~10 frames (for optimization).
2. Image sent to `POST /api/v1/detection/detect-base64`.
3. `SCRFD` detects bounding boxes + 5 landmarks.
4. If the confidence is high enough, `app.js` crops the image and sends it to `POST /api/v1/recognition/extract-features`.
5. `ArcFace` performs Umeyama alignment, runs inference, and returns a 512D embedding.
6. The embedding is sent to `POST /api/v1/recognition/search`.
7. `pgvector_client.py` queries PostgreSQL using L2 distance. If distance `< 1.0`, a match is returned.

### 3.2 Tracking Pipeline
1. Embeddings are quantized and SHA-256 hashed to create a persistent `hash_id`.
2. As a face moves, `tracking.py` continuously updates its embedding history.
3. If a face is tracked for `CONFIRM_FRAME_THRESHOLD` (10 frames) and maintains high cosine similarity (>0.50 consistency), it is promoted to `CONFIRMED`.
4. If the face belongs to a registered user, it is linked. If not, it receives an `ANON_` ID.
5. If the person switches cameras (e.g., from Floor 1 to Floor 2), a `FloorVisit` is concluded, and a cross-camera handoff is logged.

### 3.3 Security & Role-Based Access Control
- `admin`: Can wipe the DB, change roles, disable accounts.
- `operator`: Can register/delete faces, cameras, and tokens.
- `viewer`: Can only read tracking stats and logs.
- `user`: Base permission.
- Unauthorized JWTs are rejected via FastAPI `Depends()` chains.

---

## 4. Git, Environment & Repository Constraints

### Git Configuration
- **Local Path:** `c:\Users\metehan\Desktop\bitirmedeneme\AvmSmart`
- **Remote Origin:** `https://github.com/metehanyamann/AVM_Smart.git`
- **Current Branch:** `main`
- **Ignore Rules (.gitignore):** 
  - Excludes `__pycache__`, `.venv`, `.sqlite`.
  - Excludes `.env` to protect database passwords and JWT secrets.
  - **CRITICAL - ONNX MODELS:** `*.onnx` is hardcoded into `.gitignore`. `w600k_r50.onnx` is 166MB, which caused a GitHub push rejection. They exist in `backend/infrastructure/models/` locally but are completely excluded from Git tracking. **Never attempt to push them via standard git commands.**

### Environment Setup (`.env`)
- `DATABASE_URL`: `postgresql://postgres:<PASS>@localhost:5432/avm_db`
- `SECRET_KEY`, `ALGORITHM` (HS256), `ACCESS_TOKEN_EXPIRE_MINUTES`.
- AI Thresholds: `CONFIDENCE_THRESHOLD=0.5`, `MATCH_THRESHOLD=0.3`.

---

## 5. User's Current Focus & Context
The user is actively engaged in the transition from Milvus to PostgreSQL (pgvector).
**Currently Open / Active Files:**
1. `backend/infrastructure/database.py`
2. `backend/infrastructure/pgvector_client.py`
3. `.env`
4. `run.py`

**Analysis of Focus:** They are likely finalizing the SQLAlchemy ORM models, adjusting the `l2_distance` thresholds in `pgvector_client.py`, or debugging database connection strings in the `.env` file before executing `run.py` to test the backend locally.
