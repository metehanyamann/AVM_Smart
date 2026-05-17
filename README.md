# AVM Smart Track API

Gercek zamanli yuz tanima, takip ve yetkilendirme sistemi.
FastAPI tabanli REST API ile RetinaFace (SCRFD) yuz tespiti, ArcFace yuz tanima, coklu kamera takibi, JWT kimlik dogrulama ve yuz token uretimi.

---

## Proje Yapisi

```
4 Proje API/
├── app/
│   ├── main.py                  # FastAPI uygulama giris noktasi
│   ├── config.py                # Pydantic-settings ile yapilandirma
│   ├── api/v1/
│   │   ├── auth.py              # JWT giris/kayit/rol endpoint'leri
│   │   ├── detection.py         # Yuz tespit endpoint'leri
│   │   ├── recognition.py       # Yuz tanima ve arama endpoint'leri
│   │   ├── users.py             # Kullanici ve yuz kayit yonetimi
│   │   ├── tracking.py          # Gercek zamanli takip endpoint'leri
│   │   ├── face_tokens.py       # Yuz token uretim/dogrulama
│   │   ├── vectors.py           # Vektor islemleri
│   │   ├── health.py            # Sistem sagligi
│   │   └── schemas.py           # Pydantic istek/yanit modelleri
│   ├── db/
│   │   ├── milvus_client.py     # Milvus vektor DB istemcisi + SQLite fallback
│   │   └── sqlite_vector_store.py  # SQLite tabanli vektor deposu
│   ├── services/
│   │   ├── onnx_models.py       # SCRFD + ArcFace ONNX model yukleyici
│   │   ├── face_detection.py    # Yuz tespit servisi (RetinaFace/SCRFD)
│   │   ├── feature_extraction.py # Ozellik cikarma servisi (ArcFace)
│   │   ├── face_search.py       # Benzer yuz arama servisi (L2 mesafe)
│   │   ├── user_service.py      # Kullanici/yuz embedding yonetimi
│   │   ├── auth_service.py      # JWT token + bcrypt sifre yonetimi
│   │   ├── tracking.py          # Coklu kamera takip motoru
│   │   └── face_token.py        # Yuz token uretim/dogrulama motoru
│   └── static/
│       ├── index.html           # Web arayuzu (HTML/CSS)
│       └── app.js               # Frontend mantigi (JS)
├── docker-compose.yml           # Milvus + etcd + MinIO container'lari
├── requirements.txt             # Python bagimliliklari
├── run.py                       # Uygulama baslatici
└── .env                         # Ortam degiskenleri
```

---

## Kurulum ve Calistirma

### Gereksinimler

- Python 3.11+
- PostgreSQL (Veritabanı için)

### 1. Sanal Ortam ve Bagimliliklar

```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Veritabanı Ayarları (PostgreSQL)

Yerel bilgisayarınıza PostgreSQL kurduktan sonra `avm_db` adında boş bir veritabanı oluşturun (pgAdmin veya psql ile).

Daha sonra `.env` dosyasını açarak PostgreSQL şifrenizi girin:
```env
DATABASE_URL=postgresql://postgres:SENIN_SIFREN@localhost:5432/avm_db
```

*Not: Vektör arama işlemleri için sistem otomatik olarak SQLite (yerel dosya) kullanacaktır.*

```powershell
python run.py
```

Uygulama `http://localhost:8000` adresinde calisir.
- **Web Arayuzu**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs

---

## Ozellikler

### 1. Yuz Tespiti (RetinaFace / SCRFD)

| Ozellik | Detay |
|---------|-------|
| **Model** | SCRFD (RetinaFace) - `det_10g.onnx` |
| **Backend** | ONNX Runtime (dogrudan calisim) |
| **Giris boyutu** | 320x320 (performans icin optimize) |
| **Cikti** | Yuz kutusu (x, y, w, h) + 5 nokta landmark |
| **Fallback** | OpenCV Haar Cascade |
| **Endpoint** | `POST /api/v1/detection/detect-base64` |

Model ilk calistirmada otomatik indirilir (buffalo_l.zip).

### 2. Yuz Tanima (ArcFace)

| Ozellik | Detay |
|---------|-------|
| **Model** | ArcFace w600k_r50 - `w600k_r50.onnx` |
| **Backend** | ONNX Runtime |
| **Embedding** | 512 boyutlu, L2 normalize |
| **Hizalama** | Umeyama algoritmasi ile yuz hizalama |
| **Fallback** | Histogram + LBP (klasik yontem) |
| **Endpoint** | `POST /api/v1/recognition/extract-features` |
| **Arama** | `POST /api/v1/recognition/search` |

**Esik degerleri (L2 mesafe):**
- ArcFace: `< 1.0` = ayni kisi
- Histogram+LBP: `< 0.55` = ayni kisi

### 3. Vektor Veritabani

| | Milvus | SQLite (Fallback) |
|---|--------|-------------------|
| **Index** | IVF_FLAT (256 cluster) | Yok (brute-force) |
| **Metrik** | L2 (Euclidean) | L2 (Euclidean) |
| **10K kayit arama** | ~5ms | ~500ms |
| **1M kayit arama** | ~10ms | Dakikalar |
| **Kurulum** | Docker gerekli | Sifir kurulum |

Sistem baslarken Milvus'a baglanmaya calisir. Basarisiz olursa otomatik SQLite'a duser.

### 4. Kimlik Dogrulama ve Yetkilendirme (JWT + RBAC)

**Roller (yuksekten dusuge):** `admin` > `operator` > `viewer` > `user`

| Islem | Gerekli Rol |
|-------|-------------|
| Veritabani temizle, kullanici rolleri degistir | `admin` |
| Yuz kaydet/sil, kamera kaydet/sil, token uret/iptal | `operator` |
| Listeleme, goruntuleme, token dogrulama | `viewer` |
| Yuz tespiti, tanima (frontend) | Herkese acik |

**Varsayilan hesap:** `admin` / `admin123`

**Endpoint'ler:**
- `POST /api/v1/auth/login` - Giris yap, JWT token al
- `POST /api/v1/auth/register` - Yeni hesap olustur
- `GET /api/v1/auth/me` - Mevcut kullanici bilgisi
- `POST /api/v1/auth/logout` - Cikis (token kara listeye alinir)
- `POST /api/v1/auth/verify` - Token gecerliligi kontrol

### 5. Gercek Zamanli Takip

Birden fazla kamera arasinda ayni kisinin izlenmesi.

**Endpoint'ler:**
- `POST /api/v1/tracking/cameras` - Kamera kaydet
- `DELETE /api/v1/tracking/cameras/{id}` - Kamera sil
- `POST /api/v1/tracking/update` - Takip guncelle
- `GET /api/v1/tracking/active-tracks` - Aktif izler
- `GET /api/v1/tracking/person/{id}/trail` - Kisi guzergahi
- `GET /api/v1/tracking/stats` - Takip istatistikleri

### 6. Yuz Token Uretimi

Tanilan yuzler icin dogrulanabilir token olusturma ve yonetim.

**Endpoint'ler:**
- `POST /api/v1/face-tokens/generate` - Token uret
- `POST /api/v1/face-tokens/verify` - Token dogrula
- `DELETE /api/v1/face-tokens/{id}` - Token iptal et
- `GET /api/v1/face-tokens/person/{id}` - Kisi token'lari
- `GET /api/v1/face-tokens/stats` - Token istatistikleri

### 7. Web Arayuzu

`http://localhost:8000` adresinden erisilebilir.

- **Canli kamera** ile gercek zamanli yuz tespiti
- Yesil kutu = eslesen kisi, beyaz kutu = bilinmeyen, sari kutu = secili
- **Giris sistemi** (sag ust) - JWT entegrasyonu
- **Kisi kaydet** butonu veya SPACE tusu
- **Veritabani temizle** butonu (admin yetkisi gerekli)
- Kayitli kisiler listesi ve silme
- Islem suresi, FPS, istatistikler

---

## API Dokumantasyonu

Uygulama calisirken interaktif API dokumantasyonu:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Modulleri

| Modul | Prefix | Aciklama |
|-------|--------|----------|
| Authentication | `/api/v1/auth` | JWT giris, kayit, rol yonetimi |
| Detection | `/api/v1/detection` | Yuz tespiti (RetinaFace/SCRFD) |
| Recognition | `/api/v1/recognition` | Ozellik cikarma ve arama (ArcFace) |
| Users | `/api/v1/users` | Kullanici ve yuz kayit yonetimi |
| Tracking | `/api/v1/tracking` | Coklu kamera gercek zamanli takip |
| Face Tokens | `/api/v1/face-tokens` | Yuz token uretim/dogrulama |
| Vectors | `/api/v1/vectors` | Vektor islemleri |
| Health | `/api/v1/health` | Sistem sagligi |

---

## Ortam Degiskenleri (.env)

| Degisken | Varsayilan | Aciklama |
|----------|-----------|----------|
| `APP_NAME` | AVM Smart Track API | Uygulama adi |
| `DEBUG` | True | Hata ayiklama modu |
| `HOST` | 0.0.0.0 | Sunucu adresi |
| `PORT` | 8000 | Sunucu portu |
| `MILVUS_HOST` | localhost | Milvus sunucu adresi |
| `MILVUS_PORT` | 19530 | Milvus sunucu portu |
| `SECRET_KEY` | (degistirin) | JWT imzalama anahtari |
| `ALGORITHM` | HS256 | JWT algoritma |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 30 | Token gecerlilik suresi (dk) |
| `FACE_DETECTION_MODEL` | retinaface | Yuz tespit modeli |
| `FACE_RECOGNITION_MODEL` | arcface | Yuz tanima modeli |
| `CONFIDENCE_THRESHOLD` | 0.5 | Minimum tespit guveni |
| `MATCH_THRESHOLD` | 0.3 | Esleme mesafe esigi |
| `LOG_LEVEL` | INFO | Log seviyesi |

---

## Teknik Mimari

```
Tarayici (Frontend)
    |
    |  HTTP/REST
    v
FastAPI (main.py)
    |
    ├── /api/v1/auth       --> auth_service.py (JWT + bcrypt)
    ├── /api/v1/detection   --> face_detection.py --> onnx_models.py (SCRFD)
    ├── /api/v1/recognition --> feature_extraction.py --> onnx_models.py (ArcFace)
    │                       --> face_search.py --> milvus_client.py
    ├── /api/v1/users       --> user_service.py --> milvus_client.py
    ├── /api/v1/tracking    --> tracking.py (in-memory state)
    └── /api/v1/face-tokens --> face_token.py (HMAC tokens)
                                    |
                                    v
                            Milvus (Docker) veya SQLite (fallback)
```

### Veri Akisi: Yuz Tanima

```
1. Frontend: Kamera goruntusu (480px, JPEG 0.7) --> /detection/detect-base64
2. Backend: SCRFD modeli yuz kutusu dondurur
3. Frontend: Yuz bolgesini keser --> /recognition/extract-features
4. Backend: ArcFace modeli 512D embedding uretir
5. Frontend: Embedding ile --> /recognition/search
6. Backend: Milvus/SQLite'ta L2 mesafe ile arama yapar
7. Frontend: Sonuc (isim + mesafe) ile kutu cizer
```

---

## Performans Optimizasyonlari

| Optimizasyon | Etki |
|-------------|------|
| SCRFD giris boyutu 640x640 -> 320x320 | ~4x hiz artisi |
| Feature extraction'da cift SCRFD cagrisi kaldirildi | Her yuz basina ~200ms tasarruf |
| Frontend frame isleme sikligi azaltildi (her 10 frame) | CPU yuku dustu |
| Gonderilen goruntu max 480px + JPEG 0.7 | Ag trafigi azaldi |
| Yuz koordinatlari olcekleme duzeltmesi | Kutu dogru pozisyonda |
| Dinamik esik (ArcFace: 1.0, Histogram: 0.55) | Model bazli dogruluk |
| Static dosyalar icin no-cache middleware | Tarayici cache sorunu cozuldu |
| Lazy service initialization | Modeller yuklenmeden servis baslamiyor |

---

## Ilk Halinden Guncel Hale Degisiklikler

### Eklenen Yeni Moduller

| Dosya | Aciklama |
|-------|----------|
| `app/services/onnx_models.py` | SCRFD + ArcFace ONNX model indirme ve cikarim |
| `app/services/auth_service.py` | JWT, bcrypt, kullanici yonetimi servisi |
| `app/services/tracking.py` | Coklu kamera gercek zamanli takip servisi |
| `app/services/face_token.py` | Yuz token uretim/dogrulama servisi |
| `app/api/v1/auth.py` | Auth API endpoint'leri + require_role |
| `app/api/v1/tracking.py` | Takip API endpoint'leri |
| `app/api/v1/face_tokens.py` | Face token API endpoint'leri |

### Guncellenen Moduller

| Dosya | Degisiklik |
|-------|-----------|
| `app/main.py` | Milvus baglanti denemesi, ONNX model indirme, servis reinit, cache middleware |
| `app/services/face_detection.py` | Haar Cascade -> SCRFD/RetinaFace ONNX entegrasyonu |
| `app/services/feature_extraction.py` | Histogram+LBP -> ArcFace ONNX entegrasyonu |
| `app/services/face_search.py` | Dinamik esik degeri (model bazli) |
| `app/db/milvus_client.py` | Baglanti timeout/retry iyilestirmesi, pymilvus v2.6 uyumu |
| `app/api/v1/detection.py` | Lazy loading |
| `app/api/v1/recognition.py` | Lazy loading |
| `app/api/v1/users.py` | RBAC yetkilendirme + lazy loading |
| `app/api/v1/schemas.py` | Auth semalari eklendi |
| `app/static/index.html` | Login UI, cache-bust |
| `app/static/app.js` | Auth entegrasyonu, koordinat duzeltme, performans |
| `app/config.py` | JWT, tracking, face_token ayarlari |
| `docker-compose.yml` | Milvus v2.6.13 standalone (etcd + MinIO) |

### Model Degisiklikleri

| | Onceki (Ilk Hal) | Guncel |
|---|-----------------|--------|
| **Yuz Tespiti** | Haar Cascade (OpenCV, 1990'lar) | SCRFD/RetinaFace (ONNX, derin ogrenme) |
| **Yuz Tanima** | Histogram + LBP (klasik) | ArcFace w600k_r50 (ONNX, derin ogrenme) |
| **Vektor DB** | Milvus (calismiyordu) | Milvus v2.6.13 (Docker) + SQLite fallback |
| **Kimlik Dogrulama** | Yoktu | JWT + bcrypt + RBAC (4 rol) |
| **Takip** | Yoktu | Coklu kamera gercek zamanli takip |
| **Yuz Token** | Yoktu | HMAC tabanli token uretim/dogrulama |
| **Frontend** | Temel iskelet | Tam islevsel (giris, kayit, canli algilama) |
