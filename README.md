# rps-online
rps v2
# README.md untuk dokumentasi
# File: README.md
# Gunting Batu Kertas Online dengan Hand Tracking

## Deskripsi
Aplikasi permainan Gunting Batu Kertas multiplayer online real-time dengan teknologi hand tracking menggunakan MediaPipe.

## Teknologi yang Digunakan
- **Backend**: FastAPI + WebSocket
- **Frontend**: Streamlit + WebRTC  
- **Computer Vision**: MediaPipe
- **Deployment**: Railway (Backend) + Streamlit Cloud (Frontend)

## Struktur Proyek
```
gbk-online/
├── backend/
│   ├── main.py           # FastAPI server dengan WebSocket
│   ├── requirements.txt   # Dependencies backend
│   ├── Procfile          # Konfigurasi Railway
│   └── railway.json      # Konfigurasi deployment
├── frontend/
│   ├── app.py            # Streamlit app dengan WebRTC
│   ├── requirements.txt   # Dependencies frontend
│   ├── packages.txt      # System dependencies
│   └── .streamlit/
│       └── config.toml   # Konfigurasi Streamlit
└── README.md
```

## Cara Deployment

### Backend (Railway)
1. Push kode backend ke GitHub
2. Connect repo ke Railway
3. Set environment variables jika diperlukan
4. Deploy akan otomatis berjalan

### Frontend (Streamlit Cloud)
1. Push kode frontend ke GitHub
2. Connect repo ke Streamlit Cloud
3. Set backend WebSocket URL di app
4. Deploy akan otomatis berjalan

## Cara Menjalankan Lokal

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend  
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Alur Permainan
1. Pemain masuk lobby
2. Create atau Join room
3. Deteksi gesture menggunakan kamera
4. Server evaluasi hasil
5. Update skor dan main lagi