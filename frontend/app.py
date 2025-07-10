# Frontend Streamlit untuk GBK Online
# File: frontend/app.py
# Connected to: wss://illustrious-achievement-production-b825.up.railway.app/ws

import streamlit as st
import numpy as np
import json
import time
import base64
import io
from PIL import Image
from collections import deque, Counter
from enum import Enum
from typing import Deque, Optional, Dict

# --- PASTIKAN st.set_page_config() ADALAH PERINTAH STREAMLIT PERTAMA ---
st.set_page_config(
    page_title="Gunting Batu Kertas Online",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# -------------------------------------------------------------------

# Try importing MediaPipe and OpenCV
# Ini akan dijalankan setelah set_page_config
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV tidak tersedia. Menggunakan mode simulasi.")

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.warning("⚠️ MediaPipe tidak tersedia. Deteksi gesture akan disimulasikan.")

# Backend WebSocket URL - SUDAH FIXED!
# Ambil dari st.secrets untuk deployment, default localhost untuk lokal
BACKEND_WS_URL = st.secrets.get("BACKEND_WS_URL", "ws://localhost:8000/ws") # Perhatikan 'ws://' dan endpoint '/ws'

# --- Helper Functions (di luar main Streamlit render loop jika mungkin) ---

# Fungsi untuk klasifikasi gesture (dari gesture_utils.py)
class RPSMove(str, Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    NONE = "none"

def classify_gesture(hand_landmarks) -> Optional[str]:
    """Klasifikasi gesture dari MediaPipe landmarks"""
    if not hand_landmarks or not MEDIAPIPE_AVAILABLE:
        return None
    
    landmarks = hand_landmarks.landmark
    fingers_up = []
    
    # Thumb (jempol) - perhatikan orientasi x untuk jempol
    # Asumsi tangan kanan, jika tangan kiri perlu dibalik
    if landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x:
        fingers_up.append(1) # Jempol ke kanan (terbuka)
    else:
        fingers_up.append(0) # Jempol ke kiri (tertekuk)
    
    # Other fingers (telunjuk, tengah, manis, kelingking)
    # Cek apakah ujung jari lebih tinggi dari sendi di bawahnya
    for finger_tip_idx, finger_pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if landmarks[finger_tip_idx].y < landmarks[finger_pip_idx].y:
            fingers_up.append(1) # Jari terangkat
        else:
            fingers_up.append(0) # Jari ditekuk
    
    # Logika klasifikasi
    if fingers_up == [0, 0, 0, 0, 0]: # Semua jari ditekuk (termasuk jempol)
        return RPSMove.ROCK.value
    elif fingers_up == [1, 1, 1, 1, 1]: # Semua jari terangkat
        return RPSMove.PAPER.value
    elif fingers_up == [0, 1, 1, 0, 0]: # Telunjuk & tengah terangkat, lainnya ditekuk
        return RPSMove.SCISSORS.value
    
    return RPSMove.NONE.value

# --- State Management ---
# Menggunakan st.session_state untuk semua state game
if "game_state" not in st.session_state:
    st.session_state.game_state = {
        "room_id": None,
        "player_id": None,
        "player_name": None,
        "game_status": "lobby", # lobby, waiting, playing, result
        "current_gesture": None,
        "gesture_locked": False,
        "opponent_name": None,
        "scores": {"player": 0, "opponent": 0}, # Inisialisasi skor
        "last_result": None,
        "ws_connected": False,
        "opponent_frame_base64": None, # Simpan frame lawan dalam base64
        "player_role": None # 'A' atau 'B'
    }

# --- UI Components ---
def show_lobby():
    """Tampilan lobby"""
    st.markdown('<h1 class="game-title">🎮 Gunting Batu Kertas Online</h1>', unsafe_allow_html=True)
    st.markdown("### Selamat datang! Mainkan game klasik dengan teknologi modern")
    
    with st.expander("ℹ️ Status Koneksi & Info Umum"):
        st.info(f"Backend WebSocket: {BACKEND_WS_URL}")
        if CV2_AVAILABLE:
            st.success("✅ OpenCV tersedia - Camera support aktif.")
        else:
            st.warning("⚠️ OpenCV tidak tersedia - Mode simulasi gesture aktif.")
        if MEDIAPIPE_AVAILABLE:
            st.success("✅ MediaPipe tersedia - Hand tracking aktif.")
        else:
            st.warning("⚠️ MediaPipe tidak tersedia - Gesture akan disimulasikan.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 Buat Room Baru")
        with st.form("create_room_form"):
            # Gunakan st.session_state untuk nilai default yang persisten
            player_name_create = st.text_input("Nama Anda", 
                                                value=st.session_state.game_state.get("player_name", f"Player_{np.random.randint(1000, 9999)}"),
                                                max_chars=20, key="create_name_input")
            create_submitted = st.form_submit_button("Buat Room", type="primary")
            
            if create_submitted and player_name_create:
                st.session_state.game_state["player_name"] = player_name_create
                # Generate player_id unik di frontend
                st.session_state.game_state["player_id"] = f"p_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
                
                # Kirim event create_room ke backend via WebSocket
                if st.session_state.ws_client_instance: # Pastikan instance WS sudah ada
                    st.session_state.ws_client_instance.message_queue.put({
                        "event": "create_room",
                        "player_id": st.session_state.game_state["player_id"],
                        "player_name": player_name_create
                    })
                    st.session_state.game_state["game_status"] = "connecting"
                    st.success("🚀 Membangun koneksi dan membuat room...")
                    st.rerun() # Rerun untuk update UI status
                else:
                    st.error("WebSocket client belum siap. Coba refresh halaman.")

    with col2:
        st.markdown("### 🚪 Gabung Room")
        with st.form("join_room_form"):
            room_id_input = st.text_input("Room ID", 
                                          placeholder="Masukkan 8 karakter Room ID",
                                          max_chars=8, key="join_room_input")
            player_name_join = st.text_input("Nama Anda",
                                              value=st.session_state.game_state.get("player_name", f"Player_{np.random.randint(1000, 9999)}"),
                                              max_chars=20, key="join_name_input")
            join_submitted = st.form_submit_button("Gabung Room", type="primary", disabled=not room_id_input)
            
            if join_submitted and room_id_input and player_name_join:
                st.session_state.game_state["player_name"] = player_name_join
                st.session_state.game_state["player_id"] = f"p_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
                
                if st.session_state.ws_client_instance: # Pastikan instance WS sudah ada
                    st.session_state.ws_client_instance.message_queue.put({
                        "event": "join_room",
                        "room_id": room_id_input,
                        "player_id": st.session_state.game_state["player_id"],
                        "player_name": player_name_join
                    })
                    st.session_state.game_state["game_status"] = "connecting"
                    st.success("🔗 Membangun koneksi dan bergabung ke room...")
                    st.rerun() # Rerun untuk update UI status
                else:
                    st.error("WebSocket client belum siap. Coba refresh halaman.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Gunakan gesture tangan: ✊ Batu | ✋ Kertas | ✌️ Gunting</p>
    </div>
    """, unsafe_allow_html=True)

def show_game_room():
    """Tampilan game room"""
    st.markdown('<h1 class="game-title">🎮 Gunting Batu Kertas Online</h1>', unsafe_allow_html=True)
    
    # Room info
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("**📍 Room ID**")
        if st.session_state.game_state["room_id"]:
            st.markdown(f'<div class="room-code">{st.session_state.game_state["room_id"]}</div>', unsafe_allow_html=True)
    
    with col2:
        status_emoji = {
            "waiting": "⏳ Menunggu Lawan...",
            "ready": "🔔 Bersiap...",
            "playing": "🎮 Bermain!",
            "result": "🏆 Hasil"
        }
        status = st.session_state.game_state["game_status"]
        st.markdown(f"### {status_emoji.get(status, status)}")
    
    with col3:
        st.markdown("**👥 Lawan**")
        if st.session_state.game_state["opponent_name"]:
            st.info(f"🎮 {st.session_state.game_state['opponent_name']}")
        else:
            st.warning("⏳ Menunggu...")
    
    st.markdown("---")
    
    # Display scores
    st.markdown("### 📊 Skor Saat Ini")
    score_cols = st.columns(2)
    with score_cols[0]:
        st.metric("Skor Anda", st.session_state.game_state["scores"].get(st.session_state.game_state["player_id"], 0))
    with score_cols[1]:
        st.metric(f"Skor {st.session_state.game_state['opponent_name'] or 'Lawan'}", 
                  st.session_state.game_state["scores"].get(st.session_state.game_state["opponent_id"], 0)) # Asumsi backend kirim opponent_id

    # Game content
    if st.session_state.game_state["game_status"] in ["playing", "ready"]:
        col_player_cam, col_opponent_cam = st.columns(2)

        with col_player_cam:
            st.markdown("### 📹 Kamera Anda")
            player_cam_placeholder = st.empty()
            gesture_display_placeholder = st.empty()

            # Inisialisasi MediaPipe Hands di sini, di main thread Streamlit
            if "mp_hands_instance" not in st.session_state and MEDIAPIPE_AVAILABLE:
                st.session_state.mp_hands_instance = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
            
            # Gunakan st.camera_input untuk Streamlit-native camera
            # Atau WebRTC streamer jika ingin lebih advanced
            if CV2_AVAILABLE and MEDIAPIPE_AVAILABLE:
                img_file_buffer = st.camera_input("Ambil Gambar Gesture Anda", key="player_cam_input")
                
                if img_file_buffer is not None:
                    bytes_data = img_file_buffer.getvalue()
                    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Proses frame dengan MediaPipe
                    results = st.session_state.mp_hands_instance.process(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
                    
                    detected_gesture = None
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(cv2_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            detected_gesture = classify_gesture(hand_landmarks)
                    
                    player_cam_placeholder.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    if detected_gesture and not st.session_state.game_state["gesture_locked"]:
                        st.session_state.game_state["current_gesture"] = detected_gesture
                        gesture_display_placeholder.success(f"Gesture Terdeteksi: {detected_gesture.upper()}")
                        
                        # Tombol kunci pilihan
                        if st.button(f"Kunci Pilihan: {detected_gesture.upper()}", key="lock_gesture_btn", type="primary"):
                            st.session_state.game_state["gesture_locked"] = True
                            st.session_state.message_queue.put({
                                "event": "player_move",
                                "room_id": st.session_state.game_state["room_id"],
                                "player_id": st.session_state.game_state["player_id"],
                                "move": detected_gesture
                            })
                            st.success("Pilihan Anda terkunci!")
                            st.rerun() # Rerun untuk update UI
                    elif st.session_state.game_state["gesture_locked"]:
                        gesture_display_placeholder.info(f"Pilihan Anda: {st.session_state.game_state['current_gesture'].upper()} - Menunggu Lawan...")
                    else:
                        gesture_display_placeholder.info("Tunjukkan gesture Anda ke kamera...")
                else:
                    player_cam_placeholder.info("Kamera tidak aktif atau MediaPipe/OpenCV tidak tersedia.")
                    # Fallback untuk simulasi gesture jika kamera tidak tersedia
                    if not st.session_state.game_state["gesture_locked"]:
                        sim_gesture = st.radio("Pilih Gesture (Simulasi):", ["rock", "paper", "scissors"], horizontal=True)
                        if st.button(f"Kunci Pilihan Simulasi: {sim_gesture.upper()}", key="lock_sim_gesture_btn", type="primary"):
                            st.session_state.game_state["current_gesture"] = sim_gesture
                            st.session_state.game_state["gesture_locked"] = True
                            st.session_state.message_queue.put({
                                "event": "player_move",
                                "room_id": st.session_state.game_state["room_id"],
                                "player_id": st.session_state.game_state["player_id"],
                                "move": sim_gesture
                            })
                            st.success("Pilihan simulasi terkunci!")
                            st.rerun()
                    else:
                        gesture_display_placeholder.info(f"Pilihan Anda: {st.session_state.game_state['current_gesture'].upper()} - Menunggu Lawan...")


        with col_opponent_cam:
            st.markdown("### 📹 Kamera Lawan")
            opponent_cam_placeholder = st.empty()
            if st.session_state.game_state["opponent_frame_base64"]:
                frame = decode_frame(st.session_state.game_state["opponent_frame_base64"])
                if frame is not None:
                    opponent_cam_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                else:
                    opponent_cam_placeholder.info("Gagal menampilkan video lawan.")
            else:
                opponent_cam_placeholder.info("📷 Menunggu video lawan...")
        
    elif st.session_state.game_state["game_status"] == "result":
        show_result()
        
    elif st.session_state.game_state["game_status"] == "waiting":
        st.info("⏳ Menunggu pemain lain bergabung atau memulai putaran baru...")
        st.markdown("### 📋 Cara Bermain:")
        st.write("1. Bagikan Room ID di atas ke teman Anda")
        st.write("2. Tunggu mereka bergabung")
        st.write("3. Game akan otomatis dimulai setelah kedua pemain siap!")
    
    elif st.session_state.game_state["game_status"] == "connecting":
        st.info("⏳ Sedang menghubungkan ke server...")
        st.empty() # Placeholder untuk mencegah UI berkedip terlalu cepat

def show_result():
    """Tampilan hasil permainan"""
    if st.session_state.game_state["last_result"]:
        result = st.session_state.game_state["last_result"]
        
        st.markdown("### 🏆 Hasil Permainan")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        moves = result.get("moves", {})
        player_id = st.session_state.game_state["player_id"]
        
        gesture_emoji = {
            "rock": "✊",
            "paper": "✋",
            "scissors": "✌️"
        }
        
        # Display moves
        player_move_val = moves.get(player_id, "?")
        opponent_move_val = moves.get(st.session_state.game_state["opponent_id"], "?") # Asumsi opponent_id ada di game_state

        with col1:
            st.markdown("### Anda")
            st.markdown(f'<h1 style="text-align:center">{gesture_emoji.get(player_move_val, "?")}</h1>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align:center;font-size:24px">{player_move_val.upper()}</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### VS")
            
            # Display result
            winner_id = result.get("winner_id")
            if winner_id == player_id:
                st.markdown('<h2 style="text-align:center;color:green">🎉</h2>', unsafe_allow_html=True)
            elif winner_id is None:
                st.markdown('<h2 style="text-align:center;color:orange">🤝</h2>', unsafe_allow_html=True)
            else:
                st.markdown('<h2 style="text-align:center;color:red">😔</h2>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"### {st.session_state.game_state['opponent_name'] or 'Lawan'}")
            st.markdown(f'<h1 style="text-align:center">{gesture_emoji.get(opponent_move_val, "?")}</h1>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align:center;font-size:24px">{opponent_move_val.upper()}</p>', unsafe_allow_html=True)
        
        # Result message
        st.markdown("---")
        winner_id = result.get("winner_id")
        if winner_id == player_id:
            st.success("# 🎉 ANDA MENANG! 🎉")
        elif winner_id is None:
            st.info("# 🤝 SERI! 🤝")
        else:
            st.error("# 😔 ANDA KALAH! 😔")
        
        # Display scores
        st.markdown("### 📊 Skor Saat Ini")
        if st.session_state.game_state["scores"]:
            score_cols = st.columns(2)
            # Asumsi scores dari backend adalah dict {player_id: score}
            with score_cols[0]:
                st.metric("Skor Anda", st.session_state.game_state["scores"].get(st.session_state.game_state["player_id"], 0))
            with score_cols[1]:
                # Cari ID lawan di scores
                opponent_score_id = next((pid for pid in st.session_state.game_state["scores"] if pid != st.session_state.game_state["player_id"]), None)
                st.metric(f"Skor {st.session_state.game_state['opponent_name'] or 'Lawan'}", 
                          st.session_state.game_state["scores"].get(opponent_score_id, 0))
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎮 Main Lagi", type="primary", use_container_width=True):
                # Reset game state untuk putaran baru
                st.session_state.game_state["current_gesture"] = None
                st.session_state.game_state["gesture_locked"] = False
                st.session_state.game_state["game_status"] = "waiting" # Kembali ke waiting untuk putaran baru
                st.session_state.game_state["last_result"] = None
                st.session_state.game_state["opponent_frame_base64"] = None
                
                # Kirim event play_again ke backend
                st.session_state.message_queue.put({
                    "event": "play_again",
                    "room_id": st.session_state.game_state["room_id"],
                    "player_id": st.session_state.game_state["player_id"]
                })
                st.rerun()
        
        with col2:
            if st.button("🚪 Keluar Room", type="secondary", use_container_width=True):
                # Kirim event disconnect ke backend
                st.session_state.message_queue.put({
                    "event": "disconnect_player",
                    "room_id": st.session_state.game_state["room_id"],
                    "player_id": st.session_state.game_state["player_id"]
                })
                # Reset semua game state dan putuskan koneksi WS
                if st.session_state.ws_client_instance:
                    asyncio.run(st.session_state.ws_client_instance.close()) # Tutup WS connection
                st.session_state.game_state = {k: None for k in st.session_state.game_state.keys()} # Reset all
                st.session_state.game_state["game_status"] = "lobby" # Kembali ke lobby
                st.rerun()

# --- WebSocket Client (di luar main Streamlit render loop) ---
# Ini harus di luar fungsi main() karena dijalankan di thread terpisah
class WebSocketClient:
    def __init__(self, url: str, client_id: str, message_queue: queue.Queue):
        self.url = f"{url}/{client_id}"
        self.client_id = client_id
        self.message_queue = message_queue
        self.websocket = None
        self.running = False
        self.loop = asyncio.new_event_loop() # Buat event loop sendiri

    async def connect(self):
        """Koneksi ke backend WebSocket"""
        print(f"WS Client: Attempting to connect to {self.url}")
        try:
            self.websocket = await websockets.connect(self.url)
            st.session_state.game_state["ws_connected"] = True
            self.running = True
            print(f"WS Client: Connected to {self.url}")
            
            # Jalankan receive dan process outgoing secara bersamaan
            await asyncio.gather(
                self.receive_messages(),
                self.process_outgoing_messages()
            )
        except Exception as e:
            print(f"WS Client: Koneksi WebSocket gagal: {e}")
            st.session_state.game_state["ws_connected"] = False
            # Jika koneksi gagal, set status kembali ke lobby agar bisa coba lagi
            st.session_state.game_state["game_status"] = "lobby" 
            st.error(f"Koneksi WebSocket gagal: {e}. Pastikan backend berjalan.")
            st.rerun() # Rerun untuk menampilkan error dan kembali ke lobby
        finally:
            self.running = False
            st.session_state.game_state["ws_connected"] = False
            print(f"WS Client: Disconnected from {self.url}")
            st.rerun() # Rerun untuk update UI bahwa WS terputus

    async def receive_messages(self):
        """Terima pesan dari server"""
        try:
            while self.running and self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                event = data.get("event")
                
                print(f"WS Client: Received event: {event} with data: {data}") # Debugging
                
                # Update game state berdasarkan event dari server
                if event == "room_created":
                    st.session_state.game_state["room_id"] = data["room_id"]
                    st.session_state.game_state["player_role"] = data["role"] # Simpan role
                    st.session_state.game_state["game_status"] = "waiting"
                    st.session_state.game_state["scores"] = {"player": 0, "opponent": 0} # Reset skor
                
                elif event == "joined_room":
                    st.session_state.game_state["room_id"] = data["room_id"]
                    st.session_state.game_state["player_role"] = data["role"] # Simpan role
                    st.session_state.game_state["game_status"] = "waiting"
                    st.session_state.game_state["scores"] = {"player": 0, "opponent": 0} # Reset skor

                elif event == "player_joined" or event == "game_state_update": # Backend harus mengirim ini saat pemain join
                    st.session_state.game_state["game_status"] = data.get("status", "waiting")
                    players_info = data.get("players", {})
                    
                    # Update player names and roles
                    st.session_state.game_state["opponent_name"] = None
                    st.session_state.game_state["opponent_id"] = None
                    
                    for pid, pinfo in players_info.items():
                        if pinfo["id"] == st.session_state.game_state["player_id"]:
                            st.session_state.game_state["player_name"] = pinfo["name"]
                        else:
                            st.session_state.game_state["opponent_name"] = pinfo["name"]
                            st.session_state.game_state["opponent_id"] = pinfo["id"]

                elif event == "game_ready":
                    st.session_state.game_state["game_status"] = "playing" # Langsung playing
                    st.session_state.game_state["gesture_locked"] = False
                    st.session_state.game_state["current_gesture"] = None
                    st.session_state.game_state["last_result"] = None # Reset hasil sebelumnya
                    st.session_state.game_state["opponent_frame_base64"] = None
                    st.session_state.game_state["camera_active"] = True # Aktifkan kamera
                    st.session_state.game_state["scores"] = data.get("scores", {"player": 0, "opponent": 0}) # Update skor dari server
                
                elif event == "video_frame":
                    st.session_state.game_state["opponent_frame_base64"] = data.get("frame")
                    # Tidak perlu rerun di sini, karena frame akan di-render di main loop
                
                elif event == "round_result":
                    st.session_state.game_state["last_result"] = data
                    st.session_state.game_state["game_status"] = "result"
                    st.session_state.game_state["camera_active"] = False # Matikan kamera setelah hasil
                    if "scores" in data:
                        st.session_state.game_state["scores"] = data["scores"]
                
                elif event == "player_disconnected":
                    st.session_state.game_state["game_status"] = "waiting"
                    st.session_state.game_state["opponent_name"] = None
                    st.session_state.game_state["opponent_id"] = None
                    st.session_state.game_state["opponent_frame_base64"] = None
                    st.session_state.game_state["camera_active"] = False # Matikan kamera
                    st.session_state.game_state["current_gesture"] = None # Reset gesture
                    st.session_state.game_state["gesture_locked"] = False
                    st.session_state.game_state["last_result"] = None # Reset hasil
                    st.warning("Pemain lain terputus. Menunggu pemain baru...")

                elif event == "game_reset": # Event dari backend untuk memulai putaran baru setelah "Main Lagi"
                    st.session_state.game_state["game_status"] = "playing"
                    st.session_state.game_state["current_gesture"] = None
                    st.session_state.game_state["gesture_locked"] = False
                    st.session_state.game_state["last_result"] = None
                    st.session_state.game_state["opponent_frame_base64"] = None
                    st.session_state.game_state["camera_active"] = True
                    # Skor tidak direset di sini, karena ini hanya putaran baru, bukan game baru

                st.rerun() # Penting untuk memicu update UI setelah menerima event
                
        except websockets.exceptions.ConnectionClosed:
            print("WS Client: Connection closed gracefully.")
        except Exception as e:
            print(f"WS Client: Error receiving message: {e}")
        finally:
            self.running = False
            st.session_state.game_state["ws_connected"] = False
            st.session_state.game_state["game_status"] = "lobby" # Kembali ke lobby jika WS terputus
            st.rerun() # Rerun untuk update UI status

    async def process_outgoing_messages(self):
        """Kirim pesan dari queue ke server"""
        while self.running and self.websocket:
            try:
                message = self.message_queue.get_nowait() # Coba ambil pesan tanpa blocking
                await self.websocket.send(json.dumps(message))
                print(f"WS Client: Sent message: {message}") # Debugging
            except queue.Empty:
                pass # Queue kosong, tidak ada pesan untuk dikirim
            except Exception as e:
                print(f"WS Client: Error sending message: {e}")
            await asyncio.sleep(0.05) # Delay kecil agar tidak terlalu membebani CPU

    async def close(self):
        """Tutup koneksi WebSocket"""
        if self.websocket:
            await self.websocket.close()
        self.running = False

# Fungsi untuk menjalankan asyncio loop di thread terpisah
def run_websocket_loop_in_thread(loop, client_id):
    asyncio.set_event_loop(loop)
    client = WebSocketClient(BACKEND_WS_URL, client_id, st.session_state.message_queue)
    st.session_state.ws_client_instance = client # Simpan instance client di session state
    loop.run_until_complete(client.connect())

# Camera handler (berjalan di thread terpisah)
def camera_handler():
    """Handle camera capture dan kirim frame ke queue untuk dikirim via WS"""
    if not CV2_AVAILABLE:
        # Mode simulasi jika OpenCV tidak tersedia
        while st.session_state.game_state["camera_active"]:
            if (st.session_state.game_state["game_status"] == "playing" and 
                not st.session_state.game_state["gesture_locked"]):
                
                # Simulasi gesture setelah 3 detik
                if st.session_state.get("sim_gesture_timer") is None:
                    st.session_state.sim_gesture_timer = time.time()
                
                elapsed_sim = time.time() - st.session_state.sim_gesture_timer
                if elapsed_sim >= 3:
                    gestures = ["rock", "paper", "scissors"]
                    gesture = np.random.choice(gestures)
                    st.session_state.game_state["current_gesture"] = gesture
                    st.session_state.game_state["gesture_locked"] = True
                    st.session_state.sim_gesture_timer = None # Reset timer
                    
                    if st.session_state.game_state["room_id"] and st.session_state.ws_client_instance:
                        st.session_state.message_queue.put({
                            "event": "player_move",
                            "room_id": st.session_state.game_state["room_id"],
                            "player_id": st.session_state.game_state["player_id"],
                            "move": gesture
                        })
                    st.rerun() # Memicu UI update

            time.sleep(0.1) # Jangan terlalu cepat di simulasi
        return
    
    # Mode normal dengan OpenCV
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera: Gagal membuka kamera.")
        st.session_state.game_state["camera_active"] = False
        st.error("Gagal membuka kamera. Pastikan kamera tidak digunakan aplikasi lain.")
        st.rerun()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    hands_detector = None
    if MEDIAPIPE_AVAILABLE:
        hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    gesture_start_time = None
    last_gesture = None
    
    while st.session_state.game_state["camera_active"]:
        ret, frame = cap.read()
        if not ret:
            print("Camera: Gagal membaca frame.")
            continue
        
        frame = cv2.flip(frame, 1) # Mirror frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        current_gesture_detected = None
        
        if (st.session_state.game_state["game_status"] == "playing" and 
            not st.session_state.game_state["gesture_locked"] and 
            hands_detector):
            
            results = hands_detector.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    current_gesture_detected = classify_gesture(hand_landmarks)
            
            # Logika penguncian gesture
            if current_gesture_detected:
                if current_gesture_detected != last_gesture:
                    gesture_start_time = time.time()
                    last_gesture = current_gesture_detected
                
                elapsed = time.time() - gesture_start_time if gesture_start_time else 0
                remaining_time_display = max(0, 3 - int(elapsed))
                
                cv2.putText(frame, f"Gesture: {current_gesture_detected.upper()}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Lock in: {remaining_time_display}", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if elapsed >= 3 and not st.session_state.game_state["gesture_locked"]:
                    st.session_state.game_state["gesture_locked"] = True
                    st.session_state.game_state["current_gesture"] = current_gesture_detected # Kunci gesture di state
                    
                    # Kirim gerakan ke backend
                    if st.session_state.game_state["room_id"] and st.session_state.ws_client_instance:
                        st.session_state.message_queue.put({
                            "event": "player_move",
                            "room_id": st.session_state.game_state["room_id"],
                            "player_id": st.session_state.game_state["player_id"],
                            "move": current_gesture_detected
                        })
                    cv2.putText(frame, "LOCKED!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    st.rerun() # Memicu UI update
            else:
                gesture_start_time = None
                last_gesture = None
                cv2.putText(frame, "Tunjukkan tangan Anda", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Simpan frame untuk ditampilkan di UI utama
        st.session_state.current_frame = frame # Simpan frame BGR
        
        # Kirim frame ke opponent (jika game sedang bermain)
        if st.session_state.game_state["game_status"] == "playing" and st.session_state.game_state["room_id"]:
            encoded_frame = encode_frame(frame)
            if encoded_frame and st.session_state.ws_client_instance:
                st.session_state.message_queue.put({
                    "event": "video_frame",
                    "room_id": st.session_state.game_state["room_id"],
                    "player_id": st.session_state.game_state["player_id"], # Penting: sertakan player_id
                    "frame": encoded_frame
                })
        
        time.sleep(0.01) # Kecilkan delay untuk frame rate lebih tinggi
        
    cap.release()
    if hands_detector:
        hands_detector.close()
    print("Camera: Kamera dimatikan.")

# Main app logic
def main():
    # Sidebar info
    with st.sidebar:
        st.markdown("### 🎮 GBK Online")
        
        if st.session_state.game_state["ws_connected"]:
            st.success("✅ Terhubung ke server")
        else:
            st.error("❌ Tidak terhubung")
        
        st.markdown("---")
        st.markdown("### 📊 Status")
        st.json({
            "status": st.session_state.game_state["game_status"],
            "room": st.session_state.game_state["room_id"],
            "player": st.session_state.game_state["player_name"],
            "player_id": st.session_state.game_state["player_id"],
            "role": st.session_state.game_state["player_role"],
            "opponent": st.session_state.game_state["opponent_name"],
            "opencv": CV2_AVAILABLE,
            "mediapipe": MEDIAPIPE_AVAILABLE
        })
        
        st.markdown("---")
        st.markdown("### 🎯 Cara Bermain")
        st.write("✊ Batu mengalahkan Gunting")
        st.write("✋ Kertas mengalahkan Batu")
        st.write("✌️ Gunting mengalahkan Kertas")
    
    # --- Inisialisasi WebSocket Client di main() ---
    # Ini harus dilakukan sekali saja saat aplikasi dimulai
    if st.session_state.ws_thread is None:
        st.session_state.ws_thread = threading.Thread(
            target=run_websocket_loop_in_thread,
            args=(asyncio.new_event_loop(), st.session_state.game_state["player_id"])
        )
        st.session_state.ws_thread.daemon = True # Penting agar thread mati saat Streamlit keluar
        st.session_state.ws_thread.start()
        # Beri sedikit waktu agar thread WS bisa memulai koneksi
        time.sleep(0.5) 
        st.rerun() # Rerun untuk update UI status koneksi
    # -----------------------------------------------

    # Main content display logic
    if st.session_state.game_state["game_status"] == "lobby" or not st.session_state.game_state["room_id"]:
        show_lobby()
    else:
        show_game_room()

if __name__ == "__main__":
    main()
