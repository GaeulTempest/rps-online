# Frontend Streamlit untuk GBK Online
# File: frontend/app.py

import streamlit as st
import numpy as np
import asyncio
import websockets
import json
import threading
import queue # Pastikan ini terimpor
import time
import base64
import io
from PIL import Image
from collections import deque, Counter
from enum import Enum
from typing import Optional, Dict

# --- PASTIKAN st.set_page_config() ADALAH PERINTAH STREAMLIT PERTAMA ---
st.set_page_config(
    page_title="Gunting Batu Kertas Online",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# -------------------------------------------------------------------

# Try importing OpenCV and MediaPipe with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV tidak tersedia. Menggunakan mode simulasi gesture.")

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.warning("⚠️ MediaPipe tidak tersedia. Deteksi gesture akan disimulasikan.")

# Backend WebSocket URL - GANTI DENGAN URL RAILWAY ANDA YANG BENAR!
BACKEND_WS_URL = st.secrets.get("BACKEND_WS_URL", "ws://localhost:8000/ws") 

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 18px;
    }
    .game-title { text-align: center; font-size: 48px; font-weight: bold; margin-bottom: 20px; }
    .room-code { 
        font-size: 36px; font-weight: bold; color: #FF6B6B; 
        text-align: center; padding: 10px; border: 3px solid #FF6B6B; 
        border-radius: 10px; margin: 10px 0; 
    }
    .score-display { font-size: 24px; text-align: center; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# --- State Management: Inisialisasi SEMUA atribut st.session_state di top-level ---
st.session_state.setdefault("game_state", {
    "room_id": None, "player_id": None, "player_name": None,
    "game_status": "lobby", # lobby, connecting, waiting, playing, result, disconnected
    "current_gesture": None, "gesture_locked": False,
    "opponent_name": None, "opponent_id": None,
    "scores": {"player": 0, "opponent": 0}, 
    "last_result": None, "ws_connected": False,
    "opponent_frame_base64": None, "player_role": None,
    "sim_gesture_timer": None
})
st.session_state.setdefault("message_queue", queue.Queue()) # Ini akan dilewatkan ke thread
st.session_state.setdefault("ui_update_queue", queue.Queue()) # Queue baru untuk pesan dari thread ke main UI
st.session_state.setdefault("ws_thread", None)
st.session_state.setdefault("camera_thread", None)
st.session_state.setdefault("current_frame", None)
st.session_state.setdefault("ws_client_instance", None) 
st.session_state.setdefault("mp_hands_instance", None) # Inisialisasi MediaPipe Hands instance di main thread

# --- Helper Functions ---
# Fungsi untuk klasifikasi gesture
class RPSMove(str, Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    NONE = "none"

def classify_gesture(hand_landmarks) -> Optional[str]:
    if not hand_landmarks: return None
    
    landmarks = hand_landmarks.landmark
    fingers_up = []
    
    # Jempol (index 0)
    # Anda mungkin perlu menyesuaikan ini jika ada masalah deteksi jempol atau orientasi tangan.
    if landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x:
        fingers_up.append(1) # Jempol dianggap terangkat
    else:
        fingers_up.append(0) # Jempol dianggap ditekuk
    
    # Jari lainnya (index 1-4)
    finger_tip_indices = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                          mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    finger_pip_indices = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                          mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]

    for tip_idx, pip_idx in zip(finger_tip_indices, finger_pip_indices):
        if landmarks[tip_idx].y < landmarks[pip_idx].y:
            fingers_up.append(1) # Jari terangkat
        else:
            fingers_up.append(0) # Jari ditekuk
    
    # Klasifikasi
    if fingers_up == [0, 0, 0, 0, 0]: return RPSMove.ROCK.value
    elif fingers_up == [1, 1, 1, 1, 1]: return RPSMove.PAPER.value
    elif fingers_up == [0, 1, 1, 0, 0]: return RPSMove.SCISSORS.value
    
    return RPSMove.NONE.value

def encode_frame(frame):
    try:
        if CV2_AVAILABLE:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
            return base64.b64encode(buffer).decode('utf-8')
        else:
            # Jika OpenCV tidak ada, asumsikan frame adalah numpy array RGB
            img = Image.fromarray(frame) 
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=30)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return None

def decode_frame(frame_str):
    try:
        img_data = base64.b64decode(frame_str)
        if CV2_AVAILABLE:
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Kembali ke BGR
        else:
            img = Image.open(io.BytesIO(img_data))
            return np.array(img) # Kembali ke numpy array RGB
    except Exception as e:
        print(f"Error decoding frame: {e}")
        return None

# --- WebSocket Client (dijalankan di thread terpisah) ---
class WebSocketClient:
    def __init__(self, url: str, client_id: str, message_queue: queue.Queue, ui_update_queue: queue.Queue):
        self.url = f"{url}/{client_id}"
        self.client_id = client_id
        self.message_queue = message_queue # Untuk mengirim pesan ke server
        self.ui_update_queue = ui_update_queue # Untuk mengirim pesan ke main thread (UI)
        self.websocket = None
        self.running = False
        self.loop = asyncio.new_event_loop()

    async def connect(self):
        print(f"WS Client: Attempting to connect to {self.url}")
        try:
            self.websocket = await websockets.connect(self.url)
            self.running = True
            self.ui_update_queue.put({"event": "ws_connected_status", "status": True})
            print(f"WS Client: Connected to {self.url}")
            
            await asyncio.gather(
                self.receive_messages(),
                self.process_outgoing_messages()
            )
        except Exception as e:
            print(f"WS Client: Koneksi WebSocket gagal: {e}")
            self.ui_update_queue.put({"event": "ws_connected_status", "status": False})
            self.ui_update_queue.put({"event": "error", "message": f"Koneksi WebSocket gagal: {e}. Pastikan backend berjalan."})
        finally:
            self.running = False
            if self.websocket:
                await self.websocket.close()
            self.ui_update_queue.put({"event": "ws_connected_status", "status": False})
            print(f"WS Client: Disconnected from {self.url}")
            # Streamlit akan me-rerun secara otomatis jika ada perubahan state

    async def receive_messages(self):
        try:
            while self.running and self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                self.ui_update_queue.put({"event": "backend_message", "data": data}) # Kirim ke UI queue
                
        except websockets.exceptions.ConnectionClosed:
            print("WS Client: Connection closed gracefully.")
        except Exception as e:
            print(f"WS Client: Error receiving message: {e}")
        finally:
            self.running = False # Pastikan loop berhenti
            self.ui_update_queue.put({"event": "ws_connected_status", "status": False})

    async def process_outgoing_messages(self):
        while self.running and self.websocket:
            try:
                # Ambil pesan dari message_queue (untuk dikirim ke server)
                message = self.message_queue.get_nowait() 
                await self.websocket.send(json.dumps(message))
                print(f"WS Client: Sent message: {message}")
            except queue.Empty:
                pass
            except Exception as e:
                print(f"WS Client: Error sending message: {e}")
            await asyncio.sleep(0.05) 

    async def close(self):
        self.running = False
        if self.websocket:
            await self.websocket.close()

# Fungsi untuk menjalankan asyncio loop di thread terpisah (untuk WebSocketClient)
def run_websocket_loop_in_thread(loop, client_id, message_queue_ref, ui_update_queue_ref):
    asyncio.set_event_loop(loop)
    client = WebSocketClient(BACKEND_WS_URL, client_id, message_queue_ref, ui_update_queue_ref)
    st.session_state.ws_client_instance = client # Simpan instance client
    loop.run_until_complete(client.connect())

# Camera handler (berjalan di thread terpisah)
def camera_handler_thread(message_queue_ref: queue.Queue, ui_update_queue_ref: queue.Queue, mp_hands_instance_ref):
    """Handle camera capture dan deteksi gesture di thread terpisah."""
    if not CV2_AVAILABLE:
        # Mode simulasi
        while st.session_state.game_state["camera_active"]: # Tetap pakai game_state dari session state
            if (st.session_state.game_state["game_status"] == "playing" and 
                not st.session_state.game_state["gesture_locked"]):
                
                if st.session_state.game_state["sim_gesture_timer"] is None:
                    st.session_state.game_state["sim_gesture_timer"] = time.time()
                
                elapsed_sim = time.time() - st.session_state.game_state["sim_gesture_timer"]
                
                if elapsed_sim >= 3:
                    gestures = ["rock", "paper", "scissors"]
                    gesture = np.random.choice(gestures)
                    st.session_state.game_state["current_gesture"] = gesture # Update game_state
                    st.session_state.game_state["gesture_locked"] = True # Update game_state
                    st.session_state.game_state["sim_gesture_timer"] = None 
                    
                    if st.session_state.game_state["room_id"] and st.session_state.ws_client_instance:
                        message_queue_ref.put({ # Kirim pesan via queue yang dilewatkan
                            "event": "player_move",
                            "room_id": st.session_state.game_state["room_id"],
                            "player_id": st.session_state.game_state["player_id"],
                            "move": gesture
                        })
                    ui_update_queue_ref.put({"event": "rerun_ui"}) # Memicu UI update
            else:
                 st.session_state.game_state["current_gesture"] = "Simulating..." # Tampilkan status
            
            time.sleep(0.1)
        return
    
    # Mode normal dengan OpenCV dan MediaPipe
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera: Gagal membuka kamera.")
        ui_update_queue_ref.put({"event": "camera_error", "message": "Gagal membuka kamera."})
        st.session_state.game_state["camera_active"] = False
        ui_update_queue_ref.put({"event": "rerun_ui"}) # Memicu UI update
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    hands_detector = mp_hands_instance_ref # Gunakan instance yang dilewatkan
    
    gesture_start_time = None
    last_gesture = None
    
    while st.session_state.game_state["camera_active"]: # Loop selama kamera aktif
        ret, frame = cap.read()
        if not ret:
            print("Camera: Gagal membaca frame.")
            time.sleep(0.01)
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
            if current_gesture_detected and current_gesture_detected != RPSMove.NONE.value:
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
                    st.session_state.game_state["current_gesture"] = current_gesture_detected
                    
                    if st.session_state.game_state["room_id"] and st.session_state.ws_client_instance:
                        message_queue_ref.put({ # Kirim pesan via queue yang dilewatkan
                            "event": "player_move",
                            "room_id": st.session_state.game_state["room_id"],
                            "player_id": st.session_state.game_state["player_id"],
                            "move": current_gesture_detected
                        })
                    cv2.putText(frame, "LOCKED!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    ui_update_queue_ref.put({"event": "rerun_ui"}) # Memicu UI update
            else:
                gesture_start_time = None
                last_gesture = None
                st.session_state.game_state["current_gesture"] = None 
                cv2.putText(frame, "Tunjukkan tangan Anda", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        st.session_state.current_frame = frame 
        
        # Kirim frame ke opponent
        if st.session_state.game_state["game_status"] == "playing" and st.session_state.game_state["room_id"] and st.session_state.ws_client_instance:
            encoded_frame = encode_frame(frame)
            if encoded_frame:
                message_queue_ref.put({ # Kirim pesan via queue yang dilewatkan
                    "event": "video_frame",
                    "room_id": st.session_state.game_state["room_id"],
                    "player_id": st.session_state.game_state["player_id"],
                    "frame": encoded_frame
                })
        
        time.sleep(0.01) 
        
    cap.release()
    if hands_detector:
        hands_detector.close()
    print("Camera: Kamera dimatikan.")
    ui_update_queue_ref.put({"event": "camera_stopped"})
    ui_update_queue_ref.put({"event": "rerun_ui"}) # Memicu UI update


# --- UI Functions ---
def show_lobby():
    st.markdown('<h1 class="game-title">🎮 Gunting Batu Kertas Online</h1>', unsafe_allow_html=True)
    st.markdown("### Selamat datang! Mainkan game klasik dengan teknologi modern")
    
    with st.expander("ℹ️ Status Koneksi & Info Umum"):
        st.info(f"Backend WebSocket: {BACKEND_WS_URL}")
        if st.session_state.game_state["ws_connected"]:
            st.success("✅ Koneksi server aktif.")
        else:
            st.error("❌ Koneksi server terputus.")

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
            player_name_create = st.text_input("Nama Anda", 
                                                value=st.session_state.game_state.get("player_name", f"Player_{np.random.randint(1000, 9999)}"),
                                                max_chars=20, key="create_name_input")
            create_submitted = st.form_submit_button("Buat Room", type="primary", disabled=not st.session_state.game_state["ws_connected"])
            
            if create_submitted and player_name_create:
                st.session_state.game_state["player_name"] = player_name_create
                st.session_state.game_state["player_id"] = f"p_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
                
                st.session_state.message_queue.put({
                    "event": "create_room",
                    "player_id": st.session_state.game_state["player_id"],
                    "player_name": player_name_create
                })
                st.session_state.game_state["game_status"] = "connecting" 
                st.success("🚀 Membuat room...")
                st.rerun() 
    
    with col2:
        st.markdown("### 🚪 Gabung Room")
        with st.form("join_room_form"):
            room_id_input = st.text_input("Room ID", 
                                          placeholder="Masukkan 8 karakter Room ID",
                                          max_chars=8, key="join_room_input")
            player_name_join = st.text_input("Nama Anda",
                                              value=st.session_state.game_state.get("player_name", f"Player_{np.random.randint(1000, 9999)}"),
                                              max_chars=20, key="join_name_input")
            join_submitted = st.form_submit_button("Gabung Room", type="primary", disabled=not (room_id_input and player_name_join and st.session_state.game_state["ws_connected"]))
            
            if join_submitted and room_id_input and player_name_join:
                st.session_state.game_state["player_name"] = player_name_join
                st.session_state.game_state["player_id"] = f"p_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
                
                st.session_state.message_queue.put({
                    "event": "join_room",
                    "room_id": room_id_input,
                    "player_id": st.session_state.game_state["player_id"],
                    "player_name": player_name_join
                })
                st.session_state.game_state["game_status"] = "connecting" 
                st.success("🔗 Bergabung ke room...")
                st.rerun() 
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Gunakan gesture tangan: ✊ Batu | ✋ Kertas | ✌️ Gunting</p>
    </div>
    """, unsafe_allow_html=True)

def show_game_room():
    st.markdown('<h1 class="game-title">🎮 Gunting Batu Kertas Online</h1>', unsafe_allow_html=True)
    
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
            "result": "🏆 Hasil",
            "connecting": "⏳ Menghubungkan...",
            "disconnected": "🚫 Terputus!" # Tambahkan status disconnected
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
    
    st.markdown("### 📊 Skor Saat Ini")
    score_cols = st.columns(2)
    with score_cols[0]:
        st.metric("Skor Anda", st.session_state.game_state["scores"].get(st.session_state.game_state["player_id"], 0))
    with score_cols[1]:
        opponent_score_val = 0
        if opponent_id := st.session_state.game_state.get("opponent_id"):
            opponent_score_val = st.session_state.game_state["scores"].get(opponent_id, 0)
        st.metric(f"Skor {st.session_state.game_state['opponent_name'] or 'Lawan'}", opponent_score_val)


    # Game content
    if st.session_state.game_state["game_status"] == "playing":
        col_player_cam, col_opponent_cam = st.columns(2)

        with col_player_cam:
            st.markdown("### 📹 Kamera Anda")
            player_cam_placeholder = st.empty()
            gesture_display_placeholder = st.empty()

            # Ini akan terus diperbarui oleh camera_handler_thread
            if st.session_state.current_frame is not None:
                frame_to_display = cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else st.session_state.current_frame
                player_cam_placeholder.image(frame_to_display, use_column_width=True)
            else:
                player_cam_placeholder.info("📷 Mengaktifkan kamera...")
            
            if st.session_state.game_state["current_gesture"]:
                gesture_emoji = {
                    "rock": "✊", "paper": "✋", "scissors": "✌️", "Simulating...": "⏳"
                }
                gesture_placeholder1_text = f"### {gesture_emoji.get(st.session_state.game_state['current_gesture'], '')} " \
                                          f"{st.session_state.game_state['current_gesture'].upper()}"
                
                if st.session_state.game_state["gesture_locked"]:
                    gesture_display_placeholder.success(f"{gesture_placeholder1_text} - DIKUNCI!")
                else:
                    gesture_display_placeholder.info(gesture_placeholder1_text)
            else:
                gesture_display_placeholder.info("Tunjukkan gesture Anda ke kamera...")


        with col_opponent_cam:
            st.markdown("### 📹 Kamera Lawan")
            opponent_cam_placeholder = st.empty()
            if st.session_state.game_state["opponent_frame_base64"]:
                frame = decode_frame(st.session_state.game_state["opponent_frame_base64"])
                if frame is not None:
                    frame_to_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE and frame.ndim == 3 else frame
                    opponent_cam_placeholder.image(frame_to_display, use_column_width=True)
                else:
                    opponent_cam_placeholder.info("Gagal menampilkan video lawan.")
            else:
                opponent_cam_placeholder.info("📷 Menunggu video lawan...")
        
        with st.expander("📖 Cara Bermain"):
            if CV2_AVAILABLE and MEDIAPIPE_AVAILABLE:
                st.write("1. Tunjukkan tangan Anda ke kamera (posisikan agar terlihat jelas)")
                st.write("2. Buat gesture: ✊ Batu | ✋ Kertas | ✌️ Gunting")
                st.write("3. Tahan gesture yang sama selama 3 detik untuk mengunci pilihan Anda.")
                st.write("4. Tunggu hasil setelah kedua pemain selesai.")
            else:
                st.write("1. Pilih gesture Anda secara manual.")
                st.write("2. Klik 'Kunci Pilihan Simulasi' atau tunggu auto-select.")
                st.write("3. Tunggu lawan membuat pilihan.")
                st.write("4. Lihat hasilnya!")
        
        # Penting: Memicu rerun untuk update UI terus-menerus saat bermain
        time.sleep(0.1) 
        st.rerun() # Ini akan memicu app untuk terus menerus refresh UI, penting untuk kamera dan update lawan
        
    elif st.session_state.game_state["game_status"] == "result":
        show_result()
        
    elif st.session_state.game_state["game_status"] == "waiting":
        st.info("⏳ Menunggu pemain lain bergabung atau memulai putaran baru...")
        st.markdown("### 📋 Cara Bermain:")
        st.write("1. Bagikan Room ID di atas ke teman Anda")
        st.write("2. Tunggu mereka bergabung")
        st.write("3. Game akan otomatis dimulai setelah kedua pemain siap dan putaran baru dimulai.")
    
    elif st.session_state.game_state["game_status"] == "connecting":
        st.info("⏳ Sedang menghubungkan ke server dan mencari room/pemain...")
    
    elif st.session_state.game_state["game_status"] == "disconnected":
        st.error("❌ Koneksi ke server terputus. Silakan kembali ke Lobby.")
        if st.button("Kembali ke Lobby", type="primary"):
            st.session_state.game_state = {k: None for k in st.session_state.game_state.keys()} # Reset all
            st.session_state.game_state["game_status"] = "lobby" # Kembali ke lobby
            st.rerun()

def show_result():
    if st.session_state.game_state["last_result"]:
        result = st.session_state.game_state["last_result"]
        
        st.markdown("### 🏆 Hasil Permainan")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        moves = result.get("moves", {})
        player_id = st.session_state.game_state["player_id"]
        
        gesture_emoji = {
            "rock": "✊", "paper": "✋", "scissors": "✌️", "none": "❓"
        }
        
        player_move_val = moves.get(player_id, "none")
        opponent_move_id = st.session_state.game_state.get("opponent_id", None)
        opponent_move_val = moves.get(opponent_move_id, "none") if opponent_move_id else "none"

        with col1:
            st.markdown("### Anda")
            st.markdown(f'<h1 style="text-align:center">{gesture_emoji.get(player_move_val, "?")}</h1>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align:center;font-size:24px">{player_move_val.upper()}</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### VS")
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
        
        st.markdown("---")
        winner_id = result.get("winner_id")
        if winner_id == player_id:
            st.success("# 🎉 ANDA MENANG! 🎉")
        elif winner_id is None:
            st.info("# 🤝 SERI! 🤝")
        else:
            st.error("# 😔 ANDA KALAH! 😔")
        
        st.markdown("### 📊 Skor Saat Ini")
        if st.session_state.game_state["scores"]:
            score_cols = st.columns(2)
            with score_cols[0]:
                st.metric("Skor Anda", st.session_state.game_state["scores"].get(st.session_state.game_state["player_id"], 0))
            with score_cols[1]:
                opponent_score_val = 0
                if opponent_id := st.session_state.game_state.get("opponent_id"):
                    opponent_score_val = st.session_state.game_state["scores"].get(opponent_id, 0)
                st.metric(f"Skor {st.session_state.game_state['opponent_name'] or 'Lawan'}", opponent_score_val)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎮 Main Lagi", type="primary", use_container_width=True, disabled=not st.session_state.game_state["ws_connected"]):
                st.session_state.game_state["current_gesture"] = None
                st.session_state.game_state["gesture_locked"] = False
                st.session_state.game_state["game_status"] = "playing" # Langsung playing karena sudah ada 2 pemain
                st.session_state.game_state["last_result"] = None
                st.session_state.game_state["opponent_frame_base64"] = None
                
                st.session_state.message_queue.put({
                    "event": "play_again",
                    "room_id": st.session_state.game_state["room_id"],
                    "player_id": st.session_state.game_state["player_id"]
                })
                if not st.session_state.camera_thread or not st.session_state.camera_thread.is_alive():
                    st.session_state.game_state["camera_active"] = True
                    st.session_state.camera_thread = threading.Thread(
                        target=camera_handler_thread, 
                        args=(st.session_state.message_queue, st.session_state.ui_update_queue, st.session_state.mp_hands_instance),
                        daemon=True
                    )
                    st.session_state.camera_thread.start()

                st.rerun()
        
        with col2:
            if st.button("🚪 Keluar Room", type="secondary", use_container_width=True):
                if st.session_state.game_state["ws_connected"] and st.session_state.ws_client_instance:
                    st.session_state.message_queue.put({
                        "event": "disconnect_player",
                        "room_id": st.session_state.game_state["room_id"],
                        "player_id": st.session_state.game_state["player_id"]
                    })
                    time.sleep(0.05) 
                    # Jika ingin menutup WS dari thread utama, harus pakai loop thread WS
                    asyncio.run_coroutine_threadsafe(st.session_state.ws_client_instance.close(), st.session_state.ws_client_instance.loop).result()
                    st.session_state.ws_client_instance = None 
                    
                st.session_state.game_state["camera_active"] = False
                if st.session_state.camera_thread and st.session_state.camera_thread.is_alive():
                    st.session_state.camera_thread.join(timeout=1)
                st.session_state.camera_thread = None 

                # Reset semua game state
                for key in list(st.session_state.game_state.keys()):
                    st.session_state.game_state[key] = None
                # Atur ulang default yang penting
                st.session_state.game_state["game_status"] = "lobby"
                st.session_state.game_state["scores"] = {"player": 0, "opponent": 0}

                st.rerun()

# --- Main app logic ---
def main():
    # Sidebar info
    with st.sidebar:
        st.markdown("### 🎮 GBK Online")
        
        if st.session_state.game_state["ws_connected"]:
            st.success("✅ Terhubung ke server")
        else:
            st.error("❌ Tidak terhubung")
        
        st.markdown("---")
        st.markdown("### 📊 Status Game")
        st.json(st.session_state.game_state)
        
        st.markdown("---")
        st.markdown("### 🎯 Cara Bermain")
        st.write("✊ Batu mengalahkan Gunting")
        st.write("✋ Kertas mengalahkan Batu")
        st.write("✌️ Gunting mengalahkan Kertas")
    
    # --- Inisialisasi WebSocket Client di main() (hanya sekali) ---
    if st.session_state.ws_client_instance is None:
        if st.session_state.game_state["player_id"] is None:
            st.session_state.game_state["player_id"] = f"app_client_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"

        new_loop = asyncio.new_event_loop()
        ws_thread = threading.Thread(
            target=run_websocket_loop_in_thread,
            args=(new_loop, st.session_state.game_state["player_id"], 
                  st.session_state.message_queue, st.session_state.ui_update_queue),
            daemon=True
        )
        st.session_state.ws_thread = ws_thread
        ws_thread.start()
        
        time.sleep(0.5) 
        st.rerun() 
    # -----------------------------------------------

    # --- Proses Pesan dari Queue (dari thread WS dan Camera) ---
    while not st.session_state.ui_update_queue.empty():
        message = st.session_state.ui_update_queue.get()
        event_type = message.get("event")
        data = message.get("data") # Data bisa berupa dict, string, atau bool

        print(f"Main Thread: Processing UI event: {event_type}")

        if event_type == "ws_connected_status":
            st.session_state.game_state["ws_connected"] = message["status"]
            if not message["status"]: # Jika WS terputus
                st.session_state.game_state["game_status"] = "disconnected"
        elif event_type == "camera_error":
            st.error(message["message"])
        elif event_type == "camera_stopped":
            print("Main Thread: Camera thread signaled stop.")
        elif event_type == "error": # Untuk error umum dari thread
            st.error(message["message"])
            st.session_state.game_state["game_status"] = "disconnected"
        elif event_type == "backend_message":
            # Ini adalah event utama dari backend via WS
            backend_event = data.get("event")
            
            if backend_event == "room_created":
                st.session_state.game_state["room_id"] = data["room_id"]
                st.session_state.game_state["player_role"] = data["role"]
                st.session_state.game_state["game_status"] = "waiting"
                st.session_state.game_state["scores"] = {"player": 0, "opponent": 0}
                st.success(f"Room **{st.session_state.game_state['room_id']}** berhasil dibuat! Bagikan ID ini!")
            
            elif backend_event == "joined_room":
                st.session_state.game_state["room_id"] = data["room_id"]
                st.session_state.game_state["player_role"] = data["role"]
                st.session_state.game_state["game_status"] = "waiting"
                st.session_state.game_state["scores"] = {"player": 0, "opponent": 0}
                st.success(f"Berhasil bergabung ke room **{st.session_state.game_state['room_id']}**!")

            elif backend_event == "player_joined" or backend_event == "game_state_update":
                st.session_state.game_state["game_status"] = data.get("status", st.session_state.game_state["game_status"])
                players_info = data.get("players", {})
                
                st.session_state.game_state["opponent_name"] = None
                st.session_state.game_state["opponent_id"] = None
                
                for pid, pinfo in players_info.items():
                    if pinfo["id"] == st.session_state.game_state["player_id"]:
                        st.session_state.game_state["player_name"] = pinfo["name"] 
                    else:
                        st.session_state.game_state["opponent_name"] = pinfo["name"]
                        st.session_state.game_state["opponent_id"] = pinfo["id"]
                if data.get("scores"): 
                    st.session_state.game_state["scores"] = data["scores"]

            elif backend_event == "game_ready":
                st.session_state.game_state["game_status"] = "playing" 
                st.session_state.game_state["gesture_locked"] = False
                st.session_state.game_state["current_gesture"] = None
                st.session_state.game_state["last_result"] = None
                st.session_state.game_state["opponent_frame_base64"] = None
                st.session_state.game_state["camera_active"] = True 
                st.session_state.game_state["scores"] = data.get("scores", {"player": 0, "opponent": 0})
                
                # Inisialisasi MediaPipe Hands instance di main thread
                if st.session_state.mp_hands_instance is None and MEDIAPIPE_AVAILABLE:
                    st.session_state.mp_hands_instance = mp_hands.Hands(
                        static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.7, min_tracking_confidence=0.5
                    )

                # Mulai thread kamera jika belum aktif
                if not st.session_state.camera_thread or not st.session_state.camera_thread.is_alive():
                    st.session_state.camera_thread = threading.Thread(
                        target=camera_handler_thread, 
                        args=(st.session_state.message_queue, st.session_state.ui_update_queue, st.session_state.mp_hands_instance),
                        daemon=True
                    )
                    st.session_state.camera_thread.start()
                
            elif backend_event == "video_frame":
                st.session_state.game_state["opponent_frame_base64"] = data.get("frame")

            elif backend_event == "round_result":
                st.session_state.game_state["last_result"] = data
                st.session_state.game_state["game_status"] = "result"
                st.session_state.game_state["camera_active"] = False 
                st.session_state.game_state["gesture_locked"] = False 
                if "scores" in data:
                    st.session_state.game_state["scores"] = data["scores"]
            
            elif backend_event == "player_disconnected":
                st.session_state.game_state["game_status"] = "waiting"
                st.session_state.game_state["opponent_name"] = None
                st.session_state.game_state["opponent_id"] = None
                st.session_state.game_state["opponent_frame_base64"] = None
                st.session_state.game_state["camera_active"] = False 
                st.session_state.game_state["current_gesture"] = None 
                st.session_state.game_state["gesture_locked"] = False
                st.session_state.game_state["last_result"] = None
                st.warning("Pemain lain terputus. Menunggu pemain baru...")

            elif backend_event == "game_reset": 
                st.session_state.game_state["game_status"] = "playing"
                st.session_state.game_state["current_gesture"] = None
                st.session_state.game_state["gesture_locked"] = False
                st.session_state.game_state["last_result"] = None
                st.session_state.game_state["opponent_frame_base64"] = None
                st.session_state.game_state["camera_active"] = True
                # Skor tidak direset di sini, karena ini hanya putaran baru, bukan game baru
                if not st.session_state.camera_thread or not st.session_state.camera_thread.is_alive():
                    st.session_state.camera_thread = threading.Thread(
                        target=camera_handler_thread, 
                        args=(st.session_state.message_queue, st.session_state.ui_update_queue, st.session_state.mp_hands_instance),
                        daemon=True
                    )
                    st.session_state.camera_thread.start()
                
        st.rerun() # Memicu rerun setelah memproses event dari queue
    # -----------------------------------------------

    # Main content display logic
    if st.session_state.game_state["game_status"] == "lobby" or not st.session_state.game_state["room_id"]:
        show_lobby()
    else:
        show_game_room()

# Ini adalah titik masuk utama aplikasi Streamlit
if __name__ == "__main__":
    main()
