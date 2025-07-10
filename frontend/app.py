# Frontend Streamlit untuk GBK Online
# File: frontend/app.py
# Connected to: wss://illustrious-achievement-production-b825.up.railway.app/ws

import streamlit as st
import numpy as np
import asyncio
import websockets
import json
import threading
import queue
from typing import Optional, Dict
import time
from dataclasses import dataclass
import base64
import io
from PIL import Image

# Try importing OpenCV dengan fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV tidak tersedia. Menggunakan mode simulasi.")

# Import MediaPipe
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.warning("⚠️ MediaPipe tidak tersedia. Deteksi gesture akan disimulasikan.")

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Gunting Batu Kertas Online",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Backend WebSocket URL - SUDAH FIXED!
BACKEND_WS_URL = "wss://illustrious-achievement-production-b825.up.railway.app/ws"

# Custom CSS untuk UI yang lebih menarik
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 18px;
    }
    .game-title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .room-code {
        font-size: 36px;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 10px;
        border: 3px solid #FF6B6B;
        border-radius: 10px;
        margin: 10px 0;
    }
    .score-display {
        font-size: 24px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# State management untuk game
@dataclass
class GameState:
    room_id: Optional[str] = None
    player_id: Optional[str] = None
    player_name: Optional[str] = None
    game_status: str = "lobby"  # lobby, waiting, playing, result
    current_gesture: Optional[str] = None
    gesture_locked: bool = False
    countdown_start: Optional[float] = None
    opponent_name: Optional[str] = None
    scores: Dict[str, int] = None
    last_result: Optional[Dict] = None
    ws_connected: bool = False
    opponent_frame: Optional[np.ndarray] = None
    camera_active: bool = False

# Inisialisasi session state
if "game_state" not in st.session_state:
    st.session_state.game_state = GameState()
if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue()
if "ws_thread" not in st.session_state:
    st.session_state.ws_thread = None
if "camera_thread" not in st.session_state:
    st.session_state.camera_thread = None
if "current_frame" not in st.session_state:
    st.session_state.current_frame = None

# Fungsi untuk klasifikasi gesture
def classify_gesture(hand_landmarks) -> Optional[str]:
    """Klasifikasi gesture dari MediaPipe landmarks"""
    if not hand_landmarks or not MEDIAPIPE_AVAILABLE:
        return None
    
    landmarks = hand_landmarks.landmark
    fingers_up = []
    
    # Thumb
    if landmarks[4].x < landmarks[3].x:
        fingers_up.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers_up.append(1 if landmarks[4].x > landmarks[3].x else 0)
    
    # Other fingers
    for finger_tip, finger_pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_up.append(1 if landmarks[finger_tip].y < landmarks[finger_pip].y else 0)
    
    total_fingers = sum(fingers_up)
    
    if total_fingers == 0 or total_fingers == 1:
        return "rock"
    elif total_fingers >= 4:
        return "paper"
    elif total_fingers == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:
        return "scissors"
    
    return None

# Fungsi encode/decode frame
def encode_frame(frame):
    """Encode frame ke base64"""
    try:
        if CV2_AVAILABLE:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
            return base64.b64encode(buffer).decode('utf-8')
        else:
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=30)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except:
        return None

def decode_frame(frame_str):
    """Decode base64 ke frame"""
    try:
        img_data = base64.b64decode(frame_str)
        if CV2_AVAILABLE:
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = Image.open(io.BytesIO(img_data))
            return np.array(img)
    except:
        return None

# WebSocket client handler
class WebSocketClient:
    def __init__(self, url: str, client_id: str, message_queue: queue.Queue):
        self.url = f"{url}/{client_id}"
        self.client_id = client_id
        self.message_queue = message_queue
        self.websocket = None
        self.running = False
        
    async def connect(self):
        """Koneksi ke backend WebSocket"""
        try:
            self.websocket = await websockets.connect(self.url)
            st.session_state.game_state.ws_connected = True
            self.running = True
            
            await asyncio.gather(
                self.receive_messages(),
                self.process_outgoing_messages()
            )
        except Exception as e:
            st.error(f"Koneksi WebSocket gagal: {e}")
            st.session_state.game_state.ws_connected = False
    
    async def receive_messages(self):
        """Terima pesan dari server"""
        try:
            while self.running and self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                event = data.get("event")
                
                if event == "room_created":
                    st.session_state.game_state.room_id = data["room_id"]
                    st.session_state.game_state.game_status = "waiting"
                    st.rerun()
                    
                elif event == "joined_room":
                    st.session_state.game_state.room_id = data["room_id"]
                    st.session_state.game_state.game_status = "waiting"
                    st.rerun()
                    
                elif event == "game_ready":
                    st.session_state.game_state.game_status = "ready"
                    players = data.get("players", {})
                    for pid, pinfo in players.items():
                        if pid != self.client_id:
                            st.session_state.game_state.opponent_name = pinfo["name"]
                    st.rerun()
                    
                elif event == "game_started":
                    st.session_state.game_state.game_status = "playing"
                    st.session_state.game_state.gesture_locked = False
                    st.session_state.game_state.current_gesture = None
                    st.session_state.game_state.camera_active = True
                    st.rerun()
                    
                elif event == "video_frame":
                    frame_data = data.get("frame")
                    if frame_data:
                        frame = decode_frame(frame_data)
                        if frame is not None:
                            st.session_state.game_state.opponent_frame = frame
                    
                elif event == "round_result":
                    st.session_state.game_state.last_result = data
                    st.session_state.game_state.game_status = "result"
                    st.session_state.game_state.camera_active = False
                    if "scores" in data:
                        st.session_state.game_state.scores = data["scores"]
                    st.rerun()
                    
                elif event == "player_disconnected":
                    st.session_state.game_state.game_status = "waiting"
                    st.session_state.game_state.opponent_name = None
                    st.session_state.game_state.opponent_frame = None
                    st.rerun()
                    
        except websockets.exceptions.ConnectionClosed:
            st.session_state.game_state.ws_connected = False
            self.running = False
        except Exception as e:
            print(f"Error receiving message: {e}")
    
    async def process_outgoing_messages(self):
        """Kirim pesan ke server"""
        while self.running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    if self.websocket:
                        await self.websocket.send(json.dumps(message))
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error sending message: {e}")
    
    async def close(self):
        """Tutup koneksi"""
        self.running = False
        if self.websocket:
            await self.websocket.close()

# Camera handler
def camera_handler():
    """Handle camera capture dengan fallback"""
    if not CV2_AVAILABLE:
        # Mode simulasi
        while st.session_state.game_state.camera_active:
            if st.session_state.game_state.game_status == "playing" and not st.session_state.game_state.gesture_locked:
                time.sleep(3)
                gestures = ["rock", "paper", "scissors"]
                gesture = np.random.choice(gestures)
                st.session_state.game_state.current_gesture = gesture
                st.session_state.game_state.gesture_locked = True
                
                if st.session_state.game_state.room_id:
                    st.session_state.message_queue.put({
                        "event": "player_move",
                        "room_id": st.session_state.game_state.room_id,
                        "move": gesture
                    })
            time.sleep(0.1)
        return
    
    # Mode normal dengan OpenCV
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    hands = None
    if MEDIAPIPE_AVAILABLE:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    gesture_start_time = None
    last_gesture = None
    gesture_locked = False
    frame_count = 0
    
    while st.session_state.game_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if (st.session_state.game_state.game_status == "playing" and 
            not st.session_state.game_state.gesture_locked):
            
            if hands:
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
                        
                        gesture = classify_gesture(hand_landmarks)
                        
                        if gesture:
                            if gesture != last_gesture:
                                gesture_start_time = time.time()
                                last_gesture = gesture
                            
                            st.session_state.game_state.current_gesture = gesture
                            
                            elapsed = time.time() - gesture_start_time
                            remaining = max(0, 3 - int(elapsed))
                            
                            cv2.putText(frame, f"Gesture: {gesture.upper()}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f"Lock in: {remaining}", 
                                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
                            if elapsed >= 3 and not gesture_locked:
                                gesture_locked = True
                                st.session_state.game_state.gesture_locked = True
                                
                                if st.session_state.game_state.room_id:
                                    st.session_state.message_queue.put({
                                        "event": "player_move",
                                        "room_id": st.session_state.game_state.room_id,
                                        "move": gesture
                                    })
                                
                                cv2.putText(frame, "LOCKED!", 
                                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    gesture_start_time = None
                    last_gesture = None
                    cv2.putText(frame, "Tunjukkan tangan Anda", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        st.session_state.current_frame = frame
        
        # Kirim frame ke opponent
        if frame_count % 10 == 0 and st.session_state.game_state.room_id:
            encoded_frame = encode_frame(frame)
            if encoded_frame:
                st.session_state.message_queue.put({
                    "event": "video_frame",
                    "room_id": st.session_state.game_state.room_id,
                    "frame": encoded_frame
                })
        
        frame_count += 1
        
        if st.session_state.game_state.gesture_locked != gesture_locked:
            gesture_locked = st.session_state.game_state.gesture_locked
        
        time.sleep(0.03)
    
    cap.release()
    if hands:
        hands.close()

# WebSocket thread starter
def start_websocket_client(client_id: str):
    """Start WebSocket client in separate thread"""
    async def run_client():
        client = WebSocketClient(BACKEND_WS_URL, client_id, st.session_state.message_queue)
        await client.connect()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_client())

# UI Components
def show_lobby():
    """Tampilan lobby"""
    st.markdown('<h1 class="game-title">🎮 Gunting Batu Kertas Online</h1>', unsafe_allow_html=True)
    st.markdown("### Selamat datang! Mainkan game klasik dengan teknologi modern")
    
    # Info backend connection
    with st.expander("ℹ️ Status Koneksi"):
        st.info(f"Backend: {BACKEND_WS_URL}")
        if CV2_AVAILABLE:
            st.success("✅ OpenCV tersedia - Camera ready!")
        else:
            st.warning("⚠️ OpenCV tidak tersedia - Mode simulasi aktif")
        if MEDIAPIPE_AVAILABLE:
            st.success("✅ MediaPipe tersedia - Hand tracking ready!")
        else:
            st.warning("⚠️ MediaPipe tidak tersedia - Gesture akan disimulasikan")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 Buat Room Baru")
        with st.form("create_room_form"):
            player_name_create = st.text_input("Nama Anda", 
                                              value=f"Player_{np.random.randint(1000, 9999)}",
                                              max_chars=20)
            create_submitted = st.form_submit_button("Buat Room", type="primary")
            
            if create_submitted and player_name_create:
                player_id = f"player_{int(time.time() * 1000)}"
                st.session_state.game_state.player_id = player_id
                st.session_state.game_state.player_name = player_name_create
                
                if st.session_state.ws_thread is None or not st.session_state.ws_thread.is_alive():
                    st.session_state.ws_thread = threading.Thread(
                        target=start_websocket_client,
                        args=(player_id,)
                    )
                    st.session_state.ws_thread.start()
                    time.sleep(1)
                
                st.session_state.message_queue.put({
                    "event": "create_room",
                    "player_name": player_name_create
                })
                
                st.success("🚀 Membuat room...")
    
    with col2:
        st.markdown("### 🚪 Gabung Room")
        with st.form("join_room_form"):
            room_id_input = st.text_input("Room ID", 
                                         placeholder="Masukkan 8 karakter Room ID",
                                         max_chars=8)
            player_name_join = st.text_input("Nama Anda",
                                           value=f"Player_{np.random.randint(1000, 9999)}",
                                           max_chars=20)
            join_submitted = st.form_submit_button("Gabung Room", type="primary")
            
            if join_submitted and room_id_input and player_name_join:
                player_id = f"player_{int(time.time() * 1000)}"
                st.session_state.game_state.player_id = player_id
                st.session_state.game_state.player_name = player_name_join
                
                if st.session_state.ws_thread is None or not st.session_state.ws_thread.is_alive():
                    st.session_state.ws_thread = threading.Thread(
                        target=start_websocket_client,
                        args=(player_id,)
                    )
                    st.session_state.ws_thread.start()
                    time.sleep(1)
                
                st.session_state.message_queue.put({
                    "event": "join_room",
                    "room_id": room_id_input,
                    "player_name": player_name_join
                })
                
                st.success("🔗 Bergabung ke room...")

def show_game_room():
    """Tampilan game room"""
    st.markdown('<h1 class="game-title">🎮 Gunting Batu Kertas Online</h1>', unsafe_allow_html=True)
    
    # Room info
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("**📍 Room ID**")
        if st.session_state.game_state.room_id:
            st.markdown(f'<div class="room-code">{st.session_state.game_state.room_id}</div>', 
                       unsafe_allow_html=True)
    
    with col2:
        status_emoji = {
            "waiting": "⏳ Menunggu Pemain...",
            "ready": "🔔 Bersiap...",
            "playing": "🎮 Bermain!",
            "result": "🏆 Hasil"
        }
        status = st.session_state.game_state.game_status
        st.markdown(f"### {status_emoji.get(status, status)}")
    
    with col3:
        st.markdown("**👥 Lawan**")
        if st.session_state.game_state.opponent_name:
            st.info(f"🎮 {st.session_state.game_state.opponent_name}")
        else:
            st.warning("⏳ Menunggu...")
    
    st.markdown("---")
    
    # Start camera thread
    if (st.session_state.game_state.game_status in ["playing", "ready"] and
        (st.session_state.camera_thread is None or not st.session_state.camera_thread.is_alive())):
        st.session_state.game_state.camera_active = True
        st.session_state.camera_thread = threading.Thread(target=camera_handler)
        st.session_state.camera_thread.start()
    
    # Game content
    if st.session_state.game_state.game_status in ["playing", "ready"]:
        if CV2_AVAILABLE:
            # Mode dengan video
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📹 Kamera Anda")
                video_placeholder1 = st.empty()
                gesture_placeholder1 = st.empty()
                
                if hasattr(st.session_state, 'current_frame') and st.session_state.current_frame is not None:
                    frame_rgb = cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder1.image(frame_rgb, use_column_width=True)
                else:
                    video_placeholder1.info("📷 Mengaktifkan kamera...")
                
                if st.session_state.game_state.current_gesture:
                    gesture_emoji = {
                        "rock": "✊",
                        "paper": "✋", 
                        "scissors": "✌️"
                    }
                    gesture_placeholder1.success(
                        f"### {gesture_emoji.get(st.session_state.game_state.current_gesture, '')} "
                        f"{st.session_state.game_state.current_gesture.upper()}"
                    )
                    
                    if st.session_state.game_state.gesture_locked:
                        gesture_placeholder1.success("### ✅ GESTURE DIKUNCI!")
            
            with col2:
                st.markdown("### 📹 Kamera Lawan")
                video_placeholder2 = st.empty()
                
                if st.session_state.game_state.opponent_frame is not None:
                    frame_rgb = cv2.cvtColor(st.session_state.game_state.opponent_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder2.image(frame_rgb, use_column_width=True)
                else:
                    video_placeholder2.info("📷 Menunggu video lawan...")
            
            # Auto refresh untuk update video
            time.sleep(0.1)
            st.rerun()
            
        else:
            # Mode simulasi
            st.warning("🎮 Mode Simulasi - Kamera tidak aktif")
            
            if st.session_state.game_state.game_status == "playing":
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if not st.session_state.game_state.gesture_locked:
                        st.info("⏳ Gesture akan otomatis dipilih dalam 3 detik...")
                        
                        # Manual selection
                        gesture = st.radio("Atau pilih manual:", 
                                         ["rock", "paper", "scissors"],
                                         horizontal=True)
                        
                        if st.button("🎯 Kunci Pilihan", type="primary"):
                            st.session_state.game_state.current_gesture = gesture
                            st.session_state.game_state.gesture_locked = True
                            
                            if st.session_state.game_state.room_id:
                                st.session_state.message_queue.put({
                                    "event": "player_move",
                                    "room_id": st.session_state.game_state.room_id,
                                    "move": gesture
                                })
                            st.rerun()
                    else:
                        gesture_emoji = {
                            "rock": "✊",
                            "paper": "✋", 
                            "scissors": "✌️"
                        }
                        st.success(
                            f"### {gesture_emoji.get(st.session_state.game_state.current_gesture, '')} "
                            f"{st.session_state.game_state.current_gesture.upper()} - DIKUNCI!"
                        )
                        st.info("⏳ Menunggu lawan...")
        
        # Instructions
        if st.session_state.game_state.game_status == "playing":
            with st.expander("📖 Cara Bermain"):
                if CV2_AVAILABLE and MEDIAPIPE_AVAILABLE:
                    st.write("1. Tunjukkan tangan Anda ke kamera")
                    st.write("2. Buat gesture: ✊ Batu, ✋ Kertas, atau ✌️ Gunting")
                    st.write("3. Tahan gesture selama 3 detik untuk mengunci")
                    st.write("4. Tunggu hasil setelah kedua pemain selesai")
                else:
                    st.write("1. Pilih gesture Anda")
                    st.write("2. Klik 'Kunci Pilihan' atau tunggu auto-select")
                    st.write("3. Tunggu lawan membuat pilihan")
                    st.write("4. Lihat hasil!")
    
    elif st.session_state.game_state.game_status == "result":
        show_result()
    
    elif st.session_state.game_state.game_status == "waiting":
        st.info("⏳ Menunggu pemain lain bergabung...")
        st.markdown("### 📋 Cara Bermain:")
        st.write("1. Bagikan Room ID di atas ke teman Anda")
        st.write("2. Tunggu mereka bergabung")
        st.write("3. Game akan otomatis dimulai!")

def show_result():
    """Tampilan hasil permainan"""
    if st.session_state.game_state.last_result:
        result = st.session_state.game_state.last_result
        
        st.markdown("### 🏆 Hasil Permainan")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        moves = result.get("moves", {})
        player_id = st.session_state.game_state.player_id
        
        gesture_emoji = {
            "rock": "✊",
            "paper": "✋",
            "scissors": "✌️"
        }
        
        # Display moves
        player_moves = []
        for pid, move in moves.items():
            if pid == player_id:
                player_moves.insert(0, (pid, move))  # Player di kiri
            else:
                player_moves.append((pid, move))  # Opponent di kanan
        
        with col1:
            pid, move = player_moves[0]
            st.markdown("### Anda")
            st.markdown(f'<h1 style="text-align:center">{gesture_emoji.get(move, "?")}</h1>', 
                       unsafe_allow_html=True)
            st.markdown(f'<p style="text-align:center;font-size:24px">{move.upper()}</p>', 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown("### VS")
            
            # Display result
            winner_id = result.get("winner_id")
            if winner_id == player_id:
                st.markdown('<h2 style="text-align:center;color:green">🎉</h2>', 
                           unsafe_allow_html=True)
            elif winner_id is None:
                st.markdown('<h2 style="text-align:center;color:orange">🤝</h2>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<h2 style="text-align:center;color:red">😔</h2>', 
                           unsafe_allow_html=True)
        
        with col3:
            if len(player_moves) > 1:
                pid, move = player_moves[1]
                st.markdown(f"### {st.session_state.game_state.opponent_name or 'Lawan'}")
                st.markdown(f'<h1 style="text-align:center">{gesture_emoji.get(move, "?")}</h1>', 
                           unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center;font-size:24px">{move.upper()}</p>', 
                           unsafe_allow_html=True)
        
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
        if st.session_state.game_state.scores:
            score_cols = st.columns(2)
            for i, (pid, score) in enumerate(st.session_state.game_state.scores.items()):
                with score_cols[i]:
                    if pid == player_id:
                        st.metric("Skor Anda", score, delta=1 if winner_id == player_id else 0)
                    else:
                        opponent_won = winner_id and winner_id != player_id and winner_id is not None
                        st.metric(f"Skor {st.session_state.game_state.opponent_name or 'Lawan'}", 
                                score, delta=1 if opponent_won else 0)
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎮 Main Lagi", type="primary", use_container_width=True):
                st.session_state.game_state.gesture_locked = False
                st.session_state.game_state.current_gesture = None
                st.session_state.game_state.game_status = "waiting"
                
                st.session_state.message_queue.put({
                    "event": "play_again",
                    "room_id": st.session_state.game_state.room_id
                })
                
                st.rerun()
        
        with col2:
            if st.button("🚪 Keluar", type="secondary", use_container_width=True):
                st.session_state.game_state.camera_active = False
                st.session_state.game_state = GameState()
                st.rerun()

# Main app
def main():
    # Sidebar info
    with st.sidebar:
        st.markdown("### 🎮 GBK Online")
        
        if st.session_state.game_state.ws_connected:
            st.success("✅ Terhubung ke server")
        else:
            st.error("❌ Tidak terhubung")
        
        st.markdown("---")
        st.markdown("### 📊 Status")
        st.json({
            "status": st.session_state.game_state.game_status,
            "room": st.session_state.game_state.room_id,
            "player": st.session_state.game_state.player_name,
            "opencv": CV2_AVAILABLE,
            "mediapipe": MEDIAPIPE_AVAILABLE
        })
        
        st.markdown("---")
        st.markdown("### 🎯 Cara Bermain")
        st.write("✊ Batu mengalahkan Gunting")
        st.write("✋ Kertas mengalahkan Batu")
        st.write("✌️ Gunting mengalahkan Kertas")
    
    # Main content
    if st.session_state.game_state.game_status == "lobby":
        show_lobby()
    else:
        show_game_room()

if __name__ == "__main__":
    main()
