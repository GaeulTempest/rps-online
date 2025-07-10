# Frontend Streamlit dengan Error Handling untuk OpenCV
# File: frontend/app.py
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
    layout="wide"
)

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

# URL Backend WebSocket
BACKEND_WS_URL = st.text_input(
    "Backend WebSocket URL",
    value="ws://localhost:8000/ws",
    help="Masukkan URL WebSocket backend (contoh: wss://your-app.railway.app/ws)"
)

# Fungsi simulasi gesture untuk fallback
def simulate_gesture():
    """Simulasi gesture jika OpenCV/MediaPipe tidak tersedia"""
    gestures = ["rock", "paper", "scissors"]
    return np.random.choice(gestures)

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
    
    if total_fingers == 0:
        return "rock"
    elif total_fingers == 5:
        return "paper"
    elif total_fingers == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:
        return "scissors"
    
    return None

# Fungsi encode/decode frame
def encode_frame(frame):
    """Encode frame ke base64"""
    if CV2_AVAILABLE:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        return base64.b64encode(buffer).decode('utf-8')
    else:
        # Fallback: encode PIL Image
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=50)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_frame(frame_str):
    """Decode base64 ke frame"""
    try:
        img_data = base64.b64decode(frame_str)
        if CV2_AVAILABLE:
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Fallback: use PIL
            img = Image.open(io.BytesIO(img_data))
            return np.array(img)
    except:
        return None

# WebSocket client handler (sama seperti sebelumnya)
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

# Camera handler dengan fallback
def camera_handler():
    """Handle camera capture dengan fallback jika OpenCV tidak tersedia"""
    if not CV2_AVAILABLE:
        # Mode simulasi tanpa OpenCV
        while st.session_state.game_state.camera_active:
            if st.session_state.game_state.game_status == "playing" and not st.session_state.game_state.gesture_locked:
                # Simulasi deteksi gesture
                time.sleep(3)  # Simulasi countdown
                gesture = simulate_gesture()
                st.session_state.game_state.current_gesture = gesture
                st.session_state.game_state.gesture_locked = True
                
                # Kirim move ke server
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
        
        if frame_count % 5 == 0 and st.session_state.game_state.room_id:
            encoded_frame = encode_frame(frame)
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
    st.title("🎮 Gunting Batu Kertas Online")
    st.markdown("### Selamat datang! Pilih opsi untuk memulai permainan")
    
    # Tampilkan status library
    col1, col2 = st.columns(2)
    with col1:
        if CV2_AVAILABLE:
            st.success("✅ OpenCV tersedia")
        else:
            st.warning("⚠️ OpenCV tidak tersedia - Mode simulasi")
    with col2:
        if MEDIAPIPE_AVAILABLE:
            st.success("✅ MediaPipe tersedia")
        else:
            st.warning("⚠️ MediaPipe tidak tersedia - Gesture simulasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Buat Room Baru")
        player_name_create = st.text_input("Nama Anda", key="create_name", 
                                          value=f"Player_{np.random.randint(1000, 9999)}")
        
        if st.button("Buat Room", use_container_width=True, type="primary"):
            if player_name_create:
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
                
                st.success("Membuat room...")
    
    with col2:
        st.subheader("🚪 Gabung Room")
        room_id_input = st.text_input("Room ID", key="join_room_id")
        player_name_join = st.text_input("Nama Anda", key="join_name",
                                        value=f"Player_{np.random.randint(1000, 9999)}")
        
        if st.button("Gabung Room", use_container_width=True, type="primary"):
            if room_id_input and player_name_join:
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
                
                st.success("Bergabung ke room...")

def show_game_room():
    """Tampilan game room"""
    st.title("🎮 Gunting Batu Kertas Online")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric("Room ID", st.session_state.game_state.room_id or "N/A")
        if st.session_state.game_state.room_id:
            st.caption("Bagikan ID ini ke teman Anda")
    
    with col2:
        status_emoji = {
            "waiting": "⏳",
            "ready": "🔔",
            "playing": "🎮",
            "result": "🏆"
        }
        status = st.session_state.game_state.game_status
        st.metric("Status", f"{status_emoji.get(status, '')} {status.title()}")
    
    with col3:
        if st.session_state.game_state.opponent_name:
            st.metric("Lawan", st.session_state.game_state.opponent_name)
        else:
            st.metric("Lawan", "Menunggu...")
    
    # Start camera thread
    if (st.session_state.game_state.game_status in ["playing", "ready"] and
        (st.session_state.camera_thread is None or not st.session_state.camera_thread.is_alive())):
        st.session_state.game_state.camera_active = True
        st.session_state.camera_thread = threading.Thread(target=camera_handler)
        st.session_state.camera_thread.start()
    
    # Display game content
    if st.session_state.game_state.game_status in ["playing", "ready"]:
        st.markdown("### 📹 Game View")
        
        if CV2_AVAILABLE:
            # Mode normal dengan video
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Kamera Anda**")
                if hasattr(st.session_state, 'current_frame'):
                    if CV2_AVAILABLE:
                        frame_rgb = cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = st.session_state.current_frame
                    st.image(frame_rgb, use_column_width=True)
                else:
                    st.info("Menunggu kamera...")
            
            with col2:
                st.markdown("**Kamera Lawan**")
                if st.session_state.game_state.opponent_frame is not None:
                    if CV2_AVAILABLE:
                        frame_rgb = cv2.cvtColor(st.session_state.game_state.opponent_frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = st.session_state.game_state.opponent_frame
                    st.image(frame_rgb, use_column_width=True)
                else:
                    st.info("Menunggu video lawan...")
        else:
            # Mode simulasi tanpa video
            st.warning("🎮 Mode Simulasi - Kamera tidak tersedia")
            
            if st.session_state.game_state.game_status == "playing":
                col1, col2, col3 = st.columns(3)
                
                with col2:
                    if not st.session_state.game_state.gesture_locked:
                        st.info("Gesture akan otomatis dipilih dalam 3 detik...")
                        if st.button("Pilih Gesture Manual", use_container_width=True):
                            gesture = st.selectbox("Pilih gesture:", ["rock", "paper", "scissors"])
                            st.session_state.game_state.current_gesture = gesture
                            st.session_state.game_state.gesture_locked = True
                            
                            if st.session_state.game_state.room_id:
                                st.session_state.message_queue.put({
                                    "event": "player_move",
                                    "room_id": st.session_state.game_state.room_id,
                                    "move": gesture
                                })
                    else:
                        st.success(f"✅ Gesture dikunci: {st.session_state.game_state.current_gesture}")
        
        # Display current gesture
        if st.session_state.game_state.current_gesture:
            gesture_emoji = {
                "rock": "✊",
                "paper": "✋", 
                "scissors": "✌️"
            }
            st.success(f"Gesture: {gesture_emoji.get(st.session_state.game_state.current_gesture, '')} {st.session_state.game_state.current_gesture.upper()}")
            
            if st.session_state.game_state.gesture_locked:
                st.success("✅ Gesture dikunci! Menunggu lawan...")
        
        # Instructions
        if st.session_state.game_state.game_status == "playing":
            if CV2_AVAILABLE and MEDIAPIPE_AVAILABLE:
                st.info("🎯 Tunjukkan gesture Anda ke kamera! Tahan selama 3 detik untuk mengunci pilihan.")
            else:
                st.info("🎯 Mode simulasi aktif - Gesture akan dipilih otomatis atau manual.")
    
    elif st.session_state.game_state.game_status == "result":
        show_result()

def show_result():
    """Tampilan hasil permainan"""
    if st.session_state.game_state.last_result:
        result = st.session_state.game_state.last_result
        
        st.markdown("### 🏆 Hasil Ronde")
        
        col1, col2, col3 = st.columns(3)
        
        moves = result.get("moves", {})
        player_id = st.session_state.game_state.player_id
        
        gesture_emoji = {
            "rock": "✊",
            "paper": "✋",
            "scissors": "✌️"
        }
        
        for i, (pid, move) in enumerate(moves.items()):
            with [col1, col3][i]:
                if pid == player_id:
                    st.markdown(f"**Anda**")
                else:
                    st.markdown(f"**{st.session_state.game_state.opponent_name or 'Lawan'}**")
                st.markdown(f"# {gesture_emoji.get(move, '?')}")
                st.caption(move.upper())
        
        with col2:
            st.markdown("**VS**")
            
            winner_id = result.get("winner_id")
            if winner_id == player_id:
                st.success("🎉 ANDA MENANG!")
            elif winner_id is None:
                st.info("🤝 SERI!")
            else:
                st.error("😔 ANDA KALAH!")
        
        st.markdown("### 📊 Skor")
        if st.session_state.game_state.scores:
            score_cols = st.columns(2)
            for i, (pid, score) in enumerate(st.session_state.game_state.scores.items()):
                with score_cols[i]:
                    if pid == player_id:
                        st.metric("Skor Anda", score)
                    else:
                        st.metric(f"Skor {st.session_state.game_state.opponent_name or 'Lawan'}", score)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Main Lagi", use_container_width=True, type="primary"):
                st.session_state.game_state.gesture_locked = False
                st.session_state.game_state.current_gesture = None
                st.session_state.game_state.game_status = "waiting"
                
                st.session_state.message_queue.put({
                    "event": "play_again",
                    "room_id": st.session_state.game_state.room_id
                })
                
                st.rerun()
        
        with col2:
            if st.button("Keluar", use_container_width=True):
                st.session_state.game_state.camera_active = False
                st.session_state.game_state = GameState()
                st.rerun()

# Main app
def main():
    if st.session_state.game_state.ws_connected:
        st.sidebar.success("✅ Terhubung ke server")
    else:
        st.sidebar.error("❌ Tidak terhubung ke server")
    
    if st.session_state.game_state.game_status == "lobby":
        show_lobby()
    else:
        show_game_room()
    
    with st.sidebar:
        st.markdown("### Debug Info")
        st.json({
            "game_status": st.session_state.game_state.game_status,
            "room_id": st.session_state.game_state.room_id,
            "player_id": st.session_state.game_state.player_id,
            "ws_connected": st.session_state.game_state.ws_connected,
            "cv2_available": CV2_AVAILABLE,
            "mediapipe_available": MEDIAPIPE_AVAILABLE
        })

if __name__ == "__main__":
    main()
