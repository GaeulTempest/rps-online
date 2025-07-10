# Frontend Streamlit dengan WebSocket Video Streaming (Tanpa STUN/WebRTC)
# File: frontend/app.py
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
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

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Gunting Batu Kertas Online",
    page_icon="✂️",
    layout="wide"
)

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

# URL Backend WebSocket - sesuaikan dengan deployment Railway
BACKEND_WS_URL = st.text_input(
    "Backend WebSocket URL",
    value="ws://localhost:8000/ws",
    help="Masukkan URL WebSocket backend (contoh: wss://your-app.railway.app/ws)"
)

# Fungsi untuk klasifikasi gesture berdasarkan finger detection
def classify_gesture(hand_landmarks) -> Optional[str]:
    """
    Adaptasi dari kode main.py untuk klasifikasi gesture
    Menggunakan MediaPipe hand landmarks untuk deteksi rock, paper, scissors
    """
    if not hand_landmarks:
        return None
    
    # Ambil koordinat landmarks
    landmarks = hand_landmarks.landmark
    
    # Deteksi jari yang terangkat
    fingers_up = []
    
    # Thumb (ibu jari) - cek apakah tip lebih ke kiri/kanan dari IP joint
    if landmarks[4].x < landmarks[3].x:  # Untuk tangan kanan
        fingers_up.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:  # Untuk tangan kiri
        fingers_up.append(1 if landmarks[4].x > landmarks[3].x else 0)
    
    # 4 jari lainnya - cek apakah tip lebih tinggi dari PIP joint
    for finger_tip, finger_pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_up.append(1 if landmarks[finger_tip].y < landmarks[finger_pip].y else 0)
    
    # Klasifikasi gesture berdasarkan pattern jari
    total_fingers = sum(fingers_up)
    
    if total_fingers == 0:
        return "rock"  # Semua jari tertutup = Batu
    elif total_fingers == 5:
        return "paper"  # Semua jari terbuka = Kertas
    elif total_fingers == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:
        return "scissors"  # Jari telunjuk dan tengah = Gunting
    
    return None  # Gesture tidak dikenali

# Fungsi untuk encode frame ke base64
def encode_frame(frame):
    """Encode numpy array frame ke base64 string"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return base64.b64encode(buffer).decode('utf-8')

# Fungsi untuk decode frame dari base64
def decode_frame(frame_str):
    """Decode base64 string ke numpy array frame"""
    try:
        img_data = base64.b64decode(frame_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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
            
            # Start receiving messages
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
                
                # Process berdasarkan event type
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
                    # Set opponent name
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
                    # Terima video frame dari opponent
                    frame_data = data.get("frame")
                    if frame_data:
                        frame = decode_frame(frame_data)
                        if frame is not None:
                            st.session_state.game_state.opponent_frame = frame
                    
                elif event == "round_result":
                    st.session_state.game_state.last_result = data
                    st.session_state.game_state.game_status = "result"
                    st.session_state.game_state.camera_active = False
                    # Update scores
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
                # Check queue for messages
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

# Camera handler untuk capture dan process video
def camera_handler():
    """Handle camera capture dan hand detection"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
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
        
        # Flip untuk mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Proses hand detection jika game sedang playing
        if (st.session_state.game_state.game_status == "playing" and 
            not st.session_state.game_state.gesture_locked):
            
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # Klasifikasi gesture
                    gesture = classify_gesture(hand_landmarks)
                    
                    if gesture:
                        if gesture != last_gesture:
                            gesture_start_time = time.time()
                            last_gesture = gesture
                        
                        st.session_state.game_state.current_gesture = gesture
                        
                        # Hitung countdown
                        elapsed = time.time() - gesture_start_time
                        remaining = max(0, 3 - int(elapsed))
                        
                        # Display countdown
                        cv2.putText(frame, f"Gesture: {gesture.upper()}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Lock in: {remaining}", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                        # Lock gesture setelah 3 detik
                        if elapsed >= 3 and not gesture_locked:
                            gesture_locked = True
                            st.session_state.game_state.gesture_locked = True
                            
                            # Kirim move ke server
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
        
        # Store frame untuk display
        st.session_state.current_frame = frame
        
        # Kirim frame ke opponent setiap beberapa frame (untuk efisiensi)
        if frame_count % 5 == 0 and st.session_state.game_state.room_id:
            encoded_frame = encode_frame(frame)
            st.session_state.message_queue.put({
                "event": "video_frame",
                "room_id": st.session_state.game_state.room_id,
                "frame": encoded_frame
            })
        
        frame_count += 1
        
        # Reset gesture_locked untuk ronde baru
        if st.session_state.game_state.gesture_locked != gesture_locked:
            gesture_locked = st.session_state.game_state.gesture_locked
        
        time.sleep(0.03)  # ~30 FPS
    
    cap.release()
    hands.close()

# WebSocket thread starter
def start_websocket_client(client_id: str):
    """Start WebSocket client in separate thread"""
    async def run_client():
        client = WebSocketClient(BACKEND_WS_URL, client_id, st.session_state.message_queue)
        await client.connect()
    
    # Create new event loop for thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_client())

# UI Components
def show_lobby():
    """Tampilan lobby untuk create/join room"""
    st.title("🎮 Gunting Batu Kertas Online")
    st.markdown("### Selamat datang! Pilih opsi untuk memulai permainan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Buat Room Baru")
        player_name_create = st.text_input("Nama Anda", key="create_name", 
                                          value=f"Player_{np.random.randint(1000, 9999)}")
        
        if st.button("Buat Room", use_container_width=True, type="primary"):
            if player_name_create:
                # Generate player ID
                player_id = f"player_{int(time.time() * 1000)}"
                st.session_state.game_state.player_id = player_id
                st.session_state.game_state.player_name = player_name_create
                
                # Start WebSocket connection
                if st.session_state.ws_thread is None or not st.session_state.ws_thread.is_alive():
                    st.session_state.ws_thread = threading.Thread(
                        target=start_websocket_client,
                        args=(player_id,)
                    )
                    st.session_state.ws_thread.start()
                    time.sleep(1)  # Wait for connection
                
                # Send create room message
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
                # Generate player ID
                player_id = f"player_{int(time.time() * 1000)}"
                st.session_state.game_state.player_id = player_id
                st.session_state.game_state.player_name = player_name_join
                
                # Start WebSocket connection
                if st.session_state.ws_thread is None or not st.session_state.ws_thread.is_alive():
                    st.session_state.ws_thread = threading.Thread(
                        target=start_websocket_client,
                        args=(player_id,)
                    )
                    st.session_state.ws_thread.start()
                    time.sleep(1)  # Wait for connection
                
                # Send join room message
                st.session_state.message_queue.put({
                    "event": "join_room",
                    "room_id": room_id_input,
                    "player_name": player_name_join
                })
                
                st.success("Bergabung ke room...")

def show_game_room():
    """Tampilan game room dengan video feed"""
    st.title("🎮 Gunting Batu Kertas Online")
    
    # Display room info
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
    
    # Start camera thread jika belum aktif
    if (st.session_state.game_state.game_status in ["playing", "ready"] and
        (st.session_state.camera_thread is None or not st.session_state.camera_thread.is_alive())):
        st.session_state.game_state.camera_active = True
        st.session_state.camera_thread = threading.Thread(target=camera_handler)
        st.session_state.camera_thread.start()
    
    # Display video feeds
    if st.session_state.game_state.game_status in ["playing", "ready"]:
        st.markdown("### 📹 Video Feeds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Kamera Anda**")
            # Display current frame
            if hasattr(st.session_state, 'current_frame'):
                frame_rgb = cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, use_column_width=True)
            else:
                st.info("Menunggu kamera...")
            
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
        
        with col2:
            st.markdown("**Kamera Lawan**")
            # Display opponent frame
            if st.session_state.game_state.opponent_frame is not None:
                frame_rgb = cv2.cvtColor(st.session_state.game_state.opponent_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, use_column_width=True)
            else:
                st.info("Menunggu video lawan...")
        
        # Instructions
        if st.session_state.game_state.game_status == "playing":
            st.info("🎯 Tunjukkan gesture Anda ke kamera! Tahan selama 3 detik untuk mengunci pilihan.")
    
    # Show results
    elif st.session_state.game_state.game_status == "result":
        show_result()

def show_result():
    """Tampilan hasil permainan"""
    if st.session_state.game_state.last_result:
        result = st.session_state.game_state.last_result
        
        st.markdown("### 🏆 Hasil Ronde")
        
        col1, col2, col3 = st.columns(3)
        
        # Get moves and player info
        moves = result.get("moves", {})
        player_id = st.session_state.game_state.player_id
        
        gesture_emoji = {
            "rock": "✊",
            "paper": "✋",
            "scissors": "✌️"
        }
        
        # Display moves
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
            
            # Display result
            winner_id = result.get("winner_id")
            if winner_id == player_id:
                st.success("🎉 ANDA MENANG!")
            elif winner_id is None:
                st.info("🤝 SERI!")
            else:
                st.error("😔 ANDA KALAH!")
        
        # Display scores
        st.markdown("### 📊 Skor")
        if st.session_state.game_state.scores:
            score_cols = st.columns(2)
            for i, (pid, score) in enumerate(st.session_state.game_state.scores.items()):
                with score_cols[i]:
                    if pid == player_id:
                        st.metric("Skor Anda", score)
                    else:
                        st.metric(f"Skor {st.session_state.game_state.opponent_name or 'Lawan'}", score)
        
        # Play again button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Main Lagi", use_container_width=True, type="primary"):
                # Reset state untuk ronde baru
                st.session_state.game_state.gesture_locked = False
                st.session_state.game_state.current_gesture = None
                st.session_state.game_state.game_status = "waiting"
                
                # Send play again message
                st.session_state.message_queue.put({
                    "event": "play_again",
                    "room_id": st.session_state.game_state.room_id
                })
                
                st.rerun()
        
        with col2:
            if st.button("Keluar", use_container_width=True):
                # Stop camera
                st.session_state.game_state.camera_active = False
                # Reset all state
                st.session_state.game_state = GameState()
                st.rerun()

# Main app logic
def main():
    # Check WebSocket connection status
    if st.session_state.game_state.ws_connected:
        st.sidebar.success("✅ Terhubung ke server")
    else:
        st.sidebar.error("❌ Tidak terhubung ke server")
    
    # Route based on game status
    if st.session_state.game_state.game_status == "lobby":
        show_lobby()
    else:
        show_game_room()
    
    # Debug info in sidebar
    with st.sidebar:
        st.markdown("### Debug Info")
        st.json({
            "game_status": st.session_state.game_state.game_status,
            "room_id": st.session_state.game_state.room_id,
            "player_id": st.session_state.game_state.player_id,
            "ws_connected": st.session_state.game_state.ws_connected,
            "gesture_locked": st.session_state.game_state.gesture_locked,
            "current_gesture": st.session_state.game_state.current_gesture
        })

if __name__ == "__main__":
    main()