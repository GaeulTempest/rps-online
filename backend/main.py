# Backend FastAPI dengan WebSocket untuk Game GBK Online
# File: backend/main.py

import json
import uuid
import base64
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import logging
import os # Digunakan untuk os.getenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Field, Session, SQLModel, create_engine, select # Import select
from pydantic import BaseModel # Tetap gunakan BaseModel untuk data transfer (DTO)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Setup (Persistent Storage) ---
# DATABASE_URL diambil dari environment variable di Railway
# Default ke SQLite lokal untuk development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./gbk_game.db") 

# connect_args={"check_same_thread": False} hanya untuk SQLite
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Context manager untuk session database. Menggunakan async context manager.
@asyncio.contextmanager 
async def get_db_session():
    # Session SQLModel bersifat thread-local, jadi aman digunakan di sini
    # FastAPI handles concurrency via asyncio, so no explicit Threading.Lock needed here.
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction error: {str(e)}")
            raise # Re-raise the exception after rollback
        finally:
            session.close()

# Event startup untuk membuat tabel database
@app.on_event("startup")
def on_startup():
    logger.info("Creating database tables if not exists...")
    SQLModel.metadata.create_all(engine)
    logger.info("Database tables creation complete.")

# --- Models ---
# Player model (Pydantic BaseModel for data transfer/in-memory use)
class Player(BaseModel):
    id: str
    name: str
    score: int = 0
    move: Optional[str] = None
    is_ready: bool = False
    video_enabled: bool = False

# GameRoom model (SQLModel for database persistence)
class GameRoom(SQLModel, table=True):
    # Menggunakan 'id' sebagai room_id
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], primary_key=True)
    
    # Data pemain di-flatten ke GameRoom untuk kesederhanaan 2 pemain
    p1_id: Optional[str] = None
    p1_name: Optional[str] = None
    p1_score: int = 0
    p1_move: Optional[str] = None
    p1_ready: bool = False

    p2_id: Optional[str] = None
    p2_name: Optional[str] = None
    p2_score: int = 0
    p2_move: Optional[str] = None
    p2_ready: bool = False

    game_state: str = "waiting"  # waiting, playing, result, finished
    current_round: int = 1
    max_players: int = 2 # Default, not strictly enforced by logic here
    created_at: datetime = Field(default_factory=datetime.now)

# --- Inisialisasi FastAPI ---
app = FastAPI(
    title="GBK Online Backend",
    description="Backend untuk Game Gunting Batu Kertas Online dengan Hand Tracking",
    version="1.0.0"
)

# Konfigurasi CORS - PENTING untuk Streamlit Cloud
# Anda bisa membatasi allow_origins ke domain Streamlit Anda di production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain Streamlit Anda di production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket Management ---
# Cache in-memory untuk koneksi WebSocket (sementara)
# Tetap diperlukan karena WebSockets tidak bisa langsung dari DB
active_connections: Dict[str, WebSocket] = {}
# Lock untuk melindungi akses ke active_connections
# Karena ini mutable global state yang diakses concurrent
websocket_connections_lock = asyncio.Lock() 

# Fungsi utilitas untuk evaluasi permainan
def evaluate_game(move1: str, move2: str) -> str:
    if move1 == move2: return "draw"
    
    winning_combinations = {
        ("rock", "scissors"),
        ("paper", "rock"),
        ("scissors", "paper")
    }
    
    if (move1, move2) in winning_combinations: return "player1"
    else: return "player2"

# Fungsi untuk broadcast pesan ke semua pemain dalam room
async def broadcast_to_room(room_id: str, message: dict, exclude_player: Optional[str] = None):
    async with get_db_session() as session:
        room_data = session.get(GameRoom, room_id)
        if not room_data:
            logger.warning(f"Attempted to broadcast to non-existent room: {room_id}")
            return
        
    disconnected_players_ids = []
    
    # Kumpulkan ID pemain dari DB model GameRoom
    player_ids_in_room = []
    if room_data.p1_id: player_ids_in_room.append(room_data.p1_id)
    if room_data.p2_id: player_ids_in_room.append(room_data.p2_id)

    async with websocket_connections_lock: # Lindungi active_connections saat iterasi dan modifikasi
        for player_id in player_ids_in_room:
            if player_id != exclude_player and player_id in active_connections:
                try:
                    await active_connections[player_id].send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {player_id} in room {room_id}: {e}")
                    disconnected_players_ids.append(player_id)
        
        # Clean up disconnected players (di loop terpisah setelah iterasi)
        for player_id in disconnected_players_ids:
            if player_id in active_connections:
                del active_connections[player_id]
                # Panggil handler disconnect untuk update DB & room status
                await handle_player_disconnect(player_id, room_id)

# Fungsi untuk relay video frame antar pemain
async def relay_video_frame(room_id: str, sender_id: str, frame_data: str):
    async with get_db_session() as session:
        room_data = session.get(GameRoom, room_id)
        if not room_data:
            logger.warning(f"Attempted to relay frame to non-existent room: {room_id}")
            return
        
    player_ids_in_room = []
    if room_data.p1_id: player_ids_in_room.append(room_data.p1_id)
    if room_data.p2_id: player_ids_in_room.append(room_data.p2_id)

    async with websocket_connections_lock: # Lindungi active_connections
        for player_id in player_ids_in_room:
            if player_id != sender_id and player_id in active_connections:
                try:
                    await active_connections[player_id].send_json({
                        "event": "video_frame",
                        "sender_id": sender_id, # Frontend bisa pakai ini untuk tahu siapa pengirimnya
                        "frame": frame_data
                    })
                except Exception as e:
                    logger.error(f"Error relaying video to {player_id} in room {room_id}: {e}")
                    await handle_player_disconnect(player_id, room_id)

# Handler untuk disconnect pemain
async def handle_player_disconnect(client_id: str, room_id: Optional[str] = None):
    logger.info(f"Client {client_id} disconnected or failed to send message.")
    
    room_to_update: Optional[GameRoom] = None
    # Coba temukan room pemain ini
    async with get_db_session() as session:
        if room_id:
            room_to_update = session.get(GameRoom, room_id)
        else: # Cari room jika room_id tidak diberikan
            room_query = select(GameRoom).where(
                (GameRoom.p1_id == client_id) | (GameRoom.p2_id == client_id)
            )
            room_to_update = session.exec(room_query).first()

    if room_to_update:
        player_name = ""
        is_p1 = (room_to_update.p1_id == client_id)
        is_p2 = (room_to_update.p2_id == client_id)

        if is_p1:
            player_name = room_to_update.p1_name or "Player 1"
            room_to_update.p1_id = None
            room_to_update.p1_name = None
            room_to_update.p1_ready = False
            room_to_update.p1_move = None
            # Tidak perlu reset score di sini jika mau score tetap untuk player lain di game yang sama
        elif is_p2:
            player_name = room_to_update.p2_name or "Player 2"
            room_to_update.p2_id = None
            room_to_update.p2_name = None
            room_to_update.p2_ready = False
            room_to_update.p2_move = None
            # Tidak perlu reset score di sini

        # Reset game state for the room if a player disconnects
        room_to_update.game_state = "waiting" 
        room_to_update.p1_move = None
        room_to_update.p2_move = None
        # current_round tidak direset di sini, game bisa dilanjutkan dari ronde yang sama

        async with get_db_session() as session:
            session.add(room_to_update) # Simpan perubahan ke DB

        # Hapus room dari DB jika sudah kosong
        if not room_to_update.p1_id and not room_to_update.p2_id:
            async with get_db_session() as session:
                session.delete(room_to_update)
                session.commit()
            logger.info(f"Room {room_to_update.id} deleted (empty).")
        else:
            # Beritahu pemain lain yang tersisa (jika ada)
            await broadcast_to_room(room_to_update.id, {
                "event": "player_disconnected",
                "message": f"{player_name} telah meninggalkan permainan.",
                "room_id": room_to_update.id,
                "game_state": "waiting",
                "players": { # Kirim info pemain yang tersisa
                    (room_to_update.p1_id if room_to_update.p1_id else ""): {"name": room_to_update.p1_name, "score": room_to_update.p1_score, "id": room_to_update.p1_id}, 
                    (room_to_update.p2_id if room_to_update.p2_id else ""): {"name": room_to_update.p2_name, "score": room_to_update.p2_score, "id": room_to_update.p2_id}
                },
                "scores": { # Kirim skor yang tersisa
                    (room_to_update.p1_id if room_to_update.p1_id else "p1_dummy"): room_to_update.p1_score,
                    (room_to_update.p2_id if room_to_update.p2_id else "p2_dummy"): room_to_update.p2_score
                }
            })
            logger.info(f"Player {player_name} ({client_id}) left room {room_to_update.id}.")

# Root endpoint
@app.get("/")
async def root():
    async with get_db_session() as session:
        total_rooms_db = session.exec(select(GameRoom)).count()
    return {
        "status": "online",
        "message": "GBK Online Backend is running on Railway!",
        "total_rooms_in_db": total_rooms_db,
        "active_connections_in_memory": len(active_connections),
        "websocket_url_example": "wss://illustrious-achievement-production-b825.up.railway.app/ws/{client_id}", 
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# WebSocket endpoint utama
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    Endpoint WebSocket untuk komunikasi real-time
    Menangani game events dan video streaming
    """
    async with websocket_connections_lock: # Lindungi active_connections saat menambah
        active_connections[client_id] = websocket
    logger.info(f"Client {client_id} connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Terima data dari client
            data = await websocket.receive_json()
            event = data.get("event")
            room_id = data.get("room_id")
            player_name = data.get("player_name")
            move = data.get("move")
            
            logger.info(f"Received event '{event}' from client {client_id} in room {room_id if room_id else 'N/A'}")
            
            # --- Handle CREATE_ROOM event ---
            if event == "create_room":
                new_room_id = str(uuid.uuid4())[:8]
                # Buat room dengan data pemain 1
                new_room = GameRoom(id=new_room_id, p1_id=client_id, p1_name=player_name)
                
                async with get_db_session() as session:
                    session.add(new_room)
                    session.refresh(new_room) # Refresh untuk mendapatkan default values dari DB
                
                logger.info(f"Room {new_room_id} created by {client_id}")
                
                await websocket.send_json({
                    "event": "room_created",
                    "room_id": new_room_id,
                    "player_id": client_id, 
                    "role": "player1", 
                    "message": f"Room {new_room_id} berhasil dibuat. Menunggu pemain lain..."
                })
            
            # --- Handle JOIN_ROOM event ---
            elif event == "join_room":
                async with get_db_session() as session:
                    room = session.get(GameRoom, room_id)
                    if not room:
                        await websocket.send_json({"event": "error", "message": "Room tidak ditemukan!"})
                        continue
                
                # Cek apakah room sudah penuh atau pemain sudah ada di room ini
                if room.p1_id and room.p2_id:
                    if client_id == room.p1_id or client_id == room.p2_id:
                        await websocket.send_json({"event": "joined_room", "room_id": room_id, "player_id": client_id, "role": "rejoin", "message": "Anda sudah di room ini."})
                        continue
                    else:
                        await websocket.send_json({"event": "error", "message": "Room sudah penuh!"})
                        continue
                
                player_role = ""
                if not room.p1_id: # Jika p1_id kosong
                    room.p1_id = client_id
                    room.p1_name = player_name
                    player_role = "player1"
                elif not room.p2_id: # Jika p2_id kosong
                    room.p2_id = client_id
                    room.p2_name = player_name
                    player_role = "player2"
                else: # Seharusnya tidak sampai sini jika room sudah penuh dicek di atas
                    await websocket.send_json({"event": "error", "message": "Server error: Room logic inconsistent."})
                    continue

                room.game_state = "ready" # Set state menjadi ready (menunggu kedua pemain siap)
                
                async with get_db_session() as session:
                    session.add(room) 
                    session.refresh(room) # Refresh untuk mendapatkan update dari DB
                
                logger.info(f"Player {client_id} joined room {room_id}. Role: {player_role}")
                
                await websocket.send_json({
                    "event": "joined_room",
                    "room_id": room_id,
                    "player_id": client_id,
                    "role": player_role, 
                    "message": "Berhasil bergabung ke room!"
                })
                
                # Broadcast ke semua pemain bahwa game siap dimulai
                players_info_for_broadcast = {}
                if room.p1_id: players_info_for_broadcast[room.p1_id] = {"name": room.p1_name, "score": room.p1_score, "is_ready": room.p1_ready, "id": room.p1_id}
                if room.p2_id: players_info_for_broadcast[room.p2_id] = {"name": room.p2_name, "score": room.p2_score, "is_ready": room.p2_ready, "id": room.p2_id}

                scores_for_broadcast = {}
                if room.p1_id: scores_for_broadcast[room.p1_id] = room.p1_score
                if room.p2_id: scores_for_broadcast[room.p2_id] = room.p2_score

                await broadcast_to_room(room_id, {
                    "event": "game_ready",
                    "message": "Kedua pemain sudah siap! Permainan akan dimulai...",
                    "room_id": room_id,
                    "players": players_info_for_broadcast,
                    "scores": scores_for_broadcast
                })
                
                # Set game state ke playing setelah delay
                await asyncio.sleep(2) # Beri waktu frontend untuk render "Bersiap..."
                room.game_state = "playing"
                
                async with get_db_session() as session: # Simpan perubahan ke DB
                    session.add(room)
                    session.refresh(room)

                await broadcast_to_room(room_id, {
                    "event": "game_started",
                    "message": "Permainan dimulai! Tunjukkan gesture Anda!",
                    "room_id": room_id,
                    "game_state": "playing", 
                    "current_round": room.current_round
                })
            
            # --- Handle VIDEO_FRAME event (relay video antar pemain) ---
            elif event == "video_frame":
                frame_data = data.get("frame")
                if room_id and frame_data:
                    await relay_video_frame(room_id, client_id, frame_data)
            
            # --- Handle PLAYER_MOVE event ---
            elif event == "player_move":
                async with get_db_session() as session:
                    room = session.get(GameRoom, room_id)
                    if not room:
                        await websocket.send_json({"event": "error", "message": "Room tidak ditemukan!"})
                        continue
                
                if room.game_state != "playing":
                    await websocket.send_json({"event": "error", "message": "Game belum dimulai atau sudah selesai!"})
                    continue
                
                # Simpan move pemain
                if client_id == room.p1_id:
                    room.p1_move = move
                elif client_id == room.p2_id:
                    room.p2_move = move
                else:
                    await websocket.send_json({"event": "error", "message": "Anda bukan pemain di room ini!"})
                    continue

                async with get_db_session() as session: # Simpan perubahan ke DB
                    session.add(room)
                    session.refresh(room)
                
                logger.info(f"Player {client_id} submitted move: {move}")
                
                # Cek apakah kedua pemain sudah mengirim move
                if room.p1_move and room.p2_move:
                    player1_move = room.p1_move
                    player2_move = room.p2_move
                    
                    result = evaluate_game(player1_move, player2_move)
                    
                    winner_id = None
                    if result == "player1":
                        room.p1_score += 1
                        winner_id = room.p1_id
                    elif result == "player2":
                        room.p2_score += 1
                        winner_id = room.p2_id
                    else:
                        winner_id = None # Draw
                    
                    async with get_db_session() as session: # Simpan score update ke DB
                        session.add(room)
                        session.refresh(room)

                    logger.info(f"Round {room.current_round} result for room {room_id}: {result}")
                    
                    # Buat dictionary pemain yang rapi untuk broadcast
                    players_current_state = {
                        room.p1_id: {"name": room.p1_name, "move": room.p1_move, "score": room.p1_score},
                        room.p2_id: {"name": room.p2_name, "move": room.p2_move, "score": room.p2_score}
                    }

                    await broadcast_to_room(room_id, {
                        "event": "round_result",
                        "room_id": room_id,
                        "round": room.current_round,
                        "result": result,
                        "moves": {p_id: p_data["move"] for p_id, p_data in players_current_state.items()},
                        "scores": {p_id: p_data["score"] for p_id, p_data in players_current_state.items()},
                        "winner_id": winner_id,
                        "message": f"Hasil: {result.replace('player1', players_current_state[winner_id]['name']).replace('player2', players_current_state[winner_id]['name']) if result != 'draw' else 'Seri!'}" if winner_id else "Seri!"
                    })
                    
                    # Reset moves untuk ronde berikutnya
                    room.p1_move = None
                    room.p2_move = None
                    room.current_round += 1
                    async with get_db_session() as session: # Simpan perubahan ke DB
                        session.add(room)
                        session.refresh(room)
                else:
                    await websocket.send_json({
                        "event": "move_submitted",
                        "message": f"Gesture {move} berhasil dikirim! Menunggu lawan..."
                    })
            
            # --- Handle PLAY_AGAIN event ---
            elif event == "play_again":
                room_id = data.get("room_id")
                async with get_db_session() as session:
                    room = session.get(GameRoom, room_id)
                    if not room:
                        await websocket.send_json({"event": "error", "message": "Room tidak ditemukan!"})
                        continue

                    # Set pemain sebagai ready untuk ronde baru
                    if client_id == room.p1_id:
                        room.p1_ready = True
                    elif client_id == room.p2_id:
                        room.p2_ready = True
                    
                    session.add(room) # Simpan is_ready update
                    session.refresh(room)
                    
                    # Cek apakah semua pemain ready
                    if room.p1_id and room.p2_id and room.p1_ready and room.p2_ready:
                        # Reset state untuk ronde baru
                        room.game_state = "playing"
                        room.p1_ready = False
                        room.p2_ready = False
                        room.p1_move = None
                        room.p2_move = None
                        # room.current_round sudah di-increment setelah hasil
                        
                        session.add(room) # Simpan perubahan ke DB
                        session.refresh(room)
                        
                        await broadcast_to_room(room_id, {
                            "event": "game_reset", 
                            "message": "Ronde baru dimulai!",
                            "room_id": room_id,
                            "current_round": room.current_round,
                            "game_state": "playing"
                        })
                    else:
                        await websocket.send_json({
                            "event": "waiting_ready",
                            "message": "Menunggu pemain lain untuk melanjutkan..."
                        })
            
            # --- Handle GET_ROOM_INFO event ---
            elif event == "get_room_info":
                room_id = data.get("room_id")
                async with get_db_session() as session:
                    room = session.get(GameRoom, room_id)
                    if not room:
                        await websocket.send_json({"event": "error", "message": "Room tidak ditemukan!"})
                        continue
                
                players_info = {}
                if room.p1_id: players_info[room.p1_id] = {"name": room.p1_name, "score": room.p1_score, "is_ready": room.p1_ready, "id": room.p1_id}
                if room.p2_id: players_info[room.p2_id] = {"name": room.p2_name, "score": room.p2_score, "is_ready": room.p2_ready, "id": room.p2_id}

                await websocket.send_json({
                    "event": "room_info",
                    "room_id": room_id,
                    "game_state": room.game_state,
                    "players": players_info,
                    "current_round": room.current_round,
                    "scores": { # Kirim skor untuk inisialisasi frontend
                        (room.p1_id if room.p1_id else ""): room.p1_score,
                        (room.p2_id if room.p2_id else ""): room.p2_score
                    }
                })
            
            # --- Handle DISCONNECT_PLAYER event (dari frontend) ---
            elif event == "disconnect_player":
                room_id = data.get("room_id")
                logger.info(f"Client {client_id} requested disconnect from room {room_id}")
                await handle_player_disconnect(client_id, room_id)
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected via WebSocketDisconnect event.")
        await handle_player_disconnect(client_id)
            
    except Exception as e:
        logger.error(f"Error in WebSocket communication with client {client_id}: {e}", exc_info=True)
        await handle_player_disconnect(client_id)
        try:
            await websocket.send_json({"event": "error", "message": f"Terjadi kesalahan server: {e}"})
        except: pass
    finally:
        async with websocket_connections_lock:
            if client_id in active_connections:
                del active_connections[client_id]

# Endpoint untuk mendapatkan daftar room (untuk debugging)
@app.get("/rooms")
async def get_rooms():
    async with get_db_session() as session:
        rooms_in_db = session.exec(select(GameRoom)).all()
        
        return {
            "total_rooms": len(rooms_in_db),
            "rooms": [
                {
                    "id": room.id,
                    "p1_id": room.p1_id,
                    "p1_name": room.p1_name,
                    "p2_id": room.p2_id,
                    "p2_name": room.p2_name,
                    "state": room.game_state,
                    "current_round": room.current_round,
                    "created_at": room.created_at.isoformat(),
                    "p1_score": room.p1_score,
                    "p2_score": room.p2_score
                } for room in rooms_in_db
            ]
        }

# Endpoint untuk statistik
@app.get("/stats")
async def get_stats():
    async with get_db_session() as session:
        rooms_in_db = session.exec(select(GameRoom)).all()
        
        total_players_in_db = sum(1 for room in rooms_in_db for p_id in [room.p1_id, room.p2_id] if p_id is not None)
        active_games_in_db = sum(1 for room in rooms_in_db if room.game_state == "playing")
        
        return {
            "total_rooms_in_db": len(rooms_in_db),
            "total_players_in_db": total_players_in_db,
            "active_games_in_db": active_games_in_db,
            "active_connections_in_memory": len(active_connections),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Untuk development lokal
    # Pastikan database terbuat saat startup lokal
    logger.info("Running local development startup checks...")
    SQLModel.metadata.create_all(engine)
    logger.info("Local development server starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
