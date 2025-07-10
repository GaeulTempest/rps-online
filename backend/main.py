# Backend FastAPI dengan WebSocket untuk Game GBK Online
# File: backend/main.py
# Deployed at: illustrious-achievement-production-b825.up.railway.app

import json
import uuid
import base64
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model data untuk game
class Player(BaseModel):
    id: str
    name: str
    score: int = 0
    move: Optional[str] = None
    is_ready: bool = False
    video_enabled: bool = False

class GameRoom(BaseModel):
    room_id: str
    players: Dict[str, Player] = {}
    game_state: str = "waiting"  # waiting, ready, playing, finished
    current_round: int = 1
    max_players: int = 2
    created_at: datetime = datetime.now()
    round_results: List[Dict] = []

# Inisialisasi FastAPI
app = FastAPI(
    title="GBK Online Backend",
    description="Backend untuk Game Gunting Batu Kertas Online dengan Hand Tracking",
    version="1.0.0"
)

# Konfigurasi CORS - PENTING untuk Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins untuk Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage untuk game rooms dan koneksi WebSocket
game_rooms: Dict[str, GameRoom] = {}
active_connections: Dict[str, WebSocket] = {}

# Fungsi utilitas untuk evaluasi permainan
def evaluate_game(move1: str, move2: str) -> str:
    """
    Evaluasi hasil permainan berdasarkan aturan GBK
    Returns: "player1", "player2", atau "draw"
    """
    if move1 == move2:
        return "draw"
    
    winning_combinations = {
        ("rock", "scissors"),
        ("paper", "rock"),
        ("scissors", "paper")
    }
    
    if (move1, move2) in winning_combinations:
        return "player1"
    else:
        return "player2"

# Fungsi untuk broadcast pesan ke semua pemain dalam room
async def broadcast_to_room(room_id: str, message: dict, exclude_player: Optional[str] = None):
    """Broadcast pesan ke semua pemain dalam room tertentu"""
    if room_id not in game_rooms:
        return
    
    room = game_rooms[room_id]
    disconnected_players = []
    
    for player_id in room.players:
        if player_id != exclude_player and player_id in active_connections:
            try:
                await active_connections[player_id].send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {player_id}: {e}")
                disconnected_players.append(player_id)
    
    # Clean up disconnected players
    for player_id in disconnected_players:
        if player_id in active_connections:
            del active_connections[player_id]

# Fungsi untuk relay video frame antar pemain
async def relay_video_frame(room_id: str, sender_id: str, frame_data: str):
    """Relay video frame dari satu pemain ke pemain lain dalam room"""
    if room_id not in game_rooms:
        return
    
    room = game_rooms[room_id]
    # Kirim frame hanya ke pemain lain (bukan pengirim)
    for player_id in room.players:
        if player_id != sender_id and player_id in active_connections:
            try:
                await active_connections[player_id].send_json({
                    "event": "video_frame",
                    "sender_id": sender_id,
                    "frame": frame_data
                })
            except:
                pass

# Root endpoint
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "GBK Online Backend is running on Railway!",
        "active_rooms": len(game_rooms),
        "active_connections": len(active_connections),
        "websocket_url": "wss://illustrious-achievement-production-b825.up.railway.app/ws/{client_id}",
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
    await websocket.accept()
    active_connections[client_id] = websocket
    logger.info(f"Client {client_id} connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Terima data dari client
            data = await websocket.receive_json()
            event = data.get("event")
            
            logger.info(f"Received event '{event}' from client {client_id}")
            
            # Handle CREATE_ROOM event
            if event == "create_room":
                # Generate UUID untuk room baru
                room_id = str(uuid.uuid4())[:8]  # 8 karakter untuk kemudahan
                
                # Buat room baru
                new_room = GameRoom(room_id=room_id)
                
                # Tambahkan pemain pertama
                player = Player(
                    id=client_id,
                    name=data.get("player_name", f"Player 1")
                )
                new_room.players[client_id] = player
                game_rooms[room_id] = new_room
                
                logger.info(f"Room {room_id} created by {client_id}")
                
                # Kirim response ke pembuat room
                await websocket.send_json({
                    "event": "room_created",
                    "room_id": room_id,
                    "player_number": 1,
                    "message": f"Room {room_id} berhasil dibuat. Menunggu pemain lain..."
                })
            
            # Handle JOIN_ROOM event
            elif event == "join_room":
                room_id = data.get("room_id")
                
                # Validasi room
                if room_id not in game_rooms:
                    await websocket.send_json({
                        "event": "error",
                        "message": "Room tidak ditemukan!"
                    })
                    continue
                
                room = game_rooms[room_id]
                
                # Cek apakah room sudah penuh
                if len(room.players) >= room.max_players:
                    await websocket.send_json({
                        "event": "error",
                        "message": "Room sudah penuh!"
                    })
                    continue
                
                # Tambahkan pemain kedua
                player = Player(
                    id=client_id,
                    name=data.get("player_name", f"Player {len(room.players) + 1}")
                )
                room.players[client_id] = player
                
                # Update state menjadi ready
                room.game_state = "ready"
                
                logger.info(f"Player {client_id} joined room {room_id}")
                
                # Kirim konfirmasi ke pemain yang baru join
                await websocket.send_json({
                    "event": "joined_room",
                    "room_id": room_id,
                    "player_number": len(room.players),
                    "message": "Berhasil bergabung ke room!"
                })
                
                # Broadcast ke semua pemain bahwa game siap dimulai
                await broadcast_to_room(room_id, {
                    "event": "game_ready",
                    "message": "Kedua pemain sudah siap! Permainan akan dimulai...",
                    "players": {
                        p_id: {
                            "name": p.name,
                            "score": p.score
                        } for p_id, p in room.players.items()
                    }
                })
                
                # Set game state ke playing setelah delay
                await asyncio.sleep(2)
                room.game_state = "playing"
                await broadcast_to_room(room_id, {
                    "event": "game_started",
                    "message": "Permainan dimulai! Tunjukkan gesture Anda!"
                })
            
            # Handle VIDEO_FRAME event (relay video antar pemain)
            elif event == "video_frame":
                room_id = data.get("room_id")
                frame_data = data.get("frame")
                
                if room_id and frame_data:
                    # Relay frame ke pemain lain
                    await relay_video_frame(room_id, client_id, frame_data)
            
            # Handle PLAYER_MOVE event
            elif event == "player_move":
                room_id = data.get("room_id")
                move = data.get("move")  # rock, paper, atau scissors
                
                if room_id not in game_rooms:
                    continue
                
                room = game_rooms[room_id]
                
                # Validasi game state
                if room.game_state != "playing":
                    await websocket.send_json({
                        "event": "error",
                        "message": "Game belum dimulai atau sudah selesai!"
                    })
                    continue
                
                # Simpan move pemain
                if client_id in room.players:
                    room.players[client_id].move = move
                    
                    logger.info(f"Player {client_id} submitted move: {move}")
                    
                    # Kirim konfirmasi ke pemain yang sudah submit move
                    await websocket.send_json({
                        "event": "move_submitted",
                        "message": f"Gesture {move} berhasil dikirim! Menunggu lawan..."
                    })
                    
                    # Cek apakah kedua pemain sudah mengirim move
                    all_moves = [p.move for p in room.players.values()]
                    if all(all_moves) and len(all_moves) == 2:
                        # Evaluasi hasil
                        player_ids = list(room.players.keys())
                        player1_move = room.players[player_ids[0]].move
                        player2_move = room.players[player_ids[1]].move
                        
                        result = evaluate_game(player1_move, player2_move)
                        
                        # Update score
                        if result == "player1":
                            room.players[player_ids[0]].score += 1
                            winner_id = player_ids[0]
                        elif result == "player2":
                            room.players[player_ids[1]].score += 1
                            winner_id = player_ids[1]
                        else:
                            winner_id = None
                        
                        # Simpan hasil ronde
                        round_result = {
                            "round": room.current_round,
                            "player1": {
                                "id": player_ids[0],
                                "move": player1_move,
                                "name": room.players[player_ids[0]].name
                            },
                            "player2": {
                                "id": player_ids[1],
                                "move": player2_move,
                                "name": room.players[player_ids[1]].name
                            },
                            "result": result,
                            "winner_id": winner_id
                        }
                        room.round_results.append(round_result)
                        
                        logger.info(f"Round {room.current_round} result: {result}")
                        
                        # Broadcast hasil ke kedua pemain
                        await broadcast_to_room(room_id, {
                            "event": "round_result",
                            "round": room.current_round,
                            "result": result,
                            "moves": {
                                player_ids[0]: player1_move,
                                player_ids[1]: player2_move
                            },
                            "scores": {
                                p_id: p.score for p_id, p in room.players.items()
                            },
                            "winner_id": winner_id,
                            "message": f"Hasil: {result.replace('player1', room.players[player_ids[0]].name).replace('player2', room.players[player_ids[1]].name) if result != 'draw' else 'Seri!'}"
                        })
                        
                        # Reset moves untuk ronde berikutnya
                        for player in room.players.values():
                            player.move = None
                        
                        room.current_round += 1
            
            # Handle PLAY_AGAIN event
            elif event == "play_again":
                room_id = data.get("room_id")
                
                if room_id in game_rooms:
                    room = game_rooms[room_id]
                    
                    # Set pemain sebagai ready
                    if client_id in room.players:
                        room.players[client_id].is_ready = True
                        
                        # Cek apakah semua pemain ready
                        all_ready = all(p.is_ready for p in room.players.values())
                        
                        if all_ready:
                            # Reset state untuk ronde baru
                            room.game_state = "playing"
                            for player in room.players.values():
                                player.is_ready = False
                                player.move = None
                            
                            await broadcast_to_room(room_id, {
                                "event": "new_round",
                                "message": "Ronde baru dimulai!",
                                "round": room.current_round
                            })
                        else:
                            await websocket.send_json({
                                "event": "waiting_ready",
                                "message": "Menunggu pemain lain untuk melanjutkan..."
                            })
            
            # Handle GET_ROOM_INFO event
            elif event == "get_room_info":
                room_id = data.get("room_id")
                
                if room_id in game_rooms:
                    room = game_rooms[room_id]
                    await websocket.send_json({
                        "event": "room_info",
                        "room_id": room_id,
                        "game_state": room.game_state,
                        "players": {
                            p_id: {
                                "name": p.name,
                                "score": p.score,
                                "is_ready": p.is_ready
                            } for p_id, p in room.players.items()
                        },
                        "current_round": room.current_round
                    })
            
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        
        # Handle disconnect
        if client_id in active_connections:
            del active_connections[client_id]
        
        # Hapus pemain dari room dan notify pemain lain
        for room_id, room in game_rooms.items():
            if client_id in room.players:
                player_name = room.players[client_id].name
                del room.players[client_id]
                
                # Jika room kosong, hapus room
                if len(room.players) == 0:
                    del game_rooms[room_id]
                    logger.info(f"Room {room_id} deleted (empty)")
                else:
                    # Notify pemain lain
                    room.game_state = "waiting"
                    await broadcast_to_room(room_id, {
                        "event": "player_disconnected",
                        "message": f"{player_name} telah meninggalkan permainan.",
                        "game_state": "waiting"
                    })
                break

# Endpoint untuk mendapatkan daftar room (untuk debugging)
@app.get("/rooms")
async def get_rooms():
    return {
        "total_rooms": len(game_rooms),
        "rooms": {
            room_id: {
                "players": len(room.players),
                "state": room.game_state,
                "current_round": room.current_round,
                "created_at": room.created_at.isoformat()
            } for room_id, room in game_rooms.items()
        }
    }

# Endpoint untuk statistik
@app.get("/stats")
async def get_stats():
    total_players = sum(len(room.players) for room in game_rooms.values())
    active_games = sum(1 for room in game_rooms.values() if room.game_state == "playing")
    
    return {
        "total_rooms": len(game_rooms),
        "total_players": total_players,
        "active_games": active_games,
        "active_connections": len(active_connections)
    }

if __name__ == "__main__":
    # Untuk development lokal
    uvicorn.run(app, host="0.0.0.0", port=8000)
