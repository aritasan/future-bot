#!/usr/bin/env python3
"""
Simple WebSocket server test
"""

import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def simple_websocket_handler(websocket, path):
    """Simple WebSocket handler for testing."""
    try:
        logger.info(f"Client connected: {websocket.remote_address}")
        
        # Send initial data
        initial_data = {
            "type": "initial",
            "performance_score": 75.5,
            "risk_score": 15.2,
            "stability_score": 85.0,
            "timestamp": "2025-07-31T15:10:00"
        }
        await websocket.send(json.dumps(initial_data))
        logger.info("Sent initial data")
        
        # Keep connection alive
        while True:
            await asyncio.sleep(5)
            
            # Send update data
            update_data = {
                "type": "update",
                "performance_score": 76.2,
                "risk_score": 14.8,
                "stability_score": 86.1,
                "timestamp": "2025-07-31T15:10:05"
            }
            await websocket.send(json.dumps(update_data))
            logger.info("Sent update data")
            
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")

async def start_simple_server():
    """Start a simple WebSocket server for testing."""
    try:
        server = await websockets.serve(simple_websocket_handler, "localhost", 8766)
        logger.info("Simple WebSocket server running on ws://localhost:8766")
        
        # Keep server running
        await server.wait_closed()
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(start_simple_server()) 