from fastapi import FastAPI, WebSocket
app = FastAPI()
@app.websocket("/ws/updates")
async def ws(ws: WebSocket):
    await ws.accept()
    await ws.send_text("hello")
    await ws.close()