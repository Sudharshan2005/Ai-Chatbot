import { io, Socket } from "socket.io-client";

let socket: Socket | null = null;

export function initSocket() {
  if (!socket) {
    socket = io("http://localhost:5001", {
      transports: ["websocket"], // force pure websocket
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socket.on("connect", () => {
      console.log("✅ Connected to Flask server");
      socket?.send("Hello from Next.js client!");
    });

    socket.on("message", (msg) => {
      console.log("📩 Message from server:", msg);
    });

    socket.on("disconnect", () => {
      console.log("❌ Disconnected from server");
    });

    socket.on("connect_error", (err) => {
      console.error("⚠️ Connection error:", err.message);
    });
  }
  return socket;
}
