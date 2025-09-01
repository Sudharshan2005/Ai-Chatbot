"use client";
import { useEffect } from "react";
import { initSocket } from "@/utils/socket";

export default function Home() {
  useEffect(() => {
    const socket = initSocket();
    return () => {
      socket?.disconnect();
    };
  }, []);

  return (
    <div>
      <h1>WebSocket Test</h1>
      <p>Check console logs for connection status.</p>
    </div>
  );
}