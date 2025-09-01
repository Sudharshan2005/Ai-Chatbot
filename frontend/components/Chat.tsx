"use client";
import { useEffect, useRef, useState } from "react";
import { getSocket } from "@/lib/socket";

type Msg = {
  direction: "user" | "bot" | "worker" | "system";
  text: string;
  raw?: unknown;
};

type Doc = {
    response: string;
    status: string;
}

export default function Chat({ sessionId }: { sessionId: string }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const s = getSocket();

    const onConnect = () => {
        console.log("Conncted..");
      s.emit("join", { session_id: sessionId });
    };

    const onAck = (doc: Doc) => {
      setMessages(m => [...m, { direction: "system", text: doc.response, raw: doc }]);
    };


    const onBot = (doc: Doc) => {
      setMessages(m => [...m, { direction: "bot", text: doc.response, raw: doc }]);
    };

    const onProcessed = (doc: unknown) => {
      setMessages(m => [...m, { direction: "worker", text: "Worker processed", raw: doc }]);
    };

    s.on("connect", onConnect);
    s.on("message:ack", onAck);
    s.on("message:bot", onBot);
    s.on("message:processed", onProcessed);

    return () => {
      s.off("connect", onConnect);
      s.off("message:ack", onAck);
      s.off("message:bot", onBot);
      s.off("message:processed", onProcessed);
      s.emit("leave", { session_id: sessionId });
    };
  }, [sessionId]);

  const send = async () => {
    const text = inputRef.current?.value?.trim();
    if (!text) return;

    setMessages(m => [...m, { direction: "user", text }]);

    await fetch("http://localhost:5001/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: sessionId,
        user_id: "user_123",
        org_id: "acme",
        channel: "web",
      }),
    });

    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="w-full max-w-xl mx-auto p-4 space-y-3">
      <div className="border rounded p-3 h-96 overflow-auto bg-white">
        {messages.map((m, i) => (
          <div key={i} className={`mb-2 ${m.direction === "user" ? "text-right" : "text-left"} text-black`}>
            <span className="inline-block px-3 py-2 rounded bg-gray-100">
              <b>{m.direction.toUpperCase()}:</b> {m.text}
            </span>
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <input ref={inputRef} className="border rounded flex-1 px-3 py-2" placeholder="Type a message..." />
        <button onClick={send} className="px-4 py-2 rounded bg-black text-white">Send</button>
      </div>
    </div>
  );
}
