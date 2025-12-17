export type ChatRole = "user" | "assistant"

export interface ChatHistoryItem {
  role: ChatRole
  content: string
}

export interface ChatRequestPayload {
  message: string
  history: ChatHistoryItem[]
  language?: string
  location?: string | null
  uploaded_report_text?: string | null
}

export interface ChatResponsePayload {
  reply: string
  language: string
  route: string
  model_used?: string | null
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"

export async function sendChat(
  payload: ChatRequestPayload,
): Promise<ChatResponsePayload> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Chat request failed with status ${res.status}`)
  }

  return res.json()
}



