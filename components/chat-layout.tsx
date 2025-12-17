"use client"

import { useState } from "react"
import { Sidebar } from "./sidebar"
import { ChatArea } from "./chat-area"
import { Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { sendChat, type ChatHistoryItem } from "@/lib/api"

export type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

export type Conversation = {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
}

export function ChatLayout() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isSending, setIsSending] = useState(false)

  const activeConversation = conversations.find((c) => c.id === activeConversationId) || null

  const handleNewChat = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: "New conversation",
      messages: [],
      createdAt: new Date(),
    }
    setConversations([newConversation, ...conversations])
    setActiveConversationId(newConversation.id)
    setSidebarOpen(false)
  }

  const handleSelectConversation = (id: string) => {
    setActiveConversationId(id)
    setSidebarOpen(false)
  }

  const handleDeleteConversation = (id: string) => {
    setConversations(conversations.filter((c) => c.id !== id))
    if (activeConversationId === id) {
      const remaining = conversations.filter((c) => c.id !== id)
      setActiveConversationId(remaining.length > 0 ? remaining[0].id : null)
    }
  }

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return

    let convId = activeConversationId
    let conv = conversations.find((c) => c.id === convId) || null

    // If no active conversation, create one
    if (!conv) {
      convId = Date.now().toString()
      conv = {
        id: convId,
        title: "New conversation",
        messages: [],
        createdAt: new Date(),
      }
      setConversations((prev) => [conv!, ...prev])
      setActiveConversationId(convId)
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
    }

    // Optimistic update: add user message
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: [...c.messages, userMessage],
              title:
                c.messages.length === 0
                  ? content.slice(0, 30) + (content.length > 30 ? "..." : "")
                  : c.title,
            }
          : c,
      ),
    )

    setIsSending(true)
    try {
      const current = conversations.find((c) => c.id === convId)
      const historyMessages = (current?.messages || [])
        .concat(userMessage)
        .map<ChatHistoryItem>((m) => ({
          role: m.role,
          content: m.content,
        }))

      const res = await sendChat({
        message: content,
        history: historyMessages,
        language: "en",
      })

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: res.reply,
        timestamp: new Date(),
      }

      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: [...c.messages, assistantMessage],
              }
            : c,
        ),
      )
    } catch (error) {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "Sorry, I couldn't reach the Medibot server right now. Please check the backend and try again.",
        timestamp: new Date(),
      }

      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: [...c.messages, assistantMessage],
              }
            : c,
        ),
      )
      console.error(error)
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Mobile sidebar toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-3 left-3 z-50 md:hidden"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        <span className="sr-only">Toggle sidebar</span>
      </Button>

      {/* Sidebar overlay for mobile */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-30 md:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <div
        className={`fixed md:static inset-y-0 left-0 z-40 w-64 transform transition-transform duration-300 ease-in-out ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        <Sidebar
          conversations={conversations}
          activeConversationId={activeConversationId}
          onNewChat={handleNewChat}
          onSelectConversation={handleSelectConversation}
          onDeleteConversation={handleDeleteConversation}
        />
      </div>

      {/* Main chat area */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatArea
          conversation={activeConversation}
          onSendMessage={handleSendMessage}
          isSending={isSending}
        />
      </main>
    </div>
  )
}
