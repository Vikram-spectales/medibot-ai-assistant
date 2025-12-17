"use client"

import { useRef, useEffect } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { MessageBubble } from "./message-bubble"
import { ChatInput } from "./chat-input"
import { WelcomeScreen } from "./welcome-screen"
import type { Conversation } from "./chat-layout"
import { Stethoscope } from "lucide-react"

interface ChatAreaProps {
  conversation: Conversation | null
  onSendMessage: (content: string) => void | Promise<void>
  isSending?: boolean
}

export function ChatArea({ conversation, onSendMessage, isSending }: ChatAreaProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [conversation?.messages])

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      <header className="flex items-center gap-3 px-4 py-3 border-b border-border bg-card md:px-6">
        <div className="p-1.5 rounded-lg bg-primary/10 md:hidden">
          <Stethoscope className="h-4 w-4 text-primary" />
        </div>
        <div className="md:pl-0 pl-8">
          <h2 className="font-semibold text-foreground">{conversation?.title || "MediChat AI"}</h2>
          <p className="text-xs text-muted-foreground">AI-powered medical assistant</p>
        </div>
      </header>

      {/* Messages area */}
      <ScrollArea className="flex-1 px-4 py-6 md:px-6" ref={scrollRef}>
        {!conversation || conversation.messages.length === 0 ? (
          <WelcomeScreen onSuggestionClick={onSendMessage} />
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {conversation.messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
          </div>
        )}
      </ScrollArea>

      <div className="border-t border-border bg-card p-4 md:p-6">
        <div className="max-w-3xl mx-auto">
          <ChatInput onSend={onSendMessage} disabled={isSending} />
        </div>
      </div>
    </div>
  )
}
