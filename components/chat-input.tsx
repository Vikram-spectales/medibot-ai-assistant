"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Send, Paperclip, Mic } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface ChatInputProps {
  onSend: (content: string) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [message, setMessage] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = () => {
    if (disabled) return
    if (message.trim()) {
      onSend(message.trim())
      setMessage("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`
    }
  }, [message])

  return (
    <div className="relative flex items-end gap-2 p-3 rounded-2xl bg-muted border border-border transition-all duration-200 focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20">
      <Button variant="ghost" size="icon" className="shrink-0 h-8 w-8 text-muted-foreground hover:text-foreground">
        <Paperclip className="h-4 w-4" />
        <span className="sr-only">Attach file</span>
      </Button>

      <textarea
        ref={textareaRef}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Describe your symptoms or ask a health question..."
        className={cn(
          "flex-1 bg-transparent border-none outline-none resize-none",
          "text-sm text-foreground placeholder:text-muted-foreground",
          "min-h-[24px] max-h-[150px] py-1",
        )}
        rows={1}
        disabled={disabled}
      />

      <Button variant="ghost" size="icon" className="shrink-0 h-8 w-8 text-muted-foreground hover:text-foreground">
        <Mic className="h-4 w-4" />
        <span className="sr-only">Voice input</span>
      </Button>

      <Button
        onClick={handleSubmit}
        disabled={disabled || !message.trim()}
        size="icon"
        className={cn(
          "shrink-0 h-8 w-8 rounded-full transition-all duration-200",
          message.trim()
            ? "bg-primary hover:bg-primary/90 text-primary-foreground"
            : "bg-secondary text-muted-foreground",
        )}
      >
        <Send className="h-4 w-4" />
        <span className="sr-only">Send message</span>
      </Button>
    </div>
  )
}
