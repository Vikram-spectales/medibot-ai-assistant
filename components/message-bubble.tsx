"use client"

import { User, Stethoscope } from "lucide-react"
import { cn } from "@/lib/utils"
import type { Message } from "./chat-layout"

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user"

  const formatContent = (content: string) => {
    return content.split("\n").map((line, i) => {
      const boldFormatted = line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")

      return (
        <p
          key={i}
          className={cn("text-sm leading-relaxed", line === "" && "h-3")}
          dangerouslySetInnerHTML={{ __html: boldFormatted }}
        />
      )
    })
  }

  return (
    <div
      className={cn(
        "flex gap-4 transition-all duration-300 animate-in fade-in-0 slide-in-from-bottom-2",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      <div
        className={cn(
          "shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
          isUser ? "bg-muted" : "bg-primary",
        )}
      >
        {isUser ? (
          <User className="h-4 w-4 text-foreground" />
        ) : (
          <Stethoscope className="h-4 w-4 text-primary-foreground" />
        )}
      </div>

      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3",
          isUser ? "bg-muted text-foreground rounded-tr-sm" : "bg-secondary text-foreground rounded-tl-sm",
        )}
      >
        <div className="space-y-1">{formatContent(message.content)}</div>
        <p className={cn("text-xs mt-2 text-muted-foreground", isUser ? "text-right" : "text-left")}>
          {message.timestamp.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </div>
    </div>
  )
}
