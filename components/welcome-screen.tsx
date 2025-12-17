"use client"

import { Stethoscope, Heart, Brain, Pill, Activity } from "lucide-react"
import { Button } from "@/components/ui/button"

interface WelcomeScreenProps {
  onSuggestionClick: (suggestion: string) => void
}

const suggestions = [
  {
    icon: Heart,
    text: "What are the symptoms of high blood pressure?",
  },
  {
    icon: Brain,
    text: "How can I improve my mental health?",
  },
  {
    icon: Pill,
    text: "What vitamins should I take daily?",
  },
  {
    icon: Activity,
    text: "How much exercise do I need weekly?",
  },
]

export function WelcomeScreen({ onSuggestionClick }: WelcomeScreenProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full max-w-2xl mx-auto px-4 py-8">
      <div className="flex flex-col items-center mb-10">
        <div className="p-4 rounded-2xl bg-primary/10 border border-primary/20 mb-6">
          <Stethoscope className="h-12 w-12 text-primary" />
        </div>
        <h1 className="text-3xl font-semibold text-foreground text-center text-balance">How can I help you today?</h1>
        <p className="text-muted-foreground text-center mt-3 max-w-md text-pretty">
          Ask me anything about health, symptoms, medications, or wellness tips.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
        {suggestions.map((suggestion, index) => (
          <Button
            key={index}
            variant="outline"
            className="h-auto p-4 justify-start gap-3 text-left bg-secondary/50 hover:bg-secondary border-border transition-all duration-200 hover:border-primary/50"
            onClick={() => onSuggestionClick(suggestion.text)}
          >
            <div className="p-2 rounded-lg bg-primary/10 shrink-0">
              <suggestion.icon className="h-4 w-4 text-primary" />
            </div>
            <span className="text-sm text-foreground leading-relaxed">{suggestion.text}</span>
          </Button>
        ))}
      </div>

      <div className="mt-10 p-4 rounded-xl bg-muted/50 border border-border">
        <p className="text-xs text-muted-foreground text-center leading-relaxed">
          <strong className="text-foreground">Disclaimer:</strong> MediChat AI provides general health information only.
          It is not a substitute for professional medical advice, diagnosis, or treatment.
        </p>
      </div>
    </div>
  )
}
