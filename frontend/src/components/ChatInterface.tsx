import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Sparkles } from "lucide-react";
import { ChatMessage } from "./ChatMessage";
import { StatusIndicator, type StatusStep } from "./StatusIndicator";
import { SuggestionChips } from "./SuggestionChips";
import { toast } from "sonner";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [statusSteps, setStatusSteps] = useState<StatusStep[]>([]);
  const [streamingContent, setStreamingContent] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, statusSteps, streamingContent]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setStreamingContent("");
    setStatusSteps([]);

    try {
      // Replace with your actual API endpoint
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMessage.content }),
      });

      if (!response.ok) throw new Error("Failed to fetch response");

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No reader available");

      let buffer = "";
      let assistantContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const data = JSON.parse(line);

            if (data.type === "status") {
              // Update status steps
              setStatusSteps((prev) => {
                const existingIndex = prev.findIndex(
                  (step) => step.id === data.id
                );
                if (existingIndex >= 0) {
                  const updated = [...prev];
                  updated[existingIndex] = {
                    id: data.id,
                    message: data.message,
                    status: data.status,
                  };
                  return updated;
                } else {
                  return [
                    ...prev,
                    {
                      id: data.id,
                      message: data.message,
                      status: data.status || "active",
                    },
                  ];
                }
              });
            } else if (data.type === "content") {
              // Append token to streaming content
              assistantContent += data.token;
              setStreamingContent(assistantContent);
            }
          } catch (e) {
            console.error("Error parsing JSON:", e);
          }
        }
      }

      // Add final assistant message
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: assistantContent || "I'm sorry, I couldn't process that request.",
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setStreamingContent("");
      setStatusSteps([]);
    } catch (error) {
      console.error("Error:", error);
      toast.error("Failed to get response. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
    textareaRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/30 backdrop-blur-md px-6 py-4">
        <div className="flex items-center gap-3 max-w-5xl mx-auto">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Financial Analyst AI
            </h1>
            <p className="text-sm text-muted-foreground">
              Multi-agent insights powered by AI
            </p>
          </div>
        </div>
      </header>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-6 py-8">
        <div className="max-w-5xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-20 animate-fade-in">
              <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                <Sparkles className="w-10 h-10 text-primary" />
              </div>
              <h2 className="text-3xl font-bold mb-3">
                Welcome to Financial Analyst AI
              </h2>
              <p className="text-muted-foreground max-w-md mx-auto">
                Get comprehensive financial insights powered by multiple AI
                agents working together in real-time.
              </p>
            </div>
          )}

          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              role={message.role}
              content={message.content}
            />
          ))}

          {statusSteps.length > 0 && <StatusIndicator steps={statusSteps} />}

          {streamingContent && (
            <ChatMessage
              role="assistant"
              content={streamingContent}
              isStreaming={true}
            />
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-border/50 bg-card/30 backdrop-blur-md px-6 py-6">
        <div className="max-w-5xl mx-auto">
          {messages.length === 0 && (
            <SuggestionChips
              onSuggestionClick={handleSuggestionClick}
              disabled={isLoading}
            />
          )}

          <form onSubmit={handleSubmit} className="relative">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about stocks, portfolios, market trends..."
              className="min-h-[60px] max-h-[200px] pr-14 resize-none bg-input/50 border-border focus:border-primary/50 focus:ring-2 focus:ring-primary/20 transition-all backdrop-blur-sm"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              disabled={isLoading}
            />
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim() || isLoading}
              className="absolute right-2 bottom-2 bg-gradient-to-br from-primary to-accent hover:opacity-90 transition-opacity"
            >
              <Send className="w-5 h-5" />
            </Button>
          </form>

          <p className="text-xs text-muted-foreground mt-3 text-center">
            Press Enter to send • Shift + Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
};
