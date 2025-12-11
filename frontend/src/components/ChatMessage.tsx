import { cn } from "@/lib/utils";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
}

export const ChatMessage = ({ role, content, isStreaming }: ChatMessageProps) => {
  const isUser = role === "user";

  return (
    <div
      className={cn(
        "flex w-full animate-slide-up",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-5 py-3.5 shadow-lg backdrop-blur-sm transition-all",
          isUser
            ? "bg-primary/90 text-primary-foreground ml-auto"
            : "bg-card/80 border border-border"
        )}
      >
        <div className="text-[15px] leading-relaxed whitespace-pre-wrap break-words">
          {content}
          {isStreaming && (
            <span className="inline-block w-2 h-4 ml-1 bg-primary animate-pulse-glow" />
          )}
        </div>
      </div>
    </div>
  );
};
