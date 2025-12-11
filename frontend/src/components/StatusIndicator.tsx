import { CheckCircle2, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export type StatusStep = {
  id: string;
  message: string;
  status: "pending" | "active" | "complete";
};

interface StatusIndicatorProps {
  steps: StatusStep[];
  className?: string;
}

export const StatusIndicator = ({ steps, className }: StatusIndicatorProps) => {
  if (steps.length === 0) return null;

  return (
    <div
      className={cn(
        "w-full max-w-[85%] rounded-2xl bg-card/60 border border-border/50 backdrop-blur-sm p-4 shadow-lg animate-slide-up",
        className
      )}
    >
      <div className="space-y-3">
        {steps.map((step) => (
          <div
            key={step.id}
            className={cn(
              "flex items-center gap-3 transition-all duration-300",
              step.status === "complete" && "opacity-60"
            )}
          >
            {step.status === "complete" && (
              <CheckCircle2 className="w-5 h-5 text-status-complete flex-shrink-0" />
            )}
            {step.status === "active" && (
              <Loader2 className="w-5 h-5 text-status-active animate-spin flex-shrink-0" />
            )}
            {step.status === "pending" && (
              <div className="w-5 h-5 rounded-full border-2 border-status-pending flex-shrink-0" />
            )}
            <span
              className={cn(
                "text-[14px] transition-colors",
                step.status === "active" && "text-foreground font-medium",
                step.status === "pending" && "text-muted-foreground",
                step.status === "complete" && "text-muted-foreground"
              )}
            >
              {step.message}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
