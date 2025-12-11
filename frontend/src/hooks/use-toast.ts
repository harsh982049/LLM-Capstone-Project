import { toast as sonnerToast } from "sonner";

// drop-in shim so existing calls like `const { toast } = useToast()` keep working
export function useToast() {
  return {
    toasts: [],                  // kept for compatibility; Sonner manages its own queue
    toast: sonnerToast,          // usage: toast("Saved"); toast.success("Done"); etc.
    dismiss: sonnerToast.dismiss // usage: dismiss(id) or dismiss() to clear all
  };
}

// direct named export for convenience parity with old API
export const toast = sonnerToast;
