import { Toaster as Sonner, type ToasterProps } from "sonner";

/**
 * Themed Toaster — always uses "dark" to match IncidentLens's dark UI.
 * (The original shadcn template relied on next-themes; we hard-code instead.)
 */
const Toaster = ({ ...props }: ToasterProps) => {
  return (
    <Sonner
      theme="dark"
      className="toaster group"
      style={
        {
          "--normal-bg": "var(--popover)",
          "--normal-text": "var(--popover-foreground)",
          "--normal-border": "var(--border)",
        } as React.CSSProperties
      }
      {...props}
    />
  );
};

export { Toaster };
