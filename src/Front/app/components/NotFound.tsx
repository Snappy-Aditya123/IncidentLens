import { Link } from "react-router";
import { AlertTriangle } from "lucide-react";
import { Button } from "./ui/button";

export function NotFound() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <AlertTriangle className="w-16 h-16 mx-auto mb-4 text-yellow-500" />
        <h1 className="mb-2">404 - Page Not Found</h1>
        <p className="text-slate-400 mb-6">The page you're looking for doesn't exist.</p>
        <Link to="/">
          <Button>Go to Dashboard</Button>
        </Link>
      </div>
    </div>
  );
}
