import { Outlet } from "react-router";

export function Root() {
  return (
    <div className="min-h-screen bg-slate-950">
      <Outlet />
    </div>
  );
}
