import { createBrowserRouter } from "react-router";
import { Root } from "./components/Root";
import { Dashboard } from "./components/Dashboard";
import { Investigation } from "./components/Investigation";
import { NotFound } from "./components/NotFound";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Root,
    children: [
      { index: true, Component: Dashboard },
      { path: "investigation/:incidentId", Component: Investigation },
      { path: "*", Component: NotFound },
    ],
  },
]);
