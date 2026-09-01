import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// No <React.StrictMode>: its dev-only remount discards a throwaway generation of
// wasm objects, which made step-sampling calls occasionally trap in `npm run dev`.
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
