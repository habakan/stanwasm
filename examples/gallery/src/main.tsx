import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// No <React.StrictMode> here: its dev-only mount→cleanup→remount simulation
// creates and immediately discards a throwaway generation of wasm objects
// on every mount, which made the MCMC Race tab's step-sampling wasm calls
// occasionally trap in `npm run dev` (never reproduced in a production
// build — see the comment on the animation effect in McmcRace.tsx). Fine to
// leave off for a demo app that isn't shipped as a library other code
// depends on.
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
