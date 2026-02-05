import Navbar from "../Navbar/Navbar";
import "./AppLayout.css";

export default function AppLayout({ children }) {
  return (
    <div className="app-layout">
      <Navbar />
      <div className="app-content">{children}</div>
    </div>
  );
}
