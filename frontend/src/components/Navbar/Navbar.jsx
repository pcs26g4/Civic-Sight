import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./Navbar.css";
import logo from "../../assets/logo.png";


export default function Navbar() {
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const user = JSON.parse(localStorage.getItem("user"));

  function handleLogout() {
    localStorage.clear();
    navigate("/");
  }

  return (
    <nav className="navbar">
      {/* LEFT: Logo + Brand */}
      <div className="nav-left">
        <div className="logo">
        <img src={logo}  />
        
        </div>
        <h3 ><b>CivicSight</b></h3>
      </div>

      {/* HAMBURGER (mobile only) */}
      <button className="hamburger" onClick={() => setMenuOpen(!menuOpen)}>
        ☰
      </button>

      {/* RIGHT: Links + Logout */}
      <div className={`nav-right ${menuOpen ? "open" : ""}`}>
        {/* USER SPECIFIC */}
        {user?.role === "USER" && (
          <Link to="/home" onClick={() => setMenuOpen(false)}>
             Home
            </Link>
          
        )}

        {/* INSPECTOR SPECIFIC */}
        {user?.role === "INSPECTOR" && (
          <Link to="/inspector" onClick={() => setMenuOpen(false)}>
            Dashboard
          </Link>
        )}

        {/* COMMON LINKS (ACCESSIBLE TO BOTH) */}
        {(user?.role === "USER" || user?.role === "INSPECTOR") && (
          <>
          <Link to="/dashboard" onClick={() => setMenuOpen(false)}>
            Dashboard
          </Link>
            
            <Link to="/map" onClick={() => setMenuOpen(false)}>
              Map
            </Link>
            <Link to="/complaints" onClick={() => setMenuOpen(false)}>
              Complaints
            </Link>
            <Link to="/tickets" onClick={() => setMenuOpen(false)}>
              Ticket Log
            </Link>
            <Link to="/live" onClick={() => setMenuOpen(false)}>
              Live Monitoring
            </Link>
          </>
        )}

        <button
          className="logout-btn"
          title="clcik here to logout"
          onClick={() => {
            setMenuOpen(false);
            handleLogout();
            
          }}
        >
          Logout
        </button>
      </div>
    </nav>
  );
}
