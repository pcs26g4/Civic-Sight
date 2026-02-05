import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Landing.css";
import { login, signup } from "../../services/api";
import logo from "../../assets/logo.png";

function Landing() {
    const [showModal, setShowModal] = useState(false);
    const [isLogin, setIsLogin] = useState(true);
    const [showErrorPopup, setShowErrorPopup] = useState(false);
    const [errorMessage, setErrorMessage] = useState("");
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const navigate = useNavigate();

    // Refs to clear form fields
    const loginFormRef = useRef(null);
    const signupFormRef = useRef(null);

    useEffect(() => {
        if (showModal) {
            clearFormFields();
        }
    }, [showModal, isLogin]);

    useEffect(() => {
        // Close mobile menu when resizing to desktop
        const handleResize = () => {
            if (window.innerWidth > 768) {
                setIsMobileMenuOpen(false);
            }
        };
        
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const clearFormFields = () => {
        if (loginFormRef.current) loginFormRef.current.reset();
        if (signupFormRef.current) signupFormRef.current.reset();
    };

 /* ================= LOGIN ================= */
const handleLogin = async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const email = formData.get("login_email_field");
    const password = formData.get("login_password_field");

    // Debug: Log environment variables
    console.log("Environment variables check:");
    console.log("VITE_ADMIN_EMAIL:", import.meta.env.VITE_ADMIN_EMAIL);
    console.log("VITE_ADMIN_PASSWORD:", import.meta.env.VITE_ADMIN_PASSWORD);
    console.log("VITE_ADMIN_NAME:", import.meta.env.VITE_ADMIN_NAME);
    console.log("User entered:", { email, password });

    // Admin login with environment variables
    const adminEmail = import.meta.env.VITE_ADMIN_EMAIL;
    const adminPassword = import.meta.env.VITE_ADMIN_PASSWORD;
    const adminName = import.meta.env.VITE_ADMIN_NAME;

    // Check if environment variables are defined
    if (adminEmail && adminPassword && adminName) {
        console.log("Comparing:", email, "===", adminEmail, "&&", password, "===", adminPassword);
        if (email === adminEmail && password === adminPassword) {
            console.log("Admin login successful");
            localStorage.setItem(
                "user", 
                JSON.stringify({ 
                    role: "ADMIN", 
                    name: adminName,
                    email: adminEmail
                })
            );
            navigate("/admin");
            return;
        }
    } else {
        console.warn("Admin credentials not configured in environment variables");
    }

    // Regular user login
    try {
        const response = await login(email, password);
        const user = response.user;

        localStorage.setItem("user", JSON.stringify(user));

        if (user.role === "INSPECTOR") navigate("/inspector");
        else navigate("/home");

        setShowModal(false);
        clearFormFields();
        setIsMobileMenuOpen(false);
    } catch (err) {
        setErrorMessage(err.message || "Invalid credentials");
        setShowErrorPopup(true);
    }
};

    /* ================= SIGNUP ================= */
    const handleSignup = async (e) => {
        e.preventDefault();

        const formData = new FormData(e.target);
        const name = formData.get("signup_name_field");
        const email = formData.get("signup_email_field");
        const password = formData.get("signup_password_field");

        try {
            // Call signup API with only required fields
            // Role will be handled by backend (default to CITIZEN)
            await signup(name, email, password);
            setShowModal(false);
            clearFormFields();
            setErrorMessage("Signup successful! Please login.");
            setShowErrorPopup(true);
            setIsLogin(true);
            setIsMobileMenuOpen(false);
        } catch (err) {
            setErrorMessage(err.message || "Signup failed");
            setShowErrorPopup(true);
        }
    };

    // Open modal and close mobile menu
    const openModal = (isLoginMode) => {
        setIsLogin(isLoginMode);
        setShowModal(true);
        setIsMobileMenuOpen(false);
        clearFormFields();
    };

    // Open admin modal
    const openAdminModal = () => {
        setIsLogin(true);
        setShowModal(true);
        setIsMobileMenuOpen(false);
        clearFormFields();
    };

    // Handle key press for accessibility
    const handleKeyPress = (e, action) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            action();
        }
    };

    return (
        <>
            {/* TOP NAVIGATION */}
            <nav className="top-nav">
                <div className="nav-left">
                    <div className="logo">
                        <img src={logo} alt="CivicSight Logo" />
                    </div>
                    <h3><b>CivicSight</b></h3>
                </div>

                {/* DESKTOP NAV BUTTONS */}
                <div className="nav-right desktop-nav">
                    <button
                        className="nav-btn admin-btn-nav"
                        title="only admin login is allowed"
                        onClick={openAdminModal}
                        aria-label="Admin Login"
                    >
                        Admin Login
                    </button>

                    <button
                        className="nav-btn login-btn-nav"
                        title="user/inspector login"
                        onClick={() => openModal(true)}
                        aria-label="Login"
                    >
                        Login
                    </button>

                    <button
                        className="nav-btn signup-btn-nav"
                        title="user signup"
                        onClick={() => openModal(false)}
                        aria-label="Signup"
                    >
                        Signup
                    </button>
                </div>

                {/* MOBILE HAMBURGER MENU */}
                <button 
                    className="mobile-menu-btn"
                    onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                    aria-label="Toggle menu"
                    aria-expanded={isMobileMenuOpen}
                >
                    ☰
                </button>

                {/* MOBILE DROPDOWN MENU */}
                <div className={`mobile-nav-dropdown ${isMobileMenuOpen ? 'show' : ''}`}>
                    <button
                        className="mobile-nav-btn admin-btn-nav"
                        onClick={openAdminModal}
                        aria-label="Admin Login"
                    >
                        Admin Login
                    </button>

                    <button
                        className="mobile-nav-btn login-btn-nav"
                        onClick={() => openModal(true)}
                        aria-label="Login"
                    >
                        Login
                    </button>

                    <button
                        className="mobile-nav-btn signup-btn-nav"
                        onClick={() => openModal(false)}
                        aria-label="Signup"
                    >
                        Signup
                    </button>
                </div>
            </nav>

            {/* MAIN PAGE */}
            <section className="hero-wrapper">
                <div className="hero-container">
                    <div className="hero-left">
                        <span className="org-label">[ Municipal Governance System ]</span>

                        <h1 className="hero-title">
                            Managing Municipal Deviations <br /> With Accuracy &
                            Accountability
                        </h1>

                        <p className="hero-subtext">
                            A unified digital platform for tracking, resolving, and monitoring
                            municipal deviations efficiently.
                        </p>

                        <button
                            className="cta-btn"
                            title="login to view dashboard"
                            onClick={() => openModal(true)}
                            aria-label="View Dashboard"
                        >
                            View Dashboard →
                        </button>

                        <div className="features-row">
                            <div className="feature-item">
                                <h4>Complaint Tracking</h4>
                                <p>Real-time logging and monitoring of deviations.</p>
                            </div>
                            <div className="feature-item">
                                <h4>Smart Assignment</h4>
                                <p>Auto-assign complaints by category & location.</p>
                            </div>
                            <div className="feature-item">
                                <h4>Resolution Insights</h4>
                                <p>Performance analytics with SLA tracking.</p>
                            </div>
                        </div>
                    </div>

                    <div className="hero-right">
                        <div className="video-wrapper">
                            <video
                                className="hero-video"
                                src="/landing-video2.mp4"
                                autoPlay
                                loop
                                muted
                                playsInline
                                aria-label="Demo video showing CivicSight platform in action"
                            >
                                <track kind="captions" src="#" label="English captions" />
                            </video>
                        </div>
                    </div>
                </div>
            </section>

            {/* LOGIN / SIGNUP MODAL */}
            <div 
                className={`modal-overlay ${showModal ? '' : 'hidden'}`}
                role="dialog"
                aria-modal="true"
                aria-label={isLogin ? "Login modal" : "Signup modal"}
            >
                <div className="modal-box">
                    <button 
                        className="modal-close" 
                        onClick={() => {
                            setShowModal(false);
                            setShowErrorPopup(false);
                            setErrorMessage("");
                        }}
                        aria-label="Close modal"
                    >
                        ×
                    </button>

                    {/* LOGIN FORM */}
                    {isLogin ? (
                        <>
                            <h2>Login</h2>

                            <form ref={loginFormRef} onSubmit={handleLogin}>
                                <input
                                    name="login_email_field"
                                    type="email"
                                    placeholder="Email"
                                    required
                                    className="input-field"
                                    autoComplete="email"
                                    aria-label="Email address"
                                />
                                <input
                                    name="login_password_field"
                                    type="password"
                                    placeholder="Password"
                                    required
                                    className="input-field"
                                    autoComplete="current-password"
                                    aria-label="Password"
                                />

                                <button 
                                    type="submit" 
                                    className="login-btn"
                                    aria-label="Submit login"
                                >
                                    Login
                                </button>
                            </form>

                            <p className="switch-text">
                                Don't have an account?
                                <span 
                                    onClick={() => setIsLogin(false)}
                                    onKeyPress={(e) => handleKeyPress(e, () => setIsLogin(false))}
                                    role="button"
                                    tabIndex={0}
                                    aria-label="Switch to signup form"
                                >
                                    Signup
                                </span>
                            </p>
                        </>
                    ) : (
                        <>
                            <h2>Signup</h2>

                            <form ref={signupFormRef} onSubmit={handleSignup}>
                                <input
                                    name="signup_name_field"
                                    type="text"
                                    required
                                    placeholder="Full Name"
                                    className="input-field"
                                    autoComplete="name"
                                    aria-label="Full name"
                                />
                                <input
                                    name="signup_email_field"
                                    type="email"
                                    required
                                    placeholder="Email"
                                    className="input-field"
                                    autoComplete="email"
                                    aria-label="Email address"
                                />
                                <input
                                    name="signup_password_field"
                                    type="password"
                                    required
                                    placeholder="Password"
                                    className="input-field"
                                    autoComplete="new-password"
                                    aria-label="Create password"
                                />

                                <button 
                                    type="submit" 
                                    className="login-btn"
                                    aria-label="Create account"
                                >
                                    Create Account
                                </button>
                            </form>

                            <p className="switch-text">
                                Already have an account?
                                <span 
                                    onClick={() => setIsLogin(true)}
                                    onKeyPress={(e) => handleKeyPress(e, () => setIsLogin(true))}
                                    role="button"
                                    tabIndex={0}
                                    aria-label="Switch to login form"
                                >
                                    Login
                                </span>
                            </p>
                        </>
                    )}
                </div>
            </div>

            {/* ERROR POPUP MODAL */}
            {showErrorPopup && (
                <div 
                    className="modal-overlay" 
                    onClick={() => setShowErrorPopup(false)}
                    role="dialog"
                    aria-modal="true"
                    aria-label="Error message"
                >
                    <div 
                        className="modal-box" 
                        style={{ maxWidth: "380px" }} 
                        onClick={(e) => e.stopPropagation()}
                    >
                        <h3 
                            style={{ 
                                marginBottom: "10px", 
                                color: errorMessage.includes("successful") ? "#10b981" : "#b00000" 
                            }}
                        >
                            {errorMessage.includes("successful") ? "Success" : "Login Failed"}
                        </h3>

                        <p style={{ marginBottom: "20px" }}>
                            {errorMessage}
                        </p>

                        <div style={{ textAlign: "right" }}>
                            <button
                                onClick={() => {
                                    setShowErrorPopup(false);
                                    setErrorMessage("");
                                    if (errorMessage.includes("successful")) {
                                        setShowModal(true);
                                    }
                                }}
                                style={{
                                    background: "#2563eb",
                                    color: "#fff",
                                    border: "none",
                                    padding: "10px 20px",
                                    borderRadius: "6px",
                                    cursor: "pointer",
                                    fontWeight: "600",
                                    fontSize: "15px"
                                }}
                                aria-label="Close error message"
                            >
                                OK
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}

export default Landing;