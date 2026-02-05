import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import "./App.css";

/* ===== PAGES ===== */
import Landing from "./components/pages/Landing";
import Dashboard from "./components/pages/Dashboard";
import MapView from "./components/pages/MapView";
import ComplaintPage from "./components/pages/ComplaintPage";
import AnalysePage from "./components/pages/AnalysePage";
import TicketLog from "./components/pages/TicketLog";
import LiveMonitoring from "./components/pages/LiveMonitoring";
import AdminPage from "./components/pages/AdminPage";
import InspectorPage from "./components/pages/InspectorPage";
import HomePage from "./components/pages/HomePage";

/* ===== LAYOUT ===== */
import AppLayout from "./components/Layout/AppLayout";

/* ===============================
   ROLE-BASED ROUTE GUARD
================================ */
const RequireRole = ({ role, children }) => {
  const user = JSON.parse(localStorage.getItem("user"));

  if (!user || user.role !== role) {
    return <Navigate to="/" replace />;
  }

  return children;
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* ===== LANDING & AUTH ===== */}
        <Route path="/" element={<Landing />} />

        {/* ===== USER ROUTES ===== */}
        <Route
          path="/home"
          element={
            <RequireRole role="USER">
              <AppLayout>
                <HomePage />
              </AppLayout>
            </RequireRole>
          }
        />
        <Route
          path="/dashboard"
          element={
            <RequireRole role="USER">
              <AppLayout>
                <Dashboard />
              </AppLayout>
            </RequireRole>
          }
        />

        <Route
          path="/map"
          element={
            <RequireRole role="USER">
              <AppLayout>
                <MapView />
              </AppLayout>
            </RequireRole>
          }
        />

        <Route
          path="/complaints"
          element={
            <RequireRole role="USER">
              <AppLayout>
                <ComplaintPage />
              </AppLayout>
            </RequireRole>
          }
        />

        <Route
          path="/analyse"
          element={
            <RequireRole role="USER">
              <AppLayout>
                <AnalysePage />
              </AppLayout>
            </RequireRole>
          }
        />

        <Route
          path="/tickets"
          element={
            <RequireRole role="USER">
              <AppLayout>
                <TicketLog />
              </AppLayout>
            </RequireRole>
          }
        />

        <Route
          path="/live"
          element={
            <RequireRole role="USER">
              <AppLayout>
                <LiveMonitoring />
              </AppLayout>
            </RequireRole>
          }
        />

        {/* ===== INSPECTOR ===== */}
        <Route
          path="/inspector"
          element={
            <RequireRole role="INSPECTOR">
              <InspectorPage />
            </RequireRole>
          }
        />

        {/* ===== ADMIN ===== */}
        <Route
          path="/admin"
          element={
            <RequireRole role="ADMIN">
              <AdminPage />
            </RequireRole>
          }
        />

        {/* ===== FALLBACK ===== */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
