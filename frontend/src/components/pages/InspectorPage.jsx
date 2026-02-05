import { useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/InspectorPage.css";
import TicketLog from "./TicketLog";
import { getTickets, getImageUrl, resolveSubTicket } from "../../services/api";

// Popup Component
const Popup = ({ message, type = "info", onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 5000); // Increased to 5 seconds since we have OK button
    return () => clearTimeout(timer);
  }, [onClose]);

  const bgColor = type === "error" ? "#fef2f2" : 
                  type === "success" ? "#f0fdf4" : 
                  "#f0f9ff";
  const textColor = type === "error" ? "#dc2626" : 
                    type === "success" ? "#16a34a" : 
                    "#0284c7";
  const borderColor = type === "error" ? "#fecaca" : 
                      type === "success" ? "#bbf7d0" : 
                      "#bae6fd";

  return (
    <div className="popup-overlay">
      <div 
        className="popup-container"
        style={{
          backgroundColor: bgColor,
          border: `1px solid ${borderColor}`,
          borderLeft: `4px solid ${textColor}`
        }}
      >
        <div className="popup-content">
          <div className="popup-icon">
            {type === "error" ? "❌" : 
             type === "success" ? "✅" : "ℹ️"}
          </div>
          <div className="popup-message">{message}</div>
        </div>
        <div className="popup-actions">
          <button className="popup-ok-btn" onClick={onClose}>
            OK
          </button>
        </div>
      </div>
    </div>
  );
};

export default function InspectorPage() {
    const [view, setView] = useState("complaints");
    const [complaints, setComplaints] = useState([]);
    const [departmentComplaints, setDepartmentComplaints] = useState([]);
    const [inspectorDepartment, setInspectorDepartment] = useState("");
    const [inspectorName, setInspectorName] = useState("");
    const [statusFilter, setStatusFilter] = useState("all");
    const [dateFilter, setDateFilter] = useState("all");
    const [selectedDate, setSelectedDate] = useState("");
    const [popup, setPopup] = useState(null);
    const navigate = useNavigate();

    // Map issue types to departments
    const ISSUE_TO_DEPARTMENT = {
    "garbage": "Garbage",
    "garbageoverflow": "Garbage",
    "animalcarcas": "Garbage",
    "animalcarcass": "Garbage",
    "street_debris": "Garbage",
    "streetdebris": "Garbage",
    
    "pathholes": "Roads",
    "pothole": "Roads",
    "roadcracks": "Roads",
    "waterpuddles": "Roads",
    "openmanhole": "Roads",
    "sandonroad": "Roads",
    "streethawker": "Roads",
    "waterleak": "Roads",
    "watercontamination": "Roads",
    "drainage": "Roads",
    "poweroutage": "Roads",
    "streetlight": "Roads",
    "water_puddles": "Roads",
    };

    // DUMMY SLA TIMES
    const SLA_RULES = {
        pathholes: 12,
        garbage: 8,
        streetdebris: 10,
        street_debris: 10,
        animalcarcas: 12,
        animalcarcass: 12,
        waterleak: 24,
        watercontamination: 12,
        poweroutage: 6,
        streetlight: 24,
        drainage: 24,
    };

    // Show popup function
    const showPopup = (message, type = "info") => {
        setPopup({ message, type });
    };

    // Close popup function
    const closePopup = () => {
        setPopup(null);
    };

    // Get current logged-in inspector
    useEffect(() => {
        const user = JSON.parse(localStorage.getItem("user"));
        console.log("User from localStorage:", user); // Debug

        if (!user || user.role !== "INSPECTOR") {
            showPopup("Access denied. Please login as inspector.", "error");
            navigate("/");
            return;
        }

        if (user.department) {
            setInspectorDepartment(user.department);
        } else {
            console.warn("Inspector has no department assigned");
        }
        
        // Set inspector name
        if (user.name) {
            setInspectorName(user.name);
        } else if (user.username) {
            // Fallback to username if name not available
            setInspectorName(user.username);
        } else if (user.email) {
            // Fallback to email if neither name nor username available
            setInspectorName(user.email.split('@')[0]); // Use part before @
        } else {
            setInspectorName("Inspector");
        }
    }, [navigate]);

    useEffect(() => {
        let intervalId;

        const loadComplaints = async () => {
            try {
                const user = JSON.parse(localStorage.getItem("user"));
                if (!user || !inspectorDepartment) return;

                const response = await getTickets();
                const extracted = [];
                const departmentExtracted = [];

                (response.tickets || []).forEach(ticket => {
                    (ticket.sub_tickets || []).forEach(st => {
                        const issue = st.issue_type ? st.issue_type.toLowerCase() : "unknown";
                        const slaHours = SLA_RULES[issue] || 24; // Default 24h if unknown

                        // Determine department based on issue type
                        // Make sure to handle potential undefined issue_type
                        let complaintDepartment = ISSUE_TO_DEPARTMENT[issue] || "General";

                        // Optional: If you want to force specific logic for unmapped issues
                        if (issue.includes("garbage") || issue.includes("debris") || issue.includes("waste") || issue.includes("animal")) {
                            complaintDepartment = "Garbage";
                        } else {
                            complaintDepartment = "Roads";
                        }

                        const complaint = {
                            id: st.sub_id,
                            issue_type: issue,
                            department: complaintDepartment,
                            confidence: st.confidence,
                            image: st.image_id,
                            mediaType: st.media_type,
                            date: st.created_at,
                            status: st.status || "open",
                            latitude: st.latitude,
                            longitude: st.longitude,
                            area: ticket.area,
                            district: ticket.district,
                            slaHours,
                            ticketId: ticket.ticket_id,
                            timestamp: new Date(st.created_at).getTime(),
                            formattedDate: new Date(st.created_at).toISOString().split('T')[0], // YYYY-MM-DD format
                            assigned_to: st.assigned_to
                        };

                        extracted.push(complaint);

                        // Filter complaints by inspector's department AND assignment
                        // Normalize strings to avoid case mismatch issues
                        if (
                            complaintDepartment &&
                            inspectorDepartment &&
                            complaintDepartment.toLowerCase() === inspectorDepartment.toLowerCase() &&
                            st.assigned_to === user.id
                        ) {
                            departmentExtracted.push(complaint);
                        }
                    });
                });

                // Sort all complaints by date (newest first)
                const sortedExtracted = extracted.sort((a, b) => b.timestamp - a.timestamp);

                // Sort department complaints by date (newest first)
                const sortedDepartmentExtracted = departmentExtracted.sort((a, b) => b.timestamp - a.timestamp);

                setComplaints(sortedExtracted);
                setDepartmentComplaints(sortedDepartmentExtracted);
            } catch (err) {
                console.error("Error:", err);
                showPopup("Failed to load complaints. Please try again.", "error");
            }
        };

        if (inspectorDepartment) {
            loadComplaints();
            // Poll every 5 seconds for real-time updates
            intervalId = setInterval(loadComplaints, 5000);
        }

        return () => clearInterval(intervalId);
    }, [inspectorDepartment]);

    // Filter complaints based on selected status AND date
    const filteredComplaints = useMemo(() => {
        let filtered = departmentComplaints;

        // Apply status filter
        if (statusFilter !== "all") {
            filtered = filtered.filter(complaint => complaint.status === statusFilter);
        }

        // Apply date filter
        if (dateFilter === "specific" && selectedDate) {
            const selectedDateObj = new Date(selectedDate);
            filtered = filtered.filter(complaint => {
                const complaintDate = new Date(complaint.date);
                return complaintDate.toDateString() === selectedDateObj.toDateString();
            });
        } else if (dateFilter !== "all") {
            const today = new Date();
            const yesterday = new Date(today);
            yesterday.setDate(yesterday.getDate() - 1);
            const last7Days = new Date(today);
            last7Days.setDate(last7Days.getDate() - 7);
            const last30Days = new Date(today);
            last30Days.setDate(last30Days.getDate() - 30);

            filtered = filtered.filter(complaint => {
                const complaintDate = new Date(complaint.date);

                switch (dateFilter) {
                    case "today":
                        return complaintDate.toDateString() === today.toDateString();
                    case "yesterday":
                        return complaintDate.toDateString() === yesterday.toDateString();
                    case "last7days":
                        return complaintDate >= last7Days;
                    case "last30days":
                        return complaintDate >= last30Days;
                    default:
                        return true;
                }
            });
        }

        // Sort by date (newest first) after filtering
        return filtered.sort((a, b) => b.timestamp - a.timestamp);
    }, [departmentComplaints, statusFilter, dateFilter, selectedDate]);

    // Get complaint stats for current department
    const getDepartmentStats = () => {
        const stats = {
            all: departmentComplaints.length,
            open: 0,
            in_progress: 0,
            resolved: 0
        };

        departmentComplaints.forEach(complaint => {
            if (complaint.status === "open") stats.open++;
            else if (complaint.status === "in_progress") stats.in_progress++;
            else if (complaint.status === "resolved") stats.resolved++;
        });

        return stats;
    };

    const departmentStats = getDepartmentStats();

    // Get date range stats
    const getDateRangeStats = () => {
        const today = new Date();
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);
        const last7Days = new Date(today);
        last7Days.setDate(last7Days.getDate() - 7);
        const last30Days = new Date(today);
        last30Days.setDate(last30Days.getDate() - 30);

        return {
            today: departmentComplaints.filter(c => new Date(c.date).toDateString() === today.toDateString()).length,
            yesterday: departmentComplaints.filter(c => new Date(c.date).toDateString() === yesterday.toDateString()).length,
            last7days: departmentComplaints.filter(c => new Date(c.date) >= last7Days).length,
            last30days: departmentComplaints.filter(c => new Date(c.date) >= last30Days).length
        };
    };

    const dateRangeStats = getDateRangeStats();

    // Logout handler
    const handleLogout = () => {
        localStorage.clear();
        setTimeout(() => {
            navigate("/");
        }, 1500);
    };

    // Update status
    const updateStatus = async (id, newStatus) => {
        try {
            const user = JSON.parse(localStorage.getItem("user"));
            const inspectorName = user ? user.name : null;

            // Call API to update status in backend
            await resolveSubTicket(id, newStatus, null, "", inspectorName);

            // Update in department complaints
            const updatedDepartmentComplaints = departmentComplaints.map(c =>
                c.id === id ? { ...c, status: newStatus } : c
            );

            // Sort after update
            const sortedUpdated = updatedDepartmentComplaints.sort((a, b) => b.timestamp - a.timestamp);
            setDepartmentComplaints(sortedUpdated);

            // Also update in full complaints list if needed
            const updatedAllComplaints = complaints.map(c =>
                c.id === id ? { ...c, status: newStatus } : c
            ).sort((a, b) => b.timestamp - a.timestamp);

            setComplaints(updatedAllComplaints);

            // Save to localStorage for persistence (optional now since backend is updated)
            localStorage.setItem("updated_complaints", JSON.stringify(updatedAllComplaints));

            // Show success message
            showPopup(`Status updated to ${newStatus.replace('_', ' ')} successfully!`, "success");
        } catch (error) {
            console.error("Failed to update status:", error);
            showPopup("Failed to update status. Please try again.", "error");
        }
    };

    // Check SLA Breach
    const checkSLABreach = (date, slaHours) => {
        if (!date) return false;
        const created = new Date(date);
        const hoursPassed = (new Date() - created) / 36e5;
        return hoursPassed > slaHours;
    };

    // Handle status filter change
    const handleStatusFilterChange = (e) => {
        setStatusFilter(e.target.value);
    };

    // Handle date filter change
    const handleDateFilterChange = (e) => {
        const value = e.target.value;
        setDateFilter(value);
        if (value !== "specific") {
            setSelectedDate("");
        }
    };

    // Handle specific date change
    const handleSpecificDateChange = (e) => {
        setSelectedDate(e.target.value);
        if (dateFilter !== "specific") {
            setDateFilter("specific");
        }
    };

    // Clear all filters
    const clearAllFilters = () => {
        setStatusFilter("all");
        setDateFilter("all");
        setSelectedDate("");
    };

    // Get status display text
    const getStatusDisplayText = (status) => {
        const statusMap = {
            all: "All",
            open: "Open",
            in_progress: "In Progress",
            resolved: "Resolved"
        };
        return statusMap[status] || status;
    };

    // Get date filter display text
    const getDateFilterDisplayText = () => {
        if (dateFilter === "all") return "All Dates";
        if (dateFilter === "specific" && selectedDate) {
            return new Date(selectedDate).toLocaleDateString('en-GB', {
                day: '2-digit',
                month: '2-digit',
                year: 'numeric'
            });
        }

        const dateMap = {
            today: "Today",
            yesterday: "Yesterday",
            last7days: "Last 7 Days",
            last30days: "Last 30 Days",
            specific: "Specific Date"
        };
        return dateMap[dateFilter] || dateFilter;
    };

    // Format relative time (e.g., "2 hours ago")
    const getRelativeTime = (dateString) => {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffDays = Math.floor(diffHours / 24);

        if (diffDays > 0) {
            return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
        } else if (diffHours > 0) {
            return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        } else {
            const diffMinutes = Math.floor(diffMs / (1000 * 60));
            return diffMinutes < 1 ? "Just now" : `${diffMinutes} minute${diffMinutes > 1 ? 's' : ''} ago`;
        }
    };

    // Format date to dd-mm-yyyy
    const formatDateInput = (dateString) => {
        const date = new Date(dateString);
        const day = String(date.getDate()).padStart(2, '0');
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const year = date.getFullYear();
        return `${year}-${month}-${day}`;
    };

    return (
        <div className="inspector-container">
            {/* Popup Notification */}
            {popup && (
                <Popup 
                    message={popup.message} 
                    type={popup.type} 
                    onClose={closePopup}
                />
            )}

            {/* HEADER */}
            <div className="inspector-header">
                <div className="inspector-header-left">
                    <h2 className="inspector-title">Inspector Dashboard</h2>
                   
                    <div className="department-badge">
                        Department: {inspectorDepartment || "Not Assigned"}
                    </div>
                </div>
                <div className="inspector-header-right">
                    <div className="header-user">
                        Welcome, <span>{inspectorName}</span>
                    </div>
                    <button className="logout-btn" onClick={handleLogout}>
                        Logout
                    </button>
                </div>
            </div>

            {/* DEPARTMENT STATS */}
            {inspectorDepartment && (
                <div className="department-stats">
                    <div
                        className={`stat-card ${statusFilter === "all" && dateFilter === "all" ? "stat-active" : ""}`}
                        onClick={clearAllFilters}
                        style={{ cursor: "pointer" }}
                    >
                        <h3>{departmentStats.all}</h3>
                        <p>Total Complaints</p>
                    </div>
                    <div
                        className={`stat-card ${statusFilter === "open" ? "stat-active" : ""}`}
                        onClick={() => {
                            setStatusFilter("open");
                            setDateFilter("all");
                            setSelectedDate("");
                        }}
                        style={{ cursor: "pointer" }}
                    >
                        <h3 style={{ color: '#fbbf24' }}>{departmentStats.open}</h3>
                        <p>Open</p>
                    </div>
                    <div
                        className={`stat-card ${statusFilter === "in_progress" ? "stat-active" : ""}`}
                        onClick={() => {
                            setStatusFilter("in_progress");
                            setDateFilter("all");
                            setSelectedDate("");
                        }}
                        style={{ cursor: "pointer" }}
                    >
                        <h3 style={{ color: '#60a5fa' }}>{departmentStats.in_progress}</h3>
                        <p>In Progress</p>
                    </div>
                    <div
                        className={`stat-card ${statusFilter === "resolved" ? "stat-active" : ""}`}
                        onClick={() => {
                            setStatusFilter("resolved");
                            setDateFilter("all");
                            setSelectedDate("");
                        }}
                        style={{ cursor: "pointer" }}
                    >
                        <h3 style={{ color: '#34d399' }}>{departmentStats.resolved}</h3>
                        <p>Resolved</p>
                    </div>
                </div>
            )}

            <div className="inspector-tabs">
                <button
                    onClick={() => setView("complaints")}
                    className={view === "complaints" ? "active-tab" : ""}
                >
                    Department Complaints ({filteredComplaints.length})
                </button>
                <button
                    onClick={() => setView("tickets")}
                    className={view === "tickets" ? "active-tab" : ""}
                >
                    Ticket Logs
                </button>
            </div>

            {view === "complaints" && (
                <>
                    {departmentComplaints.length === 0 ? (
                        <div className="empty-state">
                            <h4>Inspector: {inspectorName}</h4>
                            <p className="empty-text">No complaints found for your department ({inspectorDepartment})</p>
                            <p className="empty-subtext">
                                Complaints will automatically appear here when they are assigned to {inspectorDepartment} department.
                            </p>
                        </div>
                    ) : (
                        <>
                            <div className="complaints-header">
                                <div>
                                    <h3>
                                        Complaints for {inspectorDepartment} Department
                                        <div className="inspector-subtitle">
                                        </div>
                                        {(statusFilter !== "all" || dateFilter !== "all") && (
                                            <span style={{ fontSize: "14px", color: "#94a3b8", marginLeft: "10px" }}>
                                                {statusFilter !== "all" && `Status: ${getStatusDisplayText(statusFilter)}`}
                                                {dateFilter !== "all" && statusFilter !== "all" && " | "}
                                                {dateFilter !== "all" && `Date: ${getDateFilterDisplayText()}`}
                                                {` (${filteredComplaints.length})`}
                                            </span>
                                        )}
                                    </h3>
                                </div>
                                <div className="filter-controls">
                                    <div className="filter-group">
                                        <label className="filter-label">Status</label>
                                        <select
                                            className="filter-dropdown"
                                            value={statusFilter}
                                            onChange={handleStatusFilterChange}
                                        >
                                            <option value="all">All ({departmentStats.all})</option>
                                            <option value="open">Open ({departmentStats.open})</option>
                                            <option value="in_progress">In Progress ({departmentStats.in_progress})</option>
                                            <option value="resolved">Resolved ({departmentStats.resolved})</option>
                                        </select>
                                    </div>

                                    <div className="filter-group">
                                        <label className="filter-label">Date</label>
                                        <select
                                            className="filter-dropdown"
                                            value={dateFilter}
                                            onChange={handleDateFilterChange}
                                        >
                                            <option value="all">All Dates</option>
                                            <option value="today">Today ({dateRangeStats.today})</option>
                                            <option value="yesterday">Yesterday ({dateRangeStats.yesterday})</option>
                                            <option value="last7days">Last 7 Days ({dateRangeStats.last7days})</option>
                                            <option value="last30days">Last 30 Days ({dateRangeStats.last30days})</option>
                                            <option value="specific">Specific Date</option>
                                        </select>
                                    </div>

                                    {dateFilter === "specific" && (
                                        <div className="filter-group">
                                            <label className="filter-label">Select Date</label>
                                            <div className="date-input-container">
                                                <div className="date-label">DATE</div>
                                                <input
                                                    type="date"
                                                    className="date-input"
                                                    value={selectedDate}
                                                    onChange={handleSpecificDateChange}
                                                    placeholder="dd-mm-yyyy"
                                                />
                                            </div>
                                        </div>
                                    )}

                                    {(statusFilter !== "all" || dateFilter !== "all") && (
                                        <button
                                            className="clear-filter-btn"
                                            onClick={clearAllFilters}
                                        >
                                            Clear Filters
                                        </button>
                                    )}
                                </div>
                            </div>

                            {filteredComplaints.length === 0 ? (
                                <div className="empty-state">
                                    <h4>Inspector: {inspectorName}</h4>
                                    <p className="empty-text">
                                        No complaints found for the selected filters
                                    </p>
                                    <p className="empty-subtext">
                                        {statusFilter !== "all" && dateFilter !== "all"
                                            ? `No ${getStatusDisplayText(statusFilter).toLowerCase()} complaints found for ${getDateFilterDisplayText().toLowerCase()}`
                                            : statusFilter !== "all"
                                                ? `No ${getStatusDisplayText(statusFilter).toLowerCase()} complaints found`
                                                : dateFilter !== "all"
                                                    ? `No complaints found for ${getDateFilterDisplayText().toLowerCase()}`
                                                    : "No complaints found"}
                                    </p>
                                    <button
                                        className="logout-btn"
                                        onClick={clearAllFilters}
                                        style={{ marginTop: "20px", background: "linear-gradient(135deg, #3b82f6, #1d4ed8)" }}
                                    >
                                        Clear All Filters
                                    </button>
                                </div>
                            ) : (
                                <div className="complaints-grid">
                                    {filteredComplaints.map((c, index) => {
                                        const breached = checkSLABreach(c.date, c.slaHours);
                                        const hoursPassed = c.date ? ((new Date() - new Date(c.date)) / 36e5).toFixed(1) : "N/A";
                                        const relativeTime = getRelativeTime(c.date);

                                        return (
                                            <div
                                                className="complaint-card-grid"
                                                key={c.id}
                                                style={{
                                                    border: breached ? "2px solid #ef4444" : "1px solid rgba(255,255,255,0.08)",
                                                    borderLeft: breached ? "4px solid #ef4444" : "4px solid rgba(59, 130, 246, 0.5)",
                                                    position: "relative"
                                                }}
                                            >
                                                {/* New indicator for recent complaints */}
                                                {index < 3 && statusFilter === "all" && dateFilter === "all" && (
                                                    <div className="new-indicator">
                                                        {index === 0 ? "🆕 NEWEST" : index === 1 ? "🆕 RECENT" : "🆕 RECENT"}
                                                    </div>
                                                )}

                                                <div className="complaint-image-wrapper">
                                                    {c.image ? (
                                                        c.mediaType === 'video' ? (
                                                            <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                                                                <video
                                                                    src={getImageUrl(c.image)}
                                                                    className="complaint-image-grid"
                                                                    style={{ objectFit: 'cover', width: '100%', height: '100%' }}
                                                                    controls
                                                                />
                                                                <div style={{
                                                                    position: 'absolute',
                                                                    top: '8px',
                                                                    right: '8px',
                                                                    background: 'rgba(0,0,0,0.6)',
                                                                    color: 'white',
                                                                    padding: '2px 6px',
                                                                    borderRadius: '4px',
                                                                    fontSize: '10px',
                                                                    fontWeight: 'bold'
                                                                }}>
                                                                    VIDEO
                                                                </div>
                                                            </div>
                                                        ) : (
                                                            <img
                                                                src={getImageUrl(c.image)}
                                                                className="complaint-image-grid"
                                                                alt="Complaint"
                                                                onError={(e) => {
                                                                    e.target.onerror = null;
                                                                    e.target.style.display = 'none';
                                                                    e.target.parentElement.innerHTML = '<div class="no-media-box">Image Not Available</div>';
                                                                }}
                                                            />
                                                        )
                                                    ) : (
                                                        <div className="no-media-box">No Media</div>
                                                    )}
                                                </div>

                                                <div className="complaint-info-grid">
                                                    <div className="complaint-header-row">
                                                        <div>
                                                            <p><strong>ID:</strong> {c.id}</p>
                                                            <p style={{ fontSize: "12px", color: "#94a3b8", marginTop: "-5px" }}>
                                                                Ticket: {c.ticketId}
                                                            </p>
                                                        </div>
                                                        <div style={{ textAlign: "right" }}>
                                                            <span className={`status-badge status-${c.status}`}>
                                                                {c.status.replace('_', ' ')}
                                                            </span>
                                                            <p style={{ fontSize: "11px", color: "#94a3b8", marginTop: "5px" }}>
                                                                {relativeTime}
                                                            </p>
                                                        </div>
                                                    </div>

                                                    <p><strong>Issue:</strong> {c.issue_type}</p>
                                                    <p><strong>Confidence:</strong> <span style={{ color: c.confidence > 0.8 ? "#34d399" : c.confidence > 0.6 ? "#fbbf24" : "#f87171" }}>
                                                        {(c.confidence * 100).toFixed(1)}%
                                                    </span></p>

                                                    <div className="status-control">
                                                        <label><strong>Update Status:</strong></label>
                                                        <select
                                                            className="status-dropdown"
                                                            value={c.status}
                                                            onChange={(e) => updateStatus(c.id, e.target.value)}
                                                        >
                                                            <option value="open">Open</option>
                                                            <option value="in_progress">In Progress</option>
                                                            <option value="resolved">Resolved</option>
                                                        </select>
                                                    </div>

                                                    <div style={{
                                                        display: "flex",
                                                        justifyContent: "space-between",
                                                        margin: "10px 0",
                                                        padding: "10px",
                                                        background: breached ? "rgba(220, 38, 38, 0.1)" : "rgba(34, 197, 94, 0.1)",
                                                        borderRadius: "8px",
                                                        border: `1px solid ${breached ? "rgba(220, 38, 38, 0.3)" : "rgba(34, 197, 94, 0.3)"}`
                                                    }}>
                                                        <div>
                                                            <p style={{ fontSize: "12px", margin: 0, color: "#94a3b8" }}>SLA Limit</p>
                                                            <p style={{ fontSize: "14px", margin: "4px 0 0 0", fontWeight: "600" }}>{c.slaHours} hrs</p>
                                                        </div>
                                                        <div>
                                                            <p style={{ fontSize: "12px", margin: 0, color: "#94a3b8" }}>Time Passed</p>
                                                            <p style={{ fontSize: "14px", margin: "4px 0 0 0", fontWeight: "600", color: breached ? "#f87171" : "#34d399" }}>
                                                                {hoursPassed} hrs
                                                            </p>
                                                        </div>
                                                        <div>
                                                            <p style={{ fontSize: "12px", margin: 0, color: "#94a3b8" }}>Status</p>
                                                            <span className={breached ? "breach-yes" : "breach-no"}>
                                                                {breached ? "Breached" : "OK"}
                                                            </span>
                                                        </div>
                                                    </div>

                                                    <p><strong>Area:</strong> {c.area}</p>
                                                    <p><strong>District:</strong> {c.district}</p>
                                                    <p><strong>Reported:</strong> {new Date(c.date).toLocaleDateString()} at {new Date(c.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>

                                                    {c.latitude && c.longitude && (
                                                        <p style={{ marginTop: "15px" }}>
                                                            <a
                                                                href={`https://maps.google.com/?q=${c.latitude},${c.longitude}`}
                                                                target="_blank"
                                                                rel="noreferrer"
                                                                className="map-link"
                                                            >
                                                                📍 View Location on Map
                                                            </a>
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </>
                    )}
                </>
            )}

            {view === "tickets" && (
                <>
                    <div className="complaints-header">
                        <h3>All Ticket Logs</h3>
                        <div className="inspector-subtitle">
                        </div>
                    </div>
                    {/* FIXED: Added showActions={false} prop */}
                    <TicketLog showActions={false} />
                </>
            )}
        </div>
    );
}