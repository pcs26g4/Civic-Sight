import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/AdminPage.css";
import TicketLog from "./TicketLog";
import Dashboard from "./Dashboard";
import { getTickets, getUsers, createInspector, deleteUser, getInspectorActions, resolveSubTicket } from "../../services/api"; // Removed getImageUrl import

export default function AdminPage() {
  const [users, setUsers] = useState([]);
  const [actions, setActions] = useState([]);
  const [complaints, setComplaints] = useState([]);
  const [view, setView] = useState("dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const user = JSON.parse(localStorage.getItem("user"));

  /* ================= NOTIFICATION TRACKING ================= */
  const [lastSeen, setLastSeen] = useState(() => {
    return JSON.parse(localStorage.getItem("admin_last_seen")) || {
      inspectors: 0,
      complaints: 0,
      actions: 0,
      accounts: 0
    };
  });

  const markSectionAsSeen = (section, items) => {
    if (!items || items.length === 0) return;

    const maxId = Math.max(...items.map(item =>
      section === "complaints" ? (item.numericId || 0) : (item.id || 0)
    ));

    if (maxId > (lastSeen[section] || 0)) {
      const updated = { ...lastSeen, [section]: maxId };
      setLastSeen(updated);
      localStorage.setItem("admin_last_seen", JSON.stringify(updated));
    }
  };

  // Add this with your other state declarations
  const [previewMedia, setPreviewMedia] = useState(null);
  
  /* ================= STATE ================= */
  const [filterDepartment, setFilterDepartment] = useState("all");
  const [accountFilter, setAccountFilter] = useState("all");
  
  // ACTION FILTERS
  const [actionDepartment, setActionDepartment] = useState("all");
  const [actionInspector, setActionInspector] = useState("all");

  // TICKET DETAILS MODAL
  const [showTicketDetails, setShowTicketDetails] = useState(false);
  const [selectedTicketId, setSelectedTicketId] = useState(null);
  const [ticketDetails, setTicketDetails] = useState(null);
  const [ticketSubTickets, setTicketSubTickets] = useState([]);

  const [showAddInspectorModal, setShowAddInspectorModal] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);

  const [createdInspector, setCreatedInspector] = useState(null);

  // Delete Popup State
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletedInspectorName, setDeletedInspectorName] = useState("");
  const [deleteModalType, setDeleteModalType] = useState("");

  // Confirm Delete State
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);
  const [selectedInspector, setSelectedInspector] = useState(null);
    
  // Error Popup State
  const [showErrorPopup, setShowErrorPopup] = useState(false);
  const [errorPopupMessage, setErrorPopupMessage] = useState("");

  // Inspector Complaints View State
  const [showInspectorComplaintsModal, setShowInspectorComplaintsModal] = useState(false);
  const [viewingInspector, setViewingInspector] = useState(null);
  const [inspectorComplaints, setInspectorComplaints] = useState([]);
  const [inspectorComplaintsTab, setInspectorComplaintsTab] = useState("open");

  // User Complaints View State
  const [showUserComplaintsModal, setShowUserComplaintsModal] = useState(false);
  const [viewingUser, setViewingUser] = useState(null);
  const [userComplaints, setUserComplaints] = useState([]);
  const [userComplaintsTab, setUserComplaintsTab] = useState("open");

  const [newInspector, setNewInspector] = useState({
    name: "",
    email: "",
    password: "",
    department: "",
    confirmPassword: ""
  });
  
  const navigate = useNavigate();

  /* ================= SLA RULES ================= */
  const SLA_RULES = {
    pathholes: 12,
    garbage: 8,
    streetdebris: 10,
    street_debris: 10,
    animalcarcas: 12,
    animalcarcass: 12
  };

  // Department options matching the signup form
  const DEPARTMENT_OPTIONS = [
    { value: "Roads", label: "Roads Department" },
    { value: "Garbage", label: "Garbage Department" }
  ];

  /* ================= LOAD USERS & ACTIONS ================= */
  const loadData = async () => {
    try {
      // Load Users
      const usersData = await getUsers();
      if (usersData.users) {
        setUsers(usersData.users);
      }

      // Load Actions
      const actionsData = await getInspectorActions();
      if (actionsData.actions) {
        setActions(actionsData.actions);
      }
    } catch (err) {
      console.error("Error loading admin data:", err);
      setErrorPopupMessage("Failed to load admin data: " + err.message);
      setShowErrorPopup(true);
    }
  };

  useEffect(() => {
    loadData();
    // Poll for updates every 10 seconds
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, []);

  /* ================= LOAD COMPLAINTS ================= */
  const loadComplaints = async () => {
    try {
      const response = await getTickets();
      const extracted = [];

      (response.tickets || []).forEach(ticket => {
        (ticket.sub_tickets || []).forEach(st => {
          const issue = st.issue_type;
          const slaHours = SLA_RULES[issue] || 12;

          extracted.push({
            id: st.sub_id,
            numericId: st.id,
            issue_type: issue,
            confidence: st.confidence,
            image: st.image_id,
            image_url: st.image_url, // ✅ Added: Include image_url from backend
            date: st.created_at,
            status: st.status || "open",
            latitude: st.latitude,
            longitude: st.longitude,
            area: ticket.area,
            district: ticket.district,
            mediaType: st.media_type,
            slaHours,
            ticketId: ticket.ticket_id,
            user_name: ticket.user_name,
            assigned_to: st.assigned_to
          });
        });
      });

      // Sort by date (newest first)
      extracted.sort((a, b) => new Date(b.date) - new Date(a.date));

      setComplaints(extracted);
      localStorage.setItem("admin_complaints", JSON.stringify(extracted));
    } catch (err) {
      console.error("Error loading complaints:", err);
      setErrorPopupMessage("Failed to load complaints: " + err.message);
      setShowErrorPopup(true);
    }
  };

  useEffect(() => {
    loadComplaints();
    // Optional: Poll for updates every 10 seconds to keep admin view fresh too
    const interval = setInterval(loadComplaints, 10000);
    return () => clearInterval(interval);
  }, []);

  // auto-mark as seen if already in the view
  useEffect(() => {
    if (view === "inspectors") markSectionAsSeen("inspectors", inspectors);
    if (view === "complaints") markSectionAsSeen("complaints", complaints);
    if (view === "actions") markSectionAsSeen("actions", actions);
    if (view === "accounts") markSectionAsSeen("accounts", allAccounts);
  }, [view, users, complaints, actions]);

  /* ================= TICKET DETAILS FUNCTION ================= */
  const viewTicketDetails = async (ticketId) => {
    try {
      setSelectedTicketId(ticketId);
      
      // Load all tickets to find the matching one
      const response = await getTickets();
      const tickets = response.tickets || [];
      
      // Find the ticket with matching ticket_id
      const ticket = tickets.find(t => t.ticket_id === ticketId);
      
      if (ticket) {
        setTicketDetails(ticket);
        
        // Extract sub-tickets with enhanced information
        const subTickets = (ticket.sub_tickets || []).map(st => ({
          id: st.sub_id,
          numericId: st.id,
          issue_type: st.issue_type,
          confidence: st.confidence,
          image: st.image_id,
          image_url: st.image_url, // ✅ Added: Include image_url
          date: st.created_at,
          status: st.status || "open",
          latitude: st.latitude,
          longitude: st.longitude,
          mediaType: st.media_type,
          slaHours: SLA_RULES[st.issue_type] || 12,
          assigned_to: st.assigned_to,
          proof_image: st.proof_image,
          inspector_comment: st.inspector_comment,
          resolved_at: st.resolved_at,
          resolved_by: st.resolved_by
        }));
        
        setTicketSubTickets(subTickets);
      } else {
        setTicketDetails(null);
        setTicketSubTickets([]);
      }
      
      setShowTicketDetails(true);
    } catch (err) {
      console.error("Error loading ticket details:", err);
      setErrorPopupMessage("Failed to load ticket details: " + err.message);
      setShowErrorPopup(true);
    }
  };

  /* ================= VIEW USER COMPLAINTS FUNCTION ================= */
  const viewUserComplaints = (user) => {
    setViewingUser(user);
    // Find complaints raised by this user (by user_name)
    const userRaisedComplaints = complaints.filter(c => 
      c.user_name && c.user_name.toLowerCase() === user.name.toLowerCase()
    );
    setUserComplaints(userRaisedComplaints);
    setUserComplaintsTab("open");
    setShowUserComplaintsModal(true);
  };

  /* ================= VIEW INSPECTOR COMPLAINTS FUNCTION ================= */
  const viewInspectorComplaints = (inspector) => {
    setViewingInspector(inspector);
    // Find complaints assigned to this inspector ID
    const assigned = complaints.filter(c => c.assigned_to === inspector.id);
    setInspectorComplaints(assigned);
    setInspectorComplaintsTab("open");
    setShowInspectorComplaintsModal(true);
  };

  /* ================= HELPERS ================= */
  const handleLogout = () => {
    localStorage.removeItem("user");
    localStorage.removeItem("admin_complaints");
    navigate("/");
  };

  const checkSLABreach = (date, slaHours) => {
    if (!date) return false;
    const created = new Date(date);
    const hoursPassed = (new Date() - created) / 36e5;
    return hoursPassed > slaHours;
  };

  /* ================= UPDATE STATUS (ADMIN) ================= */
  const updateComplaintStatus = async (id, newStatus) => {
    try {
      // Optimistic update for UI responsiveness
      setComplaints(complaints.map(c =>
        c.id === id ? { ...c, status: newStatus } : c
      ));

      // Call API to update status in backend
      await resolveSubTicket(id, newStatus, null, "Status updated by Admin", "Admin");

      // Refresh data to ensure synchronization
      await loadComplaints();
      await loadData();

    } catch (err) {
      console.error("Failed to update status:", err);
      setErrorPopupMessage("Failed to update complaint status: " + err.message);
      setShowErrorPopup(true);
      // Revert optimistic update on failure (re-fetch would also fix it)
      loadComplaints();
      loadData();
    }
  };

  /* ================= USER MANAGEMENT ================= */
  const refreshUsers = async () => {
    try {
      const data = await getUsers();
      if (data.users) setUsers(data.users);
    } catch (err) {
      console.error("Error refreshing users:", err);
      setErrorPopupMessage("Failed to refresh users: " + err.message);
      setShowErrorPopup(true);
    }
  };

  // Updated removeInspector function with confirmation modal
  const removeInspector = (id) => {
    const inspector = users.find(u => u.id === id);
    if (inspector) {
      setSelectedInspector(inspector);
      setShowConfirmDelete(true);
    }
  };

  // Delete account function for accounts tab
  const deleteAccount = async (id, name) => {
    const account = users.find(u => u.id === id);
    if (!account) return;

    if (window.confirm(`Delete account: ${name}?\nThis action cannot be undone.`)) {
      try {
        await deleteUser(id);
        setDeletedInspectorName(name);
        setDeleteModalType("inspector");
        setShowDeleteModal(true);
        refreshUsers();
      } catch (err) {
        setErrorPopupMessage("Failed to delete account: " + err.message);
        setShowErrorPopup(true);
      }
    }
  };

  /* ================= ADD INSPECTOR FUNCTION ================= */
  const handleAddInspector = async () => {
    // Validate inputs
    if (!newInspector.name.trim() || !newInspector.email.trim() ||
      !newInspector.password || !newInspector.department) {
      setErrorPopupMessage("Please fill in all fields");
      setShowErrorPopup(true);
      return;
    }

    if (newInspector.password !== newInspector.confirmPassword) {
      setErrorPopupMessage("Passwords do not match");
      setShowErrorPopup(true);
      return;
    }

    if (newInspector.password.length < 6) {
      setErrorPopupMessage("Password must be at least 6 characters long");
      setShowErrorPopup(true);
      return;
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(newInspector.email)) {
      setErrorPopupMessage("Please enter a valid email address");
      setShowErrorPopup(true);
      return;
    }

    try {
      await createInspector(
        newInspector.name.trim(),
        newInspector.email.trim(),
        newInspector.password,
        newInspector.department
      );

      // Refresh users list
      await refreshUsers();

      // Reset form and close modal
      setNewInspector({
        name: "",
        email: "",
        password: "",
        department: "",
        confirmPassword: ""
      });
      setShowAddInspectorModal(false);

      setCreatedInspector({
        name: newInspector.name,
        email: newInspector.email,
        password: newInspector.password,
        department: newInspector.department
      });
      setShowSuccessModal(true);

    } catch (err) {
      setErrorPopupMessage(err.message || "Email already registered");
      setShowErrorPopup(true);
    }
  };

  /* ================= FILTERS & SEARCH ================= */
  const inspectors = users.filter(u => u.role === "INSPECTOR");
  const normalUsers = users.filter(u => u.role === "USER");

  // All accounts for the accounts tab
  const allAccounts = [...normalUsers, ...inspectors];

  // Get department label from value
  const getDepartmentLabel = (value) => {
    const dept = DEPARTMENT_OPTIONS.find(opt => opt.value === value);
    return dept ? dept.label : value || "Not assigned";
  };

  // ACTION FILTERS LOGIC
  const actionDepartments = [
    "all",
    ...new Set(actions.map(a => a.department).filter(Boolean)),
  ];

  const actionInspectors = [
    "all",
    ...new Set(
      actions
        .filter(a =>
          actionDepartment === "all"
            ? true
            : a.department === actionDepartment
        )
        .map(a => a.inspectorName)
        .filter(Boolean)
    ),
  ];

  const filteredActions = actions.filter(a => {
    const deptMatch =
      actionDepartment === "all" || a.department === actionDepartment;

    const inspectorMatch =
      actionInspector === "all" || a.inspectorName === actionInspector;

    return deptMatch && inspectorMatch;
  });

  // Filter accounts based on search, department, and account type
  const filteredAccounts = allAccounts.filter(account => {
    const matchesSearch = searchQuery === "" ||
      account.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      account.email.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesDepartment = filterDepartment === "all" ||
      account.department === filterDepartment ||
      (!account.department && filterDepartment === "none");

    const matchesType = accountFilter === "all" ||
      (accountFilter === "user" && account.role === "USER") ||
      (accountFilter === "inspector" && account.role === "INSPECTOR");

    return matchesSearch && matchesDepartment && matchesType;
  });

  // Filter inspectors by department
  const filteredInspectors = filterDepartment === "all"
    ? inspectors
    : inspectors.filter(i => i.department === filterDepartment);

  // Get unique departments from existing users
  const departments = [...new Set(allAccounts
    .map(a => a.department)
    .filter(Boolean)
    .filter(dept => DEPARTMENT_OPTIONS.some(opt => opt.value === dept))
  )];

  /* ================= FILTER USER COMPLAINTS BY TAB ================= */
  const filteredUserComplaints = userComplaints.filter(c => {
    switch (userComplaintsTab) {
      case "open":
        return c.status === "open" || c.status === "assigned";
      case "in_progress":
        return c.status === "in_progress";
      case "resolved":
        return c.status === "resolved" || c.status === "closed";
      default:
        return true;
    }
  });

  /* ================= FILTER INSPECTOR COMPLAINTS BY TAB ================= */
  const filteredInspectorComplaints = inspectorComplaints.filter(c => {
    switch (inspectorComplaintsTab) {
      case "open":
        return c.status === "open" || c.status === "assigned";
      case "in_progress":
        return c.status === "in_progress";
      case "resolved":
        return c.status === "resolved" || c.status === "closed";
      default:
        return true;
    }
  });

  return (
    <div className="admin-layout">
      {/* ================= SIDEBAR ================= */}
      <aside className={`admin-sidebar ${sidebarOpen ? "open" : ""}`}>
        <h2 className="sidebar-title">Admin Panel</h2>

        {/* DASHBOARD TAB */}
        <button
          className={`sidebar-btn ${view === "dashboard" ? "active" : ""}`}
          onClick={() => {
            setView("dashboard");
            setSidebarOpen(false);
          }}
        >
          <span className="btn-icon"></span>
          Dashboard
        </button>

        <button
          className={`sidebar-btn ${view === "inspectors" ? "active" : ""}`}
          onClick={() => {
            setView("inspectors");
            setSidebarOpen(false);
            markSectionAsSeen("inspectors", inspectors);
          }}
        >
          <span className="btn-icon"></span>
          Inspectors
          {inspectors.filter(i => i.id > (lastSeen.inspectors || 0)).length > 0 && (
            <span className="badge-count">
              {inspectors.filter(i => i.id > (lastSeen.inspectors || 0)).length}
            </span>
          )}
        </button>

        <button
          className={`sidebar-btn ${view === "complaints" ? "active" : ""}`}
          onClick={() => {
            setView("complaints");
            setSidebarOpen(false);
            markSectionAsSeen("complaints", complaints);
          }}
        >
          <span className="btn-icon"></span>
          Complaints
          {complaints.filter(c => c.numericId > (lastSeen.complaints || 0)).length > 0 && (
            <span className="badge-count">
              {complaints.filter(c => c.numericId > (lastSeen.complaints || 0)).length}
            </span>
          )}
        </button>

        <button
          className={`sidebar-btn ${view === "tickets" ? "active" : ""}`}
          onClick={() => { setView("tickets"); setSidebarOpen(false); }}
        >
          <span className="btn-icon"></span>
          Ticket Logs
        </button>

        <button
          className={`sidebar-btn ${view === "actions" ? "active" : ""}`}
          onClick={() => {
            setView("actions");
            setSidebarOpen(false);
            markSectionAsSeen("actions", actions);
          }}
        >
          <span className="btn-icon"></span>
          Inspector Actions
          {actions.filter(a => a.id > (lastSeen.actions || 0)).length > 0 && (
            <span className="badge-count">
              {actions.filter(a => a.id > (lastSeen.actions || 0)).length}
            </span>
          )}
        </button>

        {/* ACCOUNTS TAB */}
        <button
          className={`sidebar-btn ${view === "accounts" ? "active" : ""}`}
          onClick={() => {
            setView("accounts");
            setSidebarOpen(false);
            markSectionAsSeen("accounts", allAccounts);
          }}
        >
          <span className="btn-icon"></span>
          All Accounts
          {allAccounts.filter(a => a.id > (lastSeen.accounts || 0)).length > 0 && (
            <span className="badge-count">
              {allAccounts.filter(a => a.id > (lastSeen.accounts || 0)).length}
            </span>
          )}
        </button>

        {/* STATS SUMMARY */}
        <div className="sidebar-stats">
          <h4>Quick Stats</h4>
          <div className="stat-item">
            <span className="stat-label">Users:</span>
            <span className="stat-value">{normalUsers.length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Inspectors:</span>
            <span className="stat-value approved">{inspectors.length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Total Accounts:</span>
            <span className="stat-value">{allAccounts.length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Total Complaints:</span>
            <span className="stat-value">{complaints.length}</span>
          </div>
        </div>
      </aside>

      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* ================= TICKET DETAILS MODAL ================= */}
      {showTicketDetails && selectedTicketId && (
        <div className="modal-overlay" onClick={() => setShowTicketDetails(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '700px', maxHeight: '85vh' }}>
            <div className="modal-header">
              <h2>Ticket Details: {selectedTicketId}</h2>
            </div>

            <div className="modal-body">
              {ticketDetails ? (
                <>
                  {/* Main Ticket Information */}
                  <div className="details-section">
                    <h3>Ticket Information</h3>
                    <div className="details-grid">
                      <div className="detail-item">
                        <strong>Ticket ID:</strong>
                        <span className="ticket-id-display">{ticketDetails.ticket_id}</span>
                      </div>
                      <div className="detail-item">
                        <strong>Raised By:</strong>
                        <span>{ticketDetails.user_name || "Anonymous"}</span>
                      </div>
                      <div className="detail-item">
                        <strong>Area:</strong>
                        <span>{ticketDetails.area || "N/A"}</span>
                      </div>
                      <div className="detail-item">
                        <strong>District:</strong>
                        <span>{ticketDetails.district || "N/A"}</span>
                      </div>
                      <div className="detail-item">
                        <strong>Date & Time:</strong>
                        <span>{new Date(ticketDetails.created_at).toLocaleString()}</span>
                      </div>
                      <div className="detail-item">
                        <strong>Status:</strong>
                        <span className={`status-badge ${ticketDetails.status}`}>
                          {ticketDetails.status || 'open'}
                        </span>
                      </div>
                      <div className="detail-item">
                        <strong>Sub-Tickets:</strong>
                        <span>{(ticketDetails.sub_tickets || []).length}</span>
                      </div>
                    </div>
                  </div>

                  {/* Image Preview Section - Only show if there's at least one image */}
                  {(ticketDetails.sub_tickets || []).some(st => st.image_url) && (
                    <div className="image-preview-section">
                      <h3>Image Preview</h3>
                      <div className="image-grid">
                        {(ticketDetails.sub_tickets || []).filter(st => st.image_url).map((st, index) => (
                          <div key={index} className="image-preview-item">
                            <div className="image-container">
                              <img 
                                src={st.image_url} // ✅ Changed: Use direct image_url
                                alt={`Complaint ${index + 1}`}
                                className="preview-image"
                                onClick={() => {
                                  setPreviewMedia({
                                    url: st.image_url, // ✅ Changed
                                    type: st.media_type === 'video' ? 'video' : 'image'
                                  });
                                }}
                                onError={(e) => {
                                  e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="150"%3E%3Crect fill="%23f3f4f6" width="200" height="150"/%3E%3Ctext fill="%239ca3af" x="50%" y="50%" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E';
                                }}
                              />
                              <div className="image-overlay">
                                <span className="image-label">
                                  {st.issue_type || 'Issue'} - {(st.confidence * 100).toFixed(1)}%
                                </span>
                                <button 
                                  className="enlarge-btn"
                                  onClick={() => {
                                    setPreviewMedia({
                                      url: st.image_url, // ✅ Changed
                                      type: st.media_type === 'video' ? 'video' : 'image'
                                    });
                                  }}
                                >
                                  🔍
                                </button>
                              </div>
                            </div>
                            <div className="image-info">
                              <span className="issue-type">{st.issue_type || 'Unknown'}</span>
                              <span className="confidence">{(st.confidence * 100).toFixed(1)}% confidence</span>
                              <span className={`status-indicator ${st.status}`}>{st.status}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="empty-state">
                  <p>Ticket details not found</p>
                  <p className="error-text">Ticket ID: {selectedTicketId} may have been deleted or doesn't exist.</p>
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button
                className="btn-secondary"
                onClick={() => setShowTicketDetails(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Media Preview Modal */}
      {previewMedia && (
        <div className="modal-overlay full-media-modal" onClick={() => setPreviewMedia(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setPreviewMedia(null)}>
              &times;
            </button>
            {previewMedia.type === 'video' ? (
              <video 
                src={previewMedia.url} 
                controls 
                autoPlay 
                className="full-media"
              />
            ) : (
              <img 
                src={previewMedia.url} 
                alt="Enlarged preview" 
                className="full-media"
              />
            )}
          </div>
        </div>
      )}

      {/* ================= ADD INSPECTOR MODAL ================= */}
      {showAddInspectorModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3>Add New Inspector</h3>
              <button
                className="modal-close-btn"
                onClick={() => setShowAddInspectorModal(false)}
                aria-label="Close"
              >
                ×
              </button>
            </div>

            <div className="modal-body">
              <div className="form-group">
                <label>Full Name *</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="Enter inspector's full name"
                  value={newInspector.name}
                  onChange={(e) => setNewInspector({ ...newInspector, name: e.target.value })}
                  required
                  autoComplete="off"
                />
              </div>

              <div className="form-group">
                <label>Email Address *</label>
                <input
                  type="email"
                  className="form-input"
                  placeholder="Enter email address"
                  value={newInspector.email}
                  onChange={(e) => setNewInspector({ ...newInspector, email: e.target.value })}
                  required
                  autoComplete="off"
                />
              </div>

              <div className="form-group">
                <label>Password * (min 6 characters)</label>
                <input
                  type="password"
                  className="form-input"
                  placeholder="Set password"
                  value={newInspector.password}
                  onChange={(e) => setNewInspector({ ...newInspector, password: e.target.value })}
                  required
                  minLength="6"
                  autoComplete="new-password"
                />
              </div>

              <div className="form-group">
                <label>Confirm Password *</label>
                <input
                  type="password"
                  className="form-input"
                  placeholder="Confirm password"
                  value={newInspector.confirmPassword}
                  onChange={(e) => setNewInspector({ ...newInspector, confirmPassword: e.target.value })}
                  required
                  autoComplete="new-password"
                />
              </div>

              <div className="form-group">
                <label>Department *</label>
                <select
                  className="form-select"
                  value={newInspector.department}
                  onChange={(e) => setNewInspector({ ...newInspector, department: e.target.value })}
                  required
                >
                  <option value="" disabled>Select Department</option>
                  {DEPARTMENT_OPTIONS.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="modal-footer">
              <button
                className="btn-secondary"
                onClick={() => setShowAddInspectorModal(false)}
              >
                Cancel
              </button>
              <button
                className="btn-primary"
                onClick={handleAddInspector}
              >
                Create Inspector
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ================= ERROR POPUP ================= */}
      {showErrorPopup && (
        <div className="modal-overlay">
          <div className="modal-content" style={{ maxWidth: "420px" }}>
            <div className="modal-header">
              <h3 style={{ color: "#dc2626" }}>Action Failed</h3>
              <button
                className="modal-close-btn"
                onClick={() => setShowErrorPopup(false)}
              >
                ×
              </button>
            </div>

            <div
              className="modal-body"
              style={{ textAlign: "center", padding: "20px" }}
            >
              <p style={{ fontSize: "16px", color: "#334155" }}>
                {errorPopupMessage}
              </p>
            </div>

            <div className="modal-footer" style={{ justifyContent: "center" }}>
              <button
                className="btn-primary"
                onClick={() => setShowErrorPopup(false)}
              >
                OK
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ================= CONFIRM DELETE MODAL ================= */}
      {showConfirmDelete && selectedInspector && (
        <div className="modal-overlay">
          <div className="modal-content" style={{ maxWidth: "420px" }}>
            <div className="modal-header">
              <h3>Confirm Removal</h3>
              <button
                className="modal-close-btn"
                onClick={() => setShowConfirmDelete(false)}
              >
                ×
              </button>
            </div>

            <div className="modal-body" style={{ textAlign: "center" }}>
              <p style={{ fontSize: "16px", marginBottom: "10px" }}>
                Are you sure you want to remove this inspector?
              </p>

              <strong style={{ color: "#dc2626", fontSize: "18px" }}>
                {selectedInspector.name}
              </strong>

              <p style={{ marginTop: "10px", color: "#64748b" }}>
                This will revoke their access permanently.
              </p>
            </div>

            <div className="modal-footer">
              <button
                className="btn-secondary"
                onClick={() => setShowConfirmDelete(false)}
              >
                Cancel
              </button>

              <button
                className="btn-primary"
                style={{ background: "#dc2626" }}
                onClick={async () => {
                  try {
                    await deleteUser(selectedInspector.id);
                    setDeletedInspectorName(selectedInspector.name);
                    setDeleteModalType("inspector");
                    setShowConfirmDelete(false);
                    setShowDeleteModal(true);
                    refreshUsers();
                  } catch (err) {
                    setErrorPopupMessage("Failed to remove inspector: " + err.message);
                    setShowErrorPopup(true);
                  }
                }}
              >
                Remove Inspector
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ================= SUCCESS MODAL ================= */}
      {showSuccessModal && createdInspector && (
        <div className="modal-overlay">
          <div className="modal-content success-modal-content" style={{ maxWidth: '450px', padding: '0', overflow: 'hidden' }}>
            <div className="success-modal-header">
              <span className="success-icon-large">✅</span>
              <h2 className="success-title">Inspector Added!</h2>
              <p className="success-subtitle">Account created successfully</p>
            </div>

            <div className="modal-body" style={{ padding: '30px' }}>
              <p className="success-help-text" style={{ textAlign: 'center', marginBottom: '20px', color: '#64748b' }}>
                Please share these login credentials with the inspector.
              </p>

              <div className="credentials-box">
                <div className="credential-item">
                  <label className="credential-label">EMAIL</label>
                  <div className="credential-value">{createdInspector.email}</div>
                </div>
                <div className="credential-item">
                  <label className="credential-label">PASSWORD</label>
                  <div className="credential-value">{createdInspector.password}</div>
                </div>
                <div className="credential-item">
                  <label className="credential-label">DEPARTMENT</label>
                  <div className="credential-value">{getDepartmentLabel(createdInspector.department)}</div>
                </div>
              </div>
            </div>

            <div className="modal-footer" style={{ borderTop: 'none', paddingBottom: '30px', justifyContent: 'center' }}>
              <button
                className="btn-success-full"
                onClick={() => setShowSuccessModal(false)}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ================= DELETE SUCCESS MODAL ================= */}
      {showDeleteModal && (
        <div className="modal-overlay">
          <div className="modal-content success-modal-content" style={{ maxWidth: '400px', padding: '0', overflow: 'hidden' }}>
            {deleteModalType === "inspector" ? (
              // Inspector Delete Modal (Red)
              <>
                <div className="success-modal-header" style={{ background: 'linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)' }}>
                  <span className="success-icon-large">❌</span>
                  <h2 className="success-title">Inspector Removed</h2>
                  <p className="success-subtitle">Access revoked successfully</p>
                </div>

                <div className="modal-body" style={{ padding: '30px', textAlign: 'center' }}>
                  <p style={{ fontSize: '16px', color: '#64748b', marginBottom: '10px' }}>
                    You have successfully removed inspector:
                  </p>
                  <div style={{
                    fontSize: '20px',
                    fontWeight: '700',
                    color: '#1e293b',
                    background: '#f1f5f9',
                    padding: '10px',
                    borderRadius: '8px',
                    display: 'inline-block'
                  }}>
                    {deletedInspectorName}
                  </div>
                </div>

                <div className="modal-footer" style={{ borderTop: 'none', paddingBottom: '30px', justifyContent: 'center' }}>
                  <button
                    className="btn-success-full"
                    onClick={() => setShowDeleteModal(false)}
                    style={{ background: 'linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)', boxShadow: '0 4px 12px rgba(239, 68, 68, 0.3)' }}
                  >
                    Close
                  </button>
                </div>
              </>
            ) : (
              // Ticket Delete Modal (Blue)
              <>
                <div className="success-modal-header" style={{ background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)' }}>
                  <span className="success-icon-large">❌</span>
                  <h2 className="success-title">Ticket Deleted</h2>
                  <p className="success-subtitle">Deleted successfully</p>
                </div>

                <div className="modal-footer" style={{ borderTop: 'none', padding: '30px', justifyContent: 'center' }}>
                  <button
                    className="btn-success-full"
                    onClick={() => setShowDeleteModal(false)}
                    style={{ background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)', boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)' }}
                  >
                    Close
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ================= MAIN ================= */}
      <div className="admin-main">
        <header className="admin-topbar">
          <button className="menu-btn" onClick={() => setSidebarOpen(true)}>☰</button>
          <h2>Admin Dashboard</h2>
          <div className="topbar-actions">
            <span className="welcome-text">Welcome, {user?.name || "Admin"}</span>
            <button className="logout-btn" onClick={handleLogout}
            title="click to logout"
            >Logout</button>
          </div>
        </header>

        <div className="admin-container">
          {/* ================= DASHBOARD ================= */}
          {view === "dashboard" && (
            <div className="dashboard-view">
              <Dashboard />
            </div>
          )}

          {/* ================= INSPECTORS ================= */}
          {view === "inspectors" && (
            <>
              <div className="view-header">
                <h2>Inspector Management</h2>
                <div className="header-controls">
                  <div className="filter-controls">
                    <select
                      className="filter-select"
                      value={filterDepartment}
                      onChange={(e) => setFilterDepartment(e.target.value)}
                    >
                      <option value="all">All Departments</option>
                      {departments.map(dept => (
                        <option key={dept} value={dept}>
                          {getDepartmentLabel(dept)}
                        </option>
                      ))}
                      <option value="none">No Department</option>
                    </select>
                  </div>

                  {/* Add Inspector Button */}
                  <button
                    className="add-inspector-btn"
                    onClick={() => setShowAddInspectorModal(true)}
                  >
                    <span className="btn-icon">➕</span>
                    Add Inspector
                  </button>
                </div>
              </div>

              {/* Inspectors List */}
              <div className="admin-card">
                <div className="card-header">
                  <h3>All Inspectors ({filteredInspectors.length})</h3>
                  <span className="subtitle">Active inspectors with system access</span>
                </div>

                {filteredInspectors.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon"></div>
                    <p>No inspectors found</p>
                    {filterDepartment !== "all" && (
                      <button
                        className="clear-filter-btn"
                        onClick={() => setFilterDepartment("all")}
                      >
                        Clear Filter
                      </button>
                    )}
                    <button
                      className="add-inspector-btn-empty"
                      onClick={() => setShowAddInspectorModal(true)}
                    >
                      ➕ Add Your First Inspector
                    </button>
                  </div>
                ) : (
                  <div className="inspectors-grid">
                    {filteredInspectors.map(i => (
                      <div className="inspector-card" key={i.id}>
                        <div className="inspector-header">
                          <div className="inspector-avatar">
                            {i.name.charAt(0).toUpperCase()}
                          </div>
                          <div className="inspector-info">
                            <h4>{i.name}</h4>
                            <p className="inspector-email">{i.email}</p>
                          </div>
                        </div>

                        <div className="inspector-details">
                          <div className="detail-item">
                            <span className="detail-label">Department:</span>
                            <span className="detail-value department-badge">
                              {getDepartmentLabel(i.department)}
                            </span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">Role:</span>
                            <span className="detail-value">Inspector</span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">Status:</span>
                            <span className="detail-value active">Active</span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">Created:</span>
                            <span className="detail-value">
                              {i.createdAt ? new Date(i.createdAt).toLocaleDateString() : 'N/A'}
                            </span>
                          </div>
                        </div>

                        <div className="inspector-actions">
                          <button
                            className="action-btn view-btn"
                            onClick={() => viewInspectorComplaints(i)}
                          >
                            <span className="btn-icon"></span>
                            View Complaints
                          </button>
                          <button
                            className="action-btn remove-btn"
                            onClick={() => removeInspector(i.id)}
                          >
                            <span className="btn-icon">❌</span>
                            Remove Inspector
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}

          {/* ================= ACCOUNTS TAB ================= */}
          {view === "accounts" && (
            <>
              <div className="view-header">
                <h2>Account Management</h2>
                <div className="header-controls">
                  <div className="search-box">
                    <input
                      type="text"
                      placeholder="Search by name or email..."
                      className="search-input"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                    <span className="search-icon">🔍</span>
                  </div>
                  <div className="filter-controls">
                    <select
                      className="filter-select"
                      value={filterDepartment}
                      onChange={(e) => setFilterDepartment(e.target.value)}
                    >
                      <option value="all">All Departments</option>
                      {departments.map(dept => (
                        <option key={dept} value={dept}>
                          {getDepartmentLabel(dept)}
                        </option>
                      ))}
                      <option value="none">No Department</option>
                    </select>
                  </div>
                  <button
                    className="add-inspector-btn"
                    onClick={() => setShowAddInspectorModal(true)}
                  >
                    <span className="btn-icon">➕</span>
                    Add Inspector
                  </button>
                </div>
              </div>

              <div className="accounts-container">
                {/* Summary Cards */}
                <div className="summary-cards">
                  <div
                    className={`summary-card total-users ${accountFilter === 'all' ? 'active-filter' : ''}`}
                    onClick={() => setAccountFilter('all')}
                    style={{ cursor: 'pointer', border: accountFilter === 'all' ? '2px solid #3b82f6' : '' }}
                  >
                    <div className="summary-icon"></div>
                    <div className="summary-content">
                      <h4>Total Accounts</h4>
                      <p className="summary-count">{allAccounts.length}</p>
                    </div>
                  </div>
                  <div
                    className={`summary-card normal-users ${accountFilter === 'user' ? 'active-filter' : ''}`}
                    onClick={() => setAccountFilter('user')}
                    style={{ cursor: 'pointer', border: accountFilter === 'user' ? '2px solid #8b5cf6' : '' }}
                  >
                    <div className="summary-icon"></div>
                    <div className="summary-content">
                      <h4>Regular Users</h4>
                      <p className="summary-count">
                        {normalUsers.length}
                      </p>
                    </div>
                  </div>
                  <div
                    className={`summary-card approved-inspectors ${accountFilter === 'inspector' ? 'active-filter' : ''}`}
                    onClick={() => setAccountFilter('inspector')}
                    style={{ cursor: 'pointer', border: accountFilter === 'inspector' ? '2px solid #10b981' : '' }}
                  >
                    <div className="summary-icon"></div>
                    <div className="summary-content">
                      <h4>Inspectors</h4>
                      <p className="summary-count">
                        {inspectors.length}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Accounts Table */}
                <div className="admin-card">
                  <div className="card-header">
                    <h3>All Accounts ({filteredAccounts.length})</h3>
                    <span className="subtitle">
                      Showing {filteredAccounts.length} of {allAccounts.length} accounts
                      {searchQuery && ` matching "${searchQuery}"`}
                    </span>
                  </div>

                  {filteredAccounts.length === 0 ? (
                    <div className="empty-state">
                      <div className="empty-icon"></div>
                      <p>No accounts found</p>
                      {(searchQuery || filterDepartment !== "all") && (
                        <button
                          className="clear-filter-btn"
                          onClick={() => {
                            setSearchQuery("");
                            setFilterDepartment("all");
                          }}
                        >
                          Clear Filters
                        </button>
                      )}
                    </div>
                  ) : (
                    <div className="accounts-table-container">
                      <div className="accounts-table">
                        <div className="table-header">
                          <span>Name</span>
                          <span>Email</span>
                          <span>Department</span>
                          <span>Role</span>
                          <span>Status</span>
                          <span>Actions</span>
                        </div>

                        {filteredAccounts.map(user => (
                          <div className="table-row" key={user.id}>
                            <span className="user-name">
                              <div className="user-avatar-small">
                                {user.name.charAt(0).toUpperCase()}
                              </div>
                              {user.name}
                            </span>
                            <span className="email-cell">{user.email}</span>
                            <span>
                              {user.department ? (
                                <span className="department-tag">
                                  {getDepartmentLabel(user.department)}
                                </span>
                              ) : (
                                <span className="no-dept">—</span>
                              )}
                            </span>
                            <span>
                              {user.role === "INSPECTOR" ? (
                                <span className="badge badge-inspector">Inspector</span>
                              ) : (
                                <span className="badge badge-user">User</span>
                              )}
                            </span>
                            <span>
                              <span className="badge badge-active">Active</span>
                            </span>
                            <span className="account-actions">
                              {/* VIEW COMPLAINTS BUTTON FOR ALL USERS */}
                              <button
                                className="view-complaints-btn"
                                onClick={() => {
                                  if (user.role === "INSPECTOR") {
                                    viewInspectorComplaints(user);
                                  } else {
                                    viewUserComplaints(user);
                                  }
                                }}
                                title={`View ${user.name}'s complaints`}
                              >
                                📋
                                {/* Show complaint count badge */}
                                {(() => {
                                  const userComplaintCount = complaints.filter(c => 
                                    c.user_name && c.user_name.toLowerCase() === user.name.toLowerCase()
                                  ).length;
                                  return userComplaintCount > 0 ? (
                                    <span className="complaint-count-badge">
                                      {userComplaintCount}
                                    </span>
                                  ) : null;
                                })()}
                              </button>
                              
                              {user.role === "INSPECTOR" ? (
                                <button
                                  className="delete-account-btn"
                                  onClick={() => {
                                    setSelectedInspector(user);
                                    setShowConfirmDelete(true);
                                  }}
                                  title="Remove inspector"
                                >
                                  ❌
                                </button>
                              ) : (
                                <button
                                  className="delete-user-btn"
                                  onClick={() => {
                                    if (window.confirm(`Delete user account: ${user.name}?\nThis action cannot be undone.`)) {
                                      deleteAccount(user.id, user.name);
                                    }
                                  }}
                                  title="Delete user account"
                                >
                                  ❌
                                </button>
                              )}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}

          {/* ================= COMPLAINTS (ADMIN) ================= */}
          {view === "complaints" && (
            <>
              <div className="view-header">
                <h2>Complaints Management</h2>
                <span className="subtitle">Manage and monitor all complaints</span>
              </div>

              {complaints.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon"></div>
                  <p>No complaints found</p>
                </div>
              ) : (
                <div className="complaints-grid">
                  {complaints.map(c => {
                    const breached = checkSLABreach(c.date, c.slaHours);

                    return (
                      <div
                        key={c.id}
                        className="complaint-card-grid"
                        style={{
                          border: breached ? "2px solid #ef4444" : "1px solid #444"
                        }}
                      >
                        <div className="complaint-image-wrapper">
                          {c.image_url ? ( // ✅ Changed: Check for image_url instead of image
                            <img
                              src={c.image_url} // ✅ Changed: Use direct image_url
                              className="complaint-image-grid"
                              alt="Complaint"
                              onError={(e) => {
                                e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="150"%3E%3Crect fill="%23f3f4f6" width="200" height="150"/%3E%3Ctext fill="%239ca3af" x="50%" y="50%" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E';
                              }}
                            />
                          ) : (
                            <div className="no-media-box">No Media</div>
                          )}
                        </div>

                        <div className="complaint-info-grid">
                          <p className="raised-by-top"><strong>User:</strong> <span className="user-tag">{c.user_name || "Anonymous"}</span></p>
                          <p><strong>Sub ID:</strong> {c.id}</p>
                          <p><strong>Issue:</strong> {c.issue_type}</p>
                          <p><strong>Confidence:</strong> {(c.confidence * 100).toFixed(1)}%</p>

                          {/* 🔥 STATUS DROPDOWN (ADMIN) */}
                          <div className="status-control">
                            <strong>Status:</strong>
                            <select
                              className="status-dropdown"
                              value={c.status}
                              onChange={(e) =>
                                updateComplaintStatus(c.id, e.target.value)
                              }
                            >
                              <option value="open">Open</option>
                              <option value="assigned">Assigned</option>
                              <option value="in_progress">In Progress</option>
                              <option value="resolved">Resolved</option>
                              <option value="closed">Closed</option>
                            </select>
                          </div>

                          <p><strong>TAT (SLA):</strong> {c.slaHours} hrs</p>

                          <p>
                            <strong>SLA Breach:</strong>{" "}
                            {breached ? (
                              <span className="breach-indicator">
                                YES ⚠️
                              </span>
                            ) : (
                              <span className="no-breach">
                                NO ✔
                              </span>
                            )}
                          </p>

                          <p><strong>Area:</strong> {c.area}</p>
                          <p><strong>District:</strong> {c.district}</p>
                          <p><strong>Date:</strong> {new Date(c.date).toLocaleString()}</p>

                          {c.latitude && c.longitude && (
                            <p>
                              <strong>Location:</strong>{" "}
                              <a
                                href={`https://maps.google.com/?q=${c.latitude},${c.longitude}`}
                                target="_blank"
                                rel="noreferrer"
                                className="map-link"
                              >
                                {c.latitude.toFixed(6)}, {c.longitude.toFixed(6)}
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

          {/* ================= TICKETS ================= */}
          {view === "tickets" && (
            <div className="ticket-log-container">
              <div className="view-header">
                <h2>Ticket Logs</h2>
                <span className="subtitle">Complete audit trail of all system activities</span>
              </div>
              <div className="ticket-log-card">
                <TicketLog onDeleteSuccess={() => {
                  setDeleteModalType("ticket");
                  setShowDeleteModal(true);
                }} />
              </div>
            </div>
          )}

          {/* ================= ACTIONS ================= */}
          {view === "actions" && (
            <div className="actions-container">
              <div className="view-header">
                <h2>Inspector Actions Log</h2>
                <span className="subtitle">Track all inspector activities</span>
                
                {/* 🔥 FILTER BAR */}
                <div className="filter-bar" style={{ marginTop: '15px' }}>
                  {/* Department Filter */}
                  <select
                    className="filter-select"
                    value={actionDepartment}
                    onChange={(e) => {
                      setActionDepartment(e.target.value);
                      setActionInspector("all"); // reset inspector
                    }}
                  >
                    <option value="all">All Departments</option>
                    {actionDepartments
                      .filter(d => d !== "all")
                      .map(dept => (
                        <option key={dept} value={dept}>
                          {getDepartmentLabel(dept)}
                        </option>
                      ))}
                  </select>

                  {/* Inspector Filter */}
                  <select
                    className="filter-select"
                    value={actionInspector}
                    onChange={(e) => setActionInspector(e.target.value)}
                    disabled={actionDepartment === "all"}
                  >
                    <option value="all">All Inspectors</option>
                    {actionInspectors
                      .filter(i => i !== "all")
                      .map(ins => (
                        <option key={ins} value={ins}>
                          {ins}
                        </option>
                      ))}
                  </select>
                </div>
              </div>

              <div className="admin-card">
                <div className="card-header">
                  <h3>Recent Actions ({filteredActions.length})</h3>
                </div>

                {filteredActions.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon"></div>
                    <p>No actions recorded yet</p>
                  </div>
                ) : (
                  <div className="actions-list">
                    {filteredActions.map(a => (
                      <div className="action-item" key={a.id}>
                        <div className="action-header">
                          <div className="action-avatar">
                            {a.inspectorName?.charAt(0).toUpperCase() || 'I'}
                          </div>
                          <div className="action-info">
                            <h4>{a.inspectorName || 'Unknown Inspector'}</h4>
                            <p className="action-time">{a.time}</p>
                          </div>
                          <span className="action-type">{a.action}</span>
                        </div>

                        <div className="action-details">
                          <p className="action-description">
                            <strong>Action:</strong> {a.action}
                          </p>
                          <p className="action-ticket">
                            <strong>Ticket ID:</strong>{" "}
                            <button
                              className="ticket-id-link"
                              onClick={() => viewTicketDetails(a.ticketId)}
                              title="Click to view ticket details"
                            >
                              {a.ticketId}
                            </button>
                          </p>
                          {a.details && (
                            <p className="action-extra">
                              <strong>Details:</strong> {a.details}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* ================= INSPECTOR COMPLAINTS MODAL WITH TABS ================= */}
        {showInspectorComplaintsModal && viewingInspector && (
          <div className="modal-overlay">
            <div className="modal-content complaints-modal">
              <div className="modal-header">
                <div className="view-header-accent">
                  <span className="accent-icon"></span>
                  <div>
                    <h3>Complaints for {viewingInspector.name}</h3>
                    <p className="subtitle">{viewingInspector.department} Department • {inspectorComplaints.length} active tasks</p>
                  </div>
                </div>
                <button
                  className="modal-close-btn"
                  onClick={() => setShowInspectorComplaintsModal(false)}
                >
                  ×
                </button>
              </div>

              {/* Tab Navigation */}
              <div className="complaints-tabs">
                <button
                  className={`tab-btn ${inspectorComplaintsTab === "open" ? "active" : ""}`}
                  onClick={() => setInspectorComplaintsTab("open")}
                >
                  Open
                  <span className="tab-count">
                    {inspectorComplaints.filter(c => c.status === "open" || c.status === "assigned").length}
                  </span>
                </button>
                <button
                  className={`tab-btn ${inspectorComplaintsTab === "in_progress" ? "active" : ""}`}
                  onClick={() => setInspectorComplaintsTab("in_progress")}
                >
                  In Progress
                  <span className="tab-count">
                    {inspectorComplaints.filter(c => c.status === "in_progress").length}
                  </span>
                </button>
                <button
                  className={`tab-btn ${inspectorComplaintsTab === "resolved" ? "active" : ""}`}
                  onClick={() => setInspectorComplaintsTab("resolved")}
                >
                  Resolved
                  <span className="tab-count">
                    {inspectorComplaints.filter(c => c.status === "resolved" || c.status === "closed").length}
                  </span>
                </button>
              </div>

              <div className="modal-body inspector-complaints-list">
                {filteredInspectorComplaints.length === 0 ? (
                  <div className="empty-state-small">
                    <div className="empty-icon">📂</div>
                    <p>No {inspectorComplaintsTab} complaints for this inspector.</p>
                  </div>
                ) : (
                  <div className="inspector-complaints-grid">
                    {filteredInspectorComplaints.map(c => (
                      <div className={`mini-complaint-card status-${c.status}`} key={c.id}>
                        <div className="mini-card-main">
                          <div className="mini-card-img">
                            {c.image_url ? ( // ✅ Changed: Check image_url instead of image
                              <img 
                                src={c.image_url} // ✅ Changed: Use direct image_url
                                alt="Preview" 
                                onError={(e) => {
                                  e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Crect fill="%23ddd" width="100" height="100"/%3E%3Ctext fill="%23999" x="50%" y="50%" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E';
                                }}
                              />
                            ) : (
                              <div className="no-img-placeholder">No Image</div>
                            )}
                          </div>
                          <div className="mini-card-info">
                            <div className="mini-card-header">
                              <span className="mini-id">#{c.id}</span>
                              <span className={`mini-status ${c.status}`}>{c.status}</span>
                            </div>
                            <h4 className="mini-issue">{c.issue_type}</h4>
                            <p className="mini-location"> {c.area}, {c.district}</p>
                            <p className="mini-date"> {new Date(c.date).toLocaleDateString()}</p>
                          </div>
                        </div>
                        <div className="mini-card-footer">
                          <button
                            className="mini-view-btn"
                            onClick={() => {
                              setShowInspectorComplaintsModal(false);
                              setView("complaints");
                              setSearchQuery(c.id);
                            }}
                          >
                            View Full Details →
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="modal-footer">
                <button
                  className="btn-secondary"
                  onClick={() => setShowInspectorComplaintsModal(false)}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ================= USER COMPLAINTS MODAL WITH TABS ================= */}
        {showUserComplaintsModal && viewingUser && (
          <div className="modal-overlay">
            <div className="modal-content complaints-modal">
              <div className="modal-header">
                <div className="view-header-accent">
                  <span className="accent-icon"></span>
                  <div>
                    <h3>Complaints by {viewingUser.name}</h3>
                    <p className="subtitle">Regular User • {userComplaints.length} total complaints</p>
                  </div>
                </div>
                <button
                  className="modal-close-btn"
                  onClick={() => setShowUserComplaintsModal(false)}
                >
                  ×
                </button>
              </div>

              {/* Tab Navigation */}
              <div className="complaints-tabs">
                <button
                  className={`tab-btn ${userComplaintsTab === "open" ? "active" : ""}`}
                  onClick={() => setUserComplaintsTab("open")}
                >
                  Open
                  <span className="tab-count">
                    {userComplaints.filter(c => c.status === "open" || c.status === "assigned").length}
                  </span>
                </button>
                <button
                  className={`tab-btn ${userComplaintsTab === "in_progress" ? "active" : ""}`}
                  onClick={() => setUserComplaintsTab("in_progress")}
                >
                  In Progress
                  <span className="tab-count">
                    {userComplaints.filter(c => c.status === "in_progress").length}
                  </span>
                </button>
                <button
                  className={`tab-btn ${userComplaintsTab === "resolved" ? "active" : ""}`}
                  onClick={() => setUserComplaintsTab("resolved")}
                >
                  Resolved
                  <span className="tab-count">
                    {userComplaints.filter(c => c.status === "resolved" || c.status === "closed").length}
                  </span>
                </button>
              </div>

              <div className="modal-body inspector-complaints-list">
                {filteredUserComplaints.length === 0 ? (
                  <div className="empty-state-small">
                    <div className="empty-icon">📂</div>
                    <p>No {userComplaintsTab} complaints by this user.</p>
                  </div>
                ) : (
                  <div className="inspector-complaints-grid">
                    {filteredUserComplaints.map(c => (
                      <div className={`mini-complaint-card status-${c.status}`} key={c.id}>
                        <div className="mini-card-main">
                          <div className="mini-card-img">
                            {c.image_url ? ( // ✅ Changed: Check image_url instead of image
                              <img 
                                src={c.image_url} // ✅ Changed: Use direct image_url
                                alt="Preview" 
                                onError={(e) => {
                                  e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Crect fill="%23ddd" width="100" height="100"/%3E%3Ctext fill="%23999" x="50%" y="50%" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E';
                                }}
                              />
                            ) : (
                              <div className="no-img-placeholder">No Image</div>
                            )}
                          </div>
                          <div className="mini-card-info">
                            <div className="mini-card-header">
                              <span className="mini-id">#{c.id}</span>
                              <span className={`mini-status ${c.status}`}>{c.status}</span>
                            </div>
                            <h4 className="mini-issue">{c.issue_type}</h4>
                            <p className="mini-location"> {c.area}, {c.district}</p>
                            <p className="mini-date"> {new Date(c.date).toLocaleDateString()}</p>
                          </div>
                        </div>
                        <div className="mini-card-footer">
                          <button
                            className="mini-view-btn"
                            onClick={() => {
                              setShowUserComplaintsModal(false);
                              setView("complaints");
                              setSearchQuery(c.id);
                            }}
                          >
                            View Full Details →
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="modal-footer">
                <button
                  className="btn-secondary"
                  onClick={() => setShowUserComplaintsModal(false)}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}