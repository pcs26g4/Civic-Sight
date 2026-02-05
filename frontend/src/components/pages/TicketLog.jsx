import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/TicketLog.css";
import { getTickets, deleteTicket } from "../../services/api"; // Removed getImageUrl import

function TicketLog({ showActions = true }) { // ✅ Added showActions prop with default true
  /* ===============================
     CONSTANTS
  =============================== */
  const ITEMS_PER_PAGE = 10;

  /* ===============================
     FILTER OPTIONS - UPDATED with 9 distortions from code-2
  =============================== */
  const distortions = [
    "all",
    "sand_on_road",
    "road_cracks",
    "street_hawkers",
    "animal_carcases",
    "potholes",
    "water_puddles",
    "garbage_overflow",
    "open_manholes",
    "street_debris"
  ];

  const statuses = ["all", "open", "in_progress", "resolved", "closed"];

  /* ===============================
     STATE
  =============================== */
  const navigate = useNavigate();
  const [tickets, setTickets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [previewImage, setPreviewImage] = useState(null);
  const [showDeletePopup, setShowDeletePopup] = useState(false);
  const [ticketToDelete, setTicketToDelete] = useState(null);
  const [popup, setPopup] = useState({
    show: false,
    message: "",
    type: "success", // success | error
  });

  /* ===============================
     GET CURRENT USER
  =============================== */
  const currentUser = JSON.parse(localStorage.getItem("user"));

  /* ===============================
     FIRST COMPLAINT DATE CALCULATION
  =============================== */
  // Get valid dates from tickets (filter out invalid dates)
  const getValidDatesFromTickets = (tickets) => {
    return tickets
      .map(ticket => {
        const date = new Date(ticket.created_at);
        return isNaN(date.getTime()) ? null : date;
      })
      .filter(date => date !== null);
  };

  // Calculate the user's first complaint date only for USER role
  const validDates = getValidDatesFromTickets(tickets);
  const firstComplaintDate = (currentUser?.role === "USER" && validDates.length)
    ? new Date(Math.min(...validDates.map(date => date.getTime())))
    : null;

  // Convert to minimum date string (YYYY-01-01)
  const minDate = firstComplaintDate
    ? `${firstComplaintDate.getFullYear()}-01-01`
    : "";

  // For ADMIN and INSPECTOR, show all dates (no restriction)
  const dateInputMin = currentUser?.role === "USER" ? minDate : "";

  // For debugging/logging
  useEffect(() => {
    if (firstComplaintDate) {
      console.log("First complaint date:", firstComplaintDate.toISOString());
      console.log("Min date for input:", dateInputMin);
    }
  }, [firstComplaintDate, dateInputMin]);

  /* ===============================
     FETCH TICKETS
  =============================== */
  useEffect(() => {
    const fetchTickets = async () => {
      try {
        const user = JSON.parse(localStorage.getItem("user"));
        const filters_api = {};
        if (user && user.role === "USER") {
          filters_api.user_id = user.id;
        }

        const response = await getTickets(filters_api);
        // Flatten tickets with sub_tickets
        const allSubTickets = [];
        (response.tickets || []).forEach(ticket => {
          (ticket.sub_tickets || []).forEach(subTicket => {
            // If inspector, only show assigned tickets
            if (user && user.role === "INSPECTOR" && subTicket.assigned_to !== user.id) {
              return;
            }

            allSubTickets.push({
              ...subTicket,
              ticket_id: ticket.ticket_id,
              latitude: subTicket.latitude || ticket.latitude,
              longitude: subTicket.longitude || ticket.longitude,
              area: ticket.area,
              district: ticket.district,
              user_name: ticket.user_name
            });
          });
        });
        setTickets(allSubTickets);
      } catch (error) {
        console.error("Failed to fetch tickets:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchTickets();

    // Refresh data when user returns to this tab
    window.addEventListener('focus', fetchTickets);
    return () => window.removeEventListener('focus', fetchTickets);
  }, []);

  /* ===============================
     FILTER STATE
  =============================== */
  const [selectedDistortion, setSelectedDistortion] = useState("all");
  const [selectedStatus, setSelectedStatus] = useState("all");
  const [selectedDate, setSelectedDate] = useState("");

  /* ===============================
     FILTER VISIBILITY STATE
  =============================== */
  const [showFilters, setShowFilters] = useState(false);

  /* ===============================
     PAGINATION STATE
  =============================== */
  const [currentPage, setCurrentPage] = useState(1);

  /* ===============================
     HELPER FUNCTIONS from code-2
  =============================== */

  // Function to normalize issue_type from database to match filter values
  const normalizeIssueType = (issueType) => {
    if (!issueType) return "";

    // Convert to lowercase and remove any spaces
    const cleaned = issueType.toLowerCase().trim();

    // Direct mapping of database values to filter values
    const mapping = {
      // Water puddles variations
      "waterpuddles": "water_puddles",
      "waterpuddle": "water_puddles",
      "water_puddle": "water_puddles",
      "water_puddles": "water_puddles",

      // Open manholes variations
      "openmanholes": "open_manholes",
      "openmanhole": "open_manholes",
      "open_manholes": "open_manholes",
      "open_manhole": "open_manholes",

      // Garbage overflow variations
      "garbageoverflow": "garbage_overflow",
      "garbage_overflow": "garbage_overflow",

      // Sand on road variations
      "sandonroad": "sand_on_road",
      "sand_on_road": "sand_on_road",

      // Road cracks variations
      "roadcracks": "road_cracks",
      "roadcrack": "road_cracks",
      "road_cracks": "road_cracks",
      "road_crack": "road_cracks",

      // Street hawkers variations
      "streethawkers": "street_hawkers",
      "streethawker": "street_hawkers",
      "street_hawkers": "street_hawkers",
      "street_hawker": "street_hawkers",

      // Animal carcases variations
      "animalcarcases": "animal_carcases",
      "animalcarcass": "animal_carcases",
      "animal_carcases": "animal_carcases",
      "animal_carcass": "animal_carcases",

      // Potholes variations
      "potholes": "potholes",
      "pothole": "potholes",

      // Street debris variations
      "streetdebris": "street_debris",
      "street_debris": "street_debris",
    };

    // Return the mapped value or the original if not found
    return mapping[cleaned] || cleaned;
  };

  // Function to format distortion for display
  const formatDistortionName = (distortion) => {
    if (distortion === "all") return "ALL";

    return distortion
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  /* ===============================
     FILTER + SORT LOGIC
  =============================== */
  const filteredTickets = tickets
    .filter((ticket) => {
      // Normalize the ticket's issue_type for comparison
      const normalizedIssueType = normalizeIssueType(ticket.issue_type);

      const distortionMatch =
        selectedDistortion === "all" ||
        normalizedIssueType === selectedDistortion;

      const statusMatch =
        selectedStatus === "all" ||
        (ticket.status && ticket.status.toLowerCase() === selectedStatus.toLowerCase());

      const dateMatch =
        selectedDate === "" ||
        (ticket.created_at && ticket.created_at.startsWith(selectedDate));

      return distortionMatch && statusMatch && dateMatch;
    })
    .sort((a, b) => {
      const dateA = a.created_at ? new Date(a.created_at) : new Date(0);
      const dateB = b.created_at ? new Date(b.created_at) : new Date(0);
      return dateB - dateA;
    });

  /* ===============================
     PAGINATION LOGIC
  =============================== */
  const totalPages = Math.ceil(filteredTickets.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = startIndex + ITEMS_PER_PAGE;
  const paginatedTickets = filteredTickets.slice(startIndex, endIndex);

  const handleFilterChange = (setter) => (value) => {
    setter(value);
    setCurrentPage(1);
  };

  const handleLocationClick = (ticketId, lat, lng) => {
    if (!lat || !lng) return;
    const url = `https://www.google.com/maps?q=${lat},${lng}`;
    window.open(url, "_blank");
  };

  // Function to navigate to internal map page with selected ticket
  const handleInternalMapClick = (ticket) => {
    if (!ticket.latitude || !ticket.longitude) {
      setPopup({
        show: true,
        message: "No location data available for this ticket",
        type: "error",
      });
      return;
    }

    // Prepare the ticket data to pass to the map page
    const mapTicketData = {
      selectedTicket: ticket,
      // Include all necessary data for the map
      latitude: parseFloat(ticket.latitude),
      longitude: parseFloat(ticket.longitude),
      ticketId: ticket.sub_id || ticket.ticket_id,
      issueType: ticket.issue_type,
      area: ticket.area,
      district: ticket.district,
      status: ticket.status,
      confidence: ticket.confidence,
      created_at: ticket.created_at,
      user_name: ticket.user_name,
      // Add these for better marker display
      markerId: ticket.sub_id || `ticket-${ticket.ticket_id}`,
      isSelected: true, // Flag to indicate this ticket should be focused
      // Additional details for popup
      description: ticket.description || `Issue: ${ticket.issue_type}`,
      image_id: ticket.image_id,
      media_type: ticket.media_type,
      image_url: ticket.image_url // Add image_url for map display
    };

    // Store in sessionStorage for persistence
    sessionStorage.setItem('selectedTicketForMap', JSON.stringify(mapTicketData));
    
    // Navigate to the map page with state
    navigate('/map', { 
      state: { 
        selectedTicket: mapTicketData,
        focusOnTicket: true,
        autoZoomToTicket: true,
        centerOnTicket: true,
        zoomLevel: 15 // Higher zoom level to focus on the ticket
      } 
    });
  };

  /* ===============================
     DELETE FUNCTION
  =============================== */
  const handleDelete = async () => {
    if (!ticketToDelete) return;

    try {
      await deleteTicket(ticketToDelete);
      // Remove all sub-tickets belonging to this ticket from local state
      setTickets((prev) => prev.filter((t) => t.ticket_id !== ticketToDelete));

      setPopup({
        show: true,
        message: "Ticket deleted successfully",
        type: "success",
      });
    } catch (error) {
      console.error("Failed to delete ticket:", error);

      setPopup({
        show: true,
        message: "Failed to delete ticket. Please try again.",
        type: "error",
      });
    } finally {
      setShowDeletePopup(false);
      setTicketToDelete(null);
    }
  };

  /* ===============================
     DELETE POPUP HANDLER (from code-2)
  =============================== */
  const handleDeleteClick = (ticketId) => {
    setTicketToDelete(ticketId);
    setShowDeletePopup(true);
  };

  return (
    <div className="ticket-log">
      {/* ===============================
          FILTER BUTTON
      =============================== */}
      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          marginBottom: "12px",
        }}
      >
        <button
          onClick={() => setShowFilters((prev) => !prev)}
          style={{
            background: "#ff5a4e",
            color: "#ffffff",
            border: "none",
            padding: "10px 18px",
            borderRadius: "6px",
            fontWeight: "700",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          🔽 FILTER
        </button>
      </div>

      {/* ===============================
          FILTER BAR (TOGGLE)
      =============================== */}
      {showFilters && (
        <div className="filter-bar" style={{
          overflowX: "auto",
          whiteSpace: "nowrap",
          marginBottom: "16px",
          paddingBottom: "8px" // Space for scrollbar
        }}>
          <div style={{
            display: "flex",
            gap: "16px",
            minWidth: "max-content"
          }}>
            <div className="filter-group" style={{ minWidth: "200px" }}>
              <label className="filter-label">Distortion Type</label>
              <select
                value={selectedDistortion}
                onChange={(e) =>
                  handleFilterChange(setSelectedDistortion)(e.target.value)
                }
                style={{ width: "100%" }}
              >
                {distortions.map((dist) => (
                  <option key={dist} value={dist}>
                    {formatDistortionName(dist)}
                  </option>
                ))}
              </select>
            </div>

            <div className="filter-group" style={{ minWidth: "150px" }}>
              <label className="filter-label">Status</label>
              <select
                value={selectedStatus}
                onChange={(e) =>
                  handleFilterChange(setSelectedStatus)(e.target.value)
                }
                style={{ width: "100%" }}
              >
                {statuses.map((status) => (
                  <option key={status} value={status}>
                    {status === "all" ? "ALL" : status.replace("_", " ").toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div className="filter-group" style={{ minWidth: "150px" }}>
              <label className="filter-label">Date</label>
              <input
                type="date"
                value={selectedDate}
                min={dateInputMin}  
                onChange={(e) =>
                  handleFilterChange(setSelectedDate)(e.target.value)
                }
                style={{ width: "100%" }}
                // Commenting out the onKeyDown and onPaste to prevent manual date entry
                // Users can only select dates from the calendar picker
                onKeyDown={(e) => {
                  // Prevent typing in the date field
                  e.preventDefault();
                  return false;
                }}
                onPaste={(e) => {
                  // Prevent pasting into the date field
                  e.preventDefault();
                  return false;
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* ===============================
          TABLE CONTAINER WITH SCROLL
      =============================== */}
      <div className="table-wrapper" style={{
        overflowX: "auto",
        borderRadius: "8px",
        border: "1px solid #e0e0e0",
        maxHeight: "calc(100vh - 250px)", // Adjust based on your layout
        overflowY: "auto"
      }}>
        <table className="ticket-table" style={{
          minWidth: showActions ? "1300px" : "1220px", // ✅ Adjust width based on showActions
          width: "100%",
          tableLayout: "fixed"
        }}>
        <thead>
          <tr>
            <th style={{ width: "50px" }}>#</th>
            <th style={{ width: "100px" }}>Preview</th>
            <th style={{ width: "80px" }}>Sub ID</th>

            {currentUser?.role === "ADMIN"
              ? <th style={{ width: "120px" }}>Raised By</th>
              : null}

            <th style={{ width: "150px" }}>Issue Type</th>
            <th style={{ width: "100px" }}>Confidence</th>
            <th style={{ width: "120px" }}>Area</th>
            <th style={{ width: "120px" }}>District</th>
            <th style={{ width: "150px" }}>Date & Time</th>
            <th style={{ width: "180px" }}>Location</th>
            
            {/* ✅ Conditionally show Map column based on user role from code-2 */}
            {currentUser?.role === "USER" && (
              <th style={{ width: "100px" }}>Map</th>
            )}
            
            <th style={{ width: "100px" }}>Status</th>
            
            {/* ✅ Conditionally render Actions column header */}
            {showActions && <th style={{ width: "80px" }}>Actions</th>}
          </tr>
        </thead>

          <tbody>
            {loading ? (
              <tr>
                <td
                  colSpan={
                    currentUser?.role === "ADMIN"
                      ? (showActions ? 12 : 11) // ✅ Dynamic colspan calculation
                      : currentUser?.role === "USER"
                      ? (showActions ? 11 : 10) // ✅ Dynamic colspan calculation
                      : (showActions ? 10 : 9) // ✅ For INSPECTOR or other roles
                  }
                  className="no-data"
                >
                  Loading...
                </td>
              </tr>
            ) : paginatedTickets.length === 0 ? (
              <tr>
                <td
                  colSpan={
                    currentUser?.role === "ADMIN"
                      ? (showActions ? 12 : 11) // ✅ Dynamic colspan calculation
                      : currentUser?.role === "USER"
                      ? (showActions ? 11 : 10) // ✅ Dynamic colspan calculation
                      : (showActions ? 10 : 9) // ✅ For INSPECTOR or other roles
                  }
                  className="no-data"
                >
                  No records found
                </td>
              </tr>
            ) : (
              paginatedTickets.map((ticket, index) => {
                const lat = ticket.latitude;
                const lng = ticket.longitude;
                const imageUrl = ticket.image_url; // ✅ Changed: Use image_url instead of image_id

                return (
                  <tr key={ticket.sub_id || index}>
                    <td>{startIndex + index + 1}</td>

                    <td className="preview-cell" style={{ overflow: "hidden" }}>
                      {imageUrl ? (
                        ticket.media_type === 'video' ? (
                          <div className="preview-video-container" onClick={() => setPreviewImage({ url: imageUrl, type: 'video' })}> {/* ✅ Changed */}
                            <video
                              src={imageUrl} // ✅ Changed
                              className="preview-img clickable"
                              muted
                              playsInline
                              style={{
                                width: "100%",
                                height: "60px",
                                objectFit: "cover",
                                borderRadius: "4px"
                              }}
                            />
                            <div className="play-overlay">▶</div>
                          </div>
                        ) : (
                          <img
                            src={imageUrl} // ✅ Changed
                            alt="Complaint"
                            className="preview-img clickable"
                            title="click here for preview"
                            onClick={() => setPreviewImage({ url: imageUrl, type: 'image' })} // ✅ Changed
                            onError={(e) => {
                              e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Crect fill="%23ddd" width="100" height="100"/%3E%3Ctext fill="%23999" x="50%" y="50%" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E';
                            }}
                            style={{
                              width: "100%",
                              height: "60px",
                              objectFit: "cover",
                              borderRadius: "4px"
                            }}
                          />
                        )
                      ) : (
                        <div className="preview-img" style={{
                          background: '#ddd',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: '#999',
                          width: "100%",
                          height: "60px",
                          borderRadius: "4px"
                        }}>
                          No Media
                        </div>
                      )}
                    </td>

                    <td>{ticket.sub_id || ticket.ticket_id}</td>
                    {currentUser?.role === "ADMIN" && <td>{ticket.user_name || "Anonymous"}</td>}
                    <td>
                      {ticket.issue_type ?
                        formatDistortionName(normalizeIssueType(ticket.issue_type)) :
                        "-"
                      }
                    </td>
                    <td>
                      {ticket.confidence
                        ? `${(ticket.confidence * 100).toFixed(1)}%`
                        : "-"}
                    </td>
                    <td>{ticket.area || "-"}</td>
                    <td>{ticket.district || "-"}</td>
                    <td>
                      {ticket.created_at
                        ? new Date(ticket.created_at).toLocaleDateString()
                        : "-"}
                      <br />
                      <span style={{ fontSize: "12px", opacity: 0.7 }}>
                        {ticket.created_at
                          ? new Date(ticket.created_at).toLocaleTimeString()
                          : "-"}
                      </span>
                    </td>

                    <td>
                      {lat && lng ? (
                        <span
                          className="location-link"
                          title="view location on google maps"
                          onClick={() => handleLocationClick(ticket.sub_id || ticket.ticket_id, lat, lng)}
                          style={{
                            cursor: "pointer",
                            color: "#007bff",
                            textDecoration: "underline",
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            display: "block"
                          }}
                        >
                          {lat.toFixed(6)}, {lng.toFixed(6)}
                        </span>
                      ) : (
                        <span className="location-link" style={{ opacity: 0.7, cursor: 'default', whiteSpace: "nowrap" }}>
                          Location not available
                        </span>
                      )}
                    </td>

                    {/* NEW COLUMN FOR INTERNAL MAP NAVIGATION - Conditionally rendered from code-2 */}
                    {currentUser?.role === "USER" && (
                      <td>
                        {lat && lng ? (
                          <button
                            onClick={() => handleInternalMapClick(ticket)}
                            style={{
                              background: "#4CAF50",
                              color: "white",
                              border: "none",
                              padding: "8px 12px",
                              borderRadius: "4px",
                              cursor: "pointer",
                              fontWeight: "bold",
                              fontSize: "14px",
                              width: "100%",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              gap: "6px",
                              transition: "background-color 0.2s"
                            }}
                            onMouseOver={(e) => e.currentTarget.style.backgroundColor = "#45a049"}
                            onMouseOut={(e) => e.currentTarget.style.backgroundColor = "#4CAF50"}
                            title="View on internal map"
                          >
                            <span style={{ fontSize: "16px" }}>🗺️</span>
                            Map
                          </button>
                        ) : (
                          <span style={{
                            padding: "8px 12px",
                            background: "#f0f0f0",
                            color: "#999",
                            border: "none",
                            borderRadius: "4px",
                            display: "inline-block",
                            width: "100%",
                            textAlign: "center",
                            fontSize: "12px"
                          }}>
                            No Map
                          </span>
                        )}
                      </td>
                    )}

                    <td>
                      <span style={{
                        padding: "4px 8px",
                        borderRadius: "4px",
                        fontSize: "12px",
                        fontWeight: "600",
                        backgroundColor: 
                          ticket.status === "resolved" ? "#d4edda" :
                          ticket.status === "in_progress" ? "#fff3cd" :
                          ticket.status === "closed" ? "#d1ecf1" : "#f8d7da",
                        color: 
                          ticket.status === "resolved" ? "#155724" :
                          ticket.status === "in_progress" ? "#856404" :
                          ticket.status === "closed" ? "#0c5460" : "#721c24",
                        display: "inline-block",
                        minWidth: "80px",
                        textAlign: "center"
                      }}>
                        {ticket.status
                        ? ticket.status.replace("_", " ").toUpperCase()
                        : "-"
                      }
                      </span>
                    </td>
                    
                    {/* ✅ Conditionally render Actions button in table rows */}
                    {showActions && (
                      <td>
                        <button
                          className="delete-btn"
                          onClick={() => handleDeleteClick(ticket.ticket_id)}
                          title="Delete Ticket"
                          style={{
                            background: "#ff4d4d",
                            color: "white",
                            border: "none",
                            padding: "6px 10px",
                            borderRadius: "4px",
                            cursor: "pointer",
                            fontWeight: "bold",
                            fontSize: "14px",
                            width: "100%"
                          }}
                        >
                          ❌
                        </button>
                      </td>
                    )}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* ===============================
          PAGINATION
      =============================== */}
      {totalPages > 1 && (
        <div className="pagination" style={{
          marginTop: "20px",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: "8px",
          flexWrap: "wrap",
          overflowX: "auto",
          padding: "10px 0"
        }}>
          <button
            disabled={currentPage === 1}
            onClick={() => setCurrentPage((p) => p - 1)}
            style={{
              padding: "8px 16px",
              border: "1px solid #ddd",
              background: currentPage === 1 ? "#f5f5f5" : "#fff",
              color: currentPage === 1 ? "#999" : "#333",
              cursor: currentPage === 1 ? "not-allowed" : "pointer",
              borderRadius: "4px",
              minWidth: "80px"
            }}
          >
            Prev
          </button>

          {Array.from({ length: totalPages }, (_, i) => (
            <button
              key={i + 1}
              className={currentPage === i + 1 ? "active" : ""}
              onClick={() => setCurrentPage(i + 1)}
              style={{
                padding: "8px 12px",
                border: "1px solid #ddd",
                background: currentPage === i + 1 ? "#007bff" : "#fff",
                color: currentPage === i + 1 ? "#fff" : "#333",
                cursor: "pointer",
                borderRadius: "4px",
                minWidth: "40px"
              }}
            >
              {i + 1}
            </button>
          ))}

          <button
            disabled={currentPage === totalPages}
            onClick={() => setCurrentPage((p) => p + 1)}
            style={{
              padding: "8px 16px",
              border: "1px solid #ddd",
              background: currentPage === totalPages ? "#f5f5f5" : "#fff",
              color: currentPage === totalPages ? "#999" : "#333",
              cursor: currentPage === totalPages ? "not-allowed" : "pointer",
              borderRadius: "4px",
              minWidth: "80px"
            }}
          >
            Next
          </button>
        </div>
      )}
      {/* ===============================
          IMAGE MODAL PREVIEW
      =============================== */}
      {previewImage && (
        <div className="modal-overlay" onClick={() => setPreviewImage(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{
            maxHeight: "90vh",
            overflowY: "auto",
            maxWidth: "90vw"
          }}>
            <button className="modal-close" onClick={() => setPreviewImage(null)}>
              &times;
            </button>
            {previewImage.type === 'video' ? (
              <video 
                src={previewImage.url} 
                controls 
                autoPlay 
                className="modal-img" 
                style={{ maxHeight: "80vh", maxWidth: "100%" }}
              />
            ) : (
              <img 
                src={previewImage.url} 
                alt="Enlarged preview" 
                className="modal-img" 
                style={{ maxHeight: "80vh", maxWidth: "100%" }}
              />
            )}
          </div>
        </div>
      )}

      {/* ===============================
          DELETE CONFIRMATION POPUP (from code-2)
      =============================== */}
      {showDeletePopup && (
        <div className="modal-overlay" style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "rgba(0, 0, 0, 0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000
        }}>
          <div className="modal-content" style={{
            maxWidth: "400px",
            padding: "24px",
            borderRadius: "12px",
            boxShadow: "0 8px 32px rgba(0, 0, 0, 0.15)",
            background: "#fff",
            overflowY: "auto",
            maxHeight: "90vh"
          }}>
            <h3 style={{
              marginBottom: "16px",
              textAlign: "center",
              fontSize: "20px",
              fontWeight: "600",
              color: "#333"
            }}>
              Confirm Delete
            </h3>

            <p style={{
              marginBottom: "24px",
              textAlign: "center",
              fontSize: "15px",
              lineHeight: "1.5",
              color: "#555",
              wordBreak: "break-word"
            }}>
              Are you sure you want to delete ticket <strong style={{ color: "#333" }}>{ticketToDelete}</strong>?
            </p>

            <div style={{
              display: "flex",
              justifyContent: "center",
              gap: "16px",
              marginTop: "8px"
            }}>
              <button
                onClick={() => {
                  setShowDeletePopup(false);
                  setTicketToDelete(null);
                }}
                style={{
                  padding: "10px 24px",
                  background: "#f0f0f0",
                  color: "#555",
                  border: "1px solid #ddd",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontWeight: "500",
                  fontSize: "14px",
                  minWidth: "100px",
                  transition: "all 0.2s ease"
                }}
                onMouseOver={(e) => e.currentTarget.style.background = "#e5e5e5"}
                onMouseOut={(e) => e.currentTarget.style.background = "#f0f0f0"}
              >
                Cancel
              </button>

              <button
                onClick={handleDelete}
                style={{
                  padding: "10px 24px",
                  background: "#ff4d4d",
                  color: "#fff",
                  border: "none",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontWeight: "600",
                  fontSize: "14px",
                  minWidth: "100px",
                  transition: "all 0.2s ease"
                }}
                onMouseOver={(e) => e.currentTarget.style.background = "#ff3333"}
                onMouseOut={(e) => e.currentTarget.style.background = "#ff4d4d"}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ===============================
          SUCCESS/ERROR POPUP MODAL
      =============================== */}
      {popup.show && (
        <div className="modal-overlay" style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "rgba(0, 0, 0, 0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000
        }}>
          <div
            className="modal-content"
            style={{
              maxWidth: "360px",
              padding: "24px",
              borderRadius: "12px",
              textAlign: "center",
              boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
              background: "#fff"
            }}
          >
            <h3
              style={{
                marginBottom: "12px",
                color: popup.type === "success" ? "#2e7d32" : "#d32f2f",
              }}
            >
              {popup.type === "success" ? "Success" : "Error"}
            </h3>

            <p style={{ marginBottom: "20px", color: "#555" }}>
              {popup.message}
            </p>

            <button
              onClick={() => setPopup({ ...popup, show: false })}
              style={{
                padding: "10px 24px",
                background:
                  popup.type === "success" ? "#2e7d32" : "#d32f2f",
                color: "#fff",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                fontWeight: "600",
              }}
            >
              OK
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default TicketLog;