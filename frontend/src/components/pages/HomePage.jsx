import "../styles/HomePage.css";
import { useEffect, useState } from "react";
import { getTickets, getImageUrl } from "../../services/api";

export default function HomePage() {
  const [stats, setStats] = useState({
    total: [],
    resolved: [],
    inProgress: [],
  });
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [previewMedia, setPreviewMedia] = useState(null);
  const [allTickets, setAllTickets] = useState([]);

  const user = JSON.parse(localStorage.getItem("user"));

  useEffect(() => {
    fetchTickets();
  }, []);

  const fetchTickets = async () => {
    const filters_api = {};
    if (user?.role === "USER") filters_api.user_id = user.id;

    const res = await getTickets(filters_api);
    const tickets = res.tickets || [];
    console.log("Fetched tickets:", tickets); // Debug log
    
    // Flatten tickets with sub_tickets similar to TicketLog
    const allSubTickets = [];
    tickets.forEach(ticket => {
      (ticket.sub_tickets || []).forEach(subTicket => {
        // If inspector, only show assigned tickets
        if (user?.role === "INSPECTOR" && subTicket.assigned_to !== user.id) {
          return;
        }

        allSubTickets.push({
          ...subTicket,
          ticket_id: ticket.ticket_id,
          main_ticket_id: ticket.ticket_id,
          // CRITICAL FIX: Add fallback to parent ticket coordinates (same as TicketLog)
          latitude: subTicket.latitude || ticket.latitude,
          longitude: subTicket.longitude || ticket.longitude,
          area: ticket.area,
          district: ticket.district,
          user_name: ticket.user_name,
          created_at: subTicket.created_at || ticket.created_at,
          resolved_at: subTicket.resolved_at,
          media_type: subTicket.media_type,
          image_id: subTicket.image_id,
          confidence: subTicket.confidence,
          issue_type: subTicket.issue_type,
          status: subTicket.status
        });
      });
    });
    
    console.log("All sub tickets:", allSubTickets); // Debug log
    setAllTickets(allSubTickets);
    calculateStats(allSubTickets);
  };

  const calculateStats = (tickets) => {
    const today = new Date();
    const todayStr = today.toISOString().split("T")[0];
    const thisMonthStr = todayStr.slice(0, 7);

    const resolvedTickets = tickets.filter(
      (t) => t.status === "resolved" || t.status === "closed"
    );

    const inProgressTickets = tickets.filter(
      (t) => t.status !== "resolved" && t.status !== "closed"
    );

    const todayCompleted = resolvedTickets.filter(
      (t) => t.resolved_at?.startsWith(todayStr)
    ).length;

    const thisMonthResolved = resolvedTickets.filter(
      (t) => t.resolved_at?.startsWith(thisMonthStr)
    ).length;

    setStats({
      total: tickets,
      resolved: resolvedTickets,
      inProgress: inProgressTickets,
      todayCompleted,
      thisMonth: thisMonthResolved,
    });
  };

  const handleTicketClick = (ticketId) => {
    const clickedTicket = allTickets.find(t => 
      t.ticket_id === ticketId || t.sub_id === ticketId
    );
    
    if (clickedTicket) {
      console.log("Clicked ticket data:", clickedTicket); // Debug log
      console.log("Latitude:", clickedTicket.latitude, "Type:", typeof clickedTicket.latitude);
      console.log("Longitude:", clickedTicket.longitude, "Type:", typeof clickedTicket.longitude);
      
      setSelectedTicket(clickedTicket);
      setShowDetailsModal(true);
    }
  };

  const handleLocationClick = (ticketId, lat, lng) => {
    // Check if coordinates are valid numbers
    if (!lat || !lng || isNaN(lat) || isNaN(lng)) {
      console.log(`Invalid location data for ticket ${ticketId}: ${lat}, ${lng}`);
      alert("Location data is not available or invalid for this ticket");
      return;
    }
    
    // Convert to numbers to ensure proper formatting
    const numLat = Number(lat);
    const numLng = Number(lng);
    
    const url = `https://www.google.com/maps?q=${numLat},${numLng}`;
    window.open(url, "_blank");
  };

  const handleMediaPreview = (ticket) => {
    if (ticket.image_id) {
      setPreviewMedia({
        url: getImageUrl(ticket.image_id),
        type: ticket.media_type === 'video' ? 'video' : 'image'
      });
    }
  };

  const TicketList = ({ items }) => (
    <>
      {items.length > 0 ? (
        items.map((t) => (
          <div key={t.ticket_id || t.sub_id} className="ticket-row">
            <span 
              className="ticket-id clickable"
              onClick={() => handleTicketClick(t.ticket_id || t.sub_id)}
            >
              {t.ticket_id || t.sub_id}
            </span>
            <span className={`badge ${t.issue_type}`}>
              {t.issue_type?.replace(/_/g, " ")}
            </span>
          </div>
        ))
      ) : (
        <p className="muted">No tickets to display</p>
      )}
    </>
  );

  const formatDistortionName = (distortion) => {
    if (!distortion) return "N/A";
    return distortion
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const formatDate = (dateString) => {
    if (!dateString) return "N/A";
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  // Helper function to check if location data is valid
  const isValidLocation = (lat, lng) => {
    if (!lat || !lng) return false;
    const numLat = Number(lat);
    const numLng = Number(lng);
    return !isNaN(numLat) && !isNaN(numLng) && 
           numLat !== 0 && numLng !== 0; // Also check if not 0,0
  };

  return (
    <div className="dashboard">
      <h2 className="welcome">Welcome, {user?.name || "User"}</h2>

      <div className="cards-grid"
        title="click here to view ticket details"

      >
        {/* TOTAL TICKETS CARD - FIXED */}
        <div className="card">
          <h4>TOTAL TICKETS</h4>
          <h1>{stats.total.length}</h1>
          
          <div className="card-scrollable">
            <TicketList items={stats.total} />
          </div>
        </div>

        {/* RESOLVED TICKETS CARD - FIXED */}
        <div className="card">
          <h4>RESOLVED TICKETS</h4>
          <h1>{stats.resolved.length}</h1>
          
          <div className="card-scrollable">
            <TicketList items={stats.resolved} />
          </div>
        </div>

        {/* IN PROGRESS TICKETS CARD - FIXED */}
        <div className="card">
          <h4>IN PROGRESS TICKETS</h4>
          <h1>{stats.inProgress.length}</h1>
          
          <div className="card-scrollable">
            <TicketList items={stats.inProgress} />
          </div>
        </div>


      </div>

      {/* Ticket Details Modal */}
      {showDetailsModal && selectedTicket && (
        <div className="modal-overlay" onClick={() => setShowDetailsModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>
                Ticket Details: {selectedTicket.ticket_id || selectedTicket.sub_id}
              </h2>
            </div>

            <div className="modal-body">
              <div className="details-grid">
                <div className="detail-item">
                  <strong>SUB ID:</strong>
                  <span>{selectedTicket.sub_id || "N/A"}</span>
                </div>
                
                {user?.role === "ADMIN" && (
                  <div className="detail-item">
                    <strong>Raised By:</strong>
                    <span>{selectedTicket.user_name || "Anonymous"}</span>
                  </div>
                )}
                
                <div className="detail-item">
                  <strong>Issue Type:</strong>
                  <span className={`badge ${selectedTicket.issue_type}`}>
                    {formatDistortionName(selectedTicket.issue_type)}
                  </span>
                </div>
                
                <div className="detail-item">
                  <strong>Confidence:</strong>
                  <span>{selectedTicket.confidence ? `${(selectedTicket.confidence * 100).toFixed(1)}%` : "N/A"}</span>
                </div>
                
                <div className="detail-item">
                  <strong>Area:</strong>
                  <span>{selectedTicket.area || "N/A"}</span>
                </div>
                
                <div className="detail-item">
                  <strong>District:</strong>
                  <span>{selectedTicket.district || "N/A"}</span>
                </div>
                
                <div className="detail-item">
                  <strong>Date & Time:</strong>
                  <span>{formatDate(selectedTicket.created_at)}</span>
                </div>
                
                <div className="detail-item">
                  <strong>Location:</strong>
                  <span>
                    {isValidLocation(selectedTicket.latitude, selectedTicket.longitude) ? (
                      <span
                        className="location-link"
                        onClick={() => handleLocationClick(
                          selectedTicket.sub_id || selectedTicket.ticket_id, 
                          selectedTicket.latitude, 
                          selectedTicket.longitude
                        )}
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
                        {Number(selectedTicket.latitude).toFixed(6)}, {Number(selectedTicket.longitude).toFixed(6)}
                      </span>
                    ) : (
                      <span 
                        className="location-link" 
                        style={{ 
                          opacity: 0.7, 
                          cursor: 'default', 
                          whiteSpace: "nowrap",
                          color: "#ff4d4d"  // Red color to indicate missing location
                        }}
                      >
                        Location data not available
                      </span>
                    )}
                  </span>
                </div>
                
                <div className="detail-item">
                  <strong>Status:</strong>
                  <span className={`status-badge ${selectedTicket.status}`}>
                    {selectedTicket.status || "open"}
                  </span>
                </div>
                
                {selectedTicket.resolved_at && (
                  <div className="detail-item">
                    <strong>Resolved At:</strong>
                    <span>{formatDate(selectedTicket.resolved_at)}</span>
                  </div>
                )}
              </div>

              {/* Media Preview Section */}
              {selectedTicket.image_id && (
                <div className="media-section">
                  <h3>Media Preview</h3>
                  <div className="media-container">
                    {selectedTicket.media_type === 'video' ? (
                      <div className="preview-video-container" onClick={() => handleMediaPreview(selectedTicket)}>
                        <video
                          src={getImageUrl(selectedTicket.image_id)}
                          className="media-preview clickable"
                          muted
                          playsInline
                        />
                        <div className="play-overlay">▶</div>
                      </div>
                    ) : (
                      <img 
                        src={getImageUrl(selectedTicket.image_id)}
                        alt="Complaint"
                        className="media-preview clickable"
                        onClick={() => handleMediaPreview(selectedTicket)}
                        onError={(e) => {
                          e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="300" height="200"%3E%3Crect fill="%23ddd" width="300" height="200"/%3E%3Ctext fill="%23999" x="50%" y="50%" text-anchor="middle" dy=".3em"%3ENo Image Available%3C/text%3E%3C/svg%3E';
                        }}
                      />
                    )}
                  </div>
                </div>
              )}
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
    </div>
  );
}