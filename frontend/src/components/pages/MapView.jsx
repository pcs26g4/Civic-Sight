import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect, useState, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import "../styles/MapView.css";
import { getTickets } from "../../services/api";
import L from "leaflet";

/* Fix for default marker icon */
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
});

/* -------------------- REAL MARKER ICONS (Using CDN URLs) -------------------- */
const RED_MARKER_URL = "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png";
const BLUE_MARKER_URL = "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png";
const GREEN_MARKER_URL = "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png";
const ORANGE_MARKER_URL = "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-orange.png";

const blueIcon = L.icon({
  iconUrl: BLUE_MARKER_URL,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  shadowSize: [41, 41]
});

const redIcon = L.icon({
  iconUrl: RED_MARKER_URL,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  shadowSize: [41, 41]
});

const greenIcon = L.icon({
  iconUrl: GREEN_MARKER_URL,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  shadowSize: [41, 41]
});

const orangeIcon = L.icon({
  iconUrl: ORANGE_MARKER_URL,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  shadowSize: [41, 41]
});

const getMarkerIcon = (status = "open") => {
  const s = (status || "open").toLowerCase();
  if (s === "resolved" || s === "closed") return redIcon;
  if (s === "in_progress") return orangeIcon;
  if (s === "assigned") return greenIcon;
  return blueIcon; // default for open
};

/* -------------------- FORMAT ISSUE TYPE FOR DISPLAY -------------------- */
const formatIssueTypeForDisplay = (issueType) => {
  if (!issueType) return "Not specified";

  return issueType
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

/* -------------------- MAP UPDATER COMPONENT -------------------- */
// This component handles map view updates based on selected ticket
function MapUpdater({ selectedTicket, ticketsReady }) {
  const map = useMap();
  const hasMovedRef = useRef(false);

  useEffect(() => {
    if (selectedTicket && selectedTicket.latitude && selectedTicket.longitude && ticketsReady) {
      // We allow moving multiple times if the selected ticket changes, but prevent loops
      // Simple check: if we are already close, maybe don't move? 
      // But for now, we trust the parent to only update selectedTicket when meaningful.
      console.log("📍 Zooming to ticket:", selectedTicket.ticketId);
      map.flyTo(
        [selectedTicket.latitude, selectedTicket.longitude],
        18, // High zoom level
        {
          animate: true,
          duration: 1.5
        }
      );
    }
  }, [selectedTicket, map, ticketsReady]);

  return null;
}

/* -------------------- CONSTANTS -------------------- */
const DEFAULT_CENTER = [16.303771, 80.435537];
const DEFAULT_ZOOM = 16;

/* -------------------- MAP VIEW -------------------- */
export default function MapView() {
  const [tickets, setTickets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  /* State for selected ticket from navigation */
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedTicket, setSelectedTicket] = useState(null);
  const markerRefs = useRef({});

  /* -------------------- INIT STATE -------------------- */
  useEffect(() => {
    // Check location state first
    if (location.state?.selectedTicket) {
      console.log("📥 Received ticket from navigation state:", location.state.selectedTicket);
      setSelectedTicket(location.state.selectedTicket);

      // Clear the state from history so it doesn't persist on reload
      navigate(location.pathname, { replace: true, state: {} });
    }
  }, []);

  /* -------------------- FETCH ALL TICKETS -------------------- */
  useEffect(() => {
    const fetchAllTickets = async () => {
      try {
        setLoading(true);
        setError(null);
        console.log("🔄 Fetching ALL tickets for map...");

        // Get ALL tickets without filters - SIMILAR TO TICKETLOG
        const response = await getTickets({});

        if (!response || !response.tickets) {
          console.warn("⚠️ No tickets data in response");
          setTickets([]);
          return;
        }

        const allMarkers = [];
        const ticketsData = response.tickets || [];
        const seenLocations = new Set();

        // PROCESS EXACTLY LIKE TICKETLOG
        ticketsData.forEach(ticket => {
          const subTickets = ticket.sub_tickets || [];

          // Process sub-tickets ONLY (like TicketLog does)
          subTickets.forEach(subTicket => {
            // EXTRACT COORDINATES - EXACTLY LIKE TICKETLOG
            const lat = subTicket.latitude || ticket.latitude;
            const lng = subTicket.longitude || ticket.longitude;

            if (lat && lng) {
              const parsedLat = parseFloat(lat);
              const parsedLng = parseFloat(lng);

              if (!isNaN(parsedLat) && !isNaN(parsedLng)) {
                // Modifying checking logic to allow same location if different sub_id
                const uniqueKey = `${ticket.ticket_id}_${subTicket.sub_id}`;

                if (!seenLocations.has(uniqueKey)) {
                  seenLocations.add(uniqueKey);

                  // CREATE MARKER EXACTLY LIKE TICKETLOG
                  const marker = {
                    ticket_id: ticket.ticket_id,
                    sub_id: subTicket.sub_id,
                    latitude: parsedLat,
                    longitude: parsedLng,
                    // IMPORTANT: Get issue_type from subTicket (like TicketLog does)
                    issue_type: subTicket.issue_type, // Direct access like TicketLog
                    status: subTicket.status || ticket.status || 'open',
                    area: ticket.area || subTicket.area,
                    district: ticket.district || subTicket.district,
                    // IMPORTANT: Get confidence from subTicket (like TicketLog does)
                    confidence: subTicket.confidence || ticket.confidence,
                    created_at: subTicket.created_at || ticket.created_at,
                    user_name: ticket.user_name,
                    all_tickets_at_location: [ticket.ticket_id],
                    unique_id: uniqueKey // Add unique ID for ref lookup
                  };

                  allMarkers.push(marker);
                }
              }
            }
          });
        });

        console.log(`\n📍 TOTAL UNIQUE MARKERS: ${allMarkers.length}`);
        setTickets(allMarkers);

        if (allMarkers.length === 0) {
          setError("No complaints found with location data");
        }

      } catch (err) {
        console.error("❌ Error fetching tickets:", err);
        setError("Failed to load complaints. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    fetchAllTickets();

    const interval = setInterval(fetchAllTickets, 30000);
    return () => clearInterval(interval);
  }, []);

  /* -------------------- OPEN POPUP EFFECT -------------------- */
  useEffect(() => {
    if (selectedTicket && !loading && tickets.length > 0) {
      // Small timeout to ensure markers are rendered and refs attached
      const timer = setTimeout(() => {
        let targetId = null;

        // Exact match construction: TicketID_SubID
        if (selectedTicket.selectedTicket) {
          const raw = selectedTicket.selectedTicket;
          targetId = `${raw.ticket_id}_${raw.sub_id}`;
        }

        console.log("🎯 Attempting to open popup for:", targetId);

        let marker = markerRefs.current[targetId];

        // Fallback 1: Try finding by ID directly if simple construction failed
        if (!marker && selectedTicket.ticketId) {
          const idToFind = selectedTicket.ticketId;
          // Search in tickets array to find the matching unique_id
          const matchingTicket = tickets.find(t => t.sub_id === idToFind || t.ticket_id === idToFind);
          if (matchingTicket) {
            console.log("✅ Found marker by ID search:", matchingTicket.unique_id);
            marker = markerRefs.current[matchingTicket.unique_id];
          }
        }

        // Fallback 2: Match by coordinates if ID fails
        if (!marker && selectedTicket.latitude && selectedTicket.longitude) {
          const lat = parseFloat(selectedTicket.latitude).toFixed(6);
          const lng = parseFloat(selectedTicket.longitude).toFixed(6);
          const foundByLoc = tickets.find(t =>
            t.latitude.toFixed(6) === lat && t.longitude.toFixed(6) === lng
          );
          if (foundByLoc) {
            console.log("✅ Found fallback by location:", foundByLoc.unique_id);
            marker = markerRefs.current[foundByLoc.unique_id];
          }
        }

        if (marker) {
          console.log("✨ Opening popup!");
          marker.openPopup();
        } else {
          console.warn("⚠️ Could not find marker ref for auto-popup");
        }
      }, 800); // Increased timeout slightly to ensure rendering

      return () => clearTimeout(timer);
    }
  }, [loading, tickets, selectedTicket]);

  /* -------------------- OPEN GOOGLE MAPS -------------------- */
  const openGoogleMaps = (lat, lng, ticketId) => {
    const url = `https://www.google.com/maps?q=${lat},${lng}&z=18`;
    window.open(url, '_blank');
  };

  /* -------------------- STATISTICS -------------------- */
  const stats = {
    total: tickets.length,
    open: tickets.filter(t => (t.status || '').toLowerCase() === 'open').length,
    in_progress: tickets.filter(t => (t.status || '').toLowerCase() === 'in_progress').length,
    resolved: tickets.filter(t => (t.status || '').toLowerCase() === 'resolved').length,
    closed: tickets.filter(t => (t.status || '').toLowerCase() === 'closed').length,
    assigned: tickets.filter(t => (t.status || '').toLowerCase() === 'assigned').length
  };

  /* -------------------- RENDER -------------------- */
  return (
    <div className="map-section">
      <div className="map-header">
        <div className="header-top">
          <h3>📍 Live Complaints Map</h3>
          <button
            onClick={() => window.location.reload()}
            className="refresh-btn"
            disabled={loading}
          >
            {loading ? '🔄 Loading...' : '🔄 Refresh'}
          </button>
        </div>
      </div>

      <div className="map-shell">
        {loading ? (
          <div className="loading-state">
            <div className="loading-spinner"></div>
            <p>Loading complaints from database...</p>
          </div>
        ) : error ? (
          <div className="error-state">
            <div className="error-icon">⚠️</div>
            <h4>{error}</h4>
            <button onClick={() => window.location.reload()} className="retry-btn">
              Retry
            </button>
          </div>
        ) : tickets.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📍</div>
            <h4>No Complaints Found</h4>
            <p>Add complaints with location coordinates to see them on the map.</p>
          </div>
        ) : (
          <>
            <MapContainer
              center={DEFAULT_CENTER}
              zoom={DEFAULT_ZOOM}
              className="map"
              scrollWheelZoom={true}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='© OpenStreetMap contributors'
              />

              <MapUpdater
                selectedTicket={selectedTicket}
                ticketsReady={!loading && tickets.length > 0}
              />

              {/* Render all markers */}
              {tickets.map((ticket, index) => (
                <Marker
                  key={`${ticket.unique_id}-${index}`}
                  ref={(el) => {
                    if (el) markerRefs.current[ticket.unique_id] = el;
                  }}
                  position={[ticket.latitude, ticket.longitude]}
                  icon={getMarkerIcon(ticket.status)}
                >
                  <Popup>
                    <div className="popup-content">
                      <div className="popup-header">
                        <h4>
                          Complaint #{ticket.ticket_id}
                          {ticket.sub_id && ` (Sub: ${ticket.sub_id})`}
                        </h4>
                        <span className={`status-badge ${(ticket.status || 'open').toLowerCase().replace('_', '-')}`}>
                          {ticket.status || 'OPEN'}
                        </span>
                      </div>

                      <div className="popup-issue-type">
                        <strong className="issue-type-label">Issue Type:</strong>
                        <span className="issue-type-value">
                          {formatIssueTypeForDisplay(ticket.issue_type)}
                        </span>
                        {ticket.confidence && (
                          <div className="confidence-badge">
                            Confidence: {(ticket.confidence * 100).toFixed(1)}%
                          </div>
                        )}
                      </div>

                      <div className="popup-details">
                        <div className="detail-row">
                          <span className="label">Location:</span>
                          <span className="value coordinates">
                            {ticket.latitude.toFixed(6)}, {ticket.longitude.toFixed(6)}
                          </span>
                        </div>

                        {ticket.area && (
                          <div className="detail-row">
                            <span className="label">Area:</span>
                            <span className="value">{ticket.area}</span>
                          </div>
                        )}

                        {ticket.district && (
                          <div className="detail-row">
                            <span className="label">District:</span>
                            <span className="value">{ticket.district}</span>
                          </div>
                        )}

                        {ticket.user_name && (
                          <div className="detail-row">
                            <span className="label">Reported by:</span>
                            <span className="value">{ticket.user_name}</span>
                          </div>
                        )}

                        {ticket.created_at && (
                          <div className="detail-row">
                            <span className="label">Created:</span>
                            <span className="value">
                              {new Date(ticket.created_at).toLocaleDateString()}
                            </span>
                          </div>
                        )}
                      </div>

                      <button
                        onClick={() => openGoogleMaps(ticket.latitude, ticket.longitude, ticket.ticket_id)}
                        className="gmaps-btn"
                      >
                        Open in Google Maps
                      </button>
                    </div>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>

            <div className="map-overlay">
              <div className="markers-count">
                <strong>{tickets.length}</strong> locations • <strong>{stats.open + stats.in_progress + stats.assigned}</strong> active
              </div>
            </div>
          </>
        )}

        {/* Legend */}
        <div className="map-legend">
          <h4>Marker Legend</h4>
          <div className="legend-item">
            <div className="legend-icon blue"></div>
            <span>Open</span>
          </div>
          <div className="legend-item">
            <div className="legend-icon orange"></div>
            <span>In Progress</span>
          </div>
          <div className="legend-item">
            <div className="legend-icon green"></div>
            <span>Assigned</span>
          </div>
          <div className="legend-item">
            <div className="legend-icon red"></div>
            <span>Resolved/Closed</span>
          </div>
          <div className="legend-instruction">
            <small>Click marker for details • Click button for Google Maps</small>
          </div>
        </div>
      </div>
    </div>
  );
}