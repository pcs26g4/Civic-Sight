import { useEffect, useMemo, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import "../styles/AnalysePage.css";
import {
  updateTicketLocation,
  geocodeLocation
} from "../../services/api";

function AnalysePage() {
  const navigate = useNavigate();
  const location = useLocation();

  // Files received from ComplaintPage (after backend upload)
  const uploadedFiles = location.state?.uploadedFiles || [];
  const uploadResponse = location.state?.uploadResponse || null;

  const [showPopup, setShowPopup] = useState(false);
  const [showFailedPopup, setShowFailedPopup] = useState(false);
  const [manualCoords, setManualCoords] = useState({});
  const [manualDetails, setManualDetails] = useState({});
  const [updating, setUpdating] = useState(false);

  // Resolution time mapping based on issue type
  const getResolutionTimeByIssueType = (issueType) => {
    const type = (issueType || "").toLowerCase().trim();
    
    if (type.includes("garbage") || type.includes("overflow")) {
      return "6 hours";
    } else if (type.includes("pothole")) {
      return "24 hours";
    } else if (type.includes("street") && type.includes("debris")) {
      return "6 hours";
    }
    return "Standard (3 days)";
  };

  // Get badge color based on resolution time
  const getResolutionBadgeClass = (resolutionTime) => {
    if (resolutionTime === "6 hours") {
      return "resolution-urgent";
    } else if (resolutionTime === "24 hours") {
      return "resolution-high";
    } else if (resolutionTime === "Standard (3 days)") {
      return "resolution-standard";
    }
    return "resolution-other";
  };

  // Format confidence score as percentage
  const formatConfidence = (confidence) => {
    if (confidence === undefined || confidence === null) return "-";
    
    // If confidence is already a number between 0-1, convert to percentage
    if (typeof confidence === 'number' && confidence <= 1) {
      return `${(confidence * 100).toFixed(1)}%`;
    }
    
    // If it's already a percentage string, return as-is
    if (typeof confidence === 'string' && confidence.includes('%')) {
      return confidence;
    }
    
    // Try to convert to number and format
    const num = parseFloat(confidence);
    if (!isNaN(num)) {
      return num <= 1 ? `${(num * 100).toFixed(1)}%` : `${num.toFixed(1)}%`;
    }
    
    return `${confidence}`;
  };

  // Get confidence badge class based on confidence level
  const getConfidenceBadgeClass = (confidence) => {
    if (confidence === undefined || confidence === null) return "confidence-low";
    
    const num = typeof confidence === 'number' ? confidence : parseFloat(confidence);
    
    if (isNaN(num)) return "confidence-low";
    
    // If confidence is already between 0-1, use as-is, otherwise convert from percentage
    const confidenceValue = num <= 1 ? num : num / 100;
    
    if (confidenceValue >= 0.9) return "confidence-high";
    if (confidenceValue >= 0.7) return "confidence-medium";
    return "confidence-low";
  };

  // CREATE PREVIEW DATA FROM BACKEND RESPONSE
  const previewFiles = useMemo(() => {
    // If we have a backend response, use it to show the ACTUAL results
    if (uploadResponse?.tickets_created) {
      const items = [];

      uploadResponse.tickets_created.forEach(ticket => {
        ticket.sub_tickets.forEach(sub => {
          // Calculate resolution time based on issue type
          const resolutionTime = getResolutionTimeByIssueType(sub.issue_type);
          
          // Add successfully saved images
          if (sub.images && sub.images.length > 0) {
            sub.images.forEach(img => {
              items.push({
                id: img.id,
                ticket_id: ticket.ticket_id,
                sub_id: sub.sub_id,
                issue_type: sub.issue_type,
                authority: sub.authority,
                status: "open",
                latitude: sub.latitude || ticket.latitude,
                longitude: sub.longitude || ticket.longitude,
                area: ticket.area,
                district: ticket.district,
                media_count: sub.media_count,
                type: img.media_type,
                confidence: img.confidence,
                confidence_formatted: formatConfidence(img.confidence),
                confidence_badge_class: getConfidenceBadgeClass(img.confidence),
                previewUrl: img.image_url, // ✅ CHANGED: Use Cloudinary URL directly
                isRejected: false,
                resolution_time: resolutionTime,
                resolution_badge_class: getResolutionBadgeClass(resolutionTime)
              });
            });
          }

          // Add rejected items for visibility
          if (sub.rejected_items && sub.rejected_items.length > 0) {
            sub.rejected_items.forEach((rej, idx) => {
              items.push({
                id: `rej-${sub.sub_id}-${idx}`,
                ticket_id: ticket.ticket_id,
                sub_id: sub.sub_id,
                issue_type: sub.issue_type,
                authority: sub.authority,
                status: "REJECTED",
                latitude: rej.latitude,
                longitude: rej.longitude,
                media_count: sub.media_count,
                type: rej.media_type || "image",
                confidence: rej.confidence || null,
                confidence_formatted: formatConfidence(rej.confidence),
                confidence_badge_class: getConfidenceBadgeClass(rej.confidence),
                previewUrl: null,
                isRejected: true,
                rejectionReason: rej.message,
                fileName: rej.file_name,
                resolution_time: "N/A",
                resolution_badge_class: "resolution-na"
              });
            });
          }
        });

        // Add rejected items for visibility (ticket level - items with NO valid issue detected)
        if (ticket.rejected_items && ticket.rejected_items.length > 0) {
          ticket.rejected_items.forEach((rej, idx) => {
            items.push({
              id: `rej-ticket-${ticket.ticket_id}-${idx}`,
              ticket_id: ticket.ticket_id,
              sub_id: "N/A",
              issue_type: "Unknown",
              authority: "Unclassified",
              status: "REJECTED",
              latitude: rej.latitude,
              longitude: rej.longitude,
              media_count: 0,
              type: rej.media_type || "image",
              confidence: rej.confidence || null,
              confidence_formatted: formatConfidence(rej.confidence),
              confidence_badge_class: getConfidenceBadgeClass(rej.confidence),
              previewUrl: null,
              isRejected: true,
              rejectionReason: rej.message,
              fileName: rej.file_name,
              resolution_time: "N/A",
              resolution_badge_class: "resolution-na"
            });
          });
        }
      });

      return items;
    }

    // Fallback to local files (mostly for initial state or error handling)
    return uploadedFiles.map((item) => {
      const resolutionTime = getResolutionTimeByIssueType(item.issue_type);
      return {
        ...item,
        confidence_formatted: formatConfidence(item.confidence),
        confidence_badge_class: getConfidenceBadgeClass(item.confidence),
        previewUrl: item.previewUrl || (item.file ? URL.createObjectURL(item.file) : null),
        resolution_time: resolutionTime,
        resolution_badge_class: getResolutionBadgeClass(resolutionTime)
      };
    });
  }, [uploadedFiles, uploadResponse]);

  // Function to check if any valid ticket exists
  const hasValidTickets = () => {
    // Check if there's at least one non-rejected file with a valid ticket_id
    return previewFiles.some(file => 
      !file.isRejected && 
      file.ticket_id && 
      file.ticket_id !== "N/A" && 
      file.ticket_id !== "-"
    );
  };

  // ===============================
  // CLEANUP BLOB URLS (WITH CLOUDINARY GUARD)
  // ===============================
  useEffect(() => {
    return () => {
      previewFiles.forEach((item) => {
        // ✅ CHANGED: Only revoke blob URLs, NOT Cloudinary URLs
        if (item.previewUrl && item.previewUrl.startsWith("blob:")) {
          URL.revokeObjectURL(item.previewUrl);
        }
      });
    };
  }, [previewFiles]);

  // ===============================
  // HANDLERS
  // ===============================
  const handleOkClick = () => {
    setShowPopup(false);
    // After successful submission, go to Ticket Log so user can see updated tickets
    navigate("/tickets");
  };

  const handleFailedOkClick = () => {
    setShowFailedPopup(false);
    // Navigate to ticket log to show existing tickets
    navigate("/tickets");
  };

  const handleCoordChange = (ticketId, field, value) => {
    setManualCoords((prev) => ({
      ...prev,
      [ticketId]: {
        ...(prev[ticketId] || { lat: 0, lng: 0 }),
        [field]: value,
      },
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // First check if we have valid tickets
    if (!hasValidTickets()) {
      setShowFailedPopup(true);
      return;
    }
    
    setUpdating(true);

    try {
      // Find all tickets that have manual coordinate overrides
      const ticketsToUpdate = Object.keys(manualCoords);

      for (const tId of ticketsToUpdate) {
        const coords = manualCoords[tId];
        if (coords.lat !== undefined && coords.lng !== undefined) {
          await updateTicketLocation(tId, parseFloat(coords.lat), parseFloat(coords.lng));
        }
      }

      setShowPopup(true);
    } catch (err) {
      alert("Failed to update coordinates: " + err.message);
    } finally {
      setUpdating(false);
    }
  };

  // Fetch area/district when manual coordinates change
  useEffect(() => {
    const fetchManualDetails = async () => {
      const ticketIds = Object.keys(manualCoords);
      for (const id of ticketIds) {
        const { lat, lng } = manualCoords[id];
        if (lat && lng && !manualDetails[id]?.loading && (manualDetails[id]?.lat !== lat || manualDetails[id]?.lng !== lng)) {
          setManualDetails(prev => ({ ...prev, [id]: { loading: true, lat, lng } }));
          try {
            const data = await geocodeLocation(lat, lng);
            setManualDetails(prev => ({
              ...prev,
              [id]: {
                area: data.area,
                district: data.district,
                lat,
                lng,
                loading: false
              }
            }));
          } catch (error) {
            console.error("Geocoding manual entry failed:", error);
            setManualDetails(prev => ({ ...prev, [id]: { loading: false, lat, lng } }));
          }
        }
      }
    };

    const timer = setTimeout(fetchManualDetails, 1000); // Debounce
    return () => clearTimeout(timer);
  }, [manualCoords]);

  const handleLocationClick = async (ticketId, originalLat, originalLng) => {
    const manual = manualCoords[ticketId];
    const lat = (manual?.lat !== undefined && manual?.lat !== "") ? manual.lat : originalLat;
    const lng = (manual?.lng !== undefined && manual?.lng !== "") ? manual.lng : originalLng;

    if (
      lat === null ||
      lng === null ||
      lat === undefined ||
      lng === undefined ||
      Number.isNaN(Number(lat)) ||
      Number.isNaN(Number(lng)) ||
      (Number(lat) === 0 && Number(lng) === 0)
    ) {
      alert("Location not available. Please ensure coordinates are available.");
      return;
    }

    // Auto-save manual location if changed before going to map
    if (manual?.lat !== undefined || manual?.lng !== undefined) {
      try {
        setUpdating(true);
        await updateTicketLocation(ticketId, parseFloat(lat), parseFloat(lng));
      } catch (err) {
        console.error("Auto-save failed:", err);
      } finally {
        setUpdating(false);
      }
    }

    // Navigate to internal map with state
    navigate("/map", {
      state: {
        lat: Number(lat),
        lng: Number(lng),
        label: `Ticket: ${ticketId}`
      }
    });
  };

  // ===============================
  // UI
  // ===============================
  return (
    <div className="analyze-container">
      <h2 className="analyze-title">Complaint</h2>

      <p className="analyze-subtitle">
        Images and videos will be analyzed using a Deep Learning model.
      </p>

      {previewFiles.length === 0 ? (
        <p style={{ color: "black" }}>
          No new complaint was created. 
          The uploaded image/video was either already reported earlier or no valid issue was detected.
          Thank you for your concern 🙏
        </p>
      ) : (
        <div className="table-wrapper">
          <table className="analyze-table">
            <thead>
              <tr>
                <th>Preview</th>
                <th>Details</th>
              </tr>
            </thead>

            <tbody>
              {previewFiles.map((file, index) => {
                const distortionCount = file.distortionCount ?? 1;
                const shouldShowSubId =
                  previewFiles.length > 1 || distortionCount > 1;

                return (
                  <tr key={file.id || index} className={file.isRejected ? "rejected-row" : ""}>
                    <td>
                      {file.previewUrl ? (
                        file.type === "image" ? (
                          <img
                            src={file.previewUrl}
                            alt="preview"
                            className="preview-img"
                          />
                        ) : (
                          <video
                            src={file.previewUrl}
                            controls
                            className="preview-video"
                          />
                        )
                      ) : (
                        <div className="no-preview">
                          {file.isRejected ? "⚠️ Rejected" : "No Preview"}
                        </div>
                      )}
                    </td>

                    <td>
                      <div className="details-box">
                        {file.isRejected && (
                          <div className="rejection-info">
                            <strong>Status: </strong> <span className="status-rejected">REJECTED</span>
                            <p className="rejection-reason">{file.rejectionReason}</p>
                            <small>File: {file.fileName}</small>
                          </div>
                        )}

                        {!file.isRejected && (
                          <>
                            <div>
                              <strong>Issue Type: </strong> {file.issue_type ?? "-"}
                            </div>

                            <div>
                              <strong>Authority: </strong> {file.authority ?? "-"}
                            </div>

                            <div>
                              <strong>Status: </strong> {file.status ?? "-"}
                            </div>

                            <div>
                              <strong>Ticket ID: </strong> {file.ticket_id ?? "-"}
                            </div>

                            {shouldShowSubId && (
                              <div>
                                <strong>Sub ID: </strong>{" "}
                                {file.sub_id ?? `SUB-${index + 1}`}
                              </div>
                            )}

                            <div>
                              <strong>Latitude: </strong> {(manualCoords[file.ticket_id]?.lat !== undefined && manualCoords[file.ticket_id]?.lat !== "") ? manualCoords[file.ticket_id].lat : (file.latitude || "-")}
                            </div>

                            <div>
                              <strong>Longitude: </strong> {(manualCoords[file.ticket_id]?.lng !== undefined && manualCoords[file.ticket_id]?.lng !== "") ? manualCoords[file.ticket_id].lng : (file.longitude || "-")}
                            </div>

                            <div>
                              <strong>Area: </strong> {manualDetails[file.ticket_id]?.loading ? "Loading..." : (manualDetails[file.ticket_id]?.area || file.area || "-")}
                            </div>

                            <div>
                              <strong>District: </strong> {manualDetails[file.ticket_id]?.loading ? "Loading..." : (manualDetails[file.ticket_id]?.district || file.district || "-")}
                            </div>

                            <div>
                              <strong>Media Count: </strong> {file.media_count ?? 0}
                            </div>

                            <div>
                              <strong>Location: </strong>{" "}
                              <span
                                className="location-link"
                                onClick={() =>
                                  handleLocationClick(file.ticket_id, file.latitude, file.longitude)
                                }
                              >
                                📍 View on Map
                              </span>
                            </div>

                            {/* Confidence Score Display */}
                            <div className="confidence-box">
                              <strong>Confidence Score: </strong>
                              <span className={`confidence-badge ${file.confidence_badge_class}`}>
                                {file.confidence_formatted}
                              </span>
                            </div>

                            {/* Resolution Time Field */}
                            <div className="resolution-time-box">
                              <strong>Resolution Time: </strong>
                              <span className={`resolution-time-badge ${file.resolution_badge_class}`}>
                                {file.resolution_time}
                              </span>
                            </div>
                          </>
                        )}

                        <div>
                          <strong>File Type: </strong>{" "}
                          {file.type?.toUpperCase() ?? "IMAGE"}
                        </div>

                        <div className="manual-gps-box">
                          <strong>{(file.latitude === 0 || file.longitude === 0) ? "Add Location:" : "Correct GPS:"}</strong>
                          <div className="manual-gps-inputs">
                            <input
                              type="number"
                              step="any"
                              placeholder="Lat"
                              value={manualCoords[file.ticket_id]?.lat ?? ""}
                              onChange={(e) => handleCoordChange(file.ticket_id, 'lat', e.target.value)}
                            />
                            <input
                              type="number"
                              step="any"
                              placeholder="Lng"
                              value={manualCoords[file.ticket_id]?.lng ?? ""}
                              onChange={(e) => handleCoordChange(file.ticket_id, 'lng', e.target.value)}
                            />
                          </div>
                        </div>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="submit-section">
        <button
          className="submit-btn"
          onClick={handleSubmit}
          disabled={updating}
        >
          {updating ? "Registering..." : "Register Complaint"}
        </button>
      </div>

      {/* Success Popup */}
      {showPopup && (
        <div className="popup-overlay">
          <div className="popup-card success-popup">
            <div className="popup-header success">
              <div className="popup-icon success-icon">✅</div>
              <h3 className="popup-title success-title">Registered Successfully</h3>
            </div>

            <div className="popup-message">
              {uploadResponse?.message && (
                <p className="response-message">
                  {uploadResponse.message}
                </p>
              )}

              {previewFiles.length > 0 && (
                <p className="ticket-info">
                  <strong>Ticket ID:</strong>{" "}
                  <span className="ticket-id">{previewFiles[0]?.ticket_id || "-"}</span>
                </p>
              )}

              {(previewFiles.length > 1 || previewFiles.some((f) => f.sub_id)) && (
                <div className="sub-tickets-section">
                  <p className="sub-tickets-title">
                    <strong>Sub Tickets:</strong>
                  </p>
                  <ul className="sub-tickets-list">
                    {previewFiles.map((file, index) => (
                      <li key={index} className="sub-ticket-item">
                        <span className="sub-ticket-id">{file.sub_id || `SUB-${index + 1}`}</span> - 
                        <span className="sub-ticket-type">{file.issue_type}</span> - 
                        Confidence: <strong className={`confidence-badge ${file.confidence_badge_class}`}>
                          {file.confidence_formatted}
                        </strong> - 
                        Resolution: <strong className={`resolution-badge ${file.resolution_badge_class}`}>
                          {file.resolution_time}
                        </strong>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="popup-actions">
              <button className="popup-btn success-btn" onClick={handleOkClick}>
                OK
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Registration Failed Popup */}
      {showFailedPopup && (
        <div className="popup-overlay">
          <div className="popup-card failed-popup">
            <div className="popup-header failed">
              <div className="popup-icon failed-icon">❌</div>
              <h3 className="popup-title failed-title">Registration Failed</h3>
            </div>

            <div className="popup-message">
              <p className="popup-subtitle">
                No valid tickets were created. All items were either rejected or failed to generate a ticket.
              </p>
              <p className="popup-subtitle" style={{ marginTop: '10px', fontSize: '14px' }}>
                You will be redirected to view existing tickets.
              </p>
            </div>

            <div className="issues-section">
              <div className="issues-header">
                <span className="issues-title">Issues Detected</span>
                <span className="issues-count">
                  {previewFiles.filter(f => f.isRejected || !f.ticket_id || f.ticket_id === "N/A" || f.ticket_id === "-").length} issue(s)
                </span>
              </div>
              
              <div className="issues-list">
                {previewFiles.map((file, index) => (
                  (file.isRejected || !file.ticket_id || file.ticket_id === "N/A" || file.ticket_id === "-") && (
                    <div key={index} className="issue-item">
                      <div className="issue-header">
                        <span className="issue-file-name">
                          {file.fileName || `File ${index + 1}`}
                        </span>
                        {file.confidence_formatted && file.confidence_formatted !== "-" && (
                          <span className={`confidence-badge ${file.confidence_badge_class}`}>
                            Confidence: {file.confidence_formatted}
                          </span>
                        )}
                        <span className={`issue-status ${file.isRejected ? 'status-rejected' : 'status-failed'}`}>
                          {file.isRejected ? 'Rejected' : 'Failed'}
                        </span>
                      </div>
                      {file.isRejected && file.rejectionReason && (
                        <div className="issue-reason">
                          <span className="reason-label">Reason: </span>
                          {file.rejectionReason}
                        </div>
                      )}
                      {!file.isRejected && (!file.ticket_id || file.ticket_id === "N/A" || file.ticket_id === "-") && (
                        <div className="issue-reason">
                          <span className="reason-label">Reason: </span>
                          No ticket could be generated for this file
                        </div>
                      )}
                    </div>
                  )
                ))}
              </div>
            </div>

            <div className="popup-actions">
              <button 
                className="popup-btn failed-btn"
                onClick={handleFailedOkClick}
              >
                OK
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AnalysePage;