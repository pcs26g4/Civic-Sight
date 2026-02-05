import { useRef, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/ComplaintPage.css";
import { uploadComplaints } from "../../services/api";

function ComplaintPage() {
  const navigate = useNavigate();

  const imageGalleryRef = useRef(null);
  const videoGalleryRef = useRef(null);

  const [images, setImages] = useState([]);
  const [videos, setVideos] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  // Add loading state for animation
  const [showLoadingAnimation, setShowLoadingAnimation] = useState(false);

  const [location, setLocation] = useState({
    lat: null,
    lng: null,
    error: null,
  });

  /* =========================================================
      GEOLOCATION FETCHING
  ========================================================= */
  const fetchLocation = () => {
    if (!("geolocation" in navigator)) {
      setLocation((prev) => ({ ...prev, error: "Geolocation not supported" }));
      return;
    }

    setLocation((prev) => ({ ...prev, error: null, lat: null, lng: null })); // Reset state for retry

    const options = {
      enableHighAccuracy: false,
      timeout: 15000,
      maximumAge: 30000
    };

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
          error: null,
        });
        console.log("Location fetched:", position.coords.latitude, position.coords.longitude);
      },
      (error) => {
        console.error("Error fetching location:", error);
        let errorMsg = error.message;
        if (error.code === error.TIMEOUT) {
          errorMsg = "Request Timed Out. Try moving to an open area or check your system location settings.";
        } else if (error.code === error.PERMISSION_DENIED) {
          errorMsg = "Location access denied. Please allow location permissions in your browser.";
        }
        setLocation((prev) => ({ ...prev, error: errorMsg }));
      },
      options
    );
  };

  useEffect(() => {
    fetchLocation();
  }, []);

  /* =========================================================
      YOLO LIVE CAMERA 
  ========================================================= */

  const [liveYolo, setLiveYolo] = useState(false);
  const [streamUrl, setStreamUrl] = useState("");

  const openCamera = () => {
    // Pass current location to the backend
    const latParam = location.lat ? `&latitude=${location.lat}` : "";
    const lngParam = location.lng ? `&longitude=${location.lng}` : "";

    // Add timestamp to bypass browser image caching for the multipart stream
    setStreamUrl(`http://127.0.0.1:8000/api/yolo/live?t=${Date.now()}${latParam}${lngParam}`);
    setLiveYolo(true);
  };

  /* ===================== LIVE LOGS ===================== */
  const [cameraLogs, setCameraLogs] = useState([]);

  useEffect(() => {
    let eventSource;
    if (liveYolo) {
      eventSource = new EventSource("http://127.0.0.1:8000/api/yolo/events");
      eventSource.onmessage = async (e) => {
        const data = JSON.parse(e.data);
        if (data.heartbeat) return;
        console.log("Live Log Received:", data);
        setCameraLogs((prev) => [data, ...prev].slice(0, 50));

        // detect deviation and auto-stop
        if (data.capture_filename) {
          console.log("Deviation detected, capturing frame:", data.capture_filename);

          try {
            const res = await fetch(`http://127.0.0.1:8000/api/yolo/capture/${data.capture_filename}`);
            if (res.ok) {
              const blob = await res.blob();
              const file = new File([blob], data.capture_filename, { type: "image/jpeg" });
              setImages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  url: URL.createObjectURL(file),
                  file: file,
                  type: "image",
                  source: "camera", // Marked as camera source
                  lat: location.lat,
                  lng: location.lng,
                }
              ]);

              // Don't auto-stop camera - let user continue capturing
              // Only stop if user manually closes
            }
          } catch (err) {
            console.error("Failed to capture live frame:", err);
          }
        }
      };
      eventSource.onerror = () => {
        console.error("SSE Connection failed");
        eventSource.close();
      };
    } else {
      setCameraLogs([]);
    }
    return () => {
      if (eventSource) eventSource.close();
    };
  }, [liveYolo]);

  const closeCamera = async (isManual = true) => {
    try {
      // stop backend camera
      await fetch("http://127.0.0.1:8000/api/yolo/stop");
    } catch (err) {
      console.warn("Backend camera stop failed:", err);
    }

    // Reset URL and logs
    setStreamUrl("");
    setLiveYolo(false);
    setCameraLogs([]);
  };

  /* =========================================================
      IMAGE / VIDEO UPLOAD - RESTRICTED TO ONE EACH
      But camera can capture multiple images
  ========================================================= */

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);

    // Check if user already has uploaded images (not from camera)
    const uploadedImages = images.filter(img => img.source !== "camera");
    if (uploadedImages.length >= 1) {
      alert("You can only upload one image via file upload. Use the Live Camera for multiple images.");
      e.target.value = "";
      return;
    }

    // Check if user selected multiple files
    if (files.length > 1) {
      alert("Please select only one image at a time for file upload. Use the Live Camera for multiple images.");
      e.target.value = "";
      return;
    }

    // Limit to single file upload
    const file = files[0];
    
    const reader = new FileReader();
    reader.onloadend = () => {
      setImages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          url: reader.result, // ✅ MOBILE SAFE
          file: file,         // ✅ backend unchanged
          type: "image",
          source: "upload",
          lat: location.lat,
          lng: location.lng,
        }
      ]);
    };
    reader.readAsDataURL(file);
    
    e.target.value = "";
  };

  const handleVideoUpload = (e) => {
    const files = Array.from(e.target.files);

    // Check if user is trying to upload more than one video
    if (videos.length >= 1) {
      alert("You can only upload one video. Please remove the existing video before uploading a new one.");
      e.target.value = "";
      return;
    }

    // Check if user selected multiple files
    if (files.length > 1) {
      alert("Please select only one video at a time.");
      e.target.value = "";
      return;
    }

    // Limit to single file upload
    const file = files[0];
    setVideos([
      {
        id: crypto.randomUUID(),
        url: URL.createObjectURL(file),
        file: file,
        type: "video",
        lat: location.lat,
        lng: location.lng,
      }
    ]);
    e.target.value = "";
  };

  const resetFiles = () => {
    setImages([]);
    setVideos([]);
  };

  const removeImage = (id) => {
    setImages((prev) => prev.filter((img) => img.id !== id));
  };

  const removeVideo = (id) => {
    setVideos((prev) => prev.filter((vid) => vid.id !== id));
  };

  /* =========================================================
      ANALYZE & UPLOAD 
  ========================================================= */

  const handleAnalyze = async () => {
    if (images.length === 0 && videos.length === 0) {
      alert("Please upload at least one file.");
      return;
    }

    setUploading(true);
    setUploadError(null);
    setShowLoadingAnimation(true); // Show the loading animation

    try {
      const filesToUpload = [
        ...images.map((i) => i.file),
        ...videos.map((v) => v.file),
      ];

      const response = await uploadComplaints(
        filesToUpload,
        location.lat,
        location.lng
      );

      // Keep animation visible for at least 2 seconds for better UX
      setTimeout(() => {
        navigate("/analyse", {
          state: {
            uploadedFiles: images.concat(videos),
            uploadResponse: response,
          },
        });
      }, 2000);
      
    } catch (error) {
      setUploadError(error.message);
      setShowLoadingAnimation(false); // Hide animation on error
    } finally {
      // Don't setUploading(false) immediately as we're navigating
      // The component will unmount anyway
    }
  };

  /* ========================================================= */

  return (
    <div className="complaint-container">

      {/* LOADING ANIMATION OVERLAY */}
      {showLoadingAnimation && (
        <div className="loading-overlay">
          <div className="loading-content">
            <div className="loading-spinner"></div>
            <h3>Analyzing Your Evidence</h3>
            <p>Please wait while we process your images/videos...</p>
            <div className="loading-progress">
              <div className="loading-bar"></div>
            </div>
            <p className="loading-subtext">This may take a few moments depending on file sizes</p>
          </div>
        </div>
      )}

      {/* HEADER */}
      <div className="complaint-header">
        <h2>Municipal Deviation Registration</h2>
        <p>Submit visual evidence for civic issues.</p>
        <p className="upload-limit-note">
          <strong>Note:</strong> Maximum one image via file upload, but unlimited images via Live Camera. Maximum one video allowed.
        </p>
        <div className={`location-status-badge ${location.lat ? 'success' : (location.error ? 'error' : 'pending')}`}>
          {location.lat ? (
            `📍 Location Active: ${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}`
          ) : location.error ? (
            <span>
              ⚠️ {location.error}
              <button
                onClick={fetchLocation}
                style={{
                  marginLeft: '10px',
                  padding: '4px 12px',
                  background: '#ff5a4e',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}
              >
                🔄 Retry
              </button>
            </span>
          ) : (
            '⌛ Fetching location...'
          )}
        </div>
      </div>

      {/* UPLOAD SECTION */}
      <div className="card upload-card">
        <h3>Add Evidence</h3>
        <div className="upload-buttons">
          <button
            title="upload images via local storage"
            onClick={() => imageGalleryRef.current.click()}
            disabled={images.filter(img => img.source !== "camera").length >= 1}
            className={images.filter(img => img.source !== "camera").length >= 1 ? "disabled-btn" : ""}
          >
            📁 Upload Images {images.filter(img => img.source !== "camera").length >= 1 ? "(File Limit)" : ""}
          </button>
          {/* <button
            onClick={() => videoGalleryRef.current.click()}
            disabled={videos.length >= 1}
            className={videos.length >= 1 ? "disabled-btn" : ""}
          >
            🎥 Upload Videos {videos.length >= 1 ? "(Limit Reached)" : ""}
          </button> */}
          <button
            onClick={openCamera}
            className="camera-btn"
            title="capture images from camera"
          >
            📷 Live Camera (Multiple Images)
          </button>
        </div>

        <input
          ref={imageGalleryRef}
          type="file"
          hidden
          accept="image/*"
          multiple={false}
          onChange={handleImageUpload}
        />

        <input
          ref={videoGalleryRef}
          type="file"
          hidden
          accept="video/*"
          multiple={false}
          onChange={handleVideoUpload}
        />
      </div>

      {/* ================== YOLO STREAM ================== */}
      {liveYolo && (
        <div className="card camera-section">
          <div className="camera-layout">
            <div className="camera-box">
              <img
                id="yolo-stream"
                src={streamUrl}
                alt="YOLO Live"
                className="yolo-video-feed"
              />
            </div>

            <div className="detection-log-panel">
              <h4>Live Detection Output</h4>
              <div className="log-container">
                {cameraLogs.length === 0 && <p className="no-logs">Initializing AI stream...</p>}
                {cameraLogs.map((log, idx) => (
                  <div key={idx} className="log-entry">
                    <span className="log-time">[{new Date(log.time * 1000).toLocaleTimeString()}]</span>
                    <span className="log-msg">{log.message}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="camera-actions">
            <button className="close-btn" onClick={() => closeCamera(true)}>
              <span className="btn-icon">✖</span>
              <span className="btn-text">Close Live Stream</span>
            </button>

            <p className="camera-info">
              Camera will automatically capture images when deviations are detected.
              <br />
              <strong>Images captured: {images.filter(img => img.source === "camera").length}</strong>
            </p>
          </div>
        </div>
      )}

      {/* PREVIEW SECTION */}
      {(images.length > 0 || videos.length > 0) && (
        <div className="card preview">
          <h4>Review Evidence</h4>

          {images.length > 0 && (
            <>
              <div className="preview-header">
                <p>
                  Images ({images.filter(img => img.source === "camera").length} from camera +
                  {images.filter(img => img.source !== "camera").length} uploaded)
                </p>
                {images.filter(img => img.source !== "camera").length >= 1 && (
                  <span className="limit-badge">FILE UPLOAD LIMIT REACHED</span>
                )}
              </div>
              <div className="preview-grid">
                {images.map((img) => (
                  <div key={img.id} className="preview-item">
                    <img src={img.url} alt="preview" />
                    <button
                      onClick={() => removeImage(img.id)}
                      className="remove-btn"
                      title="Remove image"
                    >
                      ✖
                    </button>
                    {img.source === "camera" && (
                      <div className="source-badge">Camera Capture</div>
                    )}
                    {img.source !== "camera" && (
                      <div className="source-badge upload-badge">File Upload</div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}

          {videos.length > 0 && (
            <>
              <div className="preview-header" style={{ marginTop: "16px" }}>
                <p>Video (1/1)</p>
                <span className="limit-badge">LIMIT REACHED</span>
              </div>
              <div className="preview-grid">
                {videos.map((vid) => (
                  <div key={vid.id} className="preview-item">
                    <video src={vid.url} controls className="preview-video" />
                    <button
                      onClick={() => removeVideo(vid.id)}
                      className="remove-btn"
                      title="Remove video"
                    >
                      ✖
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* ACTION BUTTONS */}
      {(images.length > 0 || videos.length > 0) && (
        <div className="card">
          {uploadError && <div className="error-box">{uploadError}</div>}

          <div className="action-buttons">
            <button className="retry-btn" onClick={resetFiles}>
              Clear All Evidence
            </button>
            <button
              className="analyze-btn"
              onClick={handleAnalyze}
              disabled={uploading || images.filter(img => img.source !== "camera").length > 1 || videos.length > 1}
            >
              {uploading ? "Uploading..." : "Proceed to Analysis"}
            </button>
          </div>

          {(images.filter(img => img.source !== "camera").length > 1 || videos.length > 1) && (
            <div className="warning-message">
              ⚠️ File upload limits exceeded. Only one image via file upload.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ComplaintPage;