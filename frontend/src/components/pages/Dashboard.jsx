import "../styles/Dashboard.css";
import { useEffect, useState } from "react";
import { Pie, Line, Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  Filler,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
} from "chart.js";
import { getTickets } from "../../services/api";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  Filler,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
);

// Fix react-leaflet icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
});

/* -------------------- REAL MARKER ICONS FOR LIVE ACTIVITY -------------------- */
const BLUE_MARKER_URL = "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png";
const RED_MARKER_URL = "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png";
const GREEN_MARKER_URL = "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png";
const ORANGE_MARKER_URL = "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-orange.png";

// Create marker icons
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

// Function to get marker icon based on status
const getMarkerIcon = (status = "open") => {
  const s = (status || "open").toLowerCase();
  if (s === "resolved" || s === "closed") return redIcon;
  if (s === "in_progress") return orangeIcon;
  if (s === "assigned") return greenIcon;
  return blueIcon; // default for open
};

// Helper function to normalize issue types
const normalizeIssueType = (issueType) => {
  if (!issueType) return "";
  
  const cleaned = issueType.toLowerCase().trim();
  
  const mapping = {
    // Water puddles variations - consolidated to water_puddles
    "waterpuddles": "water_puddles",
    "waterpuddle": "water_puddles",
    "water_puddle": "water_puddles",
    "water puddles": "water_puddles",
    "water puddle": "water_puddles",
    "puddles": "water_puddles",
    
    // Open manholes variations
    "openmanholes": "open_manholes",
    "openmanhole": "open_manholes",
    "open_manholes": "open_manholes",
    "open_manhole": "open_manholes",
    "open manholes": "open_manholes",
    
    // Garbage overflow variations
    "garbageoverflow": "garbage_overflow",
    "garbage_overflow": "garbage_overflow",
    "garbage overflow": "garbage_overflow",
    
    // Sand on road variations
    "sandonroad": "sand_on_road",
    "sand_on_road": "sand_on_road",
    "sand on road": "sand_on_road",
    
    // Road cracks variations
    "roadcracks": "road_cracks",
    "roadcrack": "road_cracks",
    "road_cracks": "road_cracks",
    "road_crack": "road_cracks",
    "road cracks": "road_cracks",
    
    // Street hawkers variations
    "streethawkers": "street_hawkers",
    "streethawker": "street_hawkers",
    "street_hawkers": "street_hawkers",
    "street_hawker": "street_hawkers",
    "street hawkers": "street_hawkers",
    
    // Animal carcases variations
    "animalcarcases": "animal_carcases",
    "animalcarcass": "animal_carcases",
    "animal_carcases": "animal_carcases",
    "animal_carcass": "animal_carcases",
    "animal carcases": "animal_carcases",
    
    // Potholes variations
    "potholes": "potholes",
    "pothole": "potholes",
    
    // Street debris variations
    "streetdebris": "street_debris",
    "street_debris": "street_debris",
    "street debris": "street_debris",
    "debris": "street_debris",
  };
  
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

// Updated MiniMap with proper markers
const MiniMap = ({ tickets }) => {
  // Extract markers from sub_tickets
  const markers = tickets
    .flatMap((t) =>
      (t.sub_tickets || []).map((st) => ({
        ...st,
        ticket_id: t.ticket_id,
        ticket_data: t,
        lat: st.latitude || t.latitude,
        lng: st.longitude || t.longitude,
        status: st.status || t.status || 'open'
      })),
    )
    .filter((m) => m.lat && m.lng && !isNaN(parseFloat(m.lat)) && !isNaN(parseFloat(m.lng)));

  const center = [16.29974, 80.45729];

  return (
    <div className="mini-map-card">
      <h3>Live Activity</h3>
      <div className="mini-map-container">
        <MapContainer
          center={center}
          zoom={13}
          style={{ height: "100%", width: "100%" }}
          scrollWheelZoom={true}  // Enable mouse wheel zoom
          dragging={true}         // Enable map dragging
          doubleClickZoom={true}  // Enable double-click zoom
        >
          <TileLayer 
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='© OpenStreetMap contributors'
          />
          {markers.map((m, idx) => (
            <Marker
              key={`${m.ticket_id}-${m.sub_id || idx}`}
              position={[parseFloat(m.lat), parseFloat(m.lng)]}
              icon={getMarkerIcon(m.status)}
            >
              <Popup>
                <div className="mini-map-popup" style={{ padding: '5px' }}>
                  <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '3px' }}>
                    Ticket #{m.ticket_id}
                  </div>
                  <div style={{ fontSize: '13px' }}>
                    <strong>Issue:</strong> {formatDistortionName(normalizeIssueType(m.issue_type)) || "Not specified"}
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>
    </div>
  );
};

// Function to get department for an issue type
const getDepartmentForIssue = (issueType) => {
  const normalized = normalizeIssueType(issueType);
  
  // Road Department Issues
  if (
    normalized.includes("sand_on_road") ||
    normalized.includes("road_cracks") ||
    normalized.includes("potholes") ||
    normalized.includes("water_puddles") ||
    normalized.includes("open_manholes") ||
    normalized.includes("street_debris")
  ) {
    return "Roads";
  }
  
  // Garbage Department Issues
  if (
    normalized.includes("street_hawkers") ||
    normalized.includes("animal_carcases") ||
    normalized.includes("garbage_overflow")
  ) {
    return "Garbage";
  }
  
  return "Unassigned";
};

// Complete list of distortions
const allDistortions = [
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

export default function HomePage() {
  const [allTickets, setAllTickets] = useState([]);
  const [tickets, setTickets] = useState([]);
  const [loading, setLoading] = useState(true);

  const [stats, setStats] = useState({
    total: 0,
    resolved: 0,
    inprogress: 0,
    todayCompleted: 0,
    yesterdayCompleted: 0,
    thisMonth: 0,
    lastMonth: 0,
    today: 0,
    new: 0,
  });

  const [filters, setFilters] = useState({
    dateRange: "all",
    customDate: "",
    status: "all",
    priority: "all",
    sortBy: "newest",
    category: "all",
  });

  const fetchTickets = async () => {
    try {
      const response = await getTickets();
      const ticketsData = response.tickets || [];

      setAllTickets(ticketsData);
      applyFilters(ticketsData, filters);
    } catch (err) {
      console.error("Failed to fetch tickets", err);
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = (data, activeFilters) => {
    let filtered = [...data];

    // Date filters - use currentDate to avoid redeclaration
    const currentDate = new Date();
    if (activeFilters.dateRange === "7days") {
      const last7 = new Date();
      last7.setDate(currentDate.getDate() - 7);
      filtered = filtered.filter((t) => new Date(t.created_at) >= last7);
    } else if (activeFilters.dateRange === "30days") {
      const last30 = new Date();
      last30.setDate(currentDate.getDate() - 30);
      filtered = filtered.filter((t) => new Date(t.created_at) >= last30);
    } else if (activeFilters.customDate) {
      filtered = filtered.filter(
        (t) =>
          t.created_at && t.created_at.startsWith(activeFilters.customDate),
      );
    }

    // Status filter
    if (activeFilters.status !== "all") {
      filtered = filtered.filter((t) => {
        if (activeFilters.status === "resolved")
          return t.status === "resolved" || t.status === "closed";

        if (activeFilters.status === "pending")
          return t.status !== "resolved" && t.status !== "closed";

        if (activeFilters.status === "new")
          return t.status === "new" || t.status === "open";

        return true;
      });
    }

    // Category filter - Updated to handle all distortion types
    if (activeFilters.category !== "all") {
      filtered = filtered.filter(
        (t) =>
          t.sub_tickets &&
          t.sub_tickets.some((st) => {
            const normalizedIssueType = normalizeIssueType(st.issue_type);
            return normalizedIssueType === activeFilters.category;
          }),
      );
    }

    // Sorting
    filtered.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    setTickets(filtered);

    // Calculate date-based stats
    const today = new Date();
    const todayStr = today.toISOString().split("T")[0];

    // Calculate yesterday
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split("T")[0];

    // Calculate this month (from 1st of current month to today)
    const thisMonthStart = new Date(today.getFullYear(), today.getMonth(), 1);
    const thisMonthStartStr = thisMonthStart.toISOString().split("T")[0];

    // Calculate last month (from 1st to last day of previous month)
    const lastMonthStart = new Date(
      today.getFullYear(),
      today.getMonth() - 1,
      1,
    );
    const lastMonthEnd = new Date(today.getFullYear(), today.getMonth(), 0);
    const lastMonthStartStr = lastMonthStart.toISOString().split("T")[0];
    const lastMonthEndStr = lastMonthEnd.toISOString().split("T")[0];

    // Calculate stats
    const resolved = filtered.filter(
      (t) => t.status === "resolved" || t.status === "closed",
    ).length;

    const inprogress = filtered.filter(
      (t) => t.status !== "resolved" && t.status !== "closed",
    ).length;

    const todayCompleted = filtered.filter(
      (t) =>
        (t.status === "resolved" || t.status === "closed") &&
        t.resolved_at &&
        t.resolved_at.startsWith(todayStr),
    ).length;

    const yesterdayCompleted = filtered.filter(
      (t) =>
        (t.status === "resolved" || t.status === "closed") &&
        t.resolved_at &&
        t.resolved_at.startsWith(yesterdayStr),
    ).length;

    const thisMonth = filtered.filter(
      (t) =>
        (t.status === "resolved" || t.status === "closed") &&
        t.resolved_at &&
        t.resolved_at >= thisMonthStartStr,
    ).length;

    const lastMonth = filtered.filter(
      (t) =>
        (t.status === "resolved" || t.status === "closed") &&
        t.resolved_at &&
        t.resolved_at >= lastMonthStartStr &&
        t.resolved_at <= lastMonthEndStr,
    ).length;

    const todayNew = filtered.filter(
      (t) => t.created_at && t.created_at.startsWith(todayStr),
    ).length;

    setStats({
      total: filtered.length,
      resolved,
      inprogress,
      pending: filtered.length - resolved,
      todayCompleted,
      yesterdayCompleted,
      thisMonth,
      lastMonth,
      today: todayNew,
      new: filtered.filter((t) => t.status === "new" || t.status === "open")
        .length,
    });
  };

  useEffect(() => {
    fetchTickets();
    window.addEventListener("focus", fetchTickets);
    return () => window.removeEventListener("focus", fetchTickets);
  }, []);

  const handleFilterChange = (field, value) => {
    const newFilters = { ...filters, [field]: value };

    if (field === "dateRange" && value !== "custom") {
      newFilters.customDate = "";
    }
    if (field === "customDate") {
      newFilters.dateRange = "custom";
    }

    setFilters(newFilters);
    applyFilters(allTickets, newFilters);
  };

  const recentTickets = tickets
    .flatMap((t) =>
      (t.sub_tickets || []).map((st) => ({
        ...st,
        ticket_id: t.ticket_id,
        created_at: st.created_at || t.created_at,
      })),
    )
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    .slice(0, 6);

  // SLA LOGIC (FULL FIX)
  const SLA_HOURS = 12;

  const slaStatusList = tickets.map((t) => {
    const subs = t.sub_tickets || [];

    // Extract all valid sub-ticket dates
    const validDates = subs
      .map((st) => new Date(st.created_at))
      .filter((d) => !isNaN(d));

    if (validDates.length === 0) {
      return {
        ...t,
        slaStatus: "unknown",
        hours: 0,
      };
    }

    const earliest = validDates.sort((a, b) => a - b)[0];
    const hours = (new Date() - earliest) / 36e5;

    // Determine SLA status
    let slaStatus = "";

    if (t.status === "resolved" || t.status === "closed") {
      slaStatus = hours <= SLA_HOURS ? "solvedWithinSLA" : "solvedAfterSLA";
    } else {
      slaStatus = hours > SLA_HOURS ? "breached" : "onTrack";
    }

    return {
      ...t,
      hours,
      slaStatus,
    };
  });

  // Calculate distortion type distribution for pie chart
  const getDistortionDistribution = () => {
    const distribution = {};
    
    tickets.forEach(ticket => {
      (ticket.sub_tickets || []).forEach(subTicket => {
        const normalizedType = normalizeIssueType(subTicket.issue_type);
        distribution[normalizedType] = (distribution[normalizedType] || 0) + 1;
      });
    });
    
    return distribution;
  };

  const distortionDistribution = getDistortionDistribution();

  // Calculate area distribution (geographic areas with most issues)
  const getAreaDistribution = () => {
    const distribution = {};
    
    tickets.forEach(ticket => {
      const area = ticket.area || "Unknown Area";
      distribution[area] = (distribution[area] || 0) + 1;
    });
    
    return distribution;
  };

  const areaDistribution = getAreaDistribution();

  // Calculate issue types for each area
  const getIssueTypesPerArea = () => {
    const distribution = {};
    
    tickets.forEach(ticket => {
      const area = ticket.area || "Unknown Area";
      if (!distribution[area]) {
        distribution[area] = {};
      }
      
      (ticket.sub_tickets || []).forEach(subTicket => {
        const normalizedType = normalizeIssueType(subTicket.issue_type);
        distribution[area][normalizedType] = (distribution[area][normalizedType] || 0) + 1;
      });
    });
    
    return distribution;
  };

  const issueTypesPerArea = getIssueTypesPerArea();

  // Calculate authority/department workload - Shows only Roads and Garbage departments
  const getAuthorityWorkload = () => {
    const distribution = {
      "Roads": { total: 0, pending: 0, resolved: 0 },
      "Garbage": { total: 0, pending: 0, resolved: 0 }
    };
    
    tickets.forEach(ticket => {
      (ticket.sub_tickets || []).forEach(subTicket => {
        const department = getDepartmentForIssue(subTicket.issue_type);
        const status = subTicket.status || "open";
        
        // Only count Roads and Garbage departments
        if (department === "Roads" || department === "Garbage") {
          distribution[department].total += 1;
          if (status === "resolved" || status === "closed") {
            distribution[department].resolved += 1;
          } else {
            distribution[department].pending += 1;
          }
        }
      });
    });
    
    return distribution;
  };

  const authorityWorkload = getAuthorityWorkload();

  // Prepare pie chart data
  const pieData = {
    labels: ["Resolved", "Pending"],
    datasets: [
      {
        data: [stats.resolved, stats.pending],
        backgroundColor: ["#22c55e", "#ef4444"],
        hoverOffset: 18,
        borderWidth: 3,
        borderColor: "#fff",
        offset: 10,
      },
    ],
  };

  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        padding: 12,
        titleFont: { size: 14, weight: "bold" },
        bodyFont: { size: 13 },
        displayColors: true,
        boxWidth: 15,
        boxHeight: 15,
      },
    },
    elements: {
      arc: {
        shadowOffsetX: 0,
        shadowOffsetY: 8,
        shadowBlur: 20,
        shadowColor: "rgba(0, 0, 0, 0.3)",
      },
    },
    animation: {
      animateRotate: true,
      animateScale: true,
      duration: 1000,
      easing: "easeInOutQuart",
    },
  };

  return (
    <div className="home-container">
      <div className="home-content">
        {/* SNAPSHOT BAR */}
        <div className="snapshot-bar">
          <div className="snapshot-item">
            <span className="label">Today(Resolved) </span>
            <span className="value">{stats.todayCompleted}</span>
          </div>
          <div className="divider" />
          <div className="snapshot-item new">
            <span className="label">New </span>
            <span className="value">{stats.new}</span>
          </div>
          <div className="divider" />
          <div className="snapshot-item resolved">
            <span className="label">Resolved </span>
            <span className="value">{stats.resolved}</span>
          </div>
          <div className="divider" />
          <div className="snapshot-item open">
            <span className="label">Open </span>
            <span className="value">{stats.pending}</span>
          </div>
        </div>

        {/* FILTERS */}
        <div className="home-filters">
          <div className="filter-group">
            <label>Date Range</label>
            <select
              value={filters.dateRange}
              onChange={(e) => handleFilterChange("dateRange", e.target.value)}
            >
              <option value="all">All Time</option>
              <option value="7days">Last 7 Days</option>
              <option value="30days">Last 30 Days</option>
              <option value="custom">Custom Date</option>
            </select>

            {filters.dateRange === "custom" && (
              <input
                type="date"
                value={filters.customDate}
                onChange={(e) =>
                  handleFilterChange("customDate", e.target.value)
                }
              />
            )}
          </div>

          <div className="filter-group">
            <label>Status</label>
            <select
              value={filters.status}
              onChange={(e) => handleFilterChange("status", e.target.value)}
            >
              <option value="all">All Statuses</option>
              <option value="new">New / Open</option>
              <option value="pending">Pending</option>
              <option value="resolved">Resolved</option>
            </select>
          </div>

          <div className="filter-group">
            <label>Priority</label>
            <select
              value={filters.priority}
              onChange={(e) => handleFilterChange("priority", e.target.value)}
            >
              <option value="all">All Priorities</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>

          {/* Updated Category filter with all distortions */}
          <div className="filter-group">
            <label>Category</label>
            <select
              value={filters.category}
              onChange={(e) => handleFilterChange("category", e.target.value)}
            >
              <option value="all">All Categories</option>
              {allDistortions
                .filter(dist => dist !== "all")
                .map((dist) => (
                  <option key={dist} value={dist}>
                    {formatDistortionName(dist)}
                  </option>
                ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Sort By</label>
            <select
              value={filters.sortBy}
              onChange={(e) => handleFilterChange("sortBy", e.target.value)}
            >
              <option value="newest">Newest → Oldest</option>
              <option value="oldest">Oldest → Newest</option>
              <option value="severity">Severity (High→Low)</option>
            </select>
          </div>
        </div>

        {/* MAIN GRID */}
        <div className="grid-2">
          <div className="stats-grid">
            {/* Total Tickets */}
            <div className="stat-card total tickets-list-card">
              <div className="stat-header">
                <p>TOTAL TICKETS</p>
                <h2>{stats.total}</h2>
              </div>

              {tickets.length === 0 ? (
                <div className="no-tickets">No tickets found</div>
              ) : (
                <div className="tickets-scroll">
                  {tickets.map((t) => (
                    <div key={t.ticket_id} className="ticket-row">
                      <span className="ticket-id">{t.ticket_id}</span>
                      <div className="ticket-issues">
                        {(t.sub_tickets || []).map((st) => (
                          <span
                            key={st.sub_id}
                            className={`issue-pill ${normalizeIssueType(st.issue_type)}`}
                          >
                            {formatDistortionName(normalizeIssueType(st.issue_type))}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Resolved Tickets - FIXED */}
            <div className="stat-card resolved tickets-list-card">
              <div className="stat-header">
                <p>RESOLVED TICKETS</p>
                <h2>{stats.resolved}</h2>
              </div>

              {stats.resolved === 0 ? (
                <div className="no-tickets">No resolved tickets</div>
              ) : (
                <div className="tickets-scroll">
                  {tickets
                    .filter(
                      (t) => t.status === "resolved" || t.status === "closed",
                    )
                    .map((ticket) => (
                      <div key={ticket.ticket_id} className="ticket-row">
                        <span className="ticket-id">{ticket.ticket_id}</span>
                        <div className="ticket-issues">
                          {(ticket.sub_tickets || []).map((st) => (
                            <span
                              key={st.sub_id}
                              className={`issue-pill ${normalizeIssueType(st.issue_type)}`}
                            >
                              {formatDistortionName(normalizeIssueType(st.issue_type))}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </div>

            {/* In Progress Tickets */}
            <div className="stat-card pending tickets-list-card">
              <div className="stat-header">
                <p>IN PROGRESS TICKETS</p>
                <h2>{stats.inprogress}</h2>
              </div>

              {stats.inprogress === 0 ? (
                <div className="no-tickets">No in-progress tickets</div>
              ) : (
                <div className="tickets-scroll">
                  {tickets
                    .filter(
                      (t) => t.status !== "resolved" && t.status !== "closed",
                    )
                    .map((ticket) => (
                      <div key={ticket.ticket_id} className="ticket-row">
                        <span className="ticket-id">{ticket.ticket_id}</span>
                        <div className="ticket-issues">
                          {(ticket.sub_tickets || []).map((st) => (
                            <span
                              key={st.sub_id}
                              className={`issue-pill ${normalizeIssueType(st.issue_type)}`}
                            >
                              {formatDistortionName(normalizeIssueType(st.issue_type))}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </div>

            {/* Today Completed */}
            <div className="stat-card completed-today">
              <div className="stat-header">
                <p>TODAY COMPLETED</p>
                <h2>{stats.todayCompleted}</h2>
              </div>
              <div className="date-comparison">
                <small>Yesterday: {stats.yesterdayCompleted}</small>
              </div>
            </div>

            {/* This Month */}
            <div className="stat-card this-month">
              <div className="stat-header">
                <p>THIS MONTH</p>
                <h2>{stats.thisMonth}</h2>
              </div>
              <div className="date-comparison">
                <small>Last Month: {stats.lastMonth}</small>
              </div>
            </div>
          </div>

          <div className="right-column">
            <MiniMap tickets={tickets} />

            <div className="pie-card">
              <h3>Complaint Status</h3>
              <div className="pie-box">
                <Pie data={pieData} options={pieOptions} />
              </div>
            </div>
          </div>
        </div>

        {/* LINE + BAR CHART SECTION */}
        <div
          className="charts-section"
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)",
            gap: "24px",
            marginBottom: "24px",
          }}
        >
          {/* LINE CHART */}
          <div className="chart-card">
            <h3>Complaints Per Day</h3>
            <div style={{ height: "300px" }}>
              <Line
                data={{
                  labels: Object.keys(
                    tickets.reduce((acc, t) => {
                      const date = t.created_at
                        ? t.created_at.split("T")[0]
                        : "Unknown";
                      acc[date] = (acc[date] || 0) + 1;
                      return acc;
                    }, {}),
                  ).sort(),
                  datasets: [
                    {
                      label: "Complaints",
                      data: Object.keys(
                        tickets.reduce((acc, t) => {
                          const date = t.created_at
                            ? t.created_at.split("T")[0]
                            : "Unknown";
                          acc[date] = (acc[date] || 0) + 1;
                          return acc;
                        }, {}),
                      )
                        .sort()
                        .map(
                          (date) =>
                            tickets.filter(
                              (t) =>
                                t.created_at && t.created_at.startsWith(date),
                            ).length,
                        ),
                      borderColor: "#38bdf8",
                      backgroundColor: "rgba(56, 189, 248, 0.2)",
                      tension: 0.4,
                      fill: true,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: false } },
                  scales: {
                    y: {
                      grid: { color: "rgba(30, 41, 59, 0.1)" },
                      ticks: { color: "#1e293b" },
                    },
                    x: {
                      grid: { display: false },
                      ticks: { color: "#1e293b" },
                    },
                  },
                }}
              />
            </div>
          </div>

          {/* BAR CHART - Most Reported Areas */}
          <div className="chart-card">
            <h3>Most Reported Areas</h3>

            <div style={{ height: "300px", position: "relative" }}>
              <Bar
                data={{
                  labels: Object.keys(areaDistribution)
                    .sort((a, b) => areaDistribution[b] - areaDistribution[a])
                    .slice(0, 8),

                  datasets: [
                    {
                      label: "Number of Issues",
                      data: Object.keys(areaDistribution)
                        .sort((a, b) => areaDistribution[b] - areaDistribution[a])
                        .slice(0, 8)
                        .map(area => areaDistribution[area]),

                      backgroundColor: function (ctx) {
                        const gradient = ctx.chart.ctx.createLinearGradient(
                          0,
                          0,
                          0,
                          200,
                        );
                        gradient.addColorStop(0, "#ef4444");
                        gradient.addColorStop(1, "#f87171");
                        return gradient;
                      },

                      borderRadius: 8,
                      borderSkipped: false,
                      hoverBackgroundColor: "#fca5a5",
                      barThickness: 40,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { display: false },
                    tooltip: {
                      backgroundColor: "#1e293b",
                      titleColor: "#fff",
                      bodyColor: "#cbd5e1",
                      borderColor: "#334155",
                      borderWidth: 1,
                      padding: 12,
                      displayColors: false,
                      callbacks: {
                        title: function (context) {
                          return context[0].label;
                        },
                        label: function (context) {
                          return `Total Issues: ${context.parsed.y}`;
                        },
                        afterLabel: function (context) {
                          const area = context.label;
                          const issues = issueTypesPerArea[area] || {};
                          const issueLines = Object.entries(issues)
                            .sort((a, b) => b[1] - a[1])
                            .map(([issueType, count]) => {
                              return `• ${formatDistortionName(issueType)}: ${count}`;
                            });
                          return issueLines.length > 0 ? '\n' + issueLines.join('\n') : '';
                        },
                      },
                    },
                  },

                  scales: {
                    y: {
                      beginAtZero: true,
                      ticks: {
                        color: "#1e293b",
                        padding: 10,
                        stepSize: 1,
                        callback: function (value) {
                          if (Number.isInteger(value)) {
                            return value;
                          }
                        },
                      },
                      grid: {
                        display: false,
                        drawBorder: false,
                      },
                      border: {
                        display: false,
                      },
                      title: {
                        display: true,
                        text: "Number of Issues",
                        color: "#1e293b",
                        padding: { top: 0, bottom: 15 },
                      },
                    },
                    x: {
                      ticks: {
                        color: "#1e293b",
                        padding: 10,
                      },
                      grid: {
                        display: false,
                      },
                      border: {
                        display: false,
                      },
                      title: {
                        display: true,
                        text: "Areas",
                        color: "#1e293b",
                        padding: { top: 15, bottom: 0 },
                      },
                    },
                  },

                  layout: {
                    padding: {
                      top: 20,
                      right: 20,
                      bottom: 20,
                      left: 20,
                    },
                  },

                  animation: {
                    duration: 900,
                    easing: "easeOutQuad",
                  },
                }}
              />
            </div>
          </div>

          {/* BAR CHART - Distortion Type Distribution - NOW INCLUDES WATER PUDDLES */}
          <div className="chart-card">
            <h3>Most Reported Issue Types</h3>

            <div style={{ height: "300px", position: "relative" }}>
              <Bar
                data={{
                  labels: Object.keys(distortionDistribution)
                    .sort((a, b) => distortionDistribution[b] - distortionDistribution[a])
                    .slice(0, 8)
                    .map(dist => formatDistortionName(dist)),

                  datasets: [
                    {
                      label: "Number of Issues",
                      data: Object.keys(distortionDistribution)
                        .sort((a, b) => distortionDistribution[b] - distortionDistribution[a])
                        .slice(0, 8)
                        .map(dist => distortionDistribution[dist]),

                      backgroundColor: function (ctx) {
                        const gradient = ctx.chart.ctx.createLinearGradient(
                          0,
                          0,
                          0,
                          200,
                        );
                        gradient.addColorStop(0, "#6366f1");
                        gradient.addColorStop(1, "#8b5cf6");
                        return gradient;
                      },

                      borderRadius: 8,
                      borderSkipped: false,
                      hoverBackgroundColor: "#a78bfa",
                      barThickness: 40,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { display: false },
                    tooltip: {
                      backgroundColor: "#1e293b",
                      titleColor: "#fff",
                      bodyColor: "#cbd5e1",
                      borderColor: "#334155",
                      borderWidth: 1,
                      padding: 10,
                      displayColors: false,
                      callbacks: {
                        label: function (context) {
                          return `Count: ${context.parsed.y}`;
                        },
                      },
                    },
                  },

                  scales: {
                    y: {
                      beginAtZero: true,
                      ticks: {
                        color: "#1e293b",
                        padding: 10,
                        stepSize: 1,
                        callback: function (value) {
                          if (Number.isInteger(value)) {
                            return value;
                          }
                        },
                      },
                      grid: {
                        display: false,
                        drawBorder: false,
                      },
                      border: {
                        display: false,
                      },
                      title: {
                        display: true,
                        text: "Number of Issues",
                        color: "#1e293b",
                        padding: { top: 0, bottom: 15 },
                      },
                    },
                    x: {
                      ticks: {
                        color: "#1e293b",
                        padding: 10,
                      },
                      grid: {
                        display: false,
                      },
                      border: {
                        display: false,
                      },
                      title: {
                        display: true,
                        text: "Issue Types",
                        color: "#1e293b",
                        padding: { top: 15, bottom: 0 },
                      },
                    },
                  },

                  layout: {
                    padding: {
                      top: 20,
                      right: 20,
                      bottom: 20,
                      left: 20,
                    },
                  },

                  animation: {
                    duration: 900,
                    easing: "easeOutQuad",
                  },
                }}
              />
            </div>
          </div>

          {/* AUTHORITY/DEPARTMENT WORKLOAD - ONLY ROADS AND GARBAGE */}
          <div className="chart-card">
            <h3>Department Workload & Performance</h3>

            <div style={{ height: "300px", position: "relative" }}>
              <Bar
                data={{
                  labels: ["Roads", "Garbage"],

                  datasets: [
                    {
                      label: "Resolved",
                      data: ["Roads", "Garbage"].map(dept => authorityWorkload[dept]?.resolved || 0),
                      backgroundColor: "#10b981",
                      borderRadius: 8,
                      borderSkipped: false,
                    },
                    {
                      label: "Pending",
                      data: ["Roads", "Garbage"].map(dept => authorityWorkload[dept]?.pending || 0),
                      backgroundColor: "#ef4444",
                      borderRadius: 8,
                      borderSkipped: false,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  indexAxis: "y",
                  plugins: {
                    legend: {
                      display: true,
                      position: "bottom",
                      labels: {
                        color: "#1e293b",
                        padding: 15,
                        font: { size: 12, weight: "600" },
                        boxWidth: 12,
                      },
                    },
                    tooltip: {
                      backgroundColor: "#1e293b",
                      titleColor: "#fff",
                      bodyColor: "#cbd5e1",
                      borderColor: "#334155",
                      borderWidth: 1,
                      padding: 12,
                      displayColors: true,
                      callbacks: {
                        title: function (context) {
                          return context[0].label;
                        },
                        label: function (context) {
                          const dept = context.label;
                          const resolved = authorityWorkload[dept]?.resolved || 0;
                          const pending = authorityWorkload[dept]?.pending || 0;
                          const total = resolved + pending;
                          
                          if (context.dataset.label === "Resolved") {
                            return `Resolved: ${resolved} (${total > 0 ? Math.round((resolved/total)*100) : 0}%)`;
                          } else {
                            return `Pending: ${pending} (${total > 0 ? Math.round((pending/total)*100) : 0}%)`;
                          }
                        },
                      },
                    },
                  },

                  scales: {
                    x: {
                      beginAtZero: true,
                      stacked: true,
                      ticks: {
                        color: "#1e293b",
                        padding: 10,
                        stepSize: 1,
                        callback: function (value) {
                          if (Number.isInteger(value)) {
                            return value;
                          }
                        },
                      },
                      grid: {
                        display: false,
                        drawBorder: false,
                      },
                      border: {
                        display: false,
                      },
                      title: {
                        display: true,
                        text: "Number of Issues",
                        color: "#1e293b",
                      },
                    },
                    y: {
                      ticks: {
                        color: "#1e293b",
                        padding: 12,
                        font: { size: 12, weight: "600" },
                      },
                      grid: {
                        display: false,
                      },
                      border: {
                        display: false,
                      },
                    },
                  },

                  layout: {
                    padding: {
                      top: 20,
                      right: 20,
                      bottom: 20,
                      left: 20,
                    },
                  },

                  animation: {
                    duration: 900,
                    easing: "easeOutQuad",
                  },
                }}
              />
            </div>
          </div>
        </div>

        {/* RECENT ACTIVITY + SLA BREACH SIDE BY SIDE */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "24px",
            marginTop: "24px",
          }}
        >
          {/* Recent Activity */}
          <div className="activity-card">
            <h3>Recent Activity</h3>

            {loading && <p>Loading...</p>}
            {!loading && recentTickets.length === 0 && (
              <p>No recent activity</p>
            )}

            {!loading &&
              recentTickets.map((st) => {
                const normalizedType = normalizeIssueType(st.issue_type);
                let icon = "📋";
                
                // Set appropriate icons for different distortion types
                if (normalizedType.includes("pothole")) icon = "🕳️";
                else if (normalizedType.includes("garbage")) icon = "🗑️";
                else if (normalizedType.includes("water")) icon = "💧";
                else if (normalizedType.includes("sand")) icon = "🏖️";
                else if (normalizedType.includes("crack")) icon = "🔧";
                else if (normalizedType.includes("hawker")) icon = "👨‍🍳";
                else if (normalizedType.includes("animal")) icon = "🐾";
                else if (normalizedType.includes("manhole")) icon = "🕳️";
                else if (normalizedType.includes("debris")) icon = "🚧";

                return (
                  <div key={st.sub_id} className="activity-item">
                    <span className="activity-icon">
                      {icon}
                    </span>

                    <div className="activity-text">
                      <strong>{formatDistortionName(normalizedType)}</strong>
                      <span className="activity-meta">🆔 {st.ticket_id}</span>
                      <span className="activity-meta">
                        ⏰ {new Date(st.created_at).toLocaleString()}
                      </span>
                    </div>
                  </div>
                );
              })}
          </div>

          {/* SLA Breach */}
          <div className="sla-card">
            <h3>SLA Status</h3>

            {slaStatusList.length === 0 && <p>No tickets available</p>}

            <div className="sla-scroll">
              {slaStatusList.map((t) => (
                <div key={t.ticket_id} className="ticket-row sla-breach">
                  <div style={{ flex: 1 }}>
                    <span className="ticket-id" style={{ color: "#0c4a6e" }}>{t.ticket_id}</span>
                    <div
                      style={{
                        fontSize: "0.85em",
                        color: "#1e293b",
                        marginTop: "4px",
                      }}
                    >
                      {t.area || "Unknown Location"}
                    </div>
                  </div>

                  <div className="ticket-issues">
                    {(t.sub_tickets || []).map((st) => (
                      <span
                        key={st.sub_id}
                        className={`issue-pill ${normalizeIssueType(st.issue_type)}`}
                      >
                        {formatDistortionName(normalizeIssueType(st.issue_type))}
                      </span>
                    ))}
                  </div>

                  {/* SLA Status Display */}
                  <span className="sla-status">
                    {t.slaStatus === "breached" && (
                      <span style={{ color: "red", fontWeight: "bold" }}>
                        ⚠ SLA Breached
                      </span>
                    )}

                    {t.slaStatus === "onTrack" && (
                      <span style={{ color: "orange", fontWeight: "bold" }}>
                        ⏳ On Track
                      </span>
                    )}

                    {t.slaStatus === "solvedWithinSLA" && (
                      <span style={{ color: "lightgreen", fontWeight: "bold" }}>
                        ✔ Solved within SLA
                      </span>
                    )}

                    {t.slaStatus === "solvedAfterSLA" && (
                      <span style={{ color: "#ffcc00", fontWeight: "bold" }}>
                        ⚠ Resolved but after SLA
                      </span>
                    )}

                    {t.slaStatus === "unknown" && (
                      <span style={{ color: "#999", fontWeight: "bold" }}>
                        ❓ Unknown
                      </span>
                    )}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}