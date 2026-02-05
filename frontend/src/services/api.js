/**
 * API Service for MDMS Backend
 * Centralized API calls to the backend
 */

// If VITE_API_BASE_URL is set, use it. Otherwise use relative '/api' so Vite proxy works in dev.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

/**
 * Generic fetch wrapper with error handling
 */
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  };

  try {
    const response = await fetch(url, { ...defaultOptions, ...options });

    if (!response.ok) {
      const text = await response.text();
      let errorMessage = `HTTP error! status: ${response.status}`;
      try {
        const errorData = JSON.parse(text);
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else if (errorData.message) {
          errorMessage = errorData.message;
        } else if (typeof errorData === 'string') {
          errorMessage = errorData;
        }
      } catch {
        // use raw text if not json
        if (text) errorMessage = text;
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    throw error;
  }
}

/**
 * Upload files to backend
 */
export async function uploadComplaints(files, latitude, longitude) {
  const formData = new FormData();

  // Add all files to FormData
  files.forEach((file) => {
    formData.append('files', file);
  });

  // Add optional coordinates
  if (latitude !== null && latitude !== undefined) {
    formData.append('latitude', latitude);
  }
  if (longitude !== null && longitude !== undefined) {
    formData.append('longitude', longitude);
  }
  if (files[0]?.user_id) {
    formData.append('user_id', files[0].user_id);
  } else {
    // Check if user is in localStorage
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      const user = JSON.parse(storedUser);
      formData.append('user_id', user.id);
    }
  }

  const response = await fetch(`${API_BASE_URL}/api/complaints/batch`, {
    method: 'POST',
    body: formData,
    // Don't set Content-Type header - browser will set it with boundary
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorData.detail || `Upload failed: ${response.status}`);
  }

  return await response.json();
}

/**
 * Get all tickets
 */
export async function getTickets(filters = {}) {
  const params = new URLSearchParams();

  if (filters.status) params.append('status', filters.status);
  if (filters.issue_type) params.append('issue_type', filters.issue_type);
  if (filters.user_id) params.append('user_id', filters.user_id);

  const queryString = params.toString();
  const endpoint = `/api/complaints/tickets${queryString ? `?${queryString}` : ''}`;

  return apiRequest(endpoint);
}

/**
 * Get ticket by ID
 */
export async function getTicketById(ticketId) {
  return apiRequest(`/api/complaints/tickets/${ticketId}`);
}

/**
 * Get image by ID
 */
export function getImageUrl(imageId) {
  return `${API_BASE_URL}/api/complaints/images/${imageId}`;
}

/**
 * Update ticket location
 */
export async function updateTicketLocation(ticketId, latitude, longitude) {
  const formData = new FormData();
  formData.append('latitude', latitude);
  formData.append('longitude', longitude);

  const response = await fetch(`${API_BASE_URL}/api/complaints/tickets/${ticketId}/location`, {
    method: 'PATCH',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorData.detail || `Update failed: ${response.status}`);
  }

  return await response.json();
}

/**
 * YOLO Detection - Detect image
 */
export async function detectImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/yolo/detect-image`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorData.detail || `Detection failed: ${response.status}`);
  }

  return await response.json();
}

/**
 * YOLO Detection - Detect video
 */
export async function detectVideo(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/yolo/detect-video`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorData.detail || `Video detection failed: ${response.status}`);
  }

  return await response.json();
}

/**
 * Get YOLO annotated image URL
 */
export function getAnnotatedImageUrl(filename) {
  return `${API_BASE_URL}/api/yolo/annotated/${filename}`;
}

/**
 * Get YOLO original image URL
 */
export function getOriginalImageUrl(filename) {
  return `${API_BASE_URL}/api/yolo/original/${filename}`;
}

/**
 * Reverse geocode coordinates to get area and district
 */
export async function geocodeLocation(lat, lon) {
  return apiRequest(`/api/complaints/geocode?lat=${lat}&lon=${lon}`);
}

/**
 * Delete a ticket and all its associated data
 */
export async function deleteTicket(ticketId) {
  return apiRequest(`/api/complaints/tickets/${ticketId}`, {
    method: 'DELETE',
  });
}

/**
 * INSPECTOR: Get assigned tickets
 */
export async function getInspectorTickets(authority = null, status = null) {
  const params = new URLSearchParams();
  if (authority) params.append('authority', authority);
  if (status) params.append('status', status);
  return apiRequest(`/api/inspector/tickets?${params.toString()}`);
}

/**
 * INSPECTOR: Resolve Ticket with Proof
 */
export async function resolveSubTicket(subId, status, proofFile, comment = "", inspectorName = "") {
  const formData = new FormData();
  formData.append('status', status);
  if (proofFile) {
    formData.append('file', proofFile);
  }
  if (comment) {
    formData.append('comment', comment);
  }
  if (inspectorName) {
    formData.append('resolved_by', inspectorName);
  }

  const response = await fetch(`${API_BASE_URL}/api/inspector/sub-tickets/${subId}/resolve`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorData.detail || `Resolution failed: ${response.status}`);
  }

  return await response.json();
}

/**
 * AUTH: Login
 */
export async function login(email, password) {
  return apiRequest('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
}

/**
 * AUTH: Signup
 */
export async function signup(name, email, password, role, department = null) {
  return apiRequest('/api/auth/signup', {
    method: 'POST',
    body: JSON.stringify({ name, email, password, role, department }),
  });
}

/**
 * AUTH: Get All Users (Admin)
 */
export async function getUsers() {
  return apiRequest('/api/auth/users');
}

/**
 * AUTH: Approve Inspector
 */
export async function approveUser(userId) {
  return apiRequest(`/api/auth/approve/${userId}`, {
    method: 'PUT',
  });
}

/**
 * ADMIN: Get Inspector Actions
 */
export async function getInspectorActions() {
  return apiRequest('/api/admin/inspector-actions');
}

/**
 * ADMIN: Create Inspector
 */
export async function createInspector(name, email, password, department) {
  return apiRequest('/api/admin/create-inspector', {
    method: 'POST',
    body: JSON.stringify({ name, email, password, department, role: "INSPECTOR" }),
  });
}

/**
 * AUTH: Delete User
 */
export async function deleteUser(userId) {
  return apiRequest(`/api/auth/users/${userId}`, {
    method: 'DELETE',
  });
}

export default {
  uploadComplaints,
  getTickets,
  getTicketById,
  getImageUrl,
  updateTicketLocation,
  detectImage,
  detectVideo,
  getAnnotatedImageUrl,
  getOriginalImageUrl,
  geocodeLocation,
  deleteTicket,
  getInspectorTickets,
  resolveSubTicket,
  login,
  signup,
  getUsers,
  approveUser,
  deleteUser,
  createInspector,
  getInspectorActions
};
