import React from 'react';
import '../styles/LiveMonitoring.css';
import droneImg from '../../assets/images/drone_monitoring.png';

export default function LiveMonitoring() {
    return (
        <div className="live-monitoring-container">
            <div className="monitoring-content">
                <div className="coming-soon-badge">Phase 2: Coming Soon</div>

                <h1>Aerial Drone Monitoring</h1>

                <div className="drone-image-wrapper">
                    <img
                        src={droneImg}
                        alt="Advanced Surveillance Drone"
                        className="drone-preview-img"
                    />
                </div>

                <p>
                    We are integrating real-time autonomous drone surveillance to detect
                    municipal deviations from the sky. Get a bird’s-eye view of your district
                    with automated AI patrol and instant alerts.
                </p>

                <div className="feature-list">
                    <span className="feature-tag">🛰️ Real-time Streaming</span>
                    <span className="feature-tag">🤖 AI Deviation Detection</span>
                    <span className="feature-tag">📍 GPS Sync</span>
                </div>

                <div style={{ marginTop: '50px', color: '#64748b', fontSize: '0.9rem' }}>
                    Coming Soon to MDMS 2.0
                </div>
            </div>
        </div>
    );
}
