import { useState, useEffect, useRef } from 'react';
import { useAuth } from '../auth/AuthContext';
import { auth } from '../firebase';
import { signOut } from 'firebase/auth';
import { useNavigate, useLocation } from 'react-router-dom';
import BehaviorTracker from '../components/BehaviorTracker';
import PrivacyDashboard from '../components/PrivacyDashboard';
import TrustScoreChart from '../components/TrustScoreChart';
import '../styles/Dashboard.css';

export default function DashboardPage() {
  const { currentUser } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  
  // State declarations
  const [activeTab, setActiveTab] = useState('account');
  const [trustScore, setTrustScore] = useState(100);
  const [trustStatus, setTrustStatus] = useState('normal');
  const [trustScoreHistory, setTrustScoreHistory] = useState([]);
  const [safeMode, setSafeMode] = useState(false);
  const [batterySaverMode, setBatterySaverMode] = useState(false);
  const [geoLocation, setGeoLocation] = useState(null);
  const [locationError, setLocationError] = useState(null);
  const [consecutiveAnomalies, setConsecutiveAnomalies] = useState(0);
  const [showPrivacyDashboard, setShowPrivacyDashboard] = useState(false);
  const [panicKey, setPanicKey] = useState('');
  
  const behaviorData = useRef({
    clickEvents: [],
    typingPatterns: [],
    mouseMovements: [],
    deviceInfo: {},
    initialized: true
  });

  // Panic button tracking
  const panicPressCount = useRef(0);
  const lastPanicPressTime = useRef(0);

  // Update trust score history when trustScore changes
  useEffect(() => {
    const timestamp = new Date().toISOString();
    setTrustScoreHistory(prev => [
      ...prev.slice(-29), // Keep last 29 entries
      { score: trustScore, timestamp, status: trustStatus }
    ]);
  }, [trustScore, trustStatus]);

  // Panic button handler
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!panicKey || e.key !== panicKey) {
        panicPressCount.current = 0;
        return;
      }

      const currentTime = Date.now();
      if (currentTime - lastPanicPressTime.current > 1000) {
        panicPressCount.current = 0;
      }

      panicPressCount.current += 1;
      lastPanicPressTime.current = currentTime;

      if (panicPressCount.current >= 3) {
        handlePanicLogout();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [panicKey]);

  const handlePanicLogout = async () => {
    try {
      // Clear tracking data
      behaviorData.current = {
        clickEvents: [],
        typingPatterns: [],
        mouseMovements: [],
        deviceInfo: behaviorData.current.deviceInfo,
        initialized: true
      };

      // Reset states
      setTrustScore(100);
      setTrustStatus('normal');
      setTrustScoreHistory([]);

      await signOut(auth);
      navigate('/login', {
        state: { 
          message: 'You were logged out by the panic button feature'
        },
        replace: true
      });
    } catch (error) {
      console.error('Panic logout failed:', error);
    }
  };

  // Auto-logout implementation based on anomalous trust status
  useEffect(() => {
    if (trustStatus === 'anomalous') {
      setConsecutiveAnomalies(prev => {
        const newCount = prev + 1;
        if (newCount >= 3) {
          handleAutoLogout('suspicious_activity'); // Call with specific reason
          return 0;
        }
        return newCount;
      });
    } else {
      setConsecutiveAnomalies(0);
    }
  }, [trustStatus]);

  // New useEffect for trust score based auto-logout
  useEffect(() => {
    if (trustScore <= 20) {
      handleAutoLogout('low_trust_score'); // Call with specific reason
    }
  }, [trustScore]);


  const handleAutoLogout = async (reason) => {
    try {
      // Clear tracking data
      behaviorData.current = {
        clickEvents: [],
        typingPatterns: [],
        mouseMovements: [],
        deviceInfo: behaviorData.current.deviceInfo,
        initialized: true
      };

      // Reset states
      setTrustScore(100);
      setTrustStatus('normal');
      setTrustScoreHistory([]);

      let message = 'Automatic logout due to suspicious activity';
      if (reason === 'low_trust_score') {
        message = 'Automatic logout due to low trust score (<= 20%)';
      }

      await signOut(auth);
      navigate('/login', {
        state: { 
          autoLogout: true,
          message: message
        },
        replace: true
      });
    } catch (error) {
      console.error('Auto logout failed:', error);
    }
  };

  // Device info initialization
  useEffect(() => {
    const initDeviceInfo = () => {
      behaviorData.current.deviceInfo = {
        userAgent: navigator.userAgent,
        screen: `${window.screen.width}x${window.screen.height}`,
        viewportSize: `${window.innerWidth}x${window.innerHeight}`,
        browser: navigator.userAgentData?.brands?.[0]?.brand || 'unknown',
        os: navigator.userAgentData?.platform || 'unknown',
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        userId: currentUser?.uid || 'anonymous',
        safeMode,
        batterySaverMode,
        location: geoLocation ? { 
          latitude: geoLocation.latitude, 
          longitude: geoLocation.longitude 
        } : null,
        panicKey
      };
    };

    if ('geolocation' in navigator && !batterySaverMode) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setGeoLocation({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            accuracy: position.coords.accuracy
          });
          initDeviceInfo();
        },
        (error) => {
          setLocationError(error.message);
          console.error('Geolocation error:', error);
          initDeviceInfo();
        }
      );
    } else {
      setLocationError(batterySaverMode ? 'Disabled in Battery Saver Mode' : 'Geolocation not supported');
      initDeviceInfo();
    }
  }, [currentUser, safeMode, batterySaverMode, panicKey]);

  // Behavior tracking
  useEffect(() => {
    const shouldSkip = () => {
      if (batterySaverMode) return Math.random() > 0.3;
      if (safeMode) return Math.random() > 0.5;
      return false;
    };

    const handleMouseMove = (e) => {
      if (shouldSkip()) return;
      behaviorData.current.mouseMovements.push({
        x: e.clientX,
        y: e.clientY,
        timestamp: Date.now()
      });
    };

    const handleClick = (e) => {
      if (shouldSkip()) return;
      behaviorData.current.clickEvents.push({
        x: e.clientX,
        y: e.clientY,
        target: e.target.tagName,
        timestamp: Date.now(),
        hoverTime: 0
      });
    };

    const handleKeyDown = (e) => {
      if (shouldSkip()) return;
      behaviorData.current.typingPatterns.push({
        key: e.key,
        code: e.code,
        timestamp: Date.now(),
        dwellTime: null
      });
    };

    const handleKeyUp = (e) => {
      const lastKeyPress = behaviorData.current.typingPatterns.findLast(
        kp => kp.key === e.key && kp.dwellTime === null
      );
      if (lastKeyPress) {
        lastKeyPress.dwellTime = Date.now() - lastKeyPress.timestamp;
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('click', handleClick);
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('click', handleClick);
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [safeMode, batterySaverMode]);

  // Send data to backend
  const sendBehaviorData = async () => {
    if (batterySaverMode && Math.random() > 0.5) return;
    
    if (!behaviorData.current.typingPatterns.length && 
        !behaviorData.current.mouseMovements.length && 
        !behaviorData.current.clickEvents.length) {
      return;
    }

    const payload = {
      timestamp: new Date().toISOString(),
      user_email: currentUser?.email || 'anonymous',
      context: {
        safeMode,
        batterySaverMode,
        location: geoLocation,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        panicKey
      },
      data: {
        typingPatterns: behaviorData.current.typingPatterns,
        mouseMovements: behaviorData.current.mouseMovements,
        clickEvents: behaviorData.current.clickEvents,
        deviceInfo: behaviorData.current.deviceInfo,
        fingerprint: `user-${currentUser?.uid || 'anonymous'}`
      }
    };

    try {
      const response = await fetch('http://localhost:8000/api/behavior', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      
      const rawScore = data.anomaly_score;
      const normalizedScore = Math.round(rawScore * 100);
      const clampedScore = Math.min(100, Math.max(0, normalizedScore));
      
      setTrustScore(clampedScore);
      setTrustStatus(data.is_anomalous ? 'anomalous' : 'normal');

      // Reset collected data
      behaviorData.current.typingPatterns = [];
      behaviorData.current.mouseMovements = [];
      behaviorData.current.clickEvents = [];
    } catch (error) {
      console.error('Error sending behavior data:', error);
    }
  };

  useEffect(() => {
    const interval = setInterval(sendBehaviorData, batterySaverMode ? 30000 : 15000);
    return () => clearInterval(interval);
  }, [currentUser, safeMode, batterySaverMode, panicKey]);

  const handleLogout = async () => {
    try {
      await signOut(auth);
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'account':
        return (
          <div className="tab-content">
            <h3>Account Information</h3>
            <div className="account-details">
              <div className="detail-row">
                <span className="detail-label">Account Holder:</span>
                <span className="detail-value">{currentUser?.email}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Account Number:</span>
                <span className="detail-value">XXXX-XXXX-7890</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Account Type:</span>
                <span className="detail-value">Premium Savings</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Balance:</span>
                <span className="detail-value">₹10,000.00</span>
              </div>
            </div>
          </div>
        );
      case 'transfer':
        return (
          <div className="tab-content">
            <h3>Transfer Money</h3>
            <div className="transfer-form">
              <div className="form-group">
                <label>Recipient Account</label>
                <input type="text" placeholder="Enter account number" />
              </div>
              <div className="form-group">
                <label>Amount (₹)</label>
                <input type="number" placeholder="0.00" />
              </div>
              <div className="form-group">
                <label>Description</label>
                <input type="text" placeholder="Optional" />
              </div>
              <button className="transfer-button">Transfer Now</button>
            </div>
          </div>
        );
      case 'trust':
        return (
          <div className="tab-content">
            <h3>Behavioral Trust Score</h3>
            <div className="trust-score-info">
              <p>Your trust score is continuously updated based on your interaction patterns.</p>
              <BehaviorTracker 
                trustScore={trustScore}
                trustStatus={trustStatus}
              />
              
              <div className="trust-score-chart-container">
                <h4>Trust Score History</h4>
                <TrustScoreChart data={trustScoreHistory} />
              </div>
              
              <div className="context-controls">
                <h4>Security Settings</h4>
                <div className="toggle-group">
                  <label>
                    <input 
                      type="checkbox" 
                      checked={safeMode}
                      onChange={() => setSafeMode(!safeMode)}
                    />
                    <span>Safe Mode (reduces tracking intensity)</span>
                  </label>
                </div>
                <div className="toggle-group">
                  <label>
                    <input 
                      type="checkbox" 
                      checked={batterySaverMode}
                      onChange={() => setBatterySaverMode(!batterySaverMode)}
                    />
                    <span>Battery Saver Mode (reduces tracking frequency)</span>
                  </label>
                </div>
                {geoLocation ? (
                  <div className="location-info">
                    <p>Current location: {geoLocation.latitude.toFixed(4)}, {geoLocation.longitude.toFixed(4)}</p>
                  </div>
                ) : (
                  <div className="location-error">
                    {locationError || 'Location data not available'}
                  </div>
                )}
              </div>

              <button 
                className="privacy-button"
                onClick={() => setShowPrivacyDashboard(true)}
              >
                View Privacy Dashboard
              </button>

              <div className="trust-tips">
                <h4>Tips to maintain high trust score:</h4>
                <ul>
                  <li>Use consistent typing patterns</li>
                  <li>Avoid rapid, erratic mouse movements</li>
                  <li>Use the same device when possible</li>
                  <li>Keep location services enabled</li>
                </ul>
              </div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="dashboard">
      {showPrivacyDashboard && (
        <PrivacyDashboard 
          onClose={() => setShowPrivacyDashboard(false)}
          trackingSettings={{
            safeMode,
            batterySaverMode,
            locationTracking: !batterySaverMode && geoLocation !== null,
            panicKey
          }}
          onSettingsChange={(settings) => {
            setSafeMode(settings.safeMode);
            setBatterySaverMode(settings.batterySaverMode);
            setPanicKey(settings.panicKey);
          }}
        />
      )}

      <header className="dashboard-header">
        <div className="bank-logo">
          <h1>BEACON</h1>
        </div>
        <div className="user-info">
          <span className="welcome-message">Welcome, {currentUser?.email}</span>
          {currentUser && !currentUser.emailVerified && (
            <div className="verification-warning">
              Email not verified. Please check your inbox.
            </div>
          )}
          <button onClick={handleLogout} className="logout-button">
            Logout
          </button>
        </div>
      </header>

      <div className="dashboard-container">
        <nav className="dashboard-nav">
          <button
            className={`nav-button ${activeTab === 'account' ? 'active' : ''}`}
            onClick={() => setActiveTab('account')}
          >
            <i className="fas fa-user"></i> Account Info
          </button>
          <button
            className={`nav-button ${activeTab === 'transfer' ? 'active' : ''}`}
            onClick={() => setActiveTab('transfer')}
          >
            <i className="fas fa-exchange-alt"></i> Transfer Money
          </button>
          <button
            className={`nav-button ${activeTab === 'trust' ? 'active' : ''}`}
            onClick={() => setActiveTab('trust')}
          >
            <i className="fas fa-shield-alt"></i> Trust Score
          </button>
        </nav>

        <main className="dashboard-content">
          {renderTabContent()}
        </main>
      </div>
    </div>
  );
}