import React from 'react';
import '../styles/PrivacyDashboard.css';

const PrivacyDashboard = ({ onClose, trackingSettings, onSettingsChange }) => {
  const [localSettings, setLocalSettings] = React.useState(trackingSettings);
  const [panicKey, setPanicKey] = React.useState('');

  const handleSettingChange = (setting) => {
    const newSettings = {
      ...localSettings,
      [setting]: !localSettings[setting]
    };
    setLocalSettings(newSettings);
  };

  const handleKeySelect = (e) => {
    e.preventDefault();
    setPanicKey(e.key);
  };

  const handleSave = () => {
    onSettingsChange({
      ...localSettings,
      panicKey: panicKey || trackingSettings.panicKey
    });
    onClose();
  };

  React.useEffect(() => {
    setPanicKey(trackingSettings.panicKey || '');
  }, [trackingSettings.panicKey]);

  return (
    <div className="privacy-dashboard-overlay">
      <div className="privacy-dashboard">
        <div className="privacy-header">
          <h2>Privacy Dashboard</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>
        
        <div className="privacy-content">
          <div className="privacy-section">
            <h3>Data Collection Settings</h3>
            <div className="privacy-toggle">
              <label>
                <input 
                  type="checkbox" 
                  checked={localSettings.safeMode}
                  onChange={() => handleSettingChange('safeMode')}
                />
                <span>Safe Mode (reduces tracking intensity)</span>
              </label>
              <p className="toggle-description">
                When enabled, reduces the amount of behavioral data collected to protect your privacy.
              </p>
            </div>
            
            <div className="privacy-toggle">
              <label>
                <input 
                  type="checkbox" 
                  checked={localSettings.batterySaverMode}
                  onChange={() => handleSettingChange('batterySaverMode')}
                />
                <span>Battery Saver Mode</span>
              </label>
              <p className="toggle-description">
                Reduces tracking frequency to conserve battery life. Also disables location tracking.
              </p>
            </div>
          </div>
          
          <div className="privacy-section">
            <h3>Security Features</h3>
            <div className="panic-button-setting">
              <label>Panic Button Key:</label>
              <div className="key-selector">
                <input
                  type="text"
                  value={panicKey}
                  onKeyDown={handleKeySelect}
                  onClick={(e) => e.target.value = ''}
                  placeholder="Press any key"
                  readOnly
                />
                {panicKey && (
                  <button 
                    className="clear-button"
                    onClick={() => setPanicKey('')}
                  >
                    Clear
                  </button>
                )}
              </div>
              <p className="toggle-description">
                Pressing this key 3 times quickly will log you out immediately.
              </p>
            </div>
          </div>
          
          <div className="privacy-section">
            <h3>Collected Data</h3>
            <ul className="data-list">
              <li>Typing patterns (speed, rhythm, errors)</li>
              <li>Mouse movements and clicks</li>
              <li>Device information (browser, screen size)</li>
              <li>{localSettings.locationTracking ? 'Location data' : 'Location data (disabled)'}</li>
            </ul>
          </div>
          
          <div className="privacy-section">
            <h3>Data Usage</h3>
            <p>
              Your behavioral data is used solely for authentication purposes. 
              Data is encrypted and processed locally when possible.
            </p>
            <p>
              We do not share your behavioral data with third parties.
            </p>
          </div>
        </div>
        
        <div className="privacy-footer">
          <button className="save-button" onClick={handleSave}>
            Save Settings
          </button>
          <button className="cancel-button" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default PrivacyDashboard;