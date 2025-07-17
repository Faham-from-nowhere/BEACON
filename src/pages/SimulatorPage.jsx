import { useState, useRef, useEffect } from 'react';
import { saveAs } from 'file-saver';
import BehaviorTracker from '../components/BehaviorTracker';
import '../styles/Simulator.css';

export default function SimulatorPage() {
  const [behaviorMode, setBehaviorMode] = useState('normal'); // 'normal' or 'anomalous'
  const [recordedSessions, setRecordedSessions] = useState([]);
  const behaviorDataRef = useRef(null);

  // Label and save the current session
  const saveSession = () => {
    if (!behaviorDataRef.current) return;

    const sessionData = {
      label: behaviorMode,
      timestamp: new Date().toISOString(),
      data: {
        clicks: behaviorDataRef.current.clickEvents,
        keystrokes: behaviorDataRef.current.typingPatterns,
        mouseMovements: behaviorDataRef.current.mouseMovements,
        deviceInfo: behaviorDataRef.current.deviceInfo
      }
    };

    setRecordedSessions(prev => [...prev, sessionData]);
    resetTracker();
  };

  const resetTracker = () => {
    if (behaviorDataRef.current) {
      behaviorDataRef.current.clickEvents = [];
      behaviorDataRef.current.typingPatterns = [];
      behaviorDataRef.current.mouseMovements = [];
    }
  };

  const exportData = () => {
    const blob = new Blob([JSON.stringify(recordedSessions, null, 2)], {
      type: 'application/json'
    });
    saveAs(blob, `behavior_data_${Date.now()}.json`);
  };

  // Generate labeled mock data
  const generateMockData = () => {
    if (!behaviorDataRef.current) return;

    const mockClicks = Array(behaviorMode === 'normal' ? 5 : 20).fill().map(() => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      timestamp: Date.now() - Math.random() * 1000
    }));

    behaviorDataRef.current.clickEvents.push(...mockClicks);
    saveSession(); // Auto-save the mock session
  };

  return (
    <div className="simulator-container">
      <h1>Behavior Simulator</h1>
      
      <div className="control-panel">
        <div className="mode-selector">
          <label>
            Behavior Mode:
            <select 
              value={behaviorMode} 
              onChange={(e) => setBehaviorMode(e.target.value)}
            >
              <option value="normal">Normal</option>
              <option value="anomalous">Anomalous</option>
            </select>
          </label>
          <span className={`mode-indicator ${behaviorMode}`}>
            {behaviorMode.toUpperCase()}
          </span>
        </div>

        <div className="action-buttons">
          <button onClick={generateMockData} className="mock-button">
            Generate {behaviorMode} Data
          </button>
          <button onClick={saveSession} className="save-button">
            Save Current Session
          </button>
          <button 
            onClick={exportData} 
            disabled={recordedSessions.length === 0}
            className="export-button"
          >
            Export All Data ({recordedSessions.length} sessions)
          </button>
        </div>
      </div>

      <div className="data-preview">
        <h3>Current Session Data</h3>
        <div className="data-counters">
          <p>Clicks: {behaviorDataRef.current?.clickEvents?.length || 0}</p>
          <p>Keystrokes: {behaviorDataRef.current?.typingPatterns?.length || 0}</p>
          <p>Mouse Moves: {behaviorDataRef.current?.mouseMovements?.length || 0}</p>
        </div>
      </div>

      <div className="test-area">
        <h3>Test Interactions</h3>
        <div className="test-buttons">
          <button>Test Button 1</button>
          <button>Test Button 2</button>
          <input type="text" placeholder="Type here..." />
        </div>
      </div>

      <BehaviorTracker ref={behaviorDataRef} />
    </div>
  );
}