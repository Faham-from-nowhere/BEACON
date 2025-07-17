import '../styles/TrustScore.css';

const BehaviorTracker = ({ 
  trustScore, 
  trustStatus
}) => {
  const getBadge = (score) => {
    if (score >= 90) return 'gold';
    if (score >= 75) return 'silver';
    if (score >= 60) return 'bronze';
    if (score >= 40) return 'iron';
    return 'none';
  };

  const badge = getBadge(trustScore);

  return (
    <div className="trust-score-container">
      <div className="trust-score-header">
        <h3 className="trust-score-title">Current Trust Level</h3>
        <span className={`status-indicator status-${trustStatus}`}>
          {trustStatus === 'normal' ? 'Normal' : 'Anomalous'}
        </span>
      </div>

      <div className="score-display">
        <div className="score-value">{trustScore}%</div>
        <div className="score-bar">
          <div 
            className={`score-fill score-fill-${badge}`}
            style={{ width: `${trustScore}%` }}
          />
        </div>
      </div>

      <div className="badge-display">
        {badge !== 'none' && (
          <div className={`badge badge-${badge}`}>
            <div className="badge-icon">
              {badge === 'gold' && '🥇'}
              {badge === 'silver' && '🥈'}
              {badge === 'bronze' && '🥉'}
              {badge === 'iron' && '🔩'}
            </div>
            <div className="badge-label">
              {badge === 'gold' && 'Gold Trust'}
              {badge === 'silver' && 'Silver Trust'}
              {badge === 'bronze' && 'Bronze Trust'}
              {badge === 'iron' && 'Iron Trust'}
            </div>
          </div>
        )}
      </div>

      <div className="trust-score-footer">
        Last updated: {new Date().toLocaleTimeString()}
      </div>
    </div>
  );
};

export default BehaviorTracker;