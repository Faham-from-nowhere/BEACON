import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import '../styles/TrustScoreChart.css';

const TrustScoreChart = ({ data }) => {
  // Format data for the chart with proper timestamps
  const chartData = data.map((entry) => ({
    score: entry.score,
    status: entry.status,
    timestamp: new Date(entry.timestamp),
    timeString: new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }));

  // Custom tick formatter for XAxis
  const formatXAxis = (tickItem) => {
    return new Date(tickItem).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Custom tooltip formatter
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const timeString = new Date(label).toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit'
      });
      
      return (
        <div className="custom-tooltip">
          <p className="tooltip-time">{timeString}</p>
          <p className="tooltip-score">{`Score: ${payload[0].value}%`}</p>
          <p className="tooltip-status">{`Status: ${payload[0].payload.status}`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="trust-score-chart">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp" 
            tickFormatter={formatXAxis}
            label={{ value: 'Time', position: 'insideBottomRight', offset: -5 }}
          />
          <YAxis 
            domain={[0, 100]} 
            label={{ value: 'Score', angle: -90, position: 'insideLeft' }} 
          />
          <Tooltip 
            content={<CustomTooltip />}
          />
          <Line 
            type="monotone" 
            dataKey="score" 
            stroke="#8884d8" 
            strokeWidth={2}
            dot={{ r: 4 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TrustScoreChart;