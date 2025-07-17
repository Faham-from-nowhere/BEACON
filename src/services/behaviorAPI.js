import axios from 'axios';

export const sendBehaviorData = async (data) => {
  return axios.post('/api/behavior', data);
};