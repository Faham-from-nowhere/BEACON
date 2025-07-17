export const getAuthUser = () => {
    return JSON.parse(localStorage.getItem('beacon_user'));
  };