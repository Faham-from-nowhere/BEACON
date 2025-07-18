import { Navigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function PrivateRoute({ children }) {
  const { currentUser, loading } = useAuth()

  if (loading) {
    return <div>Loading...</div>
  }

  return currentUser ? children : <Navigate to="/login" replace />
}