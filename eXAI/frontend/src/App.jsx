import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import NavBar from './components/NavBar'
import PredictorPage from './pages/PredictorPage'
import ExplainabilityPage from './pages/ExplainabilityPage'

export default function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <Routes>
        <Route path="/" element={<Navigate to="/predict" replace />} />
        <Route path="/predict" element={<PredictorPage />} />
        <Route path="/explain" element={<ExplainabilityPage />} />
      </Routes>
    </BrowserRouter>
  )
}
