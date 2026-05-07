import { Navigate, Route, Routes } from 'react-router-dom'
import './App.css'
import MainMenu from './pages/MainMenu'
import TrainImages from './pages/TrainImages'
import TrainMnist from './pages/TrainMnist'

function App() {
  return (
    <div className="app-shell">
      <Routes>
        <Route path="/" element={<MainMenu />} />
        <Route path="/train-mnist" element={<TrainMnist />} />
        <Route path="/train-images" element={<TrainImages />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  )
}

export default App
