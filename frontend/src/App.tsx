import { Routes, Route, Navigate } from 'react-router-dom'
import AppLayout from './components/AppLayout'
import DashboardPage from './pages/DashboardPage'
import ArchitecturePage from './pages/ArchitecturePage'
import ProcessorLabPage from './pages/ProcessorLabPage'
import SystemStatusPage from './pages/SystemStatusPage'
import KnowledgeBasePage from './pages/KnowledgeBasePage'
import ChatWorkspacePage from './pages/ChatWorkspacePage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<AppLayout />}>
        <Route index element={<DashboardPage />} />
        <Route path="knowledge-base" element={<KnowledgeBasePage />} />
        <Route path="chat" element={<ChatWorkspacePage />} />
        <Route path="architecture" element={<ArchitecturePage />} />
        <Route path="processor-lab" element={<ProcessorLabPage />} />
        <Route path="system-status" element={<SystemStatusPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

export default App
