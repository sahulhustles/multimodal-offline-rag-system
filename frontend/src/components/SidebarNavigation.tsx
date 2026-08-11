import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Network, FlaskConical, Server, Library, MessageSquare } from 'lucide-react'
import { clsx } from 'clsx'

const navItems = [
  { name: 'Dashboard', path: '/', icon: LayoutDashboard },
  { name: 'Knowledge Base', path: '/knowledge-base', icon: Library },
  { name: 'Chat Workspace', path: '/chat', icon: MessageSquare },
  { name: 'Processor Lab', path: '/processor-lab', icon: FlaskConical },
  { name: 'System Status', path: '/system-status', icon: Server },
  { name: 'Architecture', path: '/architecture', icon: Network },
]

interface SidebarProps {
  onClose?: () => void
  isMobile?: boolean
}

export default function SidebarNavigation({ onClose, isMobile }: SidebarProps) {
  return (
    <div className={clsx("h-full flex flex-col bg-surface", !isMobile && "w-64 border-r-2 border-border")}>
      <div className="p-4 border-b-2 border-border shrink-0">
        <h1 className="text-lg font-bold text-text-main flex items-center gap-2">
          <FlaskConical className="w-5 h-5 text-brand" />
          Offline RAG Demo
        </h1>
        <p className="text-xs text-text-muted mt-1 font-mono uppercase tracking-wider font-semibold">Phase 1 & 2 Demonstrator</p>
      </div>
      
      <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.name}
              to={item.path}
              onClick={onClose}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-3 py-3 text-sm font-semibold rounded-none transition-colors border-l-2 cursor-pointer touch-target',
                  isActive
                    ? 'bg-surface-secondary text-brand border-l-brand'
                    : 'text-text-muted hover:bg-surface-hover/30 hover:text-text-main border-l-transparent'
                )
              }
            >
              <Icon className="w-5 h-5 shrink-0" />
              {item.name}
            </NavLink>
          )
        })}
      </nav>
      
      <div className="p-4 border-t-2 border-border shrink-0">
        <div className="text-xs font-semibold text-text-muted flex items-center justify-center gap-1.5 bg-surface-secondary py-2.5 rounded-none border border-border-light font-mono uppercase tracking-wider">
          <div className="w-2 h-2 rounded-full bg-status-success"></div>
          Local processing only
        </div>
      </div>
    </div>
  )
}
