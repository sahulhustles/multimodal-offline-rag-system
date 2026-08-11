import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import SidebarNavigation from './SidebarNavigation'
import { Menu, X } from 'lucide-react'

export default function AppLayout() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  return (
    <div className="flex h-screen bg-background overflow-hidden relative">
      {/* Desktop Sidebar (hidden on mobile/tablet) */}
      <div className="hidden lg:flex lg:shrink-0">
        <SidebarNavigation />
      </div>

      {/* Mobile/Tablet Collapsible Menu (Slide-over drawer) */}
      {isMobileMenuOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden bg-black/60 transition-opacity">
          {/* Menu Container */}
          <div className="relative flex flex-col w-64 max-w-xs bg-surface border-r-2 border-border shadow-[4px_0px_10px_rgba(0,0,0,0.15)] animate-slide-in">
            {/* Close Button overlayed inside menu */}
            <div className="absolute top-3 right-3 z-10">
              <button
                onClick={() => setIsMobileMenuOpen(false)}
                className="w-10 h-10 border border-border bg-surface hover:bg-surface-secondary text-text-main flex items-center justify-center rounded-sm cursor-pointer"
                aria-label="Close menu"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            {/* Nav content */}
            <div className="h-full pt-10">
              <SidebarNavigation onClose={() => setIsMobileMenuOpen(false)} isMobile />
            </div>
          </div>
          
          {/* Click outside backdrop zone */}
          <div className="flex-1" onClick={() => setIsMobileMenuOpen(false)}></div>
        </div>
      )}

      {/* Main Page Content Wrapper */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top bar visible on mobile/tablet */}
        <header className="lg:hidden h-14 bg-surface border-b-2 border-border flex items-center justify-between px-4 shrink-0">
          <button
            onClick={() => setIsMobileMenuOpen(true)}
            className="w-10 h-10 border border-border bg-surface hover:bg-surface-secondary text-text-main flex items-center justify-center rounded-sm cursor-pointer"
            aria-label="Open navigation menu"
          >
            <Menu className="w-6 h-6" />
          </button>
          
          <span className="font-bold text-sm text-text-main tracking-tight uppercase font-mono">
            Offline RAG Demo
          </span>
          <div className="w-10 h-10"></div> {/* Balanced spacing */}
        </header>

        {/* Content area */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 w-full max-w-7xl mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
