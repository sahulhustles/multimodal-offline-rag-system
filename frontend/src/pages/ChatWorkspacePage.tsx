import { useState, useEffect } from 'react'
import { MessageSquare, Library, HelpCircle, Send, CheckSquare, Square, ShieldAlert, Cpu, Sparkles, RefreshCw } from 'lucide-react'
import { demoApi } from '../api/client'
import clsx from 'clsx'

interface Message {
  sender: 'user' | 'assistant'
  text: string
  timestamp: Date
}

export default function ChatWorkspacePage() {
  const [documents, setDocuments] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [inputVal, setInputVal] = useState('')
  
  // Mobile active panel toggle state ('chat' | 'sources' | 'metrics')
  const [activePanel, setActivePanel] = useState<'chat' | 'sources' | 'metrics'>('chat')

  const fetchDocuments = async () => {
    try {
      const data = await demoApi.getLibrary()
      // Only keep completed/processed documents
      const completed = data.filter((d: any) => d.ingestion_status === 'completed')
      setDocuments(completed)
      
      // Auto-select all by default if documents exist
      if (completed.length > 0) {
        setSelectedDocIds(completed.map((d: any) => d.source_document_id))
      }
    } catch (err) {
      console.error('Failed to load library', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [])

  const handleToggleDoc = (id: string) => {
    setSelectedDocIds(prev => 
      prev.includes(id) ? prev.filter(dId => dId !== id) : [...prev, id]
    )
  }

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputVal.trim() || selectedDocIds.length === 0) return

    const userMsg = inputVal
    setInputVal('')

    // Append user message
    setMessages(prev => [...prev, {
      sender: 'user',
      text: userMsg,
      timestamp: new Date()
    }])

    // Append structured assistant placeholder response
    setTimeout(() => {
      setMessages(prev => [...prev, {
        sender: 'assistant',
        text: `Chat retrieval and grounded answer generation will be enabled in Phase 3. \n\nYour selected documents are already parsed, chunked, and stored locally in the Knowledge Base. They are ready for indexing and querying once Phase 3 is implemented.`,
        timestamp: new Date()
      }])
    }, 600)
  }

  const getSourceIcon = (type: string) => {
    switch (type) {
      case 'pdf': return 'PDF'
      case 'docx':
      case 'doc': return 'DOC'
      case 'image': return 'IMG'
      case 'audio': return 'AUD'
      case 'text_note': return 'TXT'
      default: return 'FILE'
    }
  }

  // Count active chunks in selected documents
  const activeChunksCount = selectedDocIds.reduce((acc, id) => {
    const doc = documents.find(d => d.source_document_id === id)
    return acc + (doc?.metadata?.chunk_count || 0)
  }, 0)

  return (
    <div className="h-[calc(100vh-6rem)] md:h-[calc(100vh-8rem)] flex flex-col gap-4 max-w-7xl overflow-hidden">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b-2 border-border pb-3 gap-3 shrink-0">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-text-main flex items-center gap-3">
            <MessageSquare className="w-7 h-7 md:w-8 h-8 text-brand" />
            Chat Workspace
          </h1>
          <p className="text-text-muted text-xs md:text-sm mt-1">
            Grounded local questioning preview. Search operations are scheduled for Phase 3.
          </p>
        </div>
        
        {/* Phase Info Banner */}
        <div className="flex flex-row md:flex-col items-center md:items-end justify-between md:justify-start border border-border-light p-2 md:p-0 md:border-0 rounded-sm bg-surface-secondary md:bg-transparent text-right shrink-0 gap-2">
          <span className="text-xs font-mono bg-status-success-bg text-status-success border border-status-success px-2 py-0.5 rounded-sm uppercase tracking-wider font-semibold">
            Phase 2: Processed locally
          </span>
          <span className="text-xs text-text-muted font-medium">
            Phase 3: Q&A next
          </span>
        </div>
      </div>

      {loading ? (
        <div className="flex-1 panel-2 flex items-center justify-center text-text-muted font-mono text-xs bg-surface">
          <RefreshCw className="w-5 h-5 animate-spin text-brand mr-2" />
          VERIFYING SOURCE KNOWLEDGE FOR RETRIEVABILITY...
        </div>
      ) : documents.length === 0 ? (
        /* Empty State if no processed documents exist */
        <div className="flex-1 panel-2 flex flex-col items-center justify-center text-center p-8 text-text-muted bg-surface">
          <Library className="w-12 h-12 text-border mb-3" />
          <h3 className="font-bold text-text-main text-sm">No processed data available</h3>
          <p className="text-xs max-w-xs mt-2 leading-relaxed">
            You must first upload and process files in the Knowledge Base to make them available for querying.
          </p>
          <div className="mt-4 flex gap-4 text-xs font-bold font-mono tracking-wider">
            <span className="text-status-warning bg-status-warning-bg border border-status-warning/45 px-3 py-1.5 rounded-sm">
              Status: Blocked (Library Empty)
            </span>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col gap-4 overflow-hidden min-h-0">
          
          {/* Mobile/Tablet Segmented Workspace Panel Switcher */}
          <div className="flex lg:hidden border-2 border-border bg-surface p-1 rounded-sm shrink-0">
            <button
              onClick={() => setActivePanel('sources')}
              className={clsx(
                "flex-1 py-2 text-xs font-mono uppercase tracking-wider font-bold text-center touch-target cursor-pointer",
                activePanel === 'sources' ? "bg-surface-secondary text-brand border border-border-light" : "text-text-muted"
              )}
            >
              Sources ({selectedDocIds.length})
            </button>
            <button
              onClick={() => setActivePanel('chat')}
              className={clsx(
                "flex-1 py-2 text-xs font-mono uppercase tracking-wider font-bold text-center touch-target cursor-pointer",
                activePanel === 'chat' ? "bg-surface-secondary text-brand border border-border-light" : "text-text-muted"
              )}
            >
              Chat
            </button>
            <button
              onClick={() => setActivePanel('metrics')}
              className={clsx(
                "flex-1 py-2 text-xs font-mono uppercase tracking-wider font-bold text-center touch-target cursor-pointer",
                activePanel === 'metrics' ? "bg-surface-secondary text-brand border border-border-light" : "text-text-muted"
              )}
            >
              Metrics
            </button>
          </div>

          <div className="flex-1 flex gap-6 overflow-hidden min-h-0">
            
            {/* Column 1: Knowledge Sources List */}
            <div className={clsx(
              "w-full lg:w-72 bg-surface border-2 border-border flex flex-col overflow-hidden shrink-0",
              activePanel === 'sources' ? "flex animate-fade-in" : "hidden lg:flex"
            )}>
              <div className="p-3 bg-surface-secondary border-b border-border-light shrink-0">
                <h2 className="font-bold text-xs font-mono uppercase tracking-wider text-text-muted flex items-center gap-2">
                  <Library className="w-4 h-4 text-brand" />
                  Select active sources
                </h2>
              </div>
              
              <div className="flex-1 overflow-y-auto p-3 space-y-1.5 divide-y divide-border-light">
                {documents.map(doc => {
                  const isSelected = selectedDocIds.includes(doc.source_document_id)
                  return (
                    <button
                      key={doc.source_document_id}
                      onClick={() => handleToggleDoc(doc.source_document_id)}
                      className={clsx(
                        "w-full text-left p-3 border transition-all cursor-pointer flex items-start gap-2.5 touch-target rounded-none",
                        isSelected 
                          ? "bg-surface-secondary border-brand text-text-main shadow-[2px_2px_0px_0px_rgba(79,124,255,0.06)]" 
                          : "bg-transparent border-transparent hover:bg-surface-hover/30 text-text-muted"
                      )}
                    >
                      <span className="mt-0.5 shrink-0">
                        {isSelected ? (
                          <CheckSquare className="w-4 h-4 text-brand" />
                        ) : (
                          <Square className="w-4 h-4 text-border" />
                        )}
                      </span>
                      
                      <div className="min-w-0 flex-1">
                        <div className="font-semibold text-xs truncate">
                          <span className="font-mono text-xs font-bold bg-surface-secondary border border-border-light px-1 py-0.5 mr-1 text-text-muted">
                            {getSourceIcon(doc.source_type)}
                          </span>
                          {doc.original_filename}
                        </div>
                        <div className="text-xs text-text-muted mt-1.5 flex justify-between font-mono">
                          <span>{doc.metadata?.chunk_count || 0} chunks</span>
                          <span className="capitalize">{doc.source_type.replace('_', ' ')}</span>
                        </div>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Column 2: Chat Interface */}
            <div className={clsx(
              "flex-1 bg-surface border-2 border-border flex flex-col overflow-hidden min-w-0",
              activePanel === 'chat' ? "flex animate-fade-in" : "hidden lg:flex"
            )}>
              {/* Top Active Context Bar */}
              <div className="px-4 py-3 bg-surface-secondary border-b border-border-light flex items-center justify-between shrink-0">
                <div className="text-xs text-text-muted flex items-center gap-2 font-mono">
                  <span className="w-2 h-2 rounded-full bg-[#10b981] animate-pulse"></span>
                  <span>Active: <strong>{selectedDocIds.length}</strong> sources</span>
                  <span className="text-border-light">|</span>
                  <span className="hidden sm:inline">Scope: <strong>{activeChunksCount}</strong> chunks</span>
                </div>
                <span className="text-xs text-[#f59e0b] bg-status-warning-bg border border-status-warning/45 px-2 py-0.5 rounded-sm uppercase tracking-wider font-bold font-mono">
                  Phase 3 Shell
                </span>
              </div>

              {/* Conversation Area */}
              <div className="flex-1 p-4 overflow-y-auto space-y-4">
                {messages.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-6 text-text-muted">
                    <HelpCircle className="w-10 h-10 text-border mb-2" />
                    <h4 className="font-bold text-text-main text-sm">Ask a question</h4>
                    <p className="text-xs max-w-xs mt-1 leading-relaxed">
                      Submit questions over your selected source files. Grounded responses cite retrieved local text segments.
                    </p>
                  </div>
                ) : (
                  messages.map((msg, i) => (
                    <div 
                      key={i} 
                      className={clsx(
                        "flex gap-3 max-w-[90%] sm:max-w-[80%]",
                        msg.sender === 'user' ? "ml-auto flex-row-reverse" : "mr-auto"
                      )}
                    >
                      <div className={clsx(
                        "p-3 text-xs leading-relaxed whitespace-pre-line border rounded-sm",
                        msg.sender === 'user' 
                          ? "bg-surface-secondary border-border-light text-text-main" 
                          : "bg-white border-border-light text-text-muted"
                      )}>
                        {msg.text}
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Input Form */}
              <form onSubmit={handleSendMessage} className="p-3 border-t-2 border-border bg-surface-secondary flex gap-2 shrink-0">
                <input
                  type="text"
                  placeholder={selectedDocIds.length === 0 ? "Select at least one source document to start..." : "Ask a question over selected sources..."}
                  disabled={selectedDocIds.length === 0}
                  value={inputVal}
                  onChange={(e) => setInputVal(e.target.value)}
                  className="input-field text-xs flex-1"
                />
                
                {/* Only one primary button per screen - the Send button */}
                <button 
                  type="submit" 
                  disabled={!inputVal.trim() || selectedDocIds.length === 0}
                  className="btn-primary w-11 h-11 sm:h-auto sm:px-4 p-0 flex items-center justify-center shrink-0 cursor-pointer text-xs"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>

            {/* Column 3: Retrieval Metrics Shell */}
            <div className={clsx(
              "w-full lg:w-72 bg-surface border-2 border-border flex flex-col overflow-hidden shrink-0",
              activePanel === 'metrics' ? "flex animate-fade-in" : "hidden lg:flex"
            )}>
              <div className="p-3 bg-surface-secondary border-b border-border-light shrink-0">
                <h2 className="font-bold text-xs font-mono uppercase tracking-wider text-text-muted flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-brand" />
                  Retrieval Metrics
                </h2>
              </div>
              
              <div className="flex-1 p-4 space-y-4 overflow-y-auto">
                <div className="space-y-1">
                  <label className="text-xs text-text-muted block font-bold font-mono uppercase tracking-wider">Dense Vector Search</label>
                  <div className="p-2.5 bg-surface-secondary border border-border-light text-xs text-text-muted flex items-start gap-2 rounded-sm">
                    <ShieldAlert className="w-4 h-4 text-status-warning shrink-0 mt-0.5" />
                    <div>
                      <span className="font-bold text-text-main block text-xs">SentenceTransformer (384d)</span>
                      <p className="mt-1 leading-relaxed">Query vectorization and cosine similarity matching occur in Phase 3.</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-xs text-text-muted block font-bold font-mono uppercase tracking-wider">Sparse Keyword Search</label>
                  <div className="p-2.5 bg-surface-secondary border border-border-light text-xs text-text-muted flex items-start gap-2 rounded-sm">
                    <ShieldAlert className="w-4 h-4 text-status-warning shrink-0 mt-0.5" />
                    <div>
                      <span className="font-bold text-text-main block text-xs">BM25 / Keyword Index</span>
                      <p className="mt-1 leading-relaxed">BM25 scoring and sparse index merging are disabled in Phase 2.</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-xs text-text-muted block font-bold font-mono uppercase tracking-wider">Active Retrieval Pool</label>
                  <div className="bg-surface-secondary border border-border-light p-3 text-xs space-y-2 font-mono rounded-sm">
                    <div className="flex justify-between text-text-muted">
                      <span>Selected Sources:</span>
                      <span className="text-text-main font-semibold">{selectedDocIds.length}</span>
                    </div>
                    <div className="flex justify-between text-text-muted">
                      <span>Search Scope:</span>
                      <span className="text-text-main font-semibold">{activeChunksCount} chunks</span>
                    </div>
                    <div className="flex justify-between text-text-muted">
                      <span>Retrieval Limit:</span>
                      <span className="text-text-main font-semibold">Top 5 chunks</span>
                    </div>
                  </div>
                </div>

                <div className="p-3 bg-status-warning-bg border border-status-warning/45 text-xs text-text-muted flex gap-2 rounded-sm">
                  <Sparkles className="w-4 h-4 text-brand shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-text-main">Grounded answer synthesis</strong> is currently disabled. Chat retrieval will be enabled in Phase 3.
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  )
}
