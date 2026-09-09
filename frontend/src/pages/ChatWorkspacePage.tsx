import { useState, useEffect, useRef } from 'react'
import { MessageSquare, Library, HelpCircle, Send, CheckSquare, Square, Cpu, RefreshCw, Search, FileText, Mic, Image, Clock, Hash, ChevronDown, ChevronUp, Link2, Zap, Target, AlertTriangle } from 'lucide-react'
import { demoApi, retrievalApi } from '../api/client'
import clsx from 'clsx'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LinkedRecord {
  chunk_id: string
  source_type: string
  modality: string
  text_preview: string | null
  llava_description: string | null
}

interface RetrievalResult {
  chunk_id: string
  score: number
  source_document_id: string
  source_type: string
  modality: string
  chunk_index: number
  page_number: number | null
  text_preview: string | null
  extracted_text: string | null
  transcript_text: string | null
  llava_description: string | null
  original_filename: string | null
  timestamp_start: number | null
  timestamp_end: number | null
  image_index: number | null
  image_sha256: string | null
  has_text_vector: boolean
  has_image_vector: boolean
  linked_records: LinkedRecord[]
}

interface SearchMetrics {
  query: string
  query_vector_dim: number
  embedding_model: string
  encoding_time_ms: number
  search_time_ms: number
  hydration_time_ms: number
  total_time_ms: number
  result_count: number
}

interface SearchEntry {
  id: string
  query: string
  results: RetrievalResult[]
  metrics: SearchMetrics
  timestamp: Date
  error?: string
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const getSourceIcon = (type: string) => {
  switch (type) {
    case 'pdf': return <FileText className="w-3.5 h-3.5" />
    case 'docx':
    case 'doc': return <FileText className="w-3.5 h-3.5" />
    case 'image':
    case 'screenshot':
    case 'pdf_extracted_image': return <Image className="w-3.5 h-3.5" />
    case 'audio': return <Mic className="w-3.5 h-3.5" />
    case 'text_note': return <FileText className="w-3.5 h-3.5" />
    case 'image_description':
    case 'screenshot_description':
    case 'pdf_image_description': return <Image className="w-3.5 h-3.5" />
    default: return <FileText className="w-3.5 h-3.5" />
  }
}

const getSourceLabel = (type: string) => {
  switch (type) {
    case 'pdf': return 'PDF'
    case 'docx':
    case 'doc': return 'DOC'
    case 'image': return 'IMG'
    case 'screenshot': return 'SCRN'
    case 'pdf_extracted_image': return 'PDF-IMG'
    case 'audio': return 'AUD'
    case 'text_note': return 'TXT'
    case 'image_description': return 'IMG-DESC'
    case 'screenshot_description': return 'SCRN-DESC'
    case 'pdf_image_description': return 'PDF-IMG-DESC'
    default: return 'FILE'
  }
}

const formatScore = (score: number) => {
  const pct = (score * 100).toFixed(1)
  if (score >= 0.7) return { text: `${pct}%`, color: 'text-emerald-600 bg-emerald-50 border-emerald-200' }
  if (score >= 0.4) return { text: `${pct}%`, color: 'text-amber-600 bg-amber-50 border-amber-200' }
  return { text: `${pct}%`, color: 'text-red-500 bg-red-50 border-red-200' }
}

const formatTime = (ms: number) => {
  if (ms < 1) return '<1ms'
  if (ms < 1000) return `${ms.toFixed(0)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

// ---------------------------------------------------------------------------
// Result Card
// ---------------------------------------------------------------------------

function ResultCard({ result, rank }: { result: RetrievalResult; rank: number }) {
  const [expanded, setExpanded] = useState(false)
  const [showLinked, setShowLinked] = useState(false)
  const scoreInfo = formatScore(result.score)

  const contentText = result.extracted_text || result.transcript_text || result.llava_description || ''
  const previewText = result.text_preview || contentText.slice(0, 200)
  const hasFullContent = contentText.length > 200

  return (
    <div className="border-2 border-border bg-surface transition-all hover:border-brand/30">
      {/* Card header */}
      <div className="px-3 py-2.5 bg-surface-secondary border-b border-border-light flex items-center gap-2">
        {/* Rank badge */}
        <span className="w-6 h-6 flex items-center justify-center bg-brand/10 text-brand text-xs font-bold font-mono border border-brand/20 shrink-0">
          {rank}
        </span>

        {/* Score badge */}
        <span className={clsx("px-1.5 py-0.5 text-xs font-bold font-mono border", scoreInfo.color)}>
          {scoreInfo.text}
        </span>

        {/* Source type */}
        <span className="flex items-center gap-1 text-xs font-mono text-text-muted">
          {getSourceIcon(result.source_type)}
          <span className="uppercase font-bold">{getSourceLabel(result.source_type)}</span>
        </span>

        {/* Filename */}
        <span className="text-xs text-text-muted truncate flex-1" title={result.original_filename || ''}>
          {result.original_filename || 'Unknown source'}
        </span>

        {/* Location metadata */}
        {result.page_number != null && (
          <span className="text-xs font-mono text-text-muted bg-surface border border-border-light px-1.5 py-0.5 shrink-0">
            p.{result.page_number}
          </span>
        )}
        {result.timestamp_start != null && (
          <span className="text-xs font-mono text-text-muted bg-surface border border-border-light px-1.5 py-0.5 shrink-0">
            <Clock className="w-3 h-3 inline mr-0.5" />
            {result.timestamp_start.toFixed(1)}s–{result.timestamp_end?.toFixed(1) ?? '?'}s
          </span>
        )}
        {result.chunk_index != null && (
          <span className="text-xs font-mono text-text-muted bg-surface border border-border-light px-1.5 py-0.5 shrink-0">
            <Hash className="w-3 h-3 inline mr-0.5" />
            {result.chunk_index}
          </span>
        )}
      </div>

      {/* Content preview */}
      <div className="px-3 py-3">
        <p className="text-xs text-text-main leading-relaxed whitespace-pre-line">
          {expanded ? contentText : previewText}
          {!expanded && hasFullContent && '…'}
        </p>

        {hasFullContent && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-2 text-xs text-brand font-semibold flex items-center gap-1 cursor-pointer hover:underline"
          >
            {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            {expanded ? 'Show less' : 'Show full text'}
          </button>
        )}
      </div>

      {/* Linked records */}
      {result.linked_records.length > 0 && (
        <div className="border-t border-border-light">
          <button
            onClick={() => setShowLinked(!showLinked)}
            className="w-full px-3 py-2 text-xs text-text-muted font-mono flex items-center gap-1.5 cursor-pointer hover:bg-surface-secondary/50"
          >
            <Link2 className="w-3 h-3 text-brand" />
            <span className="font-bold uppercase tracking-wider">
              {result.linked_records.length} linked {result.linked_records.length === 1 ? 'record' : 'records'}
            </span>
            {showLinked ? <ChevronUp className="w-3 h-3 ml-auto" /> : <ChevronDown className="w-3 h-3 ml-auto" />}
          </button>

          {showLinked && (
            <div className="px-3 pb-3 space-y-2">
              {result.linked_records.map((lr) => (
                <div
                  key={lr.chunk_id}
                  className="p-2 bg-surface-secondary border border-border-light text-xs"
                >
                  <div className="flex items-center gap-2 mb-1">
                    {getSourceIcon(lr.source_type)}
                    <span className="font-bold font-mono uppercase text-text-muted">{getSourceLabel(lr.source_type)}</span>
                    <span className="text-text-muted">•</span>
                    <span className="text-text-muted capitalize">{lr.modality}</span>
                  </div>
                  {lr.llava_description && (
                    <p className="text-text-main leading-relaxed mt-1">{lr.llava_description}</p>
                  )}
                  {lr.text_preview && !lr.llava_description && (
                    <p className="text-text-main leading-relaxed mt-1">{lr.text_preview}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function ChatWorkspacePage() {
  const [documents, setDocuments] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const [searches, setSearches] = useState<SearchEntry[]>([])
  const [inputVal, setInputVal] = useState('')
  const [searching, setSearching] = useState(false)
  const [latestMetrics, setLatestMetrics] = useState<SearchMetrics | null>(null)
  const resultsEndRef = useRef<HTMLDivElement>(null)

  // Mobile active panel toggle state
  const [activePanel, setActivePanel] = useState<'chat' | 'sources' | 'metrics'>('chat')

  const fetchDocuments = async () => {
    try {
      const data = await demoApi.getLibrary()
      // Only show documents that are both processed AND indexed
      const searchable = data.filter(
        (d: any) =>
          d.ingestion_status === 'completed' &&
          (d.indexing_status === 'indexed' || d.indexing_status === 'completed_with_warnings')
      )
      setDocuments(searchable)

      // Auto-select all searchable documents
      if (searchable.length > 0) {
        setSelectedDocIds(searchable.map((d: any) => d.source_document_id))
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

  // Scroll to latest results
  useEffect(() => {
    if (resultsEndRef.current) {
      resultsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [searches])

  const handleToggleDoc = (id: string) => {
    setSelectedDocIds(prev =>
      prev.includes(id) ? prev.filter(dId => dId !== id) : [...prev, id]
    )
  }

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputVal.trim() || selectedDocIds.length === 0 || searching) return

    const query = inputVal.trim()
    setInputVal('')
    setSearching(true)

    const entryId = `search-${Date.now()}`

    try {
      const response = await retrievalApi.search(query, selectedDocIds)

      const metrics: SearchMetrics = {
        query: response.query,
        query_vector_dim: response.query_vector_dim,
        embedding_model: response.embedding_model,
        encoding_time_ms: response.encoding_time_ms,
        search_time_ms: response.search_time_ms,
        hydration_time_ms: response.hydration_time_ms,
        total_time_ms: response.total_time_ms,
        result_count: response.result_count,
      }

      setLatestMetrics(metrics)
      setSearches(prev => [
        ...prev,
        {
          id: entryId,
          query,
          results: response.results,
          metrics,
          timestamp: new Date(),
        },
      ])
    } catch (err: any) {
      const errorMsg = err?.response?.data?.detail || err?.message || 'Search failed'
      setSearches(prev => [
        ...prev,
        {
          id: entryId,
          query,
          results: [],
          metrics: {
            query,
            query_vector_dim: 0,
            embedding_model: '',
            encoding_time_ms: 0,
            search_time_ms: 0,
            hydration_time_ms: 0,
            total_time_ms: 0,
            result_count: 0,
          },
          timestamp: new Date(),
          error: errorMsg,
        },
      ])
    } finally {
      setSearching(false)
    }
  }

  // Count active chunks in selected documents
  const activeChunksCount = selectedDocIds.reduce((acc, id) => {
    const doc = documents.find((d: any) => d.source_document_id === id)
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
            Semantic search over your indexed knowledge base. Retrieved evidence is ranked by relevance.
          </p>
        </div>

        {/* Phase Info Banner */}
        <div className="flex flex-row md:flex-col items-center md:items-end justify-between md:justify-start border border-border-light p-2 md:p-0 md:border-0 rounded-sm bg-surface-secondary md:bg-transparent text-right shrink-0 gap-2">
          <span className="text-xs font-mono bg-status-success-bg text-status-success border border-status-success px-2 py-0.5 rounded-sm uppercase tracking-wider font-semibold">
            Phase 4: Retrieval Active
          </span>
          <span className="text-xs text-text-muted font-medium">
            Dense Vector Search — SentenceTransformer
          </span>
        </div>
      </div>

      {loading ? (
        <div className="flex-1 panel-2 flex items-center justify-center text-text-muted font-mono text-xs bg-surface">
          <RefreshCw className="w-5 h-5 animate-spin text-brand mr-2" />
          VERIFYING INDEXED KNOWLEDGE BASE...
        </div>
      ) : documents.length === 0 ? (
        /* Empty State */
        <div className="flex-1 panel-2 flex flex-col items-center justify-center text-center p-8 text-text-muted bg-surface">
          <Library className="w-12 h-12 text-border mb-3" />
          <h3 className="font-bold text-text-main text-sm">No indexed documents available</h3>
          <p className="text-xs max-w-xs mt-2 leading-relaxed">
            Upload and index documents in the Knowledge Base before searching. Only documents with <strong>indexed</strong> status appear here.
          </p>
          <div className="mt-4 flex gap-4 text-xs font-bold font-mono tracking-wider">
            <span className="text-status-warning bg-status-warning-bg border border-status-warning/45 px-3 py-1.5 rounded-sm">
              Status: No Indexed Documents
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
              Search
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
                {documents.map((doc: any) => {
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
                            {getSourceLabel(doc.source_type)}
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

            {/* Column 2: Search Interface */}
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
                <span className="text-xs text-status-success bg-status-success-bg border border-status-success px-2 py-0.5 rounded-sm uppercase tracking-wider font-bold font-mono">
                  Retrieval Active
                </span>
              </div>

              {/* Results Area */}
              <div className="flex-1 p-4 overflow-y-auto space-y-6">
                {searches.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-6 text-text-muted">
                    <Search className="w-10 h-10 text-border mb-2" />
                    <h4 className="font-bold text-text-main text-sm">Search your knowledge base</h4>
                    <p className="text-xs max-w-xs mt-1 leading-relaxed">
                      Enter a query below to retrieve semantically similar evidence from your indexed documents.
                    </p>
                    <div className="mt-4 flex flex-wrap gap-2 max-w-sm justify-center">
                      {['What are the key findings?', 'Summarize the methodology', 'List all conclusions'].map((suggestion) => (
                        <button
                          key={suggestion}
                          onClick={() => setInputVal(suggestion)}
                          className="text-xs px-2.5 py-1.5 border border-border-light bg-surface-secondary text-text-muted hover:text-brand hover:border-brand/30 cursor-pointer transition-colors"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  searches.map((entry) => (
                    <div key={entry.id} className="space-y-3">
                      {/* Query bubble */}
                      <div className="flex justify-end">
                        <div className="px-3 py-2.5 bg-surface-secondary border-2 border-border-light text-xs text-text-main max-w-[80%]">
                          <div className="flex items-center gap-1.5 mb-1 text-text-muted">
                            <Search className="w-3 h-3" />
                            <span className="font-mono uppercase tracking-wider font-bold text-[10px]">Query</span>
                          </div>
                          {entry.query}
                        </div>
                      </div>

                      {/* Error state */}
                      {entry.error && (
                        <div className="flex gap-2 items-start p-3 bg-red-50 border-2 border-red-200 text-xs">
                          <AlertTriangle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                          <div>
                            <span className="font-bold text-red-700">Search failed</span>
                            <p className="text-red-600 mt-1">{entry.error}</p>
                          </div>
                        </div>
                      )}

                      {/* Results header */}
                      {!entry.error && (
                        <div className="flex items-center gap-2 text-xs text-text-muted font-mono">
                          <Target className="w-3.5 h-3.5 text-brand" />
                          <span className="uppercase tracking-wider font-bold">
                            Retrieved Evidence
                          </span>
                          <span className="text-border-light">—</span>
                          <span>{entry.results.length} result{entry.results.length !== 1 ? 's' : ''}</span>
                          <span className="text-border-light">•</span>
                          <span>{formatTime(entry.metrics.total_time_ms)}</span>
                        </div>
                      )}

                      {/* No results */}
                      {!entry.error && entry.results.length === 0 && (
                        <div className="p-4 bg-surface-secondary border border-border-light text-xs text-text-muted text-center">
                          <HelpCircle className="w-6 h-6 text-border mx-auto mb-2" />
                          <p className="font-semibold text-text-main">No matching evidence found</p>
                          <p className="mt-1">Try rephrasing your query or selecting additional source documents.</p>
                        </div>
                      )}

                      {/* Result cards */}
                      {entry.results.map((result, i) => (
                        <ResultCard key={result.chunk_id} result={result} rank={i + 1} />
                      ))}
                    </div>
                  ))
                )}

                {/* Searching indicator */}
                {searching && (
                  <div className="flex items-center gap-3 p-4 bg-surface-secondary border-2 border-brand/20 text-xs text-text-muted animate-pulse">
                    <RefreshCw className="w-4 h-4 text-brand animate-spin" />
                    <span className="font-mono uppercase tracking-wider font-bold">
                      Encoding query & searching vectors...
                    </span>
                  </div>
                )}

                <div ref={resultsEndRef} />
              </div>

              {/* Input Form */}
              <form onSubmit={handleSearch} className="p-3 border-t-2 border-border bg-surface-secondary flex gap-2 shrink-0">
                <input
                  type="text"
                  placeholder={selectedDocIds.length === 0 ? "Select at least one source document to start..." : "Search your knowledge base..."}
                  disabled={selectedDocIds.length === 0 || searching}
                  value={inputVal}
                  onChange={(e) => setInputVal(e.target.value)}
                  className="input-field text-xs flex-1"
                />

                <button
                  type="submit"
                  disabled={!inputVal.trim() || selectedDocIds.length === 0 || searching}
                  className="btn-primary w-11 h-11 sm:h-auto sm:px-4 p-0 flex items-center justify-center shrink-0 cursor-pointer text-xs"
                >
                  {searching ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </button>
              </form>
            </div>

            {/* Column 3: Retrieval Metrics */}
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

                {/* Dense Vector Search — active */}
                <div className="space-y-1">
                  <label className="text-xs text-text-muted block font-bold font-mono uppercase tracking-wider">Dense Vector Search</label>
                  <div className="p-2.5 bg-surface-secondary border border-border-light text-xs text-text-muted flex items-start gap-2 rounded-sm">
                    <Zap className="w-4 h-4 text-status-success shrink-0 mt-0.5" />
                    <div>
                      <span className="font-bold text-text-main block text-xs">SentenceTransformer (384d)</span>
                      <p className="mt-1 leading-relaxed">Cosine similarity search over the <code className="bg-surface px-0.5">text</code> named vector space.</p>
                    </div>
                  </div>
                </div>

                {/* Latest Search Timing */}
                {latestMetrics && (
                  <div className="space-y-1">
                    <label className="text-xs text-text-muted block font-bold font-mono uppercase tracking-wider">Latest Search Timing</label>
                    <div className="bg-surface-secondary border border-border-light p-3 text-xs space-y-2 font-mono rounded-sm">
                      <div className="flex justify-between text-text-muted">
                        <span>Query Encoding:</span>
                        <span className="text-text-main font-semibold">{formatTime(latestMetrics.encoding_time_ms)}</span>
                      </div>
                      <div className="flex justify-between text-text-muted">
                        <span>Qdrant Search:</span>
                        <span className="text-text-main font-semibold">{formatTime(latestMetrics.search_time_ms)}</span>
                      </div>
                      <div className="flex justify-between text-text-muted">
                        <span>SQLite Hydration:</span>
                        <span className="text-text-main font-semibold">{formatTime(latestMetrics.hydration_time_ms)}</span>
                      </div>
                      <div className="border-t border-border-light pt-2 flex justify-between text-text-main font-bold">
                        <span>Total:</span>
                        <span>{formatTime(latestMetrics.total_time_ms)}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Active Retrieval Pool */}
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
                      <span className="text-text-main font-semibold">Top 5</span>
                    </div>
                    {latestMetrics && (
                      <div className="flex justify-between text-text-muted">
                        <span>Last Results:</span>
                        <span className="text-text-main font-semibold">{latestMetrics.result_count} hits</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Embedding Model Info */}
                {latestMetrics && latestMetrics.embedding_model && (
                  <div className="space-y-1">
                    <label className="text-xs text-text-muted block font-bold font-mono uppercase tracking-wider">Embedding Model</label>
                    <div className="bg-surface-secondary border border-border-light p-3 text-xs space-y-2 font-mono rounded-sm">
                      <div className="flex justify-between text-text-muted">
                        <span>Model:</span>
                        <span className="text-text-main font-semibold text-right truncate max-w-[140px]" title={latestMetrics.embedding_model}>
                          {latestMetrics.embedding_model.split('/').pop()}
                        </span>
                      </div>
                      <div className="flex justify-between text-text-muted">
                        <span>Dimensions:</span>
                        <span className="text-text-main font-semibold">{latestMetrics.query_vector_dim}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Status */}
                {!latestMetrics && (
                  <div className="p-3 bg-surface-secondary border border-border-light text-xs text-text-muted flex gap-2 rounded-sm">
                    <Search className="w-4 h-4 text-brand shrink-0 mt-0.5" />
                    <div>
                      <strong className="text-text-main">Ready for search</strong>
                      <p className="mt-1 leading-relaxed">
                        Metrics will populate after your first query.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  )
}
