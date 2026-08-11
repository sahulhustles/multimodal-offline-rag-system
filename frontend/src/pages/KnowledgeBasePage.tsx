import { useState, useEffect, useRef } from 'react'
import { Library, Upload, Trash2, RefreshCw, AlertCircle, FileText, CheckCircle2, X, Layers, Plus } from 'lucide-react'
import { demoApi } from '../api/client'
import clsx from 'clsx'

export default function KnowledgeBasePage() {
  const [documents, setDocuments] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [selectedDoc, setSelectedDoc] = useState<any>(null)
  
  // Text Note Form State
  const [showNoteModal, setShowNoteModal] = useState(false)
  const [noteTitle, setNoteTitle] = useState('')
  const [noteContent, setNoteContent] = useState('')
  const [noteSubmitting, setNoteSubmitting] = useState(false)

  // Filters
  const [statusFilter, setStatusFilter] = useState('all')
  const [typeFilter, setTypeFilter] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')

  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchDocuments = async (showLoading = false) => {
    if (showLoading) setLoading(true)
    try {
      const data = await demoApi.getLibrary()
      setDocuments(data)
      
      // Update selected doc if it is in the list
      if (selectedDoc) {
        const freshDoc = data.find((d: any) => d.source_document_id === selectedDoc.source_document_id)
        if (freshDoc) setSelectedDoc(freshDoc)
      }
    } catch (err) {
      console.error('Failed to fetch documents', err)
    } finally {
      if (showLoading) setLoading(false)
    }
  }

  useEffect(() => {
    fetchDocuments(true)
  }, [])

  // Poll for document status if any document is queued or processing
  useEffect(() => {
    const hasActiveJobs = documents.some(
      doc => doc.ingestion_status === 'queued' || doc.ingestion_status === 'processing'
    )

    if (hasActiveJobs) {
      const interval = setInterval(() => {
        fetchDocuments(false)
      }, 2500)
      return () => clearInterval(interval)
    }
  }, [documents])

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setUploading(true)
    try {
      for (let i = 0; i < files.length; i++) {
        await demoApi.uploadToLibrary(files[i])
      }
      await fetchDocuments(false)
    } catch (err) {
      alert('Upload failed: ' + err)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleNoteSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!noteTitle || !noteContent) return

    setNoteSubmitting(true)
    try {
      await demoApi.uploadTextNote(noteTitle, noteContent)
      setNoteTitle('')
      setNoteContent('')
      setShowNoteModal(false)
      await fetchDocuments(false)
    } catch (err) {
      alert('Failed to save text note: ' + err)
    } finally {
      setNoteSubmitting(false)
    }
  }

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Are you sure you want to delete this document and all its local previews?')) return

    try {
      await demoApi.deleteFromLibrary(id)
      if (selectedDoc?.source_document_id === id) {
        setSelectedDoc(null)
      }
      await fetchDocuments(false)
    } catch (err) {
      alert('Delete failed')
    }
  }

  const handleClearAll = async () => {
    if (!confirm('Are you sure you want to delete ALL documents from your library?')) return

    try {
      await demoApi.clearLibrary()
      setSelectedDoc(null)
      await fetchDocuments(false)
    } catch (err) {
      alert('Failed to clear library')
    }
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

  const formatSize = (bytes: number) => {
    if (!bytes) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  // Filter documents
  const filteredDocs = documents.filter(doc => {
    const matchesSearch = doc.original_filename.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = statusFilter === 'all' || doc.ingestion_status === statusFilter
    const matchesType = typeFilter === 'all' || doc.source_type === typeFilter
    return matchesSearch && matchesStatus && matchesType
  })

  return (
    <div className="space-y-6 max-w-7xl h-full flex flex-col relative">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b-2 border-border pb-4 gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-text-main flex items-center gap-3">
            <Library className="w-8 h-8 text-brand" />
            Knowledge Base
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Persist and monitor local data sources. Files are chunked and embedded in the offline environment.
          </p>
        </div>
        <div className="flex gap-3 shrink-0">
          <button 
            onClick={() => setShowNoteModal(true)} 
            className="btn-secondary text-xs"
          >
            <Plus className="w-4 h-4" />
            Add Note
          </button>
          
          {/* Strongly emphasized primary button (limited to one on screen) */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="btn-primary text-xs"
          >
            {uploading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            Add Data to Library
          </button>
          
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="hidden" 
            multiple 
            accept=".pdf,.docx,.doc,.png,.jpg,.jpeg,.webp,.wav,.mp3,.m4a" 
          />
        </div>
      </div>

      {/* Control Bar: Search & Filters */}
      <div className="panel-1 p-3 flex flex-col sm:flex-row gap-4 items-center justify-between bg-surface">
        <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto flex-1">
          <input
            type="text"
            placeholder="Search by file name..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input-field text-xs w-full sm:w-72"
          />
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="input-field text-xs w-full sm:w-36"
          >
            <option value="all">All Types</option>
            <option value="pdf">PDF Docs</option>
            <option value="docx">Word Files</option>
            <option value="image">Images</option>
            <option value="audio">Audio Clips</option>
            <option value="text_note">Text Notes</option>
          </select>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="input-field text-xs w-full sm:w-36"
          >
            <option value="all">All Statuses</option>
            <option value="queued">Queued</option>
            <option value="processing">Processing</option>
            <option value="completed">Processed</option>
            <option value="failed">Failed</option>
          </select>
        </div>
        
        {documents.length > 0 && (
          <button 
            onClick={handleClearAll} 
            className="text-xs text-status-error hover:underline flex items-center gap-1.5 cursor-pointer font-bold font-mono tracking-wider uppercase border border-status-error/30 px-2.5 py-1.5 bg-status-error-bg hover:bg-status-error-bg/30"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Wipe Library
          </button>
        )}
      </div>

      <div className="flex flex-1 gap-6 overflow-hidden min-h-[400px]">
        {/* Document Inventory Table */}
        <div className="flex-1 panel-2 overflow-y-auto bg-surface">
          {loading ? (
            <div className="flex h-64 items-center justify-center text-text-muted font-mono text-xs">
              <RefreshCw className="w-5 h-5 animate-spin text-brand mr-2" />
              RETRIEVING SOURCE DOCUMENT REGISTRY...
            </div>
          ) : filteredDocs.length === 0 ? (
            <div className="flex flex-col h-64 items-center justify-center text-center p-8 text-text-muted">
              <Library className="w-10 h-10 mb-2 text-border" />
              <h3 className="font-semibold text-text-main text-sm">Registry is empty</h3>
              <p className="text-xs max-w-xs mt-1">
                Upload files or submit text notes above to initialize your local offline knowledge base.
              </p>
            </div>
          ) : (
            <>
              <div className="hidden md:block">
                <table className="w-full text-left border-collapse font-medium text-xs">
                  <thead>
                    <tr className="border-b border-border bg-surface-secondary text-xs text-text-muted uppercase tracking-wider font-mono">
                      <th className="p-3 w-16 text-center">Format</th>
                      <th className="p-3">Source Name</th>
                      <th className="p-3 w-28">File Size</th>
                      <th className="p-3 w-32">Status</th>
                      <th className="p-3 w-24">Chunks</th>
                      <th className="p-3 w-12"></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border-light">
                    {filteredDocs.map(doc => {
                      const isSelected = selectedDoc?.source_document_id === doc.source_document_id
                      return (
                        <tr
                          key={doc.source_document_id}
                          onClick={() => setSelectedDoc(doc)}
                          className={clsx(
                            "hover:bg-surface-hover/50 transition-colors cursor-pointer",
                            isSelected && "bg-surface-secondary border-l-2 border-brand"
                          )}
                        >
                          <td className="p-3 text-center">
                            <span className="font-mono text-xs font-bold bg-surface-secondary border border-border-light px-1.5 py-0.5 text-text-muted">
                              {getSourceIcon(doc.source_type)}
                            </span>
                          </td>
                          <td className="p-3 text-text-main font-semibold max-w-xs truncate" title={doc.original_filename}>
                            {doc.original_filename}
                          </td>
                          <td className="p-3 text-text-muted font-mono">
                            {formatSize(doc.file_size_bytes)}
                          </td>
                          <td className="p-3">
                            <span className={clsx(
                              "px-2.5 py-1 rounded-sm text-xs font-bold uppercase tracking-wider font-mono inline-flex items-center gap-1",
                              doc.ingestion_status === 'completed' && "bg-status-success-bg text-status-success border border-status-success",
                              doc.ingestion_status === 'processing' && "bg-status-warning-bg text-status-warning border border-status-warning animate-pulse",
                              doc.ingestion_status === 'queued' && "bg-[#1e1b4b]/10 text-[#4f46e5] border border-[#4f46e5]/40",
                              doc.ingestion_status === 'failed' && "bg-status-error-bg text-status-error border border-status-error"
                            )}>
                              {doc.ingestion_status}
                            </span>
                          </td>
                          <td className="p-3 text-text-muted font-mono">
                            {doc.metadata?.chunk_count ?? '-'}
                          </td>
                          <td className="p-3 text-right">
                            <button
                              onClick={(e) => handleDelete(doc.source_document_id, e)}
                              className="text-text-muted hover:text-status-error p-1 cursor-pointer transition-colors"
                              title="Delete source"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              {/* Mobile Document Cards List */}
              <div className="block md:hidden divide-y divide-border-light">
                {filteredDocs.map(doc => {
                  const isSelected = selectedDoc?.source_document_id === doc.source_document_id
                  return (
                    <div
                      key={doc.source_document_id}
                      onClick={() => setSelectedDoc(doc)}
                      className={clsx(
                        "p-4 cursor-pointer transition-colors space-y-3",
                        isSelected ? "bg-surface-secondary border-l-4 border-brand" : "hover:bg-surface-hover/30"
                      )}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-bold bg-surface-secondary border border-border-light px-2 py-0.5 text-text-muted shrink-0">
                            {getSourceIcon(doc.source_type)}
                          </span>
                          <h4 className="font-bold text-text-main text-xs break-words" style={{ overflowWrap: 'anywhere' }}>
                            {doc.original_filename}
                          </h4>
                        </div>
                        <button
                          onClick={(e) => handleDelete(doc.source_document_id, e)}
                          className="text-text-muted hover:text-status-error w-10 h-10 border border-border-light bg-surface flex items-center justify-center rounded-sm shrink-0 cursor-pointer"
                          title="Delete source"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      </div>

                      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-text-muted font-mono">
                        <span>Size: <strong className="text-text-main">{formatSize(doc.file_size_bytes)}</strong></span>
                        <span>Chunks: <strong className="text-text-main">{doc.metadata?.chunk_count ?? '-'}</strong></span>
                      </div>

                      <div>
                        <span className={clsx(
                          "px-2.5 py-1 rounded-sm text-xs font-bold uppercase tracking-wider font-mono inline-flex items-center gap-1",
                          doc.ingestion_status === 'completed' && "bg-status-success-bg text-status-success border border-status-success",
                          doc.ingestion_status === 'processing' && "bg-status-warning-bg text-status-warning border border-status-warning animate-pulse",
                          doc.ingestion_status === 'queued' && "bg-[#1e1b4b]/10 text-[#4f46e5] border border-[#4f46e5]/40",
                          doc.ingestion_status === 'failed' && "bg-status-error-bg text-status-error border border-status-error"
                        )}>
                          {doc.ingestion_status}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </div>

        {/* Selected Document Details Sidebar Drawer */}
        {selectedDoc && (
          <>
            {/* Backdrop wrapper for mobile overlay dismiss click */}
            <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSelectedDoc(null)}></div>
            <div className="fixed inset-y-0 right-0 z-50 lg:relative lg:inset-auto w-full sm:w-96 bg-surface border-l-2 lg:border-2 border-border flex flex-col overflow-hidden shadow-[4px_4px_0px_0px_#000000] lg:shadow-none h-full lg:h-auto">
            <div className="p-4 bg-surface-secondary border-b-2 border-border flex justify-between items-center">
              <h2 className="font-bold text-sm text-text-main flex items-center gap-2">
                <Layers className="w-4 h-4 text-brand" />
                Source document info
              </h2>
              <button onClick={() => setSelectedDoc(null)} className="text-text-muted hover:text-text-main cursor-pointer">
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <div className="p-4 space-y-4 flex-1 overflow-y-auto">
              <div>
                <label className="label-text">File Name</label>
                <div className="text-sm font-bold text-text-main truncate" title={selectedDoc.original_filename}>
                  {selectedDoc.original_filename}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label-text">Source Type</label>
                  <div className="text-xs font-semibold text-text-main capitalize">
                    {selectedDoc.source_type.replace('_', ' ')}
                  </div>
                </div>
                <div>
                  <label className="label-text">Byte Size</label>
                  <div className="text-xs font-mono font-medium text-text-main">
                    {formatSize(selectedDoc.file_size_bytes)}
                  </div>
                </div>
              </div>

              <div className="border-t border-border/50 pt-3">
                <label className="label-text">Status Summary</label>
                <div className="text-xs font-semibold text-text-main mt-1 flex items-center gap-1.5">
                  {selectedDoc.ingestion_status === 'completed' && (
                    <>
                      <CheckCircle2 className="w-4 h-4 text-status-success shrink-0" />
                      <span>Phase 2: Processed locally</span>
                    </>
                  )}
                  {selectedDoc.ingestion_status === 'processing' && (
                    <>
                      <RefreshCw className="w-4 h-4 text-status-warning animate-spin shrink-0" />
                      <span>Extracting and chunking contents...</span>
                    </>
                  )}
                  {selectedDoc.ingestion_status === 'queued' && (
                    <>
                      <Layers className="w-4 h-4 text-[#818cf8] shrink-0" />
                      <span>Enqueued in library worker</span>
                    </>
                  )}
                  {selectedDoc.ingestion_status === 'failed' && (
                    <>
                      <AlertCircle className="w-4 h-4 text-status-error shrink-0" />
                      <span className="text-status-error">Ingestion failed</span>
                    </>
                  )}
                </div>
                
                {selectedDoc.ingestion_status === 'completed' && (
                  <div className="text-xs font-bold text-brand uppercase tracking-wider mt-1.5">
                    Ready for future retrieval (Phase 3 next)
                  </div>
                )}
              </div>

              {selectedDoc.error_message && (
                <div className="p-3 bg-status-error-bg border border-status-error/30 text-status-error text-xs">
                  <span className="font-mono uppercase tracking-wider block font-bold">Error detail:</span>
                  <p className="mt-1 leading-relaxed">{selectedDoc.error_message}</p>
                </div>
              )}

              {selectedDoc.metadata && (
                <div className="space-y-4 border-t border-border/50 pt-3">
                  <div>
                    <label className="label-text">Model Verification</label>
                    <div className="text-xs text-text-main font-mono p-2 bg-surface-secondary border border-border-light">
                      {selectedDoc.metadata.embedding_status}
                    </div>
                  </div>

                  <div>
                    <label className="label-text">Pipeline summary</label>
                    <p className="text-xs text-text-main bg-surface-secondary p-2.5 border border-border-light leading-relaxed">
                      {selectedDoc.metadata.summary}
                    </p>
                  </div>

                  {selectedDoc.metadata.previews?.llava_description && (
                    <div>
                      <label className="label-text">LLaVA Description (Image-to-Text)</label>
                      <p className="text-xs text-text-main bg-surface-secondary p-2.5 border border-border-light max-h-32 overflow-y-auto leading-relaxed">
                        {selectedDoc.metadata.previews.llava_description}
                      </p>
                    </div>
                  )}

                  {selectedDoc.metadata.previews?.chunk_previews && selectedDoc.metadata.previews.chunk_previews.length > 0 && (
                    <div>
                      <label className="label-text">Extract Previews (First 3 chunks)</label>
                      <div className="space-y-2.5 mt-1">
                        {selectedDoc.metadata.previews.chunk_previews.slice(0, 3).map((chk: string, idx: number) => (
                          <div key={idx} className="bg-surface-secondary p-2.5 border border-border-light text-xs text-text-muted">
                            <span className="font-mono text-xs font-bold text-brand uppercase block border-b border-border-light pb-1 mb-1.5">
                              Chunk {idx + 1} (512 tokens sliding)
                            </span>
                            <p className="line-clamp-4 leading-relaxed">{chk}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </>
      )}
      </div>

      {/* Add Text Note Modal */}
      {showNoteModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center p-4 z-50">
          <div className="bg-surface border-2 border-border w-full max-w-lg overflow-hidden flex flex-col shadow-[8px_8px_0px_0px_#000000]">
            <div className="p-4 bg-surface-secondary border-b-2 border-border flex justify-between items-center">
              <h3 className="font-bold text-sm text-text-main flex items-center gap-2">
                <FileText className="w-5 h-5 text-brand" />
                Add Text Note
              </h3>
              <button 
                onClick={() => setShowNoteModal(false)}
                className="text-text-muted hover:text-text-main cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <form onSubmit={handleNoteSubmit} className="p-4 space-y-4">
              <div>
                <label className="label-text">Note Title</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. project_meeting_notes"
                  value={noteTitle}
                  onChange={(e) => setNoteTitle(e.target.value)}
                  className="input-field text-xs w-full"
                />
              </div>
              
              <div>
                <label className="label-text">Text Note Content</label>
                <textarea
                  required
                  rows={8}
                  placeholder="Insert plain text content. The parser will partition it into overlapping windows and verify embeddings locally..."
                  value={noteContent}
                  onChange={(e) => setNoteContent(e.target.value)}
                  className="input-field text-xs w-full font-mono resize-none leading-relaxed"
                />
              </div>
              
              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowNoteModal(false)}
                  className="btn-secondary text-xs"
                  disabled={noteSubmitting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn-primary text-xs"
                  disabled={noteSubmitting}
                >
                  {noteSubmitting ? 'Saving...' : 'Add Note'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
