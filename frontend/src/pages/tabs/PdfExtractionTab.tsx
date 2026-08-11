import { useState, useRef } from 'react'
import { Upload, Play, FileText, ImageIcon, AlignLeft } from 'lucide-react'
import { demoApi } from '../../api/client'

export default function PdfExtractionTab() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0]
    if (selected) {
      setFile(selected)
      setResult(null)
      setError('')
    }
  }

  const handleRun = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const res = await demoApi.processPdf(file)
      setResult(res)
    } catch (err: any) {
      setError(err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div 
            className="border-2 border-dashed border-border rounded-card p-8 flex flex-col items-center justify-center cursor-pointer hover:bg-surface-hover transition-colors min-h-[150px]"
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="application/pdf" 
              onChange={handleFileChange}
            />
            <Upload className="w-8 h-8 text-text-muted mb-2" />
            <p className="text-sm font-medium text-text-main">
              {file ? file.name : 'Click or drag PDF to upload'}
            </p>
          </div>

          <button 
            className={`btn-primary w-full ${(!file || loading) ? 'btn-disabled' : ''}`}
            onClick={handleRun}
            disabled={!file || loading}
          >
            <Play className="w-4 h-4" />
            {loading ? 'Extracting...' : 'Extract PDF Locally'}
          </button>
          
          {error && (
            <div className="p-3 bg-status-error-bg text-status-error border border-status-error rounded text-sm">
              {error}
            </div>
          )}
        </div>
      </div>

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Metric label="Total Pages" value={result.total_pages} icon={FileText} />
            <Metric label="Text Characters" value={result.extracted_text_character_count.toLocaleString()} icon={AlignLeft} />
            <Metric label="Images Extracted" value={result.extracted_image_count} icon={ImageIcon} />
            <Metric label="Total Chunks" value={result.total_chunk_count} icon={AlignLeft} />
          </div>
          
          <div className="p-3 bg-status-warning-bg border border-status-warning text-status-warning rounded text-sm text-center">
            Extracted only — Qdrant indexing is Phase 3.
          </div>

          <div className="space-y-4">
            <h3 className="font-semibold text-text-main border-b border-border pb-2">Page Results</h3>
            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
              {result.pages.map((page: any) => (
                <div key={page.page_number} className="card overflow-hidden">
                  <div className="bg-surface-secondary px-4 py-2 border-b border-border-light flex justify-between items-center text-sm">
                    <span className="font-semibold text-brand">Page {page.page_number}</span>
                    <span className="text-text-muted">{page.chunk_count} chunks • {page.text_character_count} chars</span>
                  </div>
                  <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="text-xs font-semibold text-text-muted mb-2 uppercase tracking-wider">Text Preview</h4>
                      <p className="text-sm text-text-main leading-relaxed bg-background p-3 rounded border border-border h-32 overflow-y-auto">
                        {page.extracted_text_preview || <span className="italic opacity-50">No text on this page</span>}
                      </p>
                    </div>
                    <div>
                      <h4 className="text-xs font-semibold text-text-muted mb-2 uppercase tracking-wider">Images ({page.extracted_images.length})</h4>
                      <div className="flex gap-2 overflow-x-auto h-32 items-center bg-background p-2 rounded border border-border">
                        {page.extracted_images.length === 0 ? (
                           <span className="text-sm italic opacity-50 w-full text-center">No images</span>
                        ) : (
                          page.extracted_images.map((img: any, i: number) => (
                            <div key={i} className="flex-shrink-0 relative group">
                              <img 
                                src={`http://localhost:8000${img.preview_url}`} 
                                alt={`Page ${page.page_number} img ${i}`} 
                                className="h-24 object-cover border border-border rounded"
                                onError={(e) => {
                                  // Fallback if image not found during dev
                                  (e.target as HTMLImageElement).src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect width="100" height="100" fill="%23222"/><text x="50" y="50" fill="%23666" text-anchor="middle" dy=".3em">Image</text></svg>'
                                }}
                              />
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="text-xs text-text-muted text-center mt-4">
            {result.message}
          </div>
        </div>
      )}
    </div>
  )
}

function Metric({ label, value, icon: Icon }: { label: string, value: string | number, icon: any }) {
  return (
    <div className="p-4 bg-background border border-border rounded shadow-sm flex flex-col items-center justify-center gap-1">
      <div className="text-brand mb-1"><Icon className="w-5 h-5" /></div>
      <div className="text-xl font-semibold text-text-main">{value}</div>
      <div className="text-xs text-text-muted">{label}</div>
    </div>
  )
}
