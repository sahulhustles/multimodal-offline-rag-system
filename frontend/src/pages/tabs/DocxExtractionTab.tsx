import { useState, useRef } from 'react'
import { Upload, Play, AlignLeft, Table, Heading, ImageIcon } from 'lucide-react'
import { demoApi } from '../../api/client'

export default function DocxExtractionTab() {
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
      const res = await demoApi.processDocx(file)
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
              accept=".docx" 
              onChange={handleFileChange}
            />
            <Upload className="w-8 h-8 text-text-muted mb-2" />
            <p className="text-sm font-medium text-text-main">
              {file ? file.name : 'Click or drag DOCX to upload'}
            </p>
          </div>

          <button 
            className={`btn-primary w-full ${(!file || loading) ? 'btn-disabled' : ''}`}
            onClick={handleRun}
            disabled={!file || loading}
          >
            <Play className="w-4 h-4" />
            {loading ? 'Extracting...' : 'Extract DOCX Locally'}
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
            <Metric label="Headings" value={result.heading_count} icon={Heading} />
            <Metric label="Paragraphs" value={result.paragraph_count} icon={AlignLeft} />
            <Metric label="Tables" value={result.table_count} icon={Table} />
            <Metric label="Embedded Images" value={result.embedded_image_count} icon={ImageIcon} />
          </div>
          
          <div className="p-3 bg-status-warning-bg border border-status-warning text-status-warning rounded text-sm text-center">
            Extracted only — Qdrant indexing is Phase 3.
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="card p-4">
              <h3 className="font-semibold text-brand mb-3 border-b border-border pb-2">Full Text Preview</h3>
              <p className="text-sm text-text-main leading-relaxed bg-background p-3 rounded border border-border h-[200px] overflow-y-auto whitespace-pre-wrap">
                {result.extracted_text_preview}
              </p>
            </div>
            
            <div className="card p-4 flex flex-col">
              <h3 className="font-semibold text-brand mb-3 border-b border-border pb-2">Image Assets</h3>
              <div className="flex-1 bg-background p-3 rounded border border-border flex flex-wrap gap-2 overflow-y-auto content-start h-[200px]">
                {result.extracted_images_metadata.length === 0 ? (
                   <div className="w-full h-full flex items-center justify-center text-sm italic opacity-50">No images extracted</div>
                ) : (
                  result.extracted_images_metadata.map((img: any, i: number) => (
                    <img 
                      key={i}
                      src={`http://localhost:8000${img.preview_url}`} 
                      alt={`Image ${i}`} 
                      className="h-24 object-cover border border-border rounded bg-surface"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect width="100" height="100" fill="%23222"/><text x="50" y="50" fill="%23666" text-anchor="middle" dy=".3em">Image</text></svg>'
                      }}
                    />
                  ))
                )}
              </div>
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
