import { useState, useRef } from 'react'
import { Upload, Play, Image as ImageIcon, AlertCircle } from 'lucide-react'
import { demoApi } from '../../api/client'

export default function ImageVisionTab() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0]
    if (selected) {
      setFile(selected)
      setPreview(URL.createObjectURL(selected))
      setResult(null)
      setError('')
    }
  }

  const handleRun = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const res = await demoApi.processImage(file)
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
            className="border-2 border-dashed border-border rounded-card p-8 flex flex-col items-center justify-center cursor-pointer hover:bg-surface-hover transition-colors min-h-[200px]"
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="image/png, image/jpeg, image/webp" 
              onChange={handleFileChange}
            />
            <Upload className="w-8 h-8 text-text-muted mb-2" />
            <p className="text-sm font-medium text-text-main">Click or drag image to upload</p>
            <p className="text-xs text-text-muted mt-1">PNG, JPG up to 10MB</p>
          </div>

          <button 
            className={`btn-primary w-full ${(!file || loading) ? 'btn-disabled' : ''}`}
            onClick={handleRun}
            disabled={!file || loading}
          >
            <Play className="w-4 h-4" />
            {loading ? 'Processing Locally...' : 'Process Image Locally'}
          </button>
          
          {error && (
            <div className="p-3 bg-status-error-bg text-status-error border border-status-error rounded text-sm">
              {error}
            </div>
          )}
        </div>

        <div className="flex flex-col items-center justify-center border border-border rounded-card bg-[#121212] min-h-[200px] overflow-hidden">
          {preview ? (
            <img src={preview} alt="Preview" className="max-w-full max-h-[300px] object-contain" />
          ) : (
            <div className="text-text-muted flex flex-col items-center">
              <ImageIcon className="w-8 h-8 mb-2 opacity-50" />
              <span className="text-sm">Image Preview</span>
            </div>
          )}
        </div>
      </div>

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="flex items-center gap-2 p-3 bg-surface-secondary border border-border-light rounded text-sm font-medium">
            Status: 
            <span className={result.overall_status === 'completed' ? 'text-status-success' : 'text-status-warning'}>
              {result.overall_status.toUpperCase()}
            </span>
            <span className="text-text-muted ml-auto">{result.processing_time_ms} ms</span>
          </div>
          
          {result.overall_status === 'partial_failed' && (
            <div className="p-3 bg-status-warning-bg text-status-warning border border-status-warning rounded text-sm flex items-start gap-2">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <div>
                In the full Phase 3 pipeline, only the CLIP image record would be eligible for indexing until a retry succeeds.
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="card p-4 space-y-3">
              <h3 className="font-semibold text-brand">1. CLIP Image Embedding</h3>
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">Status</span>
                <span className={result.clip.status === 'completed' ? 'text-status-success' : 'text-status-error'}>{result.clip.status}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">Model</span>
                <span>{result.clip.model}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">Dimension</span>
                <span>{result.clip.vector_dimension}-d</span>
              </div>
              <div className="bg-[#121212] p-2 rounded border border-border font-mono text-xs text-text-muted break-all">
                [{result.clip.vector_preview.map((v: number) => v.toFixed(3)).join(', ')}...]
              </div>
            </div>

            <div className="card p-4 space-y-3">
              <h3 className="font-semibold text-brand">2. LLaVA Description</h3>
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">Status</span>
                <span className={result.llava.status === 'completed' ? 'text-status-success' : 'text-status-error'}>{result.llava.status}</span>
              </div>
              {result.llava.status === 'completed' ? (
                <div className="p-2 bg-[#121212] rounded border border-border text-sm text-text-main h-24 overflow-y-auto">
                  {result.llava.description}
                </div>
              ) : (
                <div className="p-2 bg-status-error-bg text-status-error border border-status-error rounded text-sm">
                  {result.llava.error_message}
                </div>
              )}
            </div>
          </div>
          
          {result.llava.status === 'completed' && (
            <div className="card p-4 space-y-3">
              <h3 className="font-semibold text-brand">3. Description Text Embedding</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-sm"><span className="text-text-muted block">Status</span> <span className="text-status-success">{result.description_text_embedding.status}</span></div>
                <div className="text-sm"><span className="text-text-muted block">Dimension</span> <span>{result.description_text_embedding.vector_dimension}-d</span></div>
              </div>
              <div className="bg-[#121212] p-2 rounded border border-border font-mono text-xs text-text-muted break-all">
                [{result.description_text_embedding.vector_preview.map((v: number) => v.toFixed(3)).join(', ')}...]
              </div>
            </div>
          )}

          <div className="text-xs text-text-muted text-center mt-4">
            {result.message}
          </div>
        </div>
      )}
    </div>
  )
}
