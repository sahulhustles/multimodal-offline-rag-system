import { useState } from 'react'
import { Play, CheckCircle2 } from 'lucide-react'
import { demoApi } from '../../api/client'

export default function TextEmbeddingTab() {
  const [text, setText] = useState('Machine learning is a field of inquiry devoted to understanding and building methods that "learn".')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')

  const handleRun = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await demoApi.embedText(text)
      setResult(res)
    } catch (err: any) {
      setError(err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <label className="label-text">Input Text</label>
        <textarea
          className="input-field min-h-[100px] resize-y"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
        />
      </div>
      
      <button 
        className={`btn-primary ${loading ? 'btn-disabled' : ''}`}
        onClick={handleRun}
        disabled={loading}
      >
        <Play className="w-4 h-4" />
        {loading ? 'Generating...' : 'Generate Local Embedding'}
      </button>

      {error && (
        <div className="p-3 bg-status-error-bg text-status-error border border-status-error rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Metric label="Model" value={result.embedding_model_name.split('/').pop()} />
            <Metric label="Vector Dimension" value={result.vector_dimension} />
            <Metric label="Input Tokens" value={result.input_token_count} />
            <Metric label="Processing Time" value={`${result.processing_time_ms} ms`} />
          </div>

          <div className="card p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-text-main">Generated Vector (Preview)</h3>
              {result.l2_normalized && (
                <span className="flex items-center gap-1 text-xs text-status-success bg-status-success-bg px-2 py-1 rounded border border-status-success">
                  <CheckCircle2 className="w-3 h-3" />
                  L2-Normalized
                </span>
              )}
            </div>
            
            <div className="bg-[#121212] p-4 rounded border border-border font-mono text-sm text-text-muted break-all">
              [ {result.vector_preview.map((v: number) => v.toFixed(4)).join(', ')}, ... ]
            </div>
            <div className="text-xs text-text-muted italic">
              Showing first 8 values only. Full {result.vector_dimension}-dimensional vector stored in memory.
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

function Metric({ label, value }: { label: string, value: string | number }) {
  return (
    <div className="p-4 bg-background border border-border rounded shadow-sm text-center">
      <div className="text-xs text-text-muted mb-1">{label}</div>
      <div className="text-sm font-semibold text-text-main">{value}</div>
    </div>
  )
}
