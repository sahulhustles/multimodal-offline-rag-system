import { useState } from 'react'
import { CheckCircle2, Play } from 'lucide-react'
import { demoApi } from '../../api/client'

export default function TextChunkingTab() {
  const [text, setText] = useState('Machine learning is a field of inquiry devoted to understanding and building methods that "learn", that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')

  const handleRun = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await demoApi.chunkText(text)
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
          className="input-field min-h-[150px] resize-y"
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
        {loading ? 'Chunking...' : 'Run Chunking'}
      </button>

      {error && (
        <div className="p-3 bg-status-error-bg text-status-error border border-status-error rounded text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="flex items-center gap-2 p-3 bg-status-success-bg border border-status-success text-status-success rounded text-sm font-medium">
            <CheckCircle2 className="w-5 h-5" />
            Chunking configuration verified ({result.configuration.chunk_size_tokens} tokens / {result.configuration.overlap_tokens} overlap)
          </div>

          <div className="grid grid-cols-3 gap-4">
            <Metric label="Total Tokens" value={result.total_input_tokens} />
            <Metric label="Chunk Count" value={result.chunk_count} />
            <Metric label="Processing Time" value={`${result.processing_time_ms} ms`} />
          </div>

          <div className="space-y-4">
            <h3 className="font-semibold text-text-main border-b border-border pb-2">Generated Chunks</h3>
            <div className="space-y-4">
              {result.chunks.map((chunk: any) => (
                <div key={chunk.chunk_index} className="card p-4">
                  <div className="flex justify-between items-center mb-3">
                    <span className="font-mono text-sm text-brand">Chunk {chunk.chunk_index}</span>
                    <div className="flex gap-3 text-xs text-text-muted">
                      <span>{chunk.token_count} tokens</span>
                      {chunk.starts_with_overlap && (
                        <span className="text-status-warning bg-status-warning-bg px-1.5 py-0.5 rounded border border-status-warning">
                          {chunk.overlap_token_count} overlap tokens
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="text-sm text-text-main leading-relaxed">
                    {chunk.text}
                  </p>
                </div>
              ))}
            </div>
            
            <div className="text-xs text-text-muted text-center mt-4">
              {result.message}
            </div>
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
      <div className="text-xl font-semibold text-text-main">{value}</div>
    </div>
  )
}
