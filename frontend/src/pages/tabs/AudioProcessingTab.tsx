import { useState, useRef } from 'react'
import { Upload, Play, Clock, Languages, FileAudio, AlignLeft } from 'lucide-react'
import { demoApi } from '../../api/client'

export default function AudioProcessingTab() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)

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
      const res = await demoApi.processAudio(file)
      setResult(res)
    } catch (err: any) {
      setError(err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }
  
  const handleSeek = (seconds: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = seconds
      audioRef.current.play()
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
              accept=".wav,.mp3,.m4a,.webm,.ogg" 
              onChange={handleFileChange}
            />
            <Upload className="w-8 h-8 text-text-muted mb-2" />
            <p className="text-sm font-medium text-text-main">
              {file ? file.name : 'Click or drag Audio to upload'}
            </p>
            <p className="text-xs text-text-muted mt-1">WAV, MP3, M4A, WebM, OGG</p>
          </div>

          <button 
            className={`btn-primary w-full ${(!file || loading) ? 'btn-disabled' : ''}`}
            onClick={handleRun}
            disabled={!file || loading}
          >
            <Play className="w-4 h-4" />
            {loading ? 'Normalizing & Transcribing...' : 'Normalize and Transcribe Locally'}
          </button>
          
          {error && (
            <div className="p-3 bg-status-error-bg text-status-error border border-status-error rounded text-sm">
              {error}
            </div>
          )}
        </div>
        
        <div className="card p-4 flex flex-col justify-center items-center bg-[#121212]">
           {result ? (
             <div className="w-full space-y-4">
               <h3 className="font-semibold text-text-main text-center text-sm">Normalized Audio Preview (16kHz Mono WAV)</h3>
               <audio ref={audioRef} controls src={`http://localhost:8000${result.normalized_output.audio_url}`} className="w-full outline-none" />
             </div>
           ) : (
             <div className="text-text-muted text-center text-sm flex flex-col items-center gap-2">
               <FileAudio className="w-8 h-8 opacity-50" />
               Audio Player
             </div>
           )}
        </div>
      </div>

      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Metric label="Format Conversion" value={`${result.original_format} → ${result.normalized_output.format}`} icon={FileAudio} />
            <Metric label="Duration" value={`${result.total_duration_seconds.toFixed(1)}s`} icon={Clock} />
            <Metric label="Language" value={(result.whisper.detected_language || 'unknown').toUpperCase()} icon={Languages} />
            <Metric label="Segments" value={result.segment_count} icon={AlignLeft} />
          </div>
          
          <div className="p-3 bg-status-warning-bg border border-status-warning text-status-warning rounded text-sm flex flex-col sm:flex-row items-center justify-between">
            <span>Timestamped segments are prepared for text embedding and Phase 3 indexing.</span>
            <span className="font-semibold mt-2 sm:mt-0 whitespace-nowrap">Not indexed in Qdrant</span>
          </div>

          <div className="space-y-4">
            <h3 className="font-semibold text-text-main border-b border-border pb-2">Whisper Transcription ({result.whisper.model} {result.whisper.compute_type})</h3>
            <div className="bg-background border border-border rounded-card p-4 h-[300px] overflow-y-auto space-y-2">
              {result.whisper.transcription_segments.map((seg: any) => (
                <div 
                  key={seg.segment_index} 
                  className="flex gap-4 p-2 hover:bg-surface-hover rounded cursor-pointer transition-colors group"
                  onClick={() => handleSeek(seg.start_seconds)}
                >
                  <div className="flex-shrink-0 pt-0.5">
                    <span className="inline-flex items-center justify-center w-14 py-0.5 bg-brand text-white text-xs font-mono rounded font-medium shadow-sm group-hover:bg-brand-hover">
                      {formatTime(seg.start_seconds)}
                    </span>
                  </div>
                  <div className="text-sm text-text-main leading-relaxed">
                    {seg.transcript_text}
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

function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function Metric({ label, value, icon: Icon }: { label: string, value: string | number, icon: any }) {
  return (
    <div className="p-4 bg-background border border-border rounded shadow-sm flex flex-col items-center justify-center gap-1">
      <div className="text-brand mb-1"><Icon className="w-5 h-5" /></div>
      <div className="text-xl font-semibold text-text-main text-center">{value}</div>
      <div className="text-xs text-text-muted">{label}</div>
    </div>
  )
}
