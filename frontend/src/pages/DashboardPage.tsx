import { useEffect, useState } from 'react'
import { Server, Database, HardDrive, Cpu, AlertCircle, CheckCircle2, Headphones } from 'lucide-react'
import { demoApi } from '../api/client'
import clsx from 'clsx'

export default function DashboardPage() {
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    demoApi.getSystemReadiness()
      .then(res => {
        setStatus(res)
        setLoading(false)
      })
      .catch(err => {
        console.error("Status fetch failed", err)
        setStatus(null)
        setLoading(false)
      })
  }, [])

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b-2 border-border pb-4 gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-text-main">
            Offline Multimodal RAG
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Local Platform Foundation & Processing Demonstrator (Phase 1 & Phase 2)
          </p>
        </div>
        <div className="flex flex-col items-end text-right shrink-0">
          <span className="text-xs font-mono bg-status-success-bg text-status-success border border-status-success px-2.5 py-1 rounded-sm uppercase tracking-wider font-semibold">
            Phase 2: Processed locally
          </span>
          <span className="text-[10px] text-text-muted mt-1">
            Phase 3: Retrieval and grounded Q&A next
          </span>
        </div>
      </div>

      {loading ? (
        <div className="flex h-32 items-center justify-center panel-2 bg-surface">
          <div className="text-text-muted font-mono animate-pulse">Loading system readiness state...</div>
        </div>
      ) : status ? (
        <div className="space-y-6">
          {/* Status Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <StatusCard 
              title="Backend API" 
              icon={Server} 
              ok={status.phase1_health?.status === 'healthy'} 
              detail={`Port ${status.backend_info?.port || 8000}`} 
            />
            <StatusCard 
              title="SQLite Registry" 
              icon={Database} 
              ok={status.sqlite?.connected} 
              detail="SQLite Connection Live" 
            />
            <StatusCard 
              title="Qdrant Docker" 
              icon={HardDrive} 
              ok={status.qdrant?.connected} 
              detail={status.qdrant?.collection_ready ? 'Collection Active' : 'Collection Missing'} 
            />
            <StatusCard 
              title="Ollama Inference" 
              icon={Cpu} 
              ok={status.ollama?.llava_available} 
              detail={status.ollama?.connected ? 'Ollama Live & Connected' : 'Ollama Offline'} 
            />
            <StatusCard 
              title="Audio Pipeline" 
              icon={Headphones} 
              ok={status.system_dependencies?.whisper?.operational === true} 
              warning={false}
              detail={
                status.system_dependencies?.whisper?.operational
                  ? (status.system_dependencies?.whisper?.load_test_status === 'passed' ? 'Audio pipeline ready' : 'Audio pipeline operational')
                  : 'Audio pipeline unavailable'
              } 
            />
            <StatusCard 
              title="LibreOffice converter" 
              icon={AlertCircle} 
              ok={status.system_dependencies?.libreoffice?.available} 
              warning={!status.system_dependencies?.libreoffice?.available}
              detail={status.system_dependencies?.libreoffice?.available ? 'soffice path active' : 'Missing (Required for legacy .doc only)'} 
            />
          </div>

          {/* Details Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Qdrant Schema */}
            <div className="panel-2">
              <div className="px-4 py-3 bg-surface-secondary border-b-2 border-border">
                <h2 className="text-sm font-mono uppercase tracking-wider text-text-muted">Qdrant Vector Schema</h2>
              </div>
              <div className="p-4 space-y-4">
                <div className="text-xs text-text-muted">
                  Collection: <span className="font-mono text-brand font-semibold">{status.qdrant?.collection_name || 'multimodal_rag'}</span>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-3 bg-surface-secondary border border-border-light">
                    <span className="font-mono text-sm font-semibold text-text-main">text</span>
                    <span className="text-xs text-text-muted font-mono">384-D (Cosine distance)</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-surface-secondary border border-border-light">
                    <span className="font-mono text-sm font-semibold text-text-main">image</span>
                    <span className="text-xs text-text-muted font-mono">512-D (Cosine distance)</span>
                  </div>
                </div>
                <div className="p-3 bg-status-warning-bg border border-status-warning text-status-warning text-xs font-semibold uppercase tracking-wider text-center">
                  Upcoming in Phase 3: Qdrant Point Creation
                </div>
              </div>
            </div>

            {/* Model Configurations */}
            <div className="panel-2">
              <div className="px-4 py-3 bg-surface-secondary border-b-2 border-border">
                <h2 className="text-sm font-mono uppercase tracking-wider text-text-muted">Local Active Models</h2>
              </div>
              <div className="p-4">
                <ul className="space-y-2 font-mono text-xs">
                  <ModelItem title="Text Embeddings" model={status?.models?.text_embedding || 'sentence-transformers/all-MiniLM-L6-v2'} />
                  <ModelItem title="Image Embeddings" model={status?.models?.image_embedding || 'ViT-B-32'} />
                  <ModelItem title="Vision Description" model={status?.models?.vision_description || 'llava'} />
                  <ModelItem title="Transcription" model={status?.models?.transcription || 'large-v3 int8'} />
                </ul>
              </div>
            </div>
          </div>

          {/* Build Scope */}
          <div className="panel-2">
            <div className="px-4 py-3 bg-surface-secondary border-b-2 border-border">
              <h2 className="text-sm font-mono uppercase tracking-wider text-text-muted">Platform Implementation Scope</h2>
            </div>
            <div className="p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <ScopeBadge status="verified" title="Phase 1 Verified" desc="Platform Foundation" />
              <ScopeBadge status="verified" title="Phase 2 Demonstrated" desc="Processing Components" />
              <ScopeBadge status="next" title="Phase 3 Next" desc="Qdrant Point Creation" />
              <ScopeBadge status="future" title="Future Scope" desc="Retrieval & Generation" />
            </div>
          </div>

          <div className="p-3 border border-border bg-surface-secondary text-center text-xs text-text-muted font-medium">
            Ingestion workflow stores documents locally in SQLite and parses metadata.json in real-time. No points are upserted into Qdrant in this phase.
          </div>
        </div>
      ) : (
        <div className="p-4 bg-status-error-bg text-status-error border-2 border-status-error rounded-sm text-sm">
          <strong>Connection Failure:</strong> Unable to connect to the local FastAPI backend (port 8000). Ensure the backend server is running.
        </div>
      )}
    </div>
  )
}

function StatusCard({ title, icon: Icon, ok, detail, warning }: { title: string, icon: any, ok: boolean, detail: string, warning?: boolean }) {
  return (
    <div className="panel-1 p-4 flex items-start gap-4">
      <div className={clsx(
        "p-2.5 flex items-center justify-center w-10 h-10 border",
        ok && "bg-status-success-bg text-status-success border-status-success",
        !ok && !warning && "bg-status-error-bg text-status-error border-status-error",
        !ok && warning && "bg-status-warning-bg text-status-warning border-status-warning"
      )}>
        <Icon className="w-5 h-5 shrink-0" />
      </div>
      <div className="min-w-0 flex-1">
        <h3 className="font-bold text-text-main text-sm flex items-center gap-1.5">
          {title} 
          {ok ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-status-success shrink-0" />
          ) : (
            <AlertCircle className={clsx("w-3.5 h-3.5 shrink-0", warning ? "text-status-warning" : "text-status-error")} />
          )}
        </h3>
        <p className="text-text-muted text-xs mt-1 truncate" title={detail}>{detail}</p>
      </div>
    </div>
  )
}

function ModelItem({ title, model }: { title: string, model: string }) {
  return (
    <li className="flex flex-col sm:flex-row sm:justify-between sm:items-center p-2.5 bg-surface-secondary border border-border-light gap-1">
      <span className="text-text-muted">{title}:</span>
      <span className="text-brand font-semibold font-mono text-xs sm:text-right break-words" style={{ overflowWrap: 'anywhere' }}>{model}</span>
    </li>
  )
}

function ScopeBadge({ status, title, desc }: { status: 'verified' | 'next' | 'future', title: string, desc: string }) {
  let styles = ''
  if (status === 'verified') styles = 'bg-status-success-bg border-status-success text-status-success'
  else if (status === 'next') styles = 'bg-status-warning-bg border-status-warning text-status-warning shadow-[3px_3px_0px_0px_rgba(245,158,11,0.15)] font-bold'
  else styles = 'border-border text-text-muted border-dashed opacity-50'
  
  return (
    <div className={clsx("p-3 border flex flex-col justify-center items-center text-center", styles)}>
      <span className="font-bold text-sm mb-1">{title}</span>
      <span className="text-xs opacity-80 font-medium">{desc}</span>
    </div>
  )
}
