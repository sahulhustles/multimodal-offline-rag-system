import { ArrowRight, ArrowDown, Lock } from 'lucide-react'

export default function ArchitecturePage() {
  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="border-b-2 border-border pb-4">
        <h1 className="text-3xl font-bold tracking-tight text-text-main">
          System Architecture
        </h1>
        <p className="text-text-muted text-sm mt-1">
          Data flow through the offline pipeline. Audit how data is ingested, processed, indexed, and retrieved.
        </p>
      </div>

      <div className="flex flex-col md:flex-row gap-4 items-stretch mt-8">
        
        {/* INGEST */}
        <div className="flex-1 min-w-[250px] border-2 border-border bg-surface overflow-hidden flex flex-col">
          <div className="bg-surface-secondary text-text-main px-4 py-2 font-mono font-bold text-xs uppercase tracking-wider border-b border-border">
            1. Ingest
          </div>
          <div className="p-4 flex-1 space-y-3">
            <Node title="PDF / DOCX Documents" />
            <Node title="Images (PNG/JPG)" />
            <Node title="Audio (WAV/MP3)" />
            <Node title="User Query" />
          </div>
        </div>

        <ArrowRight className="hidden md:block self-center text-text-muted w-6 h-6 shrink-0" />
        <ArrowDown className="block md:hidden self-center text-text-muted w-6 h-6 shrink-0" />

        {/* PROCESS */}
        <div className="flex-1 min-w-[250px] border-2 border-border bg-surface overflow-hidden flex flex-col">
          <div className="bg-brand text-white px-4 py-2 font-mono font-bold text-xs uppercase tracking-wider border-b border-border">
            2. Process
          </div>
          <div className="p-4 flex-1 space-y-3">
            <Node title="Text Extractor" desc="PyMuPDF / python-docx. 512/50 token sliding window." />
            <Node title="Vision Encoder" desc="CLIP 512-d + LLaVA text description (384-d)." />
            <Node title="Audio Transcriber" desc="faster-whisper large-v3 int8. Offline speech-to-text." />
            <FutureNode title="Query Encoder" desc="CLIP + text embedding, dual encoder." />
          </div>
        </div>

        <ArrowRight className="hidden md:block self-center text-text-muted w-6 h-6 shrink-0" />
        <ArrowDown className="block md:hidden self-center text-text-muted w-6 h-6 shrink-0" />

        {/* INDEX */}
        <div className="flex-1 min-w-[250px] border-2 border-border bg-surface overflow-hidden flex flex-col">
          <div className="bg-status-warning-bg border-b border-status-warning text-status-warning px-4 py-2 font-mono font-bold text-xs uppercase tracking-wider">
            3. Index (Phase 3 - Next)
          </div>
          <div className="p-4 flex-1 space-y-3">
            <Node title="Unified Vector Store" desc="Qdrant local Docker" />
            <Node title="Named Vectors" desc="text: 384-d, image: 512-d" />
            <Node title="Metadata" desc="Cross-modal links & source tracking" />
          </div>
        </div>

        <ArrowRight className="hidden md:block self-center text-text-muted w-6 h-6 shrink-0" />
        <ArrowDown className="block md:hidden self-center text-text-muted w-6 h-6 shrink-0" />

        {/* RETRIEVE & GENERATE */}
        <div className="flex-1 min-w-[250px] border-2 border-border bg-surface overflow-hidden flex flex-col opacity-50 border-dashed">
          <div className="bg-surface-secondary text-text-muted px-4 py-2 font-mono font-bold text-xs uppercase tracking-wider border-b border-border">
            4. Retrieve & Generate
          </div>
          <div className="p-4 flex-1 space-y-3">
            <FutureNode title="Retrieved Context" desc="Cross-encoder reranker" />
            <FutureNode title="Local LLM (Ollama)" desc="Grounded answer with citations" />
          </div>
        </div>
      </div>
    </div>
  )
}

function Node({ title, desc }: { title: string, desc?: string }) {
  return (
    <div className="p-3 bg-surface-secondary border border-border-light text-xs">
      <div className="font-bold text-text-main">{title}</div>
      {desc && <div className="text-text-muted mt-1 leading-snug font-mono text-xs break-words" style={{ overflowWrap: 'anywhere' }}>{desc}</div>}
    </div>
  )
}

function FutureNode({ title, desc }: { title: string, desc?: string }) {
  return (
    <div className="p-3 bg-surface-secondary border border-border-light border-dashed text-xs text-text-muted relative overflow-hidden">
      <div className="flex items-center gap-2">
        <Lock className="w-3.5 h-3.5 shrink-0" />
        <div className="font-bold">{title}</div>
      </div>
      {desc && <div className="text-xs mt-1 leading-snug ml-5 font-mono break-words" style={{ overflowWrap: 'anywhere' }}>{desc}</div>}
      <div className="absolute top-0 right-0 bg-surface border-l border-b border-border-light px-1.5 py-0.5 text-xs font-mono uppercase tracking-wider font-bold">Future</div>
    </div>
  )
}
