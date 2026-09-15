import { ArrowRight, ArrowDown } from 'lucide-react'

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
          </div>
        </div>

        <ArrowRight className="hidden md:block self-center text-text-muted w-6 h-6 shrink-0" />
        <ArrowDown className="block md:hidden self-center text-text-muted w-6 h-6 shrink-0" />

        {/* INDEX */}
        <div className="flex-1 min-w-[250px] border-2 border-border bg-surface overflow-hidden flex flex-col">
          <div className="bg-status-success-bg border-b border-status-success text-status-success px-4 py-2 font-mono font-bold text-xs uppercase tracking-wider">
            3. Index (Active)
          </div>
          <div className="p-4 flex-1 space-y-3">
            <Node title="Unified Vector Store" desc="Qdrant local Docker. Deterministic UUIDv5 point IDs." />
            <Node title="Named Vectors" desc="text: 384-d (SentenceTransformer), image: 512-d (CLIP)" />
            <Node title="Cross-modal Links" desc="Two-linked-record strategy. CLIP + LLaVA description pairs." />
            <Node title="Payload Indexes" desc="source_document_id, modality, source_type, indexing_status" />
          </div>
        </div>

        <ArrowRight className="hidden md:block self-center text-text-muted w-6 h-6 shrink-0" />
        <ArrowDown className="block md:hidden self-center text-text-muted w-6 h-6 shrink-0" />

        {/* RETRIEVE */}
        <div className="flex-1 min-w-[250px] border-2 border-border bg-surface overflow-hidden flex flex-col">
          <div className="bg-status-success-bg border-b border-status-success text-status-success px-4 py-2 font-mono font-bold text-xs uppercase tracking-wider">
            4. Retrieve (Active)
          </div>
          <div className="p-4 flex-1 space-y-3">
            <Node title="Local Query Encoder" desc="SentenceTransformer 384-d query text vector." />
            <Node title="Qdrant Text Search" desc="Cosine similarity on text named vector with metadata filters." />
            <Node title="SQLite Hydration" desc="Full record hydration with cross-modal linked records." />
            <Node title="Traceable Evidence" desc="Ranked results, scores, timing metrics, zero LLM answer." />
          </div>
        </div>

        <ArrowRight className="hidden md:block self-center text-text-muted w-6 h-6 shrink-0" />
        <ArrowDown className="block md:hidden self-center text-text-muted w-6 h-6 shrink-0" />

        {/* GENERATE */}
        <div className="flex-1 min-w-[250px] border-2 border-border bg-surface overflow-hidden flex flex-col">
          <div className="bg-status-success-bg border-b border-status-success text-status-success px-4 py-2 font-mono font-bold text-xs uppercase tracking-wider">
            5. Generate (Active)
          </div>
          <div className="p-4 flex-1 space-y-3">
            <Node title="Context Assembler" desc="Score-gated, deduplicated [S#] source blocks with budget control." />
            <Node title="Local LLM (Ollama)" desc="llava via /api/chat. Temperature 0, citation-enforced system prompt." />
            <Node title="Citation Validator" desc="[S#] extraction, allowed-set validation, safe failure on invalid/missing." />
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


