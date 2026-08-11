export default function AboutPage() {
  return (
    <div className="max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-text-main">About the Project</h1>
        <p className="text-text-muted mt-1">Review-ready project explanation.</p>
      </div>

      <div className="card p-6 space-y-6">
        <section>
          <h2 className="text-lg font-semibold text-brand mb-2">Problem Statement & Objective</h2>
          <p className="text-sm text-text-main leading-relaxed">
            Standard Retrieval-Augmented Generation (RAG) systems typically rely on cloud APIs, 
            which introduces data privacy concerns and internet dependency. They also frequently 
            struggle with multimodal inputs (images, audio, complex documents). 
            This project aims to build a <strong>Fully Offline Multimodal RAG System</strong> that 
            runs entirely on local hardware, processing text, images, and audio natively without 
            sending data to external services.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-brand mb-2">Implementation Status</h2>
          <div className="bg-[#181818] rounded border border-border p-4 text-sm">
            <ul className="space-y-3">
              <li className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-status-success flex-shrink-0"></div>
                <div><span className="font-medium text-text-main">Phase 1 Complete</span> <span className="text-text-muted">— Platform Foundation (Qdrant, SQLite, API framework)</span></div>
              </li>
              <li className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-status-success flex-shrink-0"></div>
                <div><span className="font-medium text-text-main">Phase 2 Complete</span> <span className="text-text-muted">— Processing Components (Chunking, Embedding, LLaVA, Whisper)</span></div>
              </li>
              <li className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-status-warning flex-shrink-0"></div>
                <div><span className="font-medium text-text-main">Phase 3 Planned</span> <span className="text-text-muted">— Qdrant Indexing & Cross-modal Links</span></div>
              </li>
              <li className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-[#444] border border-[#555] flex-shrink-0"></div>
                <div><span className="font-medium text-text-main opacity-70">Phase 4 Planned</span> <span className="text-text-muted opacity-70">— Semantic Retrieval</span></div>
              </li>
              <li className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-[#444] border border-[#555] flex-shrink-0"></div>
                <div><span className="font-medium text-text-main opacity-70">Phase 5 Planned</span> <span className="text-text-muted opacity-70">— Query Processing</span></div>
              </li>
              <li className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-[#444] border border-[#555] flex-shrink-0"></div>
                <div><span className="font-medium text-text-main opacity-70">Phase 6 Planned</span> <span className="text-text-muted opacity-70">— LLM Generation & Chat UI</span></div>
              </li>
            </ul>
          </div>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-brand mb-2">Two-Linked-Record Strategy</h2>
          <p className="text-sm text-text-main leading-relaxed mb-3">
            To efficiently retrieve images based on text queries, we employ a dual-record strategy for images:
          </p>
          <ul className="list-disc pl-5 text-sm text-text-main space-y-2 marker:text-brand">
            <li><strong>Image Record:</strong> Processed via CLIP to produce a 512-dimensional vector.</li>
            <li><strong>Description Record:</strong> Processed via LLaVA to generate a rich text description, which is then embedded via Sentence Transformers to a 384-dimensional vector.</li>
          </ul>
          <p className="text-sm text-text-muted leading-relaxed mt-3 bg-surface-hover p-3 rounded border border-border">
            During Phase 3, these two points will be stored in Qdrant and linked together, allowing text queries to seamlessly retrieve highly relevant images through both semantic text matching and direct CLIP embedding matching.
          </p>
        </section>

        <section>
          <h2 className="text-lg font-semibold text-brand mb-2">Teacher Demo Workflow</h2>
          <ol className="list-decimal pl-5 text-sm text-text-main space-y-2 marker:text-brand font-medium">
            <li>Verify local services on the <span className="text-brand">System Status</span> page.</li>
            <li>Demonstrate 512/50 token chunking algorithm.</li>
            <li>Demonstrate 384-d local text embedding generation.</li>
            <li>Demonstrate PDF and DOCX multi-modal extraction (text + images).</li>
            <li>Demonstrate local Whisper transcription for audio.</li>
            <li>Demonstrate the Vision pipeline (CLIP + LLaVA).</li>
          </ol>
        </section>
      </div>
    </div>
  )
}
