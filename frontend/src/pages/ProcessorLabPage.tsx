import { useState } from 'react'
import { FileText, Type, Image as ImageIcon, File, Headphones, FileCode2 } from 'lucide-react'
import clsx from 'clsx'

// Import tabs
import TextChunkingTab from '../pages/tabs/TextChunkingTab'
import TextEmbeddingTab from '../pages/tabs/TextEmbeddingTab'
import ImageVisionTab from '../pages/tabs/ImageVisionTab'
import PdfExtractionTab from '../pages/tabs/PdfExtractionTab'
import DocxExtractionTab from '../pages/tabs/DocxExtractionTab'
import AudioProcessingTab from '../pages/tabs/AudioProcessingTab'

const TABS = [
  { id: 'chunking', label: 'Text Chunking', icon: Type },
  { id: 'embedding', label: 'Text Embedding', icon: FileCode2 },
  { id: 'image', label: 'Image & Vision', icon: ImageIcon },
  { id: 'pdf', label: 'PDF Processing', icon: FileText },
  { id: 'docx', label: 'DOCX Processing', icon: File },
  { id: 'audio', label: 'Audio Processing', icon: Headphones },
]

export default function ProcessorLabPage() {
  const [activeTab, setActiveTab] = useState(TABS[0].id)

  return (
    <div className="space-y-6 max-w-6xl h-full flex flex-col">
      {/* Header */}
      <div className="border-b-2 border-border pb-4">
        <h1 className="text-3xl font-bold tracking-tight text-text-main flex items-center gap-3">
          Processor Lab
        </h1>
        <p className="text-text-muted text-sm mt-1">
          Interactive test bed for Phase 2 offline components. Run isolated extraction, chunking, and embedding checks.
        </p>
      </div>

      {/* Tabs list (like index cards folders) */}
      <div className="flex gap-1 overflow-x-auto pb-0.5 scrollbar-thin">
        {TABS.map(tab => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2.5 text-xs font-mono uppercase tracking-wider font-bold border-t border-l border-r transition-all cursor-pointer whitespace-nowrap',
                isActive
                  ? 'bg-surface border-2 border-b-transparent border-border text-brand relative z-10 translate-y-[2px]'
                  : 'bg-surface-secondary border-border text-text-muted hover:text-text-main hover:bg-surface-hover'
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Content panel */}
      <div className="flex-1 bg-surface border-2 border-border p-6 min-h-[500px]">
        {activeTab === 'chunking' && <TextChunkingTab />}
        {activeTab === 'embedding' && <TextEmbeddingTab />}
        {activeTab === 'image' && <ImageVisionTab />}
        {activeTab === 'pdf' && <PdfExtractionTab />}
        {activeTab === 'docx' && <DocxExtractionTab />}
        {activeTab === 'audio' && <AudioProcessingTab />}
      </div>
    </div>
  )
}
