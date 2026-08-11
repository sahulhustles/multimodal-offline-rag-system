import { useState, useEffect } from 'react'
import { Server, Database, HardDrive, Cpu, RefreshCw, AlertCircle, CheckCircle2, Play } from 'lucide-react'
import { demoApi } from '../api/client'
import clsx from 'clsx'

export default function SystemStatusPage() {
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null)
  const [error, setError] = useState('')
  
  const [audioChecking, setAudioChecking] = useState(false)
  const [audioReadyResult, setAudioReadyResult] = useState<any>(null)
  const [audioReadyError, setAudioReadyError] = useState('')

  const fetchStatus = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await demoApi.getSystemReadiness()
      setStatus(res)
      setLastRefreshed(new Date())
    } catch (err: any) {
      setError('Failed to fetch system status. Ensure the backend is running.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const checkAudioReadiness = async () => {
    setAudioChecking(true)
    setAudioReadyError('')
    try {
      const res = await demoApi.getAudioReadiness()
      setAudioReadyResult(res)
      // refresh status to pull cached whisper state
      await fetchStatus()
    } catch (err: any) {
      setAudioReadyError('Failed to run deep audio readiness check.')
      console.error(err)
    } finally {
      setAudioChecking(false)
    }
  }

  useEffect(() => {
    fetchStatus()
  }, [])

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b-2 border-border pb-4 gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-text-main flex items-center gap-3">
            System Status
          </h1>
          <p className="text-text-muted text-sm mt-1">
            Engine diagnostic console. Auditing local models, paths, and vector storage connectivity.
          </p>
        </div>
        <button 
          onClick={fetchStatus}
          disabled={loading}
          className="btn-secondary text-xs"
        >
          <RefreshCw className={clsx("w-4 h-4", loading && "animate-spin")} />
          Refresh Registry
        </button>
      </div>

      {error && (
        <div className="p-4 bg-status-error-bg text-status-error border-2 border-status-error flex items-start gap-4 flex-col sm:flex-row sm:items-center">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <div className="flex-1">
            <h3 className="font-bold text-sm uppercase tracking-wider font-mono">Connection Failure</h3>
            <p className="text-xs opacity-90 mt-1">{error}</p>
          </div>
          <button onClick={fetchStatus} className="btn-primary text-xs whitespace-nowrap">Retry connection</button>
        </div>
      )}

      {loading && !status && !error && (
        <div className="p-12 text-center text-text-muted font-mono text-xs animate-pulse panel-2 bg-surface">
          RESOLVING RUNTIME DEPENDENCY DIAGNOSTICS...
        </div>
      )}

      {status && (
        <div className="space-y-6">
          <div className="text-xs text-text-muted font-mono">
            LAST DIAGNOSTIC CYCLE: {lastRefreshed?.toLocaleTimeString()}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <StatusPanel 
              title="Qdrant Vector Database" 
              icon={HardDrive}
              ok={status.qdrant?.connected}
              items={[
                { label: 'Connection', value: status.qdrant?.connected ? 'CONNECTED' : 'UNREACHABLE', ok: status.qdrant?.connected },
                { label: 'Collection', value: status.qdrant?.collection_name },
                { label: 'Collection Status', value: status.qdrant?.collection_ready ? 'READY' : 'MISSING', ok: status.qdrant?.collection_ready }
              ]}
            >
              {status.qdrant?.collection_ready && (
                <div className="mt-4 border-t border-border-light pt-3">
                  <h4 className="text-xs font-bold uppercase font-mono tracking-wider mb-2 text-text-muted">Named Vectors</h4>
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="text-left text-text-muted border-b border-border-light">
                         <th className="pb-2 font-semibold">Name</th>
                        <th className="pb-2 font-semibold">Dim</th>
                        <th className="pb-2 font-semibold">Metric</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border-light">
                      {status.qdrant?.named_vectors?.map((nv: any) => (
                        <tr key={nv.name}>
                          <td className="py-2 text-brand font-semibold">{nv.name}</td>
                          <td className="py-2 text-text-main">{nv.dimension}</td>
                          <td className="py-2 text-text-muted">{nv.distance}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </StatusPanel>

            <StatusPanel 
              title="Ollama Inference Tag" 
              icon={Cpu}
              ok={status.ollama?.connected && status.ollama?.llava_available}
              items={[
                { label: 'Connection', value: status.ollama?.connected ? 'CONNECTED' : 'UNREACHABLE', ok: status.ollama?.connected },
                { label: 'LLaVA Model', value: status.ollama?.llava_available ? 'AVAILABLE' : 'MISSING', ok: status.ollama?.llava_available },
                { label: 'Expected Vision model', value: status.models?.vision_description }
              ]}
            />

            <StatusPanel 
              title="Local Executable Dependencies" 
              icon={Server}
              ok={status.system_dependencies?.ffmpeg?.available}
              items={[
                { 
                  label: 'ffmpeg binary', 
                  value: status.system_dependencies?.ffmpeg?.available 
                    ? `FOUND (${status.system_dependencies?.ffmpeg?.detection_source})` 
                    : 'MISSING (Required for audio processing)', 
                  ok: status.system_dependencies?.ffmpeg?.available 
                },
                { 
                  label: 'soffice executable', 
                  value: status.system_dependencies?.libreoffice?.available 
                    ? `FOUND (${status.system_dependencies?.libreoffice?.detection_source})` 
                    : 'MISSING (Required for legacy .doc only)', 
                  ok: status.system_dependencies?.libreoffice?.available,
                  warning: !status.system_dependencies?.libreoffice?.available
                },
              ]}
            >
              <div className="mt-4 border-t border-border-light pt-3 text-xs font-mono text-text-muted space-y-1.5 leading-normal break-words" style={{ overflowWrap: 'anywhere' }}>
                <div>
                  <strong>Runtime environment:</strong> {status.system_dependencies?.runtime_environment} ({status.system_dependencies?.platform})
                </div>
                <div>
                  <strong>ffmpeg path:</strong> {status.system_dependencies?.ffmpeg?.resolved_path || 'None'}
                </div>
                <div>
                  <strong>soffice path:</strong> {status.system_dependencies?.libreoffice?.resolved_path || 'None'}
                </div>
              </div>
            </StatusPanel>
            
            <StatusPanel 
              title="Audio Pipeline Readiness" 
              icon={Play}
              ok={status.system_dependencies?.whisper?.operational}
              items={[
                { 
                  label: 'Normalization (ffmpeg)', 
                  value: status.system_dependencies?.ffmpeg?.available ? 'AVAILABLE' : 'MISSING', 
                  ok: status.system_dependencies?.ffmpeg?.available 
                },
                { 
                  label: `Transcription (${status.system_dependencies?.whisper?.model_name || 'whisper'})`, 
                  value: status.system_dependencies?.whisper?.dependency_present
                    ? (status.system_dependencies?.whisper?.load_test_status === 'passed' ? 'READY' : 'AVAILABLE (NOT DEEP TESTED)')
                    : 'MISSING', 
                  ok: status.system_dependencies?.whisper?.load_test_status === 'passed',
                  warning: status.system_dependencies?.whisper?.dependency_present && status.system_dependencies?.whisper?.load_test_status !== 'passed'
                },
              ]}
            >
              <div className="mt-4 border-t border-border-light pt-3 flex flex-col gap-3">
                <p className="text-xs text-text-muted leading-relaxed">
                  Audio pipeline requires <strong>ffmpeg</strong> for audio normalization and <strong>faster-whisper</strong> large-v3 for transcript segmentation.
                </p>
                
                {audioReadyError && (
                   <div className="text-xs text-status-error bg-status-error-bg p-2.5 border border-status-error/45 font-mono">
                     [ERROR] {audioReadyError}
                   </div>
                )}
                
                {audioReadyResult && (
                   <div className="text-xs bg-surface-secondary p-2.5 border border-border-light font-mono">
                     {audioReadyResult.audio_pipeline_ready ? (
                       <span className="text-status-success font-bold">[SUCCESS] Deep test passed. Model verified in cache.</span>
                     ) : (
                       <span className="text-status-error font-bold">[FAILURE] Deep test failed: {audioReadyResult.whisper?.error_message || 'Missing dependencies'}</span>
                     )}
                   </div>
                )}

                {/* Only one primary action here */}
                <button 
                  onClick={checkAudioReadiness}
                  disabled={audioChecking}
                  className="btn-primary self-start text-xs py-1.5"
                >
                  {audioChecking ? 'Testing...' : 'Check Audio Readiness'}
                </button>
              </div>
            </StatusPanel>

            <StatusPanel 
              title="Global Model Configurations" 
              icon={Database}
              ok={true}
              items={[
                { label: 'Text Embedding', value: status.models?.text_embedding },
                { label: 'Image Embedding', value: status.models?.image_embedding },
                { label: 'Audio Transcription', value: status.models?.transcription },
              ]}
            />
          </div>

          <div className="panel-2 p-4 bg-surface">
            <h3 className="font-bold text-sm font-mono uppercase tracking-wider text-text-muted mb-4">Raw Engine JSON Response</h3>
            <pre className="bg-surface-secondary p-4 border border-border-light text-xs text-text-muted font-mono overflow-x-auto">
              {JSON.stringify(status, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}

function StatusPanel({ title, icon: Icon, ok, items, children }: { title: string, icon: any, ok: boolean, items: any[], children?: React.ReactNode }) {
  return (
    <div className="panel-2 h-full flex flex-col bg-surface">
      <div className="px-4 py-3 bg-surface-secondary border-b-2 border-border flex items-center gap-3">
        <Icon className={clsx("w-5 h-5 shrink-0", ok ? "text-status-success" : "text-status-error")} />
        <h2 className="text-sm font-bold text-text-main uppercase tracking-wider font-mono">{title}</h2>
      </div>
      <div className="p-4 flex-1 flex flex-col justify-between">
        <ul className="space-y-2.5">
          {items.map((item, i) => (
            <li key={i} className="flex justify-between items-center text-xs border-b border-border-light pb-2 last:border-0 last:pb-0 gap-4">
              <span className="text-text-muted font-mono font-medium">{item.label}</span>
              <span className={clsx(
                "font-mono font-bold flex items-center gap-1.5 text-right uppercase tracking-wider",
                item.ok === true && "text-status-success",
                item.ok === false && !item.warning && "text-status-error",
                item.ok === false && item.warning && "text-status-warning",
                item.ok === undefined && "text-text-main"
              )}>
                {item.ok === true && <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />}
                {item.ok === false && !item.warning && <AlertCircle className="w-3.5 h-3.5 shrink-0" />}
                {item.ok === false && item.warning && <AlertCircle className="w-3.5 h-3.5 shrink-0" />}
                <span className="truncate max-w-[120px] sm:max-w-[200px]" title={item.value}>{item.value}</span>
              </span>
            </li>
          ))}
        </ul>
        {children}
      </div>
    </div>
  )
}
