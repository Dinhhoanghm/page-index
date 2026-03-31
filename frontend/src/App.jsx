import { useState, useRef, useEffect, useCallback } from 'react'

const API = import.meta.env.VITE_API_URL ?? ''

// ─── Styles ───────────────────────────────────────────────────────────────────
const s = {
  // Layout
  app: { display: 'flex', height: '100vh', overflow: 'hidden' },
  sidebar: {
    width: 260, background: '#1a1d27', borderRight: '1px solid #2d3148',
    display: 'flex', flexDirection: 'column', padding: '20px 16px', gap: 20,
    flexShrink: 0,
  },
  main: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' },

  // Sidebar elements
  logo: { fontSize: 18, fontWeight: 700, color: '#818cf8', letterSpacing: '-0.3px' },
  sectionLabel: { fontSize: 11, fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 },
  badge: (active) => ({
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    padding: '5px 14px', borderRadius: 20, fontSize: 13, fontWeight: 600, cursor: 'pointer',
    background: active ? '#4f46e5' : 'transparent',
    color: active ? '#fff' : '#94a3b8',
    border: `1px solid ${active ? '#4f46e5' : '#2d3148'}`,
    transition: 'all 0.15s',
  }),
  modeRow: { display: 'flex', gap: 8 },
  keyVal: { fontSize: 12, color: '#64748b', wordBreak: 'break-all' },
  keyMasked: { color: '#818cf8', fontFamily: 'monospace' },
  divider: { height: 1, background: '#2d3148' },

  // Upload area
  uploadArea: (dragging) => ({
    border: `2px dashed ${dragging ? '#818cf8' : '#2d3148'}`,
    borderRadius: 10, padding: '28px 16px', textAlign: 'center', cursor: 'pointer',
    background: dragging ? '#1e2038' : 'transparent', transition: 'all 0.15s',
  }),
  uploadIcon: { fontSize: 28, marginBottom: 8 },
  uploadHint: { fontSize: 13, color: '#64748b' },
  uploadLabel: { fontSize: 14, color: '#94a3b8', marginBottom: 4 },

  // Progress
  progressWrap: { background: '#1a1d27', borderRadius: 8, padding: '12px 14px', border: '1px solid #2d3148' },
  progressBar: (pct) => ({
    height: 4, borderRadius: 2, background: '#4f46e5', width: `${pct}%`, transition: 'width 0.3s',
  }),
  progressBg: { height: 4, borderRadius: 2, background: '#2d3148', marginTop: 8 },
  statusDot: (color) => ({
    width: 8, height: 8, borderRadius: '50%', background: color, display: 'inline-block', marginRight: 6,
  }),

  // Document selector
  select: {
    width: '100%', background: '#1e2038', border: '1px solid #2d3148', borderRadius: 8,
    color: '#e2e8f0', padding: '8px 10px', fontSize: 13, cursor: 'pointer', outline: 'none',
  },

  // Chat panel
  chatHeader: {
    padding: '14px 20px', borderBottom: '1px solid #2d3148',
    display: 'flex', alignItems: 'center', gap: 10, background: '#1a1d27',
  },
  chatTitle: { fontWeight: 600, fontSize: 15 },
  chatSub: { fontSize: 12, color: '#64748b' },
  messages: {
    flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: 16,
  },
  bubble: (role) => ({
    maxWidth: '78%', borderRadius: 14, padding: '12px 16px', fontSize: 14, lineHeight: 1.6,
    alignSelf: role === 'user' ? 'flex-end' : 'flex-start',
    background: role === 'user' ? '#4f46e5' : '#1e2038',
    color: role === 'user' ? '#fff' : '#e2e8f0',
    border: role === 'user' ? 'none' : '1px solid #2d3148',
    whiteSpace: 'pre-wrap', wordBreak: 'break-word',
  }),
  citation: {
    display: 'inline-flex', alignItems: 'center', padding: '1px 7px', borderRadius: 10,
    background: '#312e81', color: '#a5b4fc', fontSize: 11, fontWeight: 600, margin: '0 2px',
  },
  reasoning: {
    marginTop: 10, borderRadius: 8, background: '#0f1721', border: '1px solid #1e3a5f',
    padding: '10px 12px', fontSize: 12, color: '#7dd3fc', lineHeight: 1.5,
  },
  reasoningToggle: {
    fontSize: 11, color: '#3b82f6', cursor: 'pointer', userSelect: 'none', marginTop: 6,
  },
  sections: { marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 },
  sectionBadge: {
    fontSize: 11, padding: '2px 8px', borderRadius: 10, background: '#1e2038',
    border: '1px solid #2d3148', color: '#94a3b8',
  },

  // Input bar
  inputBar: {
    padding: '14px 20px', borderTop: '1px solid #2d3148', display: 'flex', gap: 10,
    background: '#1a1d27',
  },
  input: {
    flex: 1, background: '#0f1117', border: '1px solid #2d3148', borderRadius: 10,
    color: '#e2e8f0', padding: '10px 14px', fontSize: 14, outline: 'none',
    resize: 'none', fontFamily: 'inherit', lineHeight: 1.5, minHeight: 44, maxHeight: 140,
  },
  sendBtn: (disabled) => ({
    padding: '10px 18px', borderRadius: 10, border: 'none', fontWeight: 600, fontSize: 14,
    background: disabled ? '#2d3148' : '#4f46e5', color: disabled ? '#64748b' : '#fff',
    cursor: disabled ? 'not-allowed' : 'pointer', flexShrink: 0, transition: 'all 0.15s',
  }),

  // Empty state
  empty: {
    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', color: '#4a5568', gap: 10,
  },
  emptyIcon: { fontSize: 48 },
  emptyText: { fontSize: 15, color: '#64748b' },
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function mask(key) {
  if (!key || key.length < 8) return '••••••••'
  return key.slice(0, 4) + '••••' + key.slice(-4)
}

function statusColor(status) {
  return { completed: '#22c55e', processing: '#f59e0b', failed: '#ef4444', queued: '#f59e0b' }[status] ?? '#64748b'
}

// ─── Message component ────────────────────────────────────────────────────────
function Message({ msg }) {
  const [showReasoning, setShowReasoning] = useState(false)

  return (
    <div style={s.bubble(msg.role)}>
      <div>{msg.content}</div>

      {msg.citations && msg.citations.length > 0 && (
        <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {msg.citations.map((c, i) => (
            <span key={i} style={s.citation}>p.{c}</span>
          ))}
        </div>
      )}

      {msg.sections && msg.sections.length > 0 && (
        <div style={s.sections}>
          {msg.sections.map((sec, i) => (
            <span key={i} style={s.sectionBadge}>
              {sec.doc && <span style={{ color: '#818cf8' }}>[{sec.doc}] </span>}
              {sec.title || sec.node_id} · p.{sec.page}
            </span>
          ))}
        </div>
      )}

      {msg.reasoning && (
        <>
          <div style={s.reasoningToggle} onClick={() => setShowReasoning(v => !v)}>
            {showReasoning ? '▾ Hide reasoning' : '▸ Show reasoning'}
          </div>
          {showReasoning && <div style={s.reasoning}>{msg.reasoning}</div>}
        </>
      )}
    </div>
  )
}

// ─── Upload Panel ─────────────────────────────────────────────────────────────
function UploadPanel({ onDocReady }) {
  const [dragging, setDragging] = useState(false)
  const [uploads, setUploads] = useState([]) // [{uid, name, docId, status, progress, pageNum}]
  const fileRef = useRef()

  const patch = useCallback((uid, update) =>
    setUploads(prev => prev.map(u => u.uid === uid ? { ...u, ...update } : u)), [])

  const startUpload = useCallback(async (files) => {
    const pdfs = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'))
    if (!pdfs.length) { alert('Please select PDF file(s).'); return }

    pdfs.forEach(async (file) => {
      const uid = Math.random().toString(36).slice(2)
      setUploads(prev => [...prev, { uid, name: file.name, docId: null, status: 'uploading', progress: 10 }])

      try {
        const form = new FormData()
        form.append('file', file)
        const res = await fetch(`${API}/api/upload`, { method: 'POST', body: form })
        if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed')
        const { doc_id } = await res.json()

        patch(uid, { docId: doc_id, status: 'processing', progress: 30 })

        const poll = setInterval(async () => {
          try {
            const data = await fetch(`${API}/api/status/${doc_id}`).then(r => r.json())
            const pct = data.status === 'completed' ? 100 : Math.min(90, (Date.now() % 60000) / 600 + 30)
            patch(uid, { status: data.status, progress: pct, pageNum: data.pageNum })
            if (data.status === 'completed' || data.status === 'failed') {
              clearInterval(poll)
              if (data.status === 'completed') onDocReady({ docId: doc_id, name: file.name, pageNum: data.pageNum })
            }
          } catch { /* ignore poll errors */ }
        }, 2000)
      } catch (err) {
        patch(uid, { status: 'failed' })
        alert(`${file.name}: ${err.message}`)
      }
    })
  }, [patch, onDocReady])

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    startUpload(e.dataTransfer.files)
  }, [startUpload])

  return (
    <div>
      <div style={s.sectionLabel}>Upload PDFs</div>

      <div
        style={s.uploadArea(dragging)}
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div style={s.uploadIcon}>📄</div>
        <div style={s.uploadLabel}>Drop PDFs here</div>
        <div style={s.uploadHint}>or click to browse (multiple OK)</div>
        <input
          ref={fileRef} type="file" accept=".pdf" multiple style={{ display: 'none' }}
          onChange={(e) => startUpload(e.target.files)}
        />
      </div>

      {uploads.map(u => (
        <div key={u.uid} style={{ ...s.progressWrap, marginTop: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
            <span style={{ fontSize: 12, color: '#94a3b8', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {u.name}
            </span>
            <span style={{ fontSize: 11, display: 'flex', alignItems: 'center' }}>
              <span style={s.statusDot(statusColor(u.status))} />
              {u.status}
            </span>
          </div>
          <div style={s.progressBg}>
            <div style={s.progressBar(u.progress)} />
          </div>
          {u.pageNum && (
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{u.pageNum} pages</div>
          )}
        </div>
      ))}
    </div>
  )
}

// ─── Document Selector ────────────────────────────────────────────────────────
function DocSelector({ currentDocId, onSelect }) {
  const [docs, setDocs] = useState([])

  useEffect(() => {
    fetch(`${API}/api/documents`)
      .then(r => r.json())
      .then(d => setDocs(d.documents || []))
      .catch(() => {})
  }, [currentDocId]) // refresh when a new doc is uploaded

  const completed = docs.filter(d => d.status === 'completed')
  if (!completed.length) return null

  return (
    <div>
      <div style={s.sectionLabel}>Switch Document</div>
      <select
        style={s.select}
        value={currentDocId || ''}
        onChange={(e) => {
          const val = e.target.value
          if (val === '__all__') {
            onSelect({ docId: '__all__', name: `All Documents (${completed.length})`, pageNum: null })
            return
          }
          const doc = completed.find(d => d.id === val)
          if (doc) onSelect({ docId: doc.id, name: doc.name, pageNum: doc.pageNum })
        }}
      >
        <option value="">— select —</option>
        {completed.length > 1 && (
          <option value="__all__">All Documents ({completed.length})</option>
        )}
        {completed.map(d => (
          <option key={d.id} value={d.id}>{d.name || d.id}</option>
        ))}
      </select>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [doc, setDoc] = useState(null)       // {docId, name, pageNum}
  const [mode, setMode] = useState('auto')
  const [cite, setCite] = useState(true)
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef()
  const textareaRef = useRef()

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(scrollToBottom, [messages])

  const handleDocReady = useCallback((newDoc) => {
    setDoc(newDoc)
    setMessages([])
  }, [])

  const sendMessage = useCallback(async () => {
    const question = draft.trim()
    if (!question || !doc || streaming) return

    const userMsg = { role: 'user', content: question }
    const history = [...messages, userMsg]
    setMessages(history)
    setDraft('')
    setStreaming(true)

    // Placeholder assistant message we'll fill in
    const assistantMsg = { role: 'assistant', content: '', reasoning: null, sections: null, citations: [] }
    setMessages([...history, assistantMsg])

    try {
      const res = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id: doc.docId,
          messages: history.map(({ role, content }) => ({ role, content })),
          mode,
          cite: mode === 'auto' ? cite : false,
        }),
      })

      if (!res.ok) throw new Error('Chat request failed')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete last line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const payload = line.slice(6).trim()
          if (payload === '[DONE]') continue

          try {
            const evt = JSON.parse(payload)

            setMessages(prev => {
              const updated = [...prev]
              const last = { ...updated[updated.length - 1] }

              if (evt.text !== undefined) {
                last.content += evt.text
              }
              if (evt.reasoning !== undefined) {
                last.reasoning = evt.reasoning
              }
              if (evt.sections !== undefined) {
                last.sections = evt.sections
              }
              if (evt.error !== undefined) {
                last.content = `[Error] ${evt.error}`
              }

              updated[updated.length - 1] = last
              return updated
            })
          } catch { /* ignore parse errors */ }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = { role: 'assistant', content: `[Error] ${err.message}` }
        return updated
      })
    } finally {
      setStreaming(false)
    }
  }, [draft, doc, messages, mode, cite, streaming])

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const canSend = !!doc && !!draft.trim() && !streaming

  return (
    <div style={s.app}>
      {/* ── Sidebar ── */}
      <aside style={s.sidebar}>
        <div style={s.logo}>PageIndex RAG</div>

        <div style={s.divider} />

        {/* LLM provider key */}
        <div>
          <div style={s.sectionLabel}>LLM Provider</div>
          <div style={s.keyVal}>
            {import.meta.env.VITE_ANTHROPIC_KEY
              ? <>Anthropic: <span style={s.keyMasked}>{mask(import.meta.env.VITE_ANTHROPIC_KEY)}</span></>
              : <>OpenAI: <span style={s.keyMasked}>{mask(import.meta.env.VITE_OPENAI_KEY)}</span></>
            }
          </div>
        </div>

        <div style={s.divider} />

        {/* Mode toggle */}
        <div>
          <div style={s.sectionLabel}>Mode</div>
          <div style={s.modeRow}>
            <button style={s.badge(mode === 'auto')} onClick={() => setMode('auto')}>Auto</button>
            <button style={s.badge(mode === 'manual')} onClick={() => setMode('manual')}>Manual</button>
          </div>
          {mode === 'auto' && (
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10, fontSize: 13, color: '#94a3b8', cursor: 'pointer' }}>
              <input type="checkbox" checked={cite} onChange={e => setCite(e.target.checked)} />
              Page citations
            </label>
          )}
          {mode === 'manual' && (
            <div style={{ marginTop: 8, fontSize: 12, color: '#64748b', lineHeight: 1.4 }}>
              Uses local LLM to select sections then generate the answer. Shows reasoning trace.
            </div>
          )}
        </div>

        <div style={s.divider} />

        {/* Upload */}
        <UploadPanel onDocReady={handleDocReady} />

        <div style={s.divider} />

        {/* Doc selector */}
        <DocSelector currentDocId={doc?.docId} onSelect={handleDocReady} />
      </aside>

      {/* ── Main area ── */}
      <main style={s.main}>
        {/* Header */}
        <div style={s.chatHeader}>
          <div>
            <div style={s.chatTitle}>{doc ? doc.name : 'No document selected'}</div>
            {doc && (
              <div style={s.chatSub}>
                {doc.pageNum ? `${doc.pageNum} pages · ` : ''}{doc.docId} · {mode} mode
              </div>
            )}
          </div>
        </div>

        {/* Messages */}
        {messages.length === 0 ? (
          <div style={s.empty}>
            <div style={s.emptyIcon}>🔍</div>
            <div style={s.emptyText}>
              {doc ? 'Ask a question about your document' : 'Upload or select a PDF to get started'}
            </div>
          </div>
        ) : (
          <div style={s.messages}>
            {messages.map((msg, i) => <Message key={i} msg={msg} />)}
            {streaming && messages[messages.length - 1]?.content === '' && (
              <div style={{ alignSelf: 'flex-start', color: '#64748b', fontSize: 13 }}>Thinking…</div>
            )}
            <div ref={bottomRef} />
          </div>
        )}

        {/* Input */}
        <div style={s.inputBar}>
          <textarea
            ref={textareaRef}
            style={s.input}
            placeholder={doc ? 'Ask anything about this document… (Enter to send, Shift+Enter for newline)' : 'Upload a PDF first'}
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={!doc || streaming}
            rows={1}
          />
          <button style={s.sendBtn(!canSend)} onClick={sendMessage} disabled={!canSend}>
            {streaming ? '…' : 'Send'}
          </button>
        </div>
      </main>
    </div>
  )
}
