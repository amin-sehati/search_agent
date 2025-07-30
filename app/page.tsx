'use client'

import { useState, useEffect } from 'react'
import { Search, Clock, CheckCircle, AlertCircle, Download, RotateCcw } from 'lucide-react'

interface ProgressEvent {
  timestamp: string
  step: string
  message: string
  step_number?: number
  progress: number
}

interface ResearchResult {
  query: string
  research_questions: string[]
  summary: string
  report: string
  sources: Array<{
    title: string
    url: string
    snippet: string
    source: string
    score: number
    published_date?: string
  }>
  total_sources: number
  timestamp: string
}

interface StreamEvent {
  type: 'progress' | 'complete' | 'error'
  data?: any
  error?: string
}

export default function Home() {
  const [query, setQuery] = useState('')
  const [isResearching, setIsResearching] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState('')
  const [progressEvents, setProgressEvents] = useState<ProgressEvent[]>([])
  const [result, setResult] = useState<ResearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('questions')

  const startResearch = async () => {
    if (!query.trim()) return

    setIsResearching(true)
    setProgress(0)
    setCurrentStep('')
    setProgressEvents([])
    setResult(null)
    setError(null)

    try {
      const response = await fetch('/api/research/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query.trim() }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const eventData: StreamEvent = JSON.parse(line.slice(6))
              
              if (eventData.type === 'progress' && eventData.data) {
                const progressData = eventData.data as ProgressEvent
                setProgress(progressData.progress)
                setCurrentStep(progressData.step)
                setProgressEvents(prev => [...prev, progressData])
              } else if (eventData.type === 'complete' && eventData.data) {
                setResult(eventData.data as ResearchResult)
                setProgress(100)
                setCurrentStep('Complete')
              } else if (eventData.type === 'error') {
                setError(eventData.error || 'Unknown error occurred')
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsResearching(false)
    }
  }

  const resetResearch = () => {
    setQuery('')
    setResult(null)
    setError(null)
    setProgress(0)
    setCurrentStep('')
    setProgressEvents([])
    setActiveTab('questions')
  }

  const saveResearch = () => {
    if (!result) return

    const dataStr = JSON.stringify(result, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `research_${new Date().toISOString().slice(0, 10)}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const renderMarkdown = (text: string) => {
    return text
      .replace(/^### (.*$)/gim, '<h3 class="text-lg font-medium mb-2">$1</h3>')
      .replace(/^## (.*$)/gim, '<h2 class="text-xl font-semibold mb-3">$1</h2>')
      .replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold mb-4">$1</h1>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
      .replace(/\[(\d+)\]/g, '<sup class="text-blue-600">[$1]</sup>')
      .replace(/\n\n/g, '</p><p class="mb-4">')
      .replace(/\n/g, '<br>')
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">🔍 AI Research Assistant</h1>
        <p className="text-gray-600">Get comprehensive research reports with real-time progress updates</p>
      </div>

      {/* Query Input */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <div className="flex flex-col gap-4">
          <div>
            <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
              Research Query
            </label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Companies like Uber that failed in the past and died"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={3}
              disabled={isResearching}
            />
          </div>
          <div className="flex gap-3">
            <button
              onClick={startResearch}
              disabled={!query.trim() || isResearching}
              className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Search className="w-4 h-4" />
              {isResearching ? 'Researching...' : 'Start Research'}
            </button>
            {result && (
              <>
                <button
                  onClick={saveResearch}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                >
                  <Download className="w-4 h-4" />
                  Save
                </button>
                <button
                  onClick={resetResearch}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
                >
                  <RotateCcw className="w-4 h-4" />
                  New Research
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Progress Section */}
      {(isResearching || progressEvents.length > 0) && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5" />
            Research Progress
          </h2>
          
          {/* Progress Bar */}
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>{currentStep}</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {/* Progress Events */}
          <div className="max-h-48 overflow-y-auto">
            {progressEvents.map((event, index) => (
              <div key={index} className="flex items-start gap-2 text-sm text-gray-600 mb-1">
                <span className="text-blue-600 font-mono">[{event.timestamp}]</span>
                <span>{event.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8 flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-medium text-red-800">Error</h3>
            <p className="text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'questions', label: '📋 Research Questions', count: result.research_questions.length },
                { id: 'summary', label: '📝 Summary', count: null },
                { id: 'report', label: '📄 Full Report', count: null },
                { id: 'sources', label: '🔗 Sources', count: result.total_sources }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab.label}
                  {tab.count !== null && (
                    <span className="ml-2 bg-gray-100 text-gray-600 py-0.5 px-2 rounded-full text-xs">
                      {tab.count}
                    </span>
                  )}
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {activeTab === 'questions' && (
              <div>
                <h2 className="text-xl font-semibold mb-4">Research Questions</h2>
                <div className="space-y-3">
                  {result.research_questions.map((question, index) => (
                    <div key={index} className="flex items-start gap-3">
                      <span className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                        {index + 1}
                      </span>
                      <p className="text-gray-700">{question}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'summary' && (
              <div>
                <h2 className="text-xl font-semibold mb-4">Research Summary</h2>
                <div
                  className="markdown-content"
                  dangerouslySetInnerHTML={{
                    __html: `<p class="mb-4">${renderMarkdown(result.summary)}</p>`
                  }}
                />
              </div>
            )}

            {activeTab === 'report' && (
              <div>
                <h2 className="text-xl font-semibold mb-4">Full Research Report</h2>
                <div
                  className="markdown-content"
                  dangerouslySetInnerHTML={{
                    __html: `<p class="mb-4">${renderMarkdown(result.report)}</p>`
                  }}
                />
              </div>
            )}

            {activeTab === 'sources' && (
              <div>
                <h2 className="text-xl font-semibold mb-4">Sources ({result.total_sources})</h2>
                <div className="space-y-4">
                  {result.sources.map((source, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-start justify-between gap-4 mb-2">
                        <h3 className="font-medium text-blue-600 hover:text-blue-800">
                          <a href={source.url} target="_blank" rel="noopener noreferrer">
                            [{index + 1}] {source.title}
                          </a>
                        </h3>
                        <div className="flex items-center gap-2 text-sm text-gray-500">
                          <span className="bg-gray-100 px-2 py-1 rounded">{source.source}</span>
                          <span>Score: {source.score.toFixed(2)}</span>
                        </div>
                      </div>
                      <p className="text-gray-600 text-sm mb-2">{source.url}</p>
                      {source.published_date && (
                        <p className="text-gray-500 text-sm mb-2">Published: {source.published_date}</p>
                      )}
                      <p className="text-gray-700">
                        {source.snippet.length > 500 ? `${source.snippet.slice(0, 500)}...` : source.snippet}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Success Indicator */}
      {result && !isResearching && (
        <div className="fixed bottom-4 right-4 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2">
          <CheckCircle className="w-4 h-4" />
          Research Complete!
        </div>
      )}
    </div>
  )
}