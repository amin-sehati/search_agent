'use client'

import { useState } from 'react'
import { Search, Clock, CheckCircle, AlertCircle, Download, RotateCcw } from 'lucide-react'

interface ProgressEvent {
  timestamp: string
  step: string
  message: string
  step_number?: number
  progress: number
}

interface Company {
  name: string
  description: string
  reasoning: string
  year_established?: string
  still_in_business?: boolean
  history?: string
  future_roadmap?: string
}

interface CompanyDiscoveryResult {
  query: string
  market_topic: string
  companies: Company[]
  total_companies: number
  tavily_source_count: number
  firecrawl_source_count: number
  total_sources: number
  timestamp: string
  awaiting_user_input: boolean
  step: string
}
interface CompanyResearchResult {
  query: string
  market_topic: string
  company_pages: Record<string, string>
  total_companies: number
  timestamp: string
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
  type: 'progress' | 'complete' | 'error' | 'company_discovery'
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
  const [companyDiscovery, setCompanyDiscovery] = useState<CompanyDiscoveryResult | null>(null)
  const [companyResult, setCompanyResult] = useState<CompanyResearchResult | null>(null)
  const [companies, setCompanies] = useState<Company[]>([])
  const [isGatheringInfo, setIsGatheringInfo] = useState(false)
  const [researchState, setResearchState] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('companies')
  const [companyProfileProgress, setCompanyProfileProgress] = useState(0)
  const [companyProfileStep, setCompanyProfileStep] = useState('')
  const [companyProfileEvents, setCompanyProfileEvents] = useState<ProgressEvent[]>([])
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
              } else if (eventData.type === 'company_discovery' && eventData.data) {
                const discoveryData = eventData.data as CompanyDiscoveryResult
                setCompanyDiscovery(discoveryData)
                setCompanies(discoveryData.companies)
                setResearchState(eventData.data)
                setProgress(100)
                setCurrentStep('Company Search Complete')
              } else if (eventData.type === 'complete' && eventData.data) {
                if (eventData.data.company_pages) {
                  setCompanyResult(eventData.data as CompanyResearchResult)
                } else {
                  setResult(eventData.data as ResearchResult)
                }
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
    setCompanyDiscovery(null)
    setCompanyResult(null)
    setCompanies([])
    setResearchState(null)
    setError(null)
    setProgress(0)
    setCurrentStep('')
    setProgressEvents([])
    setIsGatheringInfo(false)
    setActiveTab('companies')
    setCompanyProfileProgress(0)
    setCompanyProfileStep('')
    setCompanyProfileEvents([])
  }

  const saveResearch = () => {
    const dataToSave = companyResult || result
    if (!dataToSave) return

    const dataStr = JSON.stringify(dataToSave, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `research_${new Date().toISOString().slice(0, 10)}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const findCompanyInfo = async () => {
    if (!researchState || companies.length === 0) return

    setIsGatheringInfo(true)
    setCompanyProfileProgress(0)
    setCompanyProfileStep('')
    setCompanyProfileEvents([])
    setError(null)

    try {
      const response = await fetch('/api/research/company-info', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          state: researchState, 
          user_companies: companies 
        }),
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
                setCompanyProfileProgress(progressData.progress)
                setCompanyProfileStep(progressData.step)
                setCompanyProfileEvents(prev => [...prev, progressData])
              } else if (eventData.type === 'complete' && eventData.data) {
                setCompanyResult(eventData.data as CompanyResearchResult)
                setCompanyProfileProgress(100)
                setCompanyProfileStep('Complete')
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
      setIsGatheringInfo(false)
    }
  }

  const addCompany = () => {
    setCompanies([...companies, {
      name: '',
      description: '',
      reasoning: ''
    }])
  }

  const removeCompany = (index: number) => {
    setCompanies(companies.filter((_, i) => i !== index))
  }

  const updateCompany = (index: number, field: keyof Company, value: string) => {
    const updatedCompanies = [...companies]
    updatedCompanies[index] = { ...updatedCompanies[index], [field]: value }
    setCompanies(updatedCompanies)
  }

  const renderMarkdown = (text: string) => {
    if (!text) return ''
    
    let html = text

    // Remove fenced code block markers (e.g., ``` or ```markdown)
    html = html.replace(/^```[\s\w]*$/gm, '')
    
    // Handle headers (must be at start of line)
    html = html.replace(/^# (.+)$/gm, '<h1 class="text-2xl font-bold mb-4 mt-6 text-gray-900">$1</h1>')
    html = html.replace(/^## (.+)$/gm, '<h2 class="text-xl font-semibold mb-3 mt-5 text-gray-900">$1</h2>')
    html = html.replace(/^### (.+)$/gm, '<h3 class="text-lg font-semibold mb-2 mt-4 text-gray-900">$1</h3>')
    
    // Handle bold and italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>')
    html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em class="italic">$1</em>')
    
    // Handle unordered lists
    html = html.replace(/^[-*+] (.+)$/gm, '<li class="ml-6 mb-1 text-gray-800">• $1</li>')
    
    // Handle ordered lists  
    html = html.replace(/^\d+\. (.+)$/gm, '<li class="ml-6 mb-1 text-gray-800 list-decimal">$1</li>')
    
    // Handle citations/references
    html = html.replace(/\[(\d+)\]/g, '<sup class="text-blue-600 text-xs">[$1]</sup>')
    
    // Handle code blocks
    html = html.replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">$1</code>')
    
    // Handle emoji status indicators
    html = html.replace(/❌/g, '<span class="text-red-600">❌</span>')
    html = html.replace(/✅/g, '<span class="text-green-600">✅</span>')
    
    // Convert line breaks and paragraphs
    html = html.replace(/\n\n+/g, '</p><p class="mb-4 text-gray-800">')
    html = html.replace(/\n/g, '<br>')
    
    // Wrap content in a paragraph only if it doesn't already start with a block element
    if (!html.match(/^<(h[1-6]|div|p|ul|ol|li)/)) {
      html = `<p class="mb-4 text-gray-800">${html}</p>`
    }
    
    // Clean up any empty paragraphs
    html = html.replace(/<p[^>]*><\/p>/g, '')
    
    return html
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-black mb-2">🔍 AI Research Assistant</h1>
        <p className="text-black">Get comprehensive research reports with real-time progress updates</p>
      </div>

      {/* Query Input */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <div className="flex flex-col gap-4">
          <div>
            <label htmlFor="query" className="block text-sm font-medium text-black mb-2">
              Research Query
            </label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Ride sharing companies that are for luxury and VIP"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
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
            {(result || companyResult) && (
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
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-black">
            <Clock className="w-5 h-5" />
            Company Discovery Progress
          </h2>
          
          {/* Progress Bar */}
          <div className="mb-4">
            <div className="flex justify-between text-sm text-black mb-1">
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
              <div key={index} className="flex items-start gap-2 text-sm text-black mb-1">
                <span className="text-blue-600 font-mono">[{event.timestamp}]</span>
                <span>{event.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Company Profile Progress Section */}
      {(isGatheringInfo || companyProfileEvents.length > 0) && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-black">
            <Clock className="w-5 h-5" />
            Company Profile Creation Progress
          </h2>
          
          {/* Progress Bar */}
          <div className="mb-4">
            <div className="flex justify-between text-sm text-black mb-1">
              <span>{companyProfileStep}</span>
              <span>{Math.round(companyProfileProgress)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-purple-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${companyProfileProgress}%` }}
              />
            </div>
          </div>

          {/* Progress Events */}
          <div className="max-h-48 overflow-y-auto">
            {companyProfileEvents.map((event, index) => (
              <div key={index} className="flex items-start gap-2 text-sm text-black mb-1">
                <span className="text-purple-600 font-mono">[{event.timestamp}]</span>
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
            <h3 className="font-medium text-black">Error</h3>
            <p className="text-black">{error}</p>
          </div>
        </div>
      )}

      {/* Company Discovery Results */}
      {companyDiscovery && !companyResult && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
            <h2 className="text-xl font-semibold flex items-center gap-2 text-black">
              🏢 Companies Found in {companyDiscovery.market_topic}
            </h2>
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <div className="flex items-center gap-2 bg-blue-50 px-3 py-1 rounded-full">
                <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                <span className="text-blue-700 font-medium">Tavily: {companyDiscovery.tavily_source_count}</span>
              </div>
              <div className="flex items-center gap-2 bg-green-50 px-3 py-1 rounded-full">
                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                <span className="text-green-700 font-medium">Firecrawl: {companyDiscovery.firecrawl_source_count}</span>
              </div>
              <div className="flex items-center gap-2 bg-gray-50 px-3 py-1 rounded-full">
                <span className="w-2 h-2 bg-gray-500 rounded-full"></span>
                <span className="text-gray-700 font-medium">Total: {companyDiscovery.total_sources}</span>
              </div>
            </div>
          </div>
          
          <div className="space-y-4 mb-6">
            {companies.map((company, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex justify-between items-start mb-2">
                  <input
                    type="text"
                    value={company.name}
                    onChange={(e) => updateCompany(index, 'name', e.target.value)}
                    className="text-lg font-medium border-b border-gray-300 focus:border-blue-500 focus:outline-none bg-transparent text-gray-900"
                    placeholder="Company Name"
                  />
                  <button
                    onClick={() => removeCompany(index)}
                    className="text-red-600 hover:text-red-800 text-sm"
                  >
                    Remove
                  </button>
                </div>
                <textarea
                  value={company.description}
                  onChange={(e) => updateCompany(index, 'description', e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded focus:border-blue-500 focus:outline-none mb-2 text-gray-900"
                  placeholder="Company Description"
                  rows={2}
                />
                <textarea
                  value={company.reasoning}
                  onChange={(e) => updateCompany(index, 'reasoning', e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded focus:border-blue-500 focus:outline-none text-gray-900"
                  placeholder="Why this company is relevant to the market"
                  rows={2}
                />
              </div>
            ))}
          </div>
          
          <div className="flex gap-3">
            <button
              onClick={addCompany}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Add Company
            </button>
            <button
              onClick={findCompanyInfo}
              disabled={isGatheringInfo || companies.length === 0}
              className="flex items-center gap-2 px-6 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Search className="w-4 h-4" />
              {isGatheringInfo ? 'Creating Profiles...' : 'Create Company Profile for All'}
            </button>
          </div>
        </div>
      )}

      {/* Company Research Results */}
      {companyResult && (
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              <button
                onClick={() => setActiveTab('companies')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'companies'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-black hover:text-black hover:border-gray-300'
                }`}
              >
                🏢 Company Pages ({companyResult.total_companies})
              </button>
            </nav>
          </div>

          <div className="p-6">
            {activeTab === 'companies' && (
              <div className="space-y-8">
                {Object.entries(companyResult.company_pages).map(([companyName, pageContent]) => (
                  <div key={companyName} className="border border-gray-200 rounded-lg p-6">
                    <div
                      className="markdown-content max-w-none"
                      dangerouslySetInnerHTML={{
                        __html: renderMarkdown(pageContent)
                      }}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Legacy Research Results */}
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
                      : 'border-transparent text-black hover:text-black hover:border-gray-300'
                  }`}
                >
                  {tab.label}
                  {tab.count !== null && (
                    <span className="ml-2 bg-gray-100 text-black py-0.5 px-2 rounded-full text-xs">
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
                <h2 className="text-xl font-semibold mb-4 text-black">Research Questions</h2>
                <div className="space-y-3">
                  {result.research_questions.map((question, index) => (
                    <div key={index} className="flex items-start gap-3">
                      <span className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                        {index + 1}
                      </span>
                      <p className="text-black">{question}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'summary' && (
              <div>
                <h2 className="text-xl font-semibold mb-4 text-black">Research Summary</h2>
                <div
                  className="markdown-content max-w-none"
                  dangerouslySetInnerHTML={{
                    __html: renderMarkdown(result.summary)
                  }}
                />
              </div>
            )}

            {activeTab === 'report' && (
              <div>
                <h2 className="text-xl font-semibold mb-4 text-black">Full Research Report</h2>
                <div
                  className="markdown-content max-w-none"
                  dangerouslySetInnerHTML={{
                    __html: renderMarkdown(result.report)
                  }}
                />
              </div>
            )}

            {activeTab === 'sources' && (
              <div>
                <h2 className="text-xl font-semibold mb-4 text-black">Sources ({result.total_sources})</h2>
                <div className="space-y-4">
                  {result.sources.map((source, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-start justify-between gap-4 mb-2">
                        <h3 className="font-medium text-blue-600 hover:text-blue-800">
                          <a href={source.url} target="_blank" rel="noopener noreferrer">
                            [{index + 1}] {source.title}
                          </a>
                        </h3>
                        <div className="flex items-center gap-2 text-sm text-black">
                          <span className="bg-gray-100 px-2 py-1 rounded">{source.source}</span>
                          <span>Score: {source.score.toFixed(2)}</span>
                        </div>
                      </div>
                      <p className="text-black text-sm mb-2">{source.url}</p>
                      {source.published_date && (
                        <p className="text-black text-sm mb-2">Published: {source.published_date}</p>
                      )}
                      <p className="text-black">
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
      {(result || companyResult) && !isResearching && !isGatheringInfo && (
        <div className="fixed bottom-4 right-4 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2">
          <CheckCircle className="w-4 h-4" />
          Research Complete!
        </div>
      )}
    </div>
  )
}
