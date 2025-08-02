import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { query } = body

    if (!query) {
      return new Response('Query is required', { status: 400 })
    }

    // Create a readable stream for Server-Sent Events
    const encoder = new TextEncoder()
    
    const readable = new ReadableStream({
      start(controller) {
        // Send initial progress event
        const initialEvent = {
          type: 'progress',
          data: {
            timestamp: new Date().toLocaleTimeString(),
            step: 'Starting',
            message: '🚀 Initializing company research...',
            progress: 5
          }
        }
        
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(initialEvent)}\n\n`))

        // Simulate the company research workflow
        setTimeout(() => {
          // Step 1: User input processing
          const step1Event = {
            type: 'progress',
            data: {
              timestamp: new Date().toLocaleTimeString(),
              step: 'User Input',
              message: '🎯 Processing market/topic query...',
              progress: 15
            }
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(step1Event)}\n\n`))
          
          setTimeout(() => {
            // Step 2: Company discovery
            const step2Event = {
              type: 'progress',
              data: {
                timestamp: new Date().toLocaleTimeString(),
                step: 'Company Discovery',
                message: '🔍 Searching for companies in the market...',
                progress: 35
              }
            }
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(step2Event)}\n\n`))
            
            setTimeout(() => {
              // Step 3: Company list ready
              const step3Event = {
                type: 'progress',
                data: {
                  timestamp: new Date().toLocaleTimeString(),
                  step: 'Company List',
                  message: '📋 Creating company list...',
                  progress: 50
                }
              }
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(step3Event)}\n\n`))
              
              setTimeout(() => {
                // Send mock company discovery result
                const discoveryResult = {
                  type: 'company_discovery',
                  data: {
                    query: query,
                    market_topic: extractMarketTopic(query),
                    companies: generateMockCompanies(query),
                    total_companies: 5,
                    tavily_source_count: 3,
                    firecrawl_source_count: 7,
                    total_sources: 8,
                    timestamp: new Date().toISOString(),
                    awaiting_user_input: true,
                    step: 'company_review'
                  }
                }
                
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(discoveryResult)}\n\n`))
                controller.close()
                
              }, 1000)
            }, 1500)
          }, 1000)
        }, 500)
      }
    })

    return new Response(readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    })
  } catch (error) {
    console.error('Stream API error:', error)
    
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    )
  }
}

function extractMarketTopic(query: string): string {
  // Simple market topic extraction logic
  if (query.toLowerCase().includes('uber') || query.toLowerCase().includes('ride') || query.toLowerCase().includes('sharing')) {
    return 'ride-sharing and transportation market'
  }
  if (query.toLowerCase().includes('food') || query.toLowerCase().includes('delivery')) {
    return 'food delivery market'
  }
  if (query.toLowerCase().includes('social') || query.toLowerCase().includes('media')) {
    return 'social media market'
  }
  if (query.toLowerCase().includes('fintech') || query.toLowerCase().includes('finance')) {
    return 'fintech market'
  }
  
  return `${query} market`
}

function generateMockCompanies(query: string): Array<{name: string, description: string, reasoning: string}> {
  const lowerQuery = query.toLowerCase()
  
  if (lowerQuery.includes('uber') || lowerQuery.includes('ride') || lowerQuery.includes('sharing')) {
    return [
      {
        name: 'Lyft',
        description: 'Ride-sharing platform connecting drivers with passengers',
        reasoning: 'Direct competitor to Uber in the ride-sharing market with similar business model'
      },
      {
        name: 'Sidecar',
        description: 'Former ride-sharing company that shut down in 2015',
        reasoning: 'Early ride-sharing company that failed to compete with Uber and Lyft'
      },
      {
        name: 'Haxi',
        description: 'Brazilian ride-sharing app',
        reasoning: 'Regional ride-sharing competitor that operated in Brazil'
      },
      {
        name: 'Juno',
        description: 'Ride-sharing service that was acquired by Gett',
        reasoning: 'NYC-focused ride-sharing company that tried to compete with Uber'
      },
      {
        name: 'Via',
        description: 'Shared ride service focusing on carpooling',
        reasoning: 'Ride-sharing company focusing on shared rides and carpooling solutions'
      }
    ]
  }
  
  // Default mock companies for other queries
  return [
    {
      name: 'Company Alpha',
      description: 'Innovative startup in the specified market',
      reasoning: 'Operates directly in the market mentioned in the query'
    },
    {
      name: 'Beta Solutions',
      description: 'Enterprise solution provider',
      reasoning: 'Provides solutions for businesses in this market sector'
    },
    {
      name: 'Gamma Technologies',
      description: 'Tech company with focus on this market',
      reasoning: 'Technology-focused company serving this specific market'
    },
    {
      name: 'Delta Ventures',
      description: 'Failed startup that shut down in 2020',
      reasoning: 'Former company that attempted to capture market share but failed'
    },
    {
      name: 'Epsilon Corp',
      description: 'Established company in this space',
      reasoning: 'Long-standing player in this market with proven track record'
    }
  ]
}