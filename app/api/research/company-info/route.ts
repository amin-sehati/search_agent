import { NextRequest } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { state, user_companies } = body

    if (!state || !user_companies) {
      return new Response('State and user_companies are required', { status: 400 })
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
            step: 'Company Info',
            message: '🔍 Gathering detailed company information...',
            progress: 60
          }
        }
        
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(initialEvent)}\n\n`))

        // Simulate company info gathering
        setTimeout(() => {
          const step1Event = {
            type: 'progress',
            data: {
              timestamp: new Date().toLocaleTimeString(),
              step: 'Company Info',
              message: '📊 Researching company histories...',
              progress: 75
            }
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(step1Event)}\n\n`))
          
          setTimeout(() => {
            const step2Event = {
              type: 'progress',
              data: {
                timestamp: new Date().toLocaleTimeString(),
                step: 'Final Synthesis',
                message: '📄 Creating company pages...',
                progress: 90
              }
            }
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(step2Event)}\n\n`))
            
            setTimeout(() => {
              // Generate mock company pages
              const companyPages = generateMockCompanyPages(user_companies)
              
              const finalResult = {
                type: 'complete',
                data: {
                  query: state.query,
                  market_topic: state.market_topic,
                  company_pages: companyPages,
                  total_companies: user_companies.length,
                  timestamp: new Date().toISOString()
                }
              }
              
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(finalResult)}\n\n`))
              controller.close()
              
            }, 1500)
          }, 1500)
        }, 1000)
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
    console.error('Company info stream API error:', error)
    
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    )
  }
}

function generateMockCompanyPages(companies: any[]): Record<string, string> {
  const pages: Record<string, string> = {}
  
  companies.forEach(company => {
    pages[company.name] = generateMockCompanyPage(company)
  })
  
  return pages
}

function generateMockCompanyPage(company: any): string {
  const isShutDown = company.name.toLowerCase().includes('sidecar') || 
                   company.name.toLowerCase().includes('delta') ||
                   company.description.toLowerCase().includes('shut down') ||
                   company.description.toLowerCase().includes('failed')
  
  return `# ${company.name}

## Company Overview
${company.description}

**Market Relevance**: ${company.reasoning}

## Year Established
${isShutDown ? 
  `Founded in ${2010 + Math.floor(Math.random() * 10)}` : 
  `Founded in ${2008 + Math.floor(Math.random() * 15)}`}

## Business Status
${isShutDown ? 
  '❌ **Shut Down** - Company ceased operations' : 
  '✅ **Active** - Currently operating'}

## Market Operations
${company.name} ${isShutDown ? 'operated' : 'operates'} in the ${extractMarketFromReasoning(company.reasoning)} space, focusing on ${company.description.toLowerCase()}.

${isShutDown ? 
  `The company faced significant challenges including competitive pressure, funding issues, and market saturation that ultimately led to its closure.` :
  `The company continues to serve customers and adapt to market changes.`}

## Company History
- **Early Stage**: ${company.name} was founded to address specific market needs
- **Growth Phase**: ${isShutDown ? 'Experienced initial growth but faced increasing competition' : 'Successfully grew user base and expanded operations'}
- **Current Status**: ${isShutDown ? 'Ceased operations due to market pressures' : 'Continues to operate and evolve'}

## Future Roadmap
${isShutDown ? 
  'No future plans as the company has shut down. Assets may have been acquired by other companies.' :
  `${company.name} plans to continue expanding its services and improving user experience.`}

## Market Position
${company.name} ${isShutDown ? 'was positioned as' : 'is positioned as'} a ${company.description.includes('competitor') ? 'direct competitor' : 'key player'} in the market, ${isShutDown ? 'but ultimately could not maintain its position.' : 'with ongoing efforts to maintain market relevance.'}

---
*Research conducted on ${new Date().toLocaleDateString()}*`
}

function extractMarketFromReasoning(reasoning: string): string {
  if (reasoning.toLowerCase().includes('ride-sharing') || reasoning.toLowerCase().includes('transportation')) {
    return 'transportation and mobility'
  }
  if (reasoning.toLowerCase().includes('food') || reasoning.toLowerCase().includes('delivery')) {
    return 'food delivery'
  }
  if (reasoning.toLowerCase().includes('social') || reasoning.toLowerCase().includes('media')) {
    return 'social media'
  }
  if (reasoning.toLowerCase().includes('fintech') || reasoning.toLowerCase().includes('finance')) {
    return 'financial technology'
  }
  
  return 'technology'
}