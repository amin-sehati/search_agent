import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  const startTime = Date.now()
  const requestId = Math.random().toString(36).substring(7)
  
  console.log(`🌐 [${requestId}] API Request received: /api/research/stream`)
  console.log(`📋 [${requestId}] Request timestamp: ${new Date().toISOString()}`)
  
  try {
    console.log(`📖 [${requestId}] Parsing request body...`)
    const body = await request.json()
    const { query } = body
    
    console.log(`🔍 [${requestId}] Query received: "${query}"`)
    console.log(`📊 [${requestId}] Request body size: ${JSON.stringify(body).length} characters`)

    if (!query) {
      console.log(`❌ [${requestId}] Query validation failed: empty query`)
      return new Response('Query is required', { status: 400 })
    }
    
    console.log(`✅ [${requestId}] Query validation passed`)
    console.log(`🚀 [${requestId}] Starting Python research process...`)

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

        // Run actual Python research system
        const pythonScript = path.join(process.cwd(), 'research_system.py')
        console.log(`🐍 [${requestId}] Python script path: ${pythonScript}`)
        console.log(`🔧 [${requestId}] Spawning Python process with args: ['python3', '${pythonScript}', '${query}']`)
        
        const pythonProcess = spawn('python3', [pythonScript, query], {
          env: { ...process.env },
          stdio: ['ignore', 'pipe', 'pipe']
        })

        console.log(`🆔 [${requestId}] Python process PID: ${pythonProcess.pid}`)

        let buffer = ''
        let hasStarted = false
        let lineCount = 0

        pythonProcess.stdout.on('data', (data) => {
          const dataStr = data.toString()
          console.log(`📤 [${requestId}] Python stdout (${dataStr.length} bytes):`, dataStr.slice(0, 200) + (dataStr.length > 200 ? '...' : ''))
          
          buffer += dataStr
          
          // Process complete lines
          const lines = buffer.split('\n')
          buffer = lines.pop() || '' // Keep incomplete line in buffer
          
          for (const line of lines) {
            lineCount++
            if (line.trim()) {
              try {
                const event = JSON.parse(line)
                console.log(`🎯 [${requestId}] JSON event (line ${lineCount}):`, event.type, event.data ? Object.keys(event.data) : 'no data')
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
                hasStarted = true
              } catch (e) {
                // Log non-JSON output
                console.log(`📋 [${requestId}] Python output (line ${lineCount}):`, line)
              }
            }
          }
        })

        pythonProcess.stderr.on('data', (data) => {
          const errorStr = data.toString()
          console.error(`❌ [${requestId}] Python stderr:`, errorStr)
        })

        pythonProcess.on('close', (code) => {
          const duration = Date.now() - startTime
          console.log(`🏁 [${requestId}] Python process closed with code: ${code} (duration: ${duration}ms)`)
          
          if (code !== 0 && !hasStarted) {
            console.log(`💥 [${requestId}] Process failed to start, sending error event`)
            const errorEvent = {
              type: 'error',
              data: {
                message: 'Research system failed to start',
                timestamp: new Date().toISOString()
              }
            }
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(errorEvent)}\n\n`))
          }
          console.log(`🔚 [${requestId}] Closing SSE stream`)
          controller.close()
        })

        pythonProcess.on('error', (error) => {
          console.error(`💥 [${requestId}] Python process error:`, error.message)
          const errorEvent = {
            type: 'error',
            data: {
              message: `Failed to start research system: ${error.message}`,
              timestamp: new Date().toISOString()
            }
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(errorEvent)}\n\n`))
          controller.close()
        })
      }
    })

    console.log(`🌊 [${requestId}] Returning SSE stream response`)
    return new Response(readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    })
  } catch (error) {
    const duration = Date.now() - startTime
    console.error(`💥 [${requestId}] Stream API error (duration: ${duration}ms):`, error)
    
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    )
  }
}

