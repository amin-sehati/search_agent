import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { password } = await request.json()
    
    // Get the password from environment variables
    const APP_PASSWORD = process.env.APP_PASSWORD
    
    // Check if environment variable is set
    if (!APP_PASSWORD) {
      console.error('APP_PASSWORD environment variable is not set')
      return NextResponse.json(
        { success: false, message: 'Server configuration error' },
        { status: 500 }
      )
    }
    
    // Verify password
    if (password === APP_PASSWORD) {
      return NextResponse.json({ success: true })
    } else {
      return NextResponse.json(
        { success: false, message: 'Invalid password' },
        { status: 401 }
      )
    }
  } catch (error) {
    console.error('Authentication error:', error)
    return NextResponse.json(
      { success: false, message: 'Authentication failed' },
      { status: 500 }
    )
  }
}