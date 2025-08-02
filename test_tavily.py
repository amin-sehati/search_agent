#!/usr/bin/env python3
"""
Test script to verify Tavily API integration and payload format
"""
import os
import asyncio
import aiohttp
import json
from config import Config

async def test_tavily_api():
    """Test Tavily API with minimal payload"""
    
    # Validate config first
    config_status = Config.validate_config()
    print(f"Config validation: {config_status}")
    
    if not Config.TAVILY_API_KEY:
        print("❌ Tavily API key not found")
        return
    
    # Test with minimal payload that should work
    base_url = "https://api.tavily.com"
    headers = {"Content-Type": "application/json"}
    
    # Test with different payload configurations
    test_payloads = [
        # Minimal payload
        {
            "api_key": Config.TAVILY_API_KEY,
            "query": "ride sharing companies",
            "max_results": 3
        },
        # Add search_depth
        {
            "api_key": Config.TAVILY_API_KEY,
            "query": "ride sharing companies", 
            "max_results": 3,
            "search_depth": "basic"
        },
        # Add include flags
        {
            "api_key": Config.TAVILY_API_KEY,
            "query": "ride sharing companies",
            "max_results": 3,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False
        },
        # Add domain arrays
        {
            "api_key": Config.TAVILY_API_KEY,
            "query": "ride sharing companies",
            "max_results": 3,
            "search_depth": "basic", 
            "include_answer": False,
            "include_raw_content": False,
            "include_domains": [],
            "exclude_domains": []
        }
    ]
    
    for i, payload in enumerate(test_payloads, 1):
        print(f"\n=== Test {i}: Payload with {list(payload.keys())} ===")
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(f"{base_url}/search", headers=headers, json=payload) as response:
                    print(f"Response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Success! Found {len(data.get('results', []))} results")
                    else:
                        error_text = await response.text()
                        print(f"❌ Error {response.status}: {error_text}")
                        
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_tavily_api())