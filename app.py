#!/usr/bin/env python3
"""
Simple command-line interface to test MCP server tools directly
"""

import sys
import os

# Add current directory to path so we can import functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🚗 Car RAG Bot Test Interface")
print("Loading server components...")

try:
    from mcpServer import get_car_info, send_to_discord, send_to_slack
    print("✅ Server loaded successfully!")
except Exception as e:
    print(f"❌ Error loading server: {e}")
    sys.exit(1)

def main():
    print("="*40)
    
    while True:
        print("\nAvailable commands:")
        print("1. query - Ask about car data")
        print("2. discord - Send message to Discord")
        print("3. slack - Send message to Slack")
        print("4. quit - Exit")
        
        cmd = input("\nEnter command: ").strip().lower()
        
        if cmd in ['quit', 'q', 'exit']:
            print("Goodbye!")
            break
            
        elif cmd in ['1', 'query', 'car']:
            query = input("Enter your car question: ")
            if query.strip():
                print("\n🔍 Searching...")
                try:
                    result = get_car_info(query)
                    print(f"\n📋 Answer:\n{result}")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
        elif cmd in ['2', 'discord']:
            message = input("Enter message (or press Enter to use last answer): ")
            try:
                result = send_to_discord(message if message.strip() else None)
                print(f"\n{result}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif cmd in ['3', 'slack']:
            message = input("Enter message (or press Enter to use last answer): ")
            try:
                result = send_to_slack(message if message.strip() else None)
                print(f"\n{result}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        else:
            print("❌ Unknown command")

if __name__ == "__main__":
    main()