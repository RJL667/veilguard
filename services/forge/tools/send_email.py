
import subprocess
import os
import sys

def run(to: str, subject: str, body: str) -> str:
    """
    Last resort: open Windows default email client via webbrowser.
    """
    
    try:
        import webbrowser
        
        # Build mailto URI safely
        mailto_uri = f"mailto:{to}?subject={subject.replace(' ', '%20')}&body={body.replace(' ', '%20')}"
        
        # Open in default email client
        webbrowser.open(mailto_uri)
        
        return f"✓ REF_PERSON_21 composer opened in your default email client for {to}"
    
    except Exception as e:
        return f"✗ Error: {str(e)}"
