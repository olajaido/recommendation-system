import http.server
import socketserver
import os
from pathlib import Path

# Set the port for the web server
PORT = 8080

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Change to the directory containing the HTML file
os.chdir(SCRIPT_DIR)

# Create a simple HTTP server
Handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("", PORT), Handler)

print(f"Serving frontend at http://localhost:{PORT}")
print("Press Ctrl+C to stop the server")

# Start the server
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped")
    httpd.server_close()