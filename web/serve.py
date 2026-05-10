from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class WasmHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
    }


if __name__ == "__main__":
    web_root = Path(__file__).resolve().parent
    import os

    os.chdir(web_root)
    server = ThreadingHTTPServer(("localhost", 8080), WasmHandler)
    print("Serving crystaldive at http://localhost:8080")
    server.serve_forever()
