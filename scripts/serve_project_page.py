"""Local project-page preview with byte ranges for video seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range = None
        request_range = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not request_range or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", request_range)
        if not match or not any(match.groups()):
            self.send_error(416, "Unsupported byte range")
            return None
        start, end = match.groups()
        if start:
            first = int(start)
            last = min(int(end) if end else size - 1, size - 1)
        else:
            first, last = max(0, size - int(end)), size - 1
        if first > last or first >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        source = path.open("rb")
        source.seek(first)
        self.byte_range = first, last
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.send_header("Content-Length", str(last - first + 1))
        self.end_headers()
        return source

    def copyfile(self, source, outputfile):
        if self.byte_range is None:
            return shutil.copyfileobj(source, outputfile)
        remaining = self.byte_range[1] - self.byte_range[0] + 1
        while remaining:
            block = source.read(min(1024 * 1024, remaining))
            if not block:
                break
            outputfile.write(block)
            remaining -= len(block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "project-page/dist"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(RangeHandler, directory=str(directory)))
    print(f"Project page: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
