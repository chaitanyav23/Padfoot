#!/usr/bin/env python3
import os
import glob
import re

HTML_DIR = os.path.dirname(os.path.abspath(__file__))

WIDGET_CSS  = '<link rel="stylesheet" href="widget.css">\\n'
WIDGET_JS   = '<script src="widget.js"></script>\\n'

def inject_widget(html_file):
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip if widget already injected
    if 'widget.js' in content:
        print(f"  Skipping {html_file}: widget already present.")
        return

    # 1. Insert widget CSS before </head>
    head_match = re.search(r'</head>', content, re.IGNORECASE)
    if head_match:
        idx = head_match.start()
        content = content[:idx] + WIDGET_CSS + content[idx:]

    # 2. Insert widget JS before </body>
    body_close_match = re.search(r'</body>', content, re.IGNORECASE)
    if body_close_match:
        idx = body_close_match.start()
        content = content[:idx] + WIDGET_JS + content[idx:]

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    html_files = glob.glob(os.path.join(HTML_DIR, "*.html"))
    for f in html_files:
        print(f"Injecting widget into {f}...")
        inject_widget(f)
    print("Done.")
