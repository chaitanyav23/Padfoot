#!/usr/bin/env python3
import os
import glob
import re

HTML_DIR = os.path.dirname(os.path.abspath(__file__))

CSS_LINK    = '<link rel="stylesheet" href="zoom_theme.css">\\n'
WIDGET_CSS  = '<link rel="stylesheet" href="widget.css">\\n'
JS_SCRIPT   = '<script src="zoom_theme.js"></script>\\n'
WIDGET_JS   = '<script src="widget.js"></script>\\n'

THEME_SELECTOR_HTML = """
<div id="theme-selector">
    <div class="swatch" style="background-color: #fff9e6;" onclick="setTheme('theme-default')" title="Default"></div>
    <div class="swatch" style="background-color: #616161;" onclick="setTheme('theme-dark')" title="Dark"></div>
    <div class="swatch" style="background-color: #e6f0ff;" onclick="setTheme('theme-blue')" title="Blue"></div>
</div>
"""

def cleanup_and_reinject(html_file):
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Remove all injected CSS links
    content = re.sub(r'<link rel="stylesheet" href="zoom_theme\.css">\\s*', '', content)
    content = re.sub(r'<link rel="stylesheet" href="widget\.css">\\s*', '', content)

    # 2. Remove all theme-selector blocks (including swatches)
    # We match the outer div and everything until its corresponding closing div.
    # Since we know the structure, we can match specifically.
    content = re.sub(r'<div id="theme-selector">.*?</div>\s*</div>\s*</div>\s*</div>', '', content, flags=re.DOTALL)
    # Also remove any leftover swatches or fragments
    content = re.sub(r'<div class="swatch".*?</div>\s*', '', content)
    content = re.sub(r'<div id="theme-selector">\s*', '', content)

    # 3. Remove all page-content wrappers
    content = re.sub(r'<div id="page-content">\s*', '', content)
    # Remove closing </div> tags that are likely ours
    content = re.sub(r'</div>\s*(?=<script src="zoom_theme\.js"|<script src="widget\.js"|</body>)', '', content)
    # Also handle the cases where they were duplicated
    content = re.sub(r'</div>\s*</div>\s*(?=<script src="zoom_theme\.js"|<script src="widget\.js"|</body>)', '', content)

    # 4. Remove all injected JS scripts
    content = re.sub(r'<script src="zoom_theme\.js"></script>\s*', '', content)
    content = re.sub(r'<script src="zoom_theme_persistent\.js"></script>\s*', '', content)
    content = re.sub(r'<script src="widget\.js"></script>\s*', '', content)

    # 5. Clean up any weird literal \n strings if they were injected
    content = content.replace('\\\\n', '\\n')

    # NOW RE-INJECT ONCE
    
    # Insert CSS before </head>
    head_match = re.search(r'</head>', content, re.IGNORECASE)
    if head_match:
        idx = head_match.start()
        content = content[:idx] + CSS_LINK + WIDGET_CSS + content[idx:]

    # Wrap body content and insert theme selector
    body_open_match  = re.search(r'<body[^>]*>', content, re.IGNORECASE)
    body_close_match = re.search(r'</body>', content, re.IGNORECASE)
    if body_open_match and body_close_match:
        body_start_tag_end = body_open_match.end()
        body_end_start     = body_close_match.start()
        body_inner         = content[body_start_tag_end:body_end_start].strip()
        
        # Remove any leading/trailing </div> that might be leftovers
        while body_inner.startswith('</div>'):
            body_inner = body_inner[6:].strip()
        while body_inner.endswith('</div>'):
            # Only remove if it's likely one of ours (empty or preceded by newline)
            body_inner = body_inner[:-6].strip()

        # Wrap existing content
        body_inner_wrapped = f'\\n<div id="page-content">\\n{body_inner}\\n</div>\\n'

        # Insert theme selector at the top of body content
        content = content[:body_start_tag_end] + THEME_SELECTOR_HTML + body_inner_wrapped + content[body_end_start:]

    # Insert JS before </body>
    body_close_match = re.search(r'</body>', content, re.IGNORECASE)
    if body_close_match:
        idx = body_close_match.start()
        content = content[:idx] + JS_SCRIPT + WIDGET_JS + content[idx:]

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    html_files = glob.glob(os.path.join(HTML_DIR, "*.html"))
    for f in html_files:
        print(f"Cleaning up {f}...")
        cleanup_and_reinject(f)
    print("Done.")
