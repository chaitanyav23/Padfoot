#!/usr/bin/env python3
import os
import glob
import re

# Directory containing HTML files
HTML_DIR = os.path.dirname(os.path.abspath(__file__))  # Always resolves to the HTML/ folder
BACKUP = True    # Make backups

# Snippets to insert
CSS_LINK    = '<link rel="stylesheet" href="zoom_theme.css">\n'
JS_SCRIPT   = '<script src="zoom_theme_persistent.js"></script>\n'
WIDGET_CSS  = '<link rel="stylesheet" href="widget.css">\n'
WIDGET_JS   = '<script src="widget.js"></script>\n'

# Theme selector HTML (swatches)
THEME_SELECTOR_HTML = '''
<div id="theme-selector">
    <div class="swatch" style="background-color: #fff9e6;" onclick="setTheme('theme-default')" title="Default"></div>
    <div class="swatch" style="background-color: #616161;" onclick="setTheme('theme-dark')" title="Dark"></div>
    <div class="swatch" style="background-color: #e6f0ff;" onclick="setTheme('theme-blue')" title="Blue"></div>
</div>
'''

# Process all HTML files in the directory
html_files = glob.glob(os.path.join(HTML_DIR, "*.html"))

for html_file in html_files:
    print(f"Processing {html_file}...")
    
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip files that already have the widget injected
    if 'widget.js' in content:
        print(f"  Skipping {html_file}: already injected.")
        continue

    # Backup original
    if BACKUP:
        backup_file = html_file + ".bak"
        with open(backup_file, "w", encoding="utf-8") as f:
            f.write(content)

    # Insert zoom CSS and widget CSS before </head> (case-insensitive)
    head_match = re.search(r'</head>', content, re.IGNORECASE)
    if head_match:
        idx = head_match.start()
        content = content[:idx] + CSS_LINK + WIDGET_CSS + content[idx:]
    else:
        print(f"  Warning: </head> not found in {html_file}")

    # Wrap body content and insert theme selector
    body_open_match  = re.search(r'<body[^>]*>', content, re.IGNORECASE)
    body_close_match = re.search(r'</body>', content, re.IGNORECASE)
    if body_open_match and body_close_match:
        body_start_tag_end = body_open_match.end()
        body_end_start     = body_close_match.start()
        body_inner         = content[body_start_tag_end:body_end_start]

        # Wrap existing content
        body_inner_wrapped = f'<div id="page-content">\n{body_inner}\n</div>'

        # Insert theme selector at the top of body content
        content = content[:body_start_tag_end] + THEME_SELECTOR_HTML + body_inner_wrapped + content[body_end_start:]
    else:
        print(f"  Warning: <body> or </body> not found in {html_file}")

    # Insert zoom JS and widget JS before </body> (case-insensitive)
    body_close_match = re.search(r'</body>', content, re.IGNORECASE)
    if body_close_match:
        idx = body_close_match.start()
        content = content[:idx] + JS_SCRIPT + WIDGET_JS + content[idx:]
    else:
        print(f"  Warning: </body> not found for JS insertion in {html_file}")

    # Write changes back to file
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(content)

print("All files processed successfully.")

