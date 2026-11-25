"""Check repository-local Markdown article/image targets and result snapshots."""
from pathlib import Path
import json
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
errors=[]
for path in ROOT.rglob('*.md'):
    if '.git' in path.parts:continue
    for raw in re.findall(r'!?\[[^\]]*\]\(([^\n]*?)\)',path.read_text(encoding='utf-8')):
        raw=raw.strip('<>')
        if not raw or raw.startswith('#'):continue
        parsed=urlsplit(raw)
        if parsed.scheme or parsed.netloc:continue
        target=(path.parent/unquote(parsed.path)).resolve()
        if not target.exists():errors.append(f'{path.relative_to(ROOT)}: missing {raw}')
for slug in ['forecasting','knowledge-search','commercial-reporting','event-prediction',
             'optimization','ml-from-scratch','clustering-noncompliance','dynamic-zoom']:
    path=ROOT/'projects'/slug
    current=json.loads(subprocess.check_output([sys.executable,str(path/'demo.py')],text=True))
    saved=json.loads((path/'results.json').read_text(encoding='utf-8'))
    if current != saved:errors.append(f'{slug}: result snapshot differs from executable output')
if errors:sys.exit('\n'.join(errors))
print('Repository-local article/image links resolve; all eight portable result snapshots match execution.')
