"""Inspect the GitHub blob page HTML, from public GET only, at an immutable commit."""
import argparse
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from check_gfm import Geometry, source_payloads


class GitHubMath(Geometry):
    def __init__(self):
        super().__init__()
        self.equations=[]
        self.table_math=0
        self.current=None

    def handle_starttag(self,tag,attrs):
        super().handle_starttag(tag,attrs)
        if tag=='math-renderer':
            classes=dict(attrs).get('class','').split()
            if 'js-inline-math' in classes: kind='inline'
            elif 'js-display-math' in classes: kind='display'
            else: raise ValueError(f'Unknown math wrapper class {classes}')
            if self.in_table: self.table_math+=1
            self.current=[kind,[]]

    def handle_data(self,data):
        if self.current is not None: self.current[1].append(data)

    def handle_endtag(self,tag):
        super().handle_endtag(tag)
        if tag=='math-renderer':
            kind,parts=self.current;self.current=None
            # GitHub's JSON richText stores e.g. TeX '>' as '&amp;gt;' in
            # the HTML node. Decode its payload after HTMLParser's first pass.
            text=unescape(''.join(parts)).strip()
            delim='$' if kind=='inline' else '$$'
            if not (text.startswith(delim) and text.endswith(delim)): raise ValueError(f'Missing math delimiter: {text}')
            self.equations.append((kind,text[len(delim):-len(delim)].strip()))


def extract_page(html):
    match=re.search(r'<script\b[^>]*data-target="react-app.embeddedData"[^>]*>(.*?)</script>',html,re.S)
    if not match: raise ValueError("Public blob page contained no GitHub document payload")
    data=json.loads(match[1])['payload']
    return data['codeViewBlobRoute']['richText']


def check(commit,doc,raw=None):
    url='https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/blob/'+commit+'/'+str(doc)
    if raw is None:
        # No request body, credentials, or cookies. Only an already-published document URL is sent.
        with urlopen(Request(url,method='GET',headers={'User-Agent':'Parity-CP-render-audit/1.0'}),timeout=30) as response:
            raw=response.read()
    parser=GitHubMath();parser.feed(extract_page(raw.decode()))
    expected,tables,table_math=source_payloads(doc.read_text())
    if parser.equations!=expected:
        if len(parser.equations)!=len(expected):
            raise ValueError(f'{doc}: {len(parser.equations)} GitHub math wrappers, expected {len(expected)}')
        i=next(i for i,v in enumerate(expected) if v!=parser.equations[i])
        raise ValueError(f'{doc}: expression {i+1} differs on GitHub: {parser.equations[i]} versus {expected[i]}')
    if parser.tables!=tables or parser.table_math!=table_math: raise ValueError(f'{doc}: GitHub table mismatch')
    return dict(document=str(doc),source_sha256=sha256(doc.read_bytes()).hexdigest(),
                url=url,github_inline_wrappers=sum(k=='inline' for k,_ in expected),
                github_display_wrappers=sum(k=='display' for k,_ in expected),
                github_table_math_nodes=table_math,github_table_shapes=[[len(row),row[0]] for row in tables],
                exact_tex_payloads_and_table_positions_verified=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('commit',help='Full public Git commit SHA')
    cli.add_argument('output',type=Path);cli.add_argument('docs',type=Path,nargs='+')
    cli.add_argument('--cached-html',type=Path,help='Optional directory with public GET responses named <basename>.html')
    args=cli.parse_args()
    if not re.fullmatch(r'[a-f0-9]{40}',args.commit): raise ValueError('Expected a full immutable commit SHA')
    results=[check(args.commit,doc,(args.cached_html/(doc.name+'.html')).read_bytes() if args.cached_html else None)
             for doc in args.docs]
    result=dict(commit=args.commit,checks='Actual GitHub server Markdown HTML. MathJax payloads and table geometry checked against source. Client-side rendered layout is separate.',documents=results)
    with args.output.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print('Public GFM validated:',[(r['document'],r['github_inline_wrappers'],r['github_display_wrappers']) for r in results])


if __name__=='__main__': main()
