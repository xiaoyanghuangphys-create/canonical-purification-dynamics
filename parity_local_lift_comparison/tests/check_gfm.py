"""Check protected TeX payloads and table geometry through an actual local GFM parser."""
import argparse
import ctypes as c
from ctypes.util import find_library
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess


class Geometry(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables=[]
        self.row=None
        self.in_table=False

    def handle_starttag(self,tag,attrs):
        if tag=='table':
            self.in_table=True; self.tables.append([])
        if self.in_table and tag=='tr': self.row=0
        if self.row is not None and tag in ('th','td'): self.row+=1

    def handle_endtag(self,tag):
        if tag=='table': self.in_table=False
        if tag=='tr' and self.row is not None:
            self.tables[-1].append(self.row); self.row=None


def source_payloads(source):
    expressions=[];table_rows=[];current=None;code=None;block=None;table_math=0
    for line in source.splitlines():
        stripped=line.strip()
        if code is not None:
            if stripped==code:
                if block is not None: expressions.append(('display','\n'.join(block)))
                code=None;block=None
            elif block is not None: block.append(line)
            continue
        f=re.fullmatch(r'(`{3,}|~{3,})(.*)',stripped)
        if f:
            code=f[1];current=None
            if f[2].strip()=='math': block=[]
            continue
        inside=stripped.startswith('|') and stripped.endswith('|')
        if inside and not re.fullmatch(r'[|\-: ]+',stripped):
            if current is None:
                current=[];table_rows.append(current)
            current.append(len(re.split(r'(?<!\\)\|',stripped))-2)
        elif not inside: current=None
        for match in re.finditer(r'\$`([^`\n]+?)`\$',line):
            expressions.append(('inline',match[1]));table_math+=int(inside)
        remainder=re.sub(r'\$`[^`\n]+?`\$|`[^`\n]*`','',line)
        if re.search(r'(?<!\\)\$',remainder): raise ValueError('Use protected inline or fenced math here: '+line[:120])
    if code: raise ValueError('Unclosed fence')
    return expressions,table_rows,table_math


def renderer():
    # Prefer an installed executable so the same checks work on Linux and macOS.
    exe=shutil.which('cmark-gfm')
    if exe:
        ver=subprocess.check_output([exe,'--version'],text=True).strip()
        return lambda data:subprocess.check_output([exe,'--extension','table'],input=data),ver
    lib=c.CDLL(find_library('cmark-gfm') or 'libcmark-gfm.so')
    def fn(name,result,*args):
        f=getattr(lib,name);f.restype=result;f.argtypes=list(args)
        return f
    ptr=c.c_void_p;char=c.c_char_p
    ver=fn('cmark_version_string',char)().decode()
    fn('cmark_gfm_core_extensions_ensure_registered',None)()
    new=fn('cmark_parser_new',ptr,c.c_int)
    extension=fn('cmark_find_syntax_extension',ptr,char)(b'table')
    if not extension: raise RuntimeError('Missing GFM table extension')
    attach=fn('cmark_parser_attach_syntax_extension',c.c_int,ptr,ptr)
    feed=fn('cmark_parser_feed',None,ptr,char,c.c_size_t)
    finish=fn('cmark_parser_finish',ptr,ptr)
    extensions=fn('cmark_parser_get_syntax_extensions',ptr,ptr)
    render=fn('cmark_render_html',ptr,ptr,c.c_int,ptr)
    free_parser=fn('cmark_parser_free',None,ptr);free_tree=fn('cmark_node_free',None,ptr)
    free=c.CDLL(None).free;free.argtypes=[ptr];free.restype=None
    def parse(data):
        parser=new(0)
        if not attach(parser,extension): raise RuntimeError('Cannot attach GFM table extension')
        feed(parser,data,len(data));tree=finish(parser)
        html=render(tree,0,extensions(parser))
        try: return c.string_at(html)
        finally: free(html);free_tree(tree);free_parser(parser)
    return parse,ver


INLINE_HTML=r'\$<code>(?P<inline>.*?)</code>\$'
DISPLAY_HTML=r'<pre><code class="language-math">(?P<display>.*?)\n</code></pre>'
def inspect_document(doc,render):
    source=doc.read_bytes()
    expected,tables,table_math=source_payloads(source.decode())
    html=render(source).decode()
    payloads=[]
    for token in re.finditer(INLINE_HTML+'|'+DISPLAY_HTML,html,re.S):
        kind=token.lastgroup
        payloads.append((kind,unescape(token[kind])))
    geometry=Geometry();geometry.feed(html)
    observed_table_math=sum(len(re.findall(INLINE_HTML,table,re.S)) for table in re.findall(r'<table>.*?</table>',html,re.S))
    if payloads!=expected: raise ValueError(f'{doc}: GFM changed a TeX payload or its order')
    if geometry.tables!=tables: raise ValueError(f'{doc}: expected {tables}, got {geometry.tables}')
    if observed_table_math!=table_math: raise ValueError(f'{doc}: GFM lost math in a table cell')
    return dict(document=str(doc),sha256=sha256(source).hexdigest(),rendered_html_sha256=sha256(html.encode()).hexdigest(),
                inline_payloads=sum(k=='inline' for k,_ in payloads),display_payloads=sum(k=='display' for k,_ in payloads),
                table_inline_payloads=observed_table_math,tables=[dict(rows=len(rows),columns=rows[0]) for rows in tables],
                exact_tex_payloads_preserved=True,table_geometry_preserved=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path);cli.add_argument('docs',type=Path,nargs='+')
    args=cli.parse_args()
    render,version=renderer()
    files=[inspect_document(doc,render) for doc in args.docs]
    result=dict(renderer='cmark-gfm',version=version,extension='table',
                scope='Offline GFM HTML output. Compared source and rendered TeX in document order, including table cells. Client-side GitHub MathJax and browser layout are outside this check.',
                documents=files)
    with args.output.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(files,indent=2))


if __name__=='__main__': main()
