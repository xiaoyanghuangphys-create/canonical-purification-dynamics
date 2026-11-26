// Validate local TeX with an installed KaTeX or the VS Code math extension.
// Use a fresh output path. Run from the repository root.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';

const [, , output, ...docs] = process.argv;
if (!output || !docs.length) throw new Error('Supply a new JSON path and one or more Markdown files');
const hash = text => crypto.createHash('sha256').update(text).digest('hex');
let render, metadata;
try {
  const {default:katex} = await import('katex');
  render = (tex,display) => katex.renderToString(tex,{displayMode:display,throwOnError:true});
  metadata = {library:'KaTeX',version:katex.version};
} catch (err) {
  if (err.code !== 'ERR_MODULE_NOT_FOUND') throw err;
  const bundle = process.env.VSCODE_KATEX_BUNDLE || '/Applications/Visual Studio Code.app/Contents/Resources/app/extensions/markdown-math/notebook-out/katex.js';
  const input = fs.readFileSync(bundle, 'utf8');
  // Register the extension's public notebook renderer; all actual math
  // parsing and MathML output are produced by its unchanged KaTeX bundle.
  const element=()=>({classList:{add(){}},content:{appendChild(){}}});
  globalThis.document={compatMode:'CSS1Compat',createElement:element,head:{appendChild(){}}};
  const {activate}=await import(pathToFileURL(bundle));
  const ruler={after(){},before(){},push(){}};
  const md={inline:{ruler},block:{ruler},core:{ruler},renderer:{rules:{}},use(plugin,options){plugin(this,options)}};
  await activate({getRenderer:async()=>({extendMarkdownIt:callback=>callback(md)})});
  render=(tex,display)=>md.renderer.rules[display?'math_block':'math_inline']([{content:tex}],0);
  metadata={library:'VS Code markdown-math / KaTeX',version:input.match(/version:"([0-9.]+)",render:/)?.[1],bundle_sha256:hash(input)};
}
let unknownRejected=false;
try {
  unknownRejected=render('\\UnknownUnconfiguredCommand',false).includes('katex-error');
} catch {unknownRejected=true;}
if (!unknownRejected || !render('x^2',true).includes('<math')) throw new Error('Renderer self-check failed');

const reports = [];
for (const doc of docs) {
  const source=fs.readFileSync(doc,'utf8');
  const report={document:doc,sha256:hash(source),inline_count:0,display_count:0,local_links:[],external_links:[],errors:[]};
  let fence=null,block=null,start=null,mathFence=false;
  const check=(tex,display,line)=>{
    try {
      const html=render(tex,display);
      if (html.includes('katex-error') || !html.includes('<math')) throw new Error('Renderer rejected expression: '+html.slice(0,180));
      ++report[display?'display_count':'inline_count'];
    } catch(err) { report.errors.push({line,tex,error:String(err)}); }
  };
  for(const [idx,line] of source.split(/\r?\n/).entries()) {
    const lineno=idx+1;
    const f=line.match(/^\s*(`{3,}|~{3,})(.*)$/);
    if (f) {
      if (fence && f[1][0]===fence[0] && f[1].length>=fence.length && !f[2].trim()) {
        if (mathFence) {check(block.join('\n'),true,start);block=null;mathFence=false;}
        fence=null;
      } else if (!fence) {
        fence=f[1];
        if (f[2].trim()==='math') {block=[];start=lineno;mathFence=true;}
      }
      continue;
    }
    if (fence) {if (mathFence) block.push(line);continue;}
    if (line.trim()==='$$') {
      if (block!==null) {check(block.join('\n'),true,start);block=null;}
      else {block=[];start=lineno;}
      continue;
    }
    if (block!==null) {block.push(line);continue;}
    // All documents in this cohort use simple file destinations, and math
    // outside code spans. Check for unmatched inline delimiters per line.
    const unprotected=line.replace(/\$`([^`\n]+?)`\$/g,(_full,tex)=>{
      check(tex,false,lineno);return '';
    });
    const clean=unprotected.replace(/(`+).*?\1/g,'');
    const pieces=clean.split(/(?<!\\)\$/);
    if (pieces.length%2!==1) report.errors.push({line:lineno,error:'Unmatched inline dollar'});
    for(let j=1;j+1<pieces.length;j+=2) {
      if (!pieces[j]) {report.errors.push({line:lineno,error:'Empty inline expression or misplaced display delimiter'});continue;}
      check(pieces[j],false,lineno);
    }
    for (const link of clean.matchAll(/!?\[[^\]]*\]\(([^)]+)\)/g)) {
      const dest=link[1].replace(/^<|>$/g,'');
      if (/^(https?:|mailto:)/.test(dest)) {report.external_links.push(dest);continue;}
      const [local,fragment]=dest.split('#');
      const target=path.resolve(path.dirname(doc), decodeURIComponent(local||path.basename(doc)));
      const exists=fs.existsSync(target);
      report.local_links.push({line:lineno,target:dest,exists});
      if (!exists) report.errors.push({line:lineno,error:'Missing local target '+dest});
      if (exists && fragment) {
        const slug=title=>title.toLowerCase().replace(/[^\p{L}\p{N}_ -]/gu,'').replace(/ /g,'-');
        const headings=Array.from(fs.readFileSync(target,'utf8').matchAll(/^#{1,6}\s+(.+)$/gm),m=>slug(m[1]));
        if (!headings.includes(decodeURIComponent(fragment))) report.errors.push({line:lineno,error:'Missing local heading '+dest});
      }
    }
  }
  if(fence||block!==null) report.errors.push({error:'Unclosed code or display-math fence'});
  reports.push(report);
}
const record={
  tool:metadata,
  scope:'Local KaTeX rendered the extracted protected/dollar-delimited inline and fenced/dollar-delimited display expressions to MathML. Checked relative links and heading targets on disk. Ordinary program code was excluded. This checks the mathematical payload, with live GitHub inspection recorded separately if available.',
  files:reports
};
fs.writeFileSync(output,JSON.stringify(record,null,2)+'\n',{flag:'wx'});
for(const r of reports) console.log(r.document,r.inline_count,r.display_count,r.local_links.length,r.errors);
if(reports.some(r=>r.errors.length)) process.exitCode=1;
