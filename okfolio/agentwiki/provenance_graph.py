"""Self-contained offline knowledge graph for the global compiler."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any


def build_provenance_graph(
    articles: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    judgements: list[dict[str, Any]],
) -> str:
    # The standalone graph only needs the first evidence excerpt for preview.
    # The complete evidence arrays remain available in refs.json and Concept
    # Markdown, so this keeps the UI responsive without weakening provenance.
    graph_refs = [
        {
            **item,
            "evidence": list(item.get("evidence", []))[:1],
        }
        for item in refs
    ]
    ref_to_concept = {
        ref_id: concept["id"]
        for concept in concepts
        for ref_id in concept["ref_ids"]
    }
    grouped: dict[tuple[str, str], dict[str, list[Any]]] = {}
    for item in judgements:
        if item.get("decision") != "related":
            continue
        left = ref_to_concept.get(item["left_ref_id"])
        right = ref_to_concept.get(item["right_ref_id"])
        if left is None or right is None or left == right:
            continue
        edge = grouped.setdefault(
            tuple(sorted((left, right))),
            {"source_refs": [], "reasons": []},
        )
        edge["source_refs"].append(
            [item["left_ref_id"], item["right_ref_id"]]
        )
        edge["reasons"].append(item["reason"])

    data = {
        "articles": articles,
        "refs": graph_refs,
        "concepts": concepts,
        "semantic_edges": [
            {"source": source, "target": target, **edge}
            for (source, target), edge in sorted(grouped.items())
        ],
        "stats": {
            "articles": len(articles),
            "refs": len(refs),
            "concepts": len(concepts),
            "multi_source": sum(
                len(concept["articles"]) > 1 for concept in concepts
            ),
            "relations": len(grouped),
            "types": dict(
                sorted(Counter(concept["type"] for concept in concepts).items())
            ),
        },
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>决策参考知识图谱</title>
<style>
:root{color-scheme:light;--bg:#f3f6fb;--panel:rgba(255,255,255,.92);--ink:#172033;--muted:#64748b;--line:#dce4f0;--brand:#3157d5;--shadow:0 18px 55px rgba(35,52,93,.12);--analysis:#4077e8;--policy:#8b5cf6;--data:#0ea5a5;--global:#f59e0b;--term:#ef5da8}
@media(prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#0b1120;--panel:rgba(15,23,42,.93);--ink:#e8eefc;--muted:#94a3b8;--line:#27344c;--brand:#7aa2ff;--shadow:0 18px 55px rgba(0,0,0,.35)}}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 18% 4%,rgba(64,119,232,.12),transparent 34%),radial-gradient(circle at 82% 96%,rgba(139,92,246,.11),transparent 30%),var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;overflow:hidden}
button,input{font:inherit}.app{height:100vh;display:grid;grid-template-rows:auto 1fr}.topbar{display:grid;grid-template-columns:1fr auto;align-items:center;gap:20px;padding:18px 24px;border-bottom:1px solid var(--line);background:var(--panel);backdrop-filter:blur(18px);z-index:4}.brand{display:flex;align-items:center;gap:13px}.mark{width:38px;height:38px;border-radius:13px;background:linear-gradient(135deg,#3157d5,#8b5cf6);box-shadow:0 10px 24px rgba(49,87,213,.28);position:relative}.mark:before,.mark:after{content:"";position:absolute;border-radius:50%;background:#fff}.mark:before{width:8px;height:8px;left:9px;top:15px;box-shadow:12px -7px 0 #fff,12px 7px 0 #fff}.mark:after{height:2px;width:16px;left:11px;top:18px;transform:rotate(-31deg);border-radius:2px}.brand h1{font-size:18px;margin:0;font-weight:650;letter-spacing:.01em}.brand p{margin:2px 0 0;color:var(--muted);font-size:12px}.metrics{display:flex;gap:22px}.metric strong{display:block;font-size:18px;line-height:1.15}.metric span{color:var(--muted);font-size:11px}.shell{min-height:0;display:grid;grid-template-columns:258px minmax(420px,1fr) 360px}.left,.detail{background:var(--panel);backdrop-filter:blur(18px);overflow:auto}.left{border-right:1px solid var(--line);padding:18px}.detail{border-left:1px solid var(--line);padding:22px}.section{margin-bottom:22px}.section-title{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:9px}.search{width:100%;border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:transparent;color:var(--ink);outline:none}.search:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(49,87,213,.12)}.type-list{display:grid;gap:7px}.type-row{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:9px;padding:7px 8px;border-radius:9px}.type-row:hover{background:rgba(100,116,139,.08)}.swatch{width:10px;height:10px;border-radius:50%}.type-row small{color:var(--muted)}.switch{display:flex;align-items:center;gap:9px;margin:9px 0;color:var(--muted)}.switch input{accent-color:var(--brand)}.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px}.btn{border:1px solid var(--line);border-radius:9px;background:transparent;color:var(--ink);padding:8px;cursor:pointer}.btn:hover{border-color:var(--brand);color:var(--brand)}.legend{font-size:12px;color:var(--muted);display:grid;gap:7px}.legend-line{display:flex;align-items:center;gap:8px}.line{width:22px;height:2px;background:var(--muted)}.line.dashed{background:repeating-linear-gradient(90deg,var(--muted) 0 4px,transparent 4px 7px)}.stage{position:relative;min-width:0;overflow:hidden}.stage canvas{display:block;width:100%;height:100%;cursor:grab}.stage canvas.dragging{cursor:grabbing}.float{position:absolute;left:18px;bottom:18px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:9px 12px;color:var(--muted);font-size:12px;box-shadow:var(--shadow);pointer-events:none}.empty{min-height:100%;display:grid;place-content:center;text-align:center;color:var(--muted)}.empty-icon{width:54px;height:54px;border:1px solid var(--line);border-radius:18px;margin:0 auto 13px;display:grid;place-items:center;font-size:22px}.detail h2{font-size:20px;line-height:1.35;margin:0 0 8px}.meta{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:15px}.pill{padding:3px 8px;border-radius:999px;background:rgba(49,87,213,.1);color:var(--brand);font-size:11px}.description{color:var(--muted);margin-bottom:20px}.detail h3{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:22px 0 9px}.body{white-space:pre-wrap;font:12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;background:rgba(100,116,139,.07);border:1px solid var(--line);border-radius:12px;padding:13px;max-height:240px;overflow:auto}.ref{border-top:1px solid var(--line);padding:11px 0}.ref:first-of-type{border-top:0}.ref b{display:block;font-size:13px}.ref small{color:var(--muted)}.quote{border-left:2px solid var(--brand);padding-left:9px;margin-top:6px;color:var(--muted);font-size:12px}.article{display:flex;gap:8px;padding:7px 0;color:var(--ink)}.article-id{color:var(--brand);font:11px ui-monospace}.related{border:1px solid var(--line);border-radius:10px;padding:9px;margin:7px 0;cursor:pointer}.related:hover{border-color:var(--brand)}.related b{display:block}.related small{color:var(--muted)}
@media(max-width:1050px){.shell{grid-template-columns:220px 1fr}.detail{position:absolute;right:0;top:75px;bottom:0;width:340px;box-shadow:var(--shadow);transform:translateX(100%);transition:.2s;z-index:5}.detail.open{transform:none}.metrics .metric:nth-child(-n+2){display:none}}@media(max-width:720px){.topbar{padding:12px}.shell{grid-template-columns:1fr}.left{display:none}.metrics{gap:10px}.detail{top:64px;width:min(92vw,360px)}.brand p{display:none}}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand"><div class="mark"></div><div><h1>决策参考知识图谱</h1><p>Concept · ConceptRef · Article 全链路溯源</p></div></div>
    <div class="metrics">
      <div class="metric"><strong id="mArticles"></strong><span>原始文章</span></div>
      <div class="metric"><strong id="mRefs"></strong><span>ConceptRef</span></div>
      <div class="metric"><strong id="mConcepts"></strong><span>最终 Concept</span></div>
      <div class="metric"><strong id="mRelations"></strong><span>语义关系</span></div>
    </div>
  </header>
  <main class="shell">
    <aside class="left">
      <div class="section"><div class="section-title">搜索知识</div><input id="search" class="search" type="search" placeholder="标题、摘要或来源…"></div>
      <div class="section"><div class="section-title">概念类型</div><div id="types" class="type-list"></div></div>
      <div class="section"><div class="section-title">显示</div><label class="switch"><input id="multiOnly" type="checkbox">仅多来源 Concept</label><label class="switch"><input id="showRelations" type="checkbox" checked>显示语义关系</label></div>
      <div class="section"><div class="section-title">视图</div><div class="btns"><button id="fit" class="btn">适应画布</button><button id="reset" class="btn">清除选择</button></div></div>
      <div class="legend"><div class="legend-line"><span class="line"></span>Concept 语义关系</div><div class="legend-line"><span class="line dashed"></span>选择后展开溯源</div><div>大节点表示由多篇 Article 联合编译。</div></div>
    </aside>
    <section class="stage"><canvas id="graph"></canvas><div id="status" class="float"></div></section>
    <aside id="detail" class="detail"><div class="empty"><div><div class="empty-icon">⌁</div><div>选择一个 Concept</div><small>查看正文、Ref 证据和来源文章</small></div></div></aside>
  </main>
</div>
<script>
const DATA=""" + payload + """;
const COLORS={"分析框架":"#4077e8","政策建议":"#8b5cf6","数据口径":"#0ea5a5","国际比较":"#f59e0b","术语解释":"#ef5da8"};
const CENTERS={"分析框架":[620,740],"政策建议":[1780,740],"数据口径":[1200,210],"国际比较":[1200,1270],"术语解释":[1200,740]};
const WORLD={w:2400,h:1480};const byRef=Object.fromEntries(DATA.refs.map(x=>[x.ref_id,x]));const byArticle=Object.fromEntries(DATA.articles.map(x=>[x.article_id,x]));const byConcept=Object.fromEntries(DATA.concepts.map(x=>[x.id,x]));
const canvas=document.getElementById("graph"),ctx=canvas.getContext("2d"),detail=document.getElementById("detail"),status=document.getElementById("status");
let dpr=1,w=0,h=0,scale=.45,ox=0,oy=0,drag=false,moved=false,last=null,selected=null,query="",multiOnly=false,showRelations=true;const enabled=new Set(Object.keys(COLORS));
const nodes=[];for(const type of Object.keys(COLORS)){const list=DATA.concepts.filter(x=>x.type===type).sort((a,b)=>b.articles.length-a.articles.length||a.title.localeCompare(b.title));const[cx,cy]=CENTERS[type];list.forEach((c,i)=>{const angle=i*2.3999632297;const radius=24+Math.sqrt(i)*24;nodes.push({...c,x:cx+Math.cos(angle)*radius,y:cy+Math.sin(angle)*radius,r:c.articles.length>1?8:4.5})})}const nodeById=Object.fromEntries(nodes.map(x=>[x.id,x]));
function visible(n){if(!enabled.has(n.type))return false;if(multiOnly&&n.articles.length<2)return false;if(!query)return true;const q=query.toLowerCase();return(n.title+" "+n.description+" "+n.articles.join(" ")).toLowerCase().includes(q)}
function resize(){const r=canvas.getBoundingClientRect();dpr=Math.min(devicePixelRatio||1,2);w=r.width;h=r.height;canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);if(!ox&&!oy)fit();draw()}new ResizeObserver(resize).observe(canvas);
function fit(){scale=Math.min(w/WORLD.w,h/WORLD.h)*.88;ox=(w-WORLD.w*scale)/2;oy=(h-WORLD.h*scale)/2;draw()}function screen(n){return{x:ox+n.x*scale,y:oy+n.y*scale}}function worldPoint(x,y){return{x:(x-ox)/scale,y:(y-oy)/scale}}
function line(a,b,color,width=1,dash=[]){ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();ctx.setLineDash([])}
function draw(){ctx.clearRect(0,0,w,h);const styles=getComputedStyle(document.documentElement),ink=styles.getPropertyValue("--ink"),muted=styles.getPropertyValue("--muted"),lineColor=styles.getPropertyValue("--line");ctx.save();for(const[type,[cx,cy]]of Object.entries(CENTERS)){if(!enabled.has(type))continue;const p=screen({x:cx,y:cy});ctx.beginPath();ctx.arc(p.x,p.y,Math.max(42,250*scale),0,Math.PI*2);ctx.fillStyle=COLORS[type]+"0d";ctx.fill();ctx.fillStyle=muted;ctx.font="500 12px system-ui";ctx.textAlign="center";ctx.fillText(type,p.x,p.y-Math.max(48,260*scale))}if(showRelations)for(const e of DATA.semantic_edges){const a=nodeById[e.source],b=nodeById[e.target];if(!a||!b||!visible(a)||!visible(b))continue;const active=selected&&(selected.id===a.id||selected.id===b.id);line(screen(a),screen(b),active?COLORS[selected.type]:lineColor,active?2:1)}for(const n of nodes){if(!visible(n))continue;const p=screen(n),active=selected&&selected.id===n.id,dim=selected&&!active&&!DATA.semantic_edges.some(e=>(e.source===selected.id&&e.target===n.id)||(e.target===selected.id&&e.source===n.id));ctx.globalAlpha=dim?.22:1;ctx.beginPath();ctx.arc(p.x,p.y,Math.max(2.4,n.r*scale*1.55),0,Math.PI*2);ctx.fillStyle=COLORS[n.type];ctx.fill();if(n.articles.length>1){ctx.strokeStyle=ink;ctx.lineWidth=active?2.5:1;ctx.stroke()}if(active||scale>1.05||(query&&visible(n))){ctx.globalAlpha=1;ctx.font=active?"500 13px system-ui":"12px system-ui";ctx.textAlign="center";ctx.fillStyle=ink;ctx.fillText(n.title.slice(0,22),p.x,p.y-12)}ctx.globalAlpha=1}if(selected&&visible(selected))drawProvenance(selected,ink,muted);ctx.restore();status.textContent=`显示 ${nodes.filter(visible).length} / ${nodes.length} 个 Concept · 缩放 ${Math.round(scale*100)}%`}
function drawProvenance(c,ink,muted){const center=screen(c),refs=c.ref_ids.map(id=>byRef[id]).filter(Boolean),articleIds=[...new Set(c.articles)],refPos={};refs.forEach((r,i)=>{const a=2*Math.PI*i/Math.max(refs.length,1)-Math.PI/2;refPos[r.ref_id]={x:center.x+Math.cos(a)*72,y:center.y+Math.sin(a)*72}});const articlePos={};articleIds.forEach((id,i)=>{const a=2*Math.PI*i/Math.max(articleIds.length,1)-Math.PI/2;articlePos[id]={x:center.x+Math.cos(a)*138,y:center.y+Math.sin(a)*138}});for(const r of refs){line(center,refPos[r.ref_id],muted,1,[4,4]);line(refPos[r.ref_id],articlePos[r.article_id],muted,1,[4,4])}for(const r of refs){const p=refPos[r.ref_id];ctx.beginPath();ctx.arc(p.x,p.y,4,0,Math.PI*2);ctx.fillStyle="#14b8a6";ctx.fill()}for(const id of articleIds){const p=articlePos[id];ctx.fillStyle="#f59e0b";ctx.fillRect(p.x-5,p.y-5,10,10);ctx.fillStyle=ink;ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText(id,p.x,p.y+18)}}
function select(n){selected=n;detail.classList.add("open");renderDetail(n);draw()}function renderDetail(c){const refs=c.ref_ids.map(id=>byRef[id]).filter(Boolean),relations=DATA.semantic_edges.filter(e=>e.source===c.id||e.target===c.id);detail.innerHTML="";const h=document.createElement("h2");h.textContent=c.title;detail.append(h);const meta=document.createElement("div");meta.className="meta";for(const text of[c.type,c.articles.length>1?`${c.articles.length} 篇联合编译`:"单来源",`${refs.length} 个 Ref`]){const s=document.createElement("span");s.className="pill";s.textContent=text;meta.append(s)}detail.append(meta);const desc=document.createElement("div");desc.className="description";desc.textContent=c.description;detail.append(desc);appendTitle("Concept 正文");const body=document.createElement("div");body.className="body";body.textContent=c.body;detail.append(body);appendTitle("ConceptRef 证据");for(const r of refs){const box=document.createElement("div");box.className="ref";const b=document.createElement("b");b.textContent=r.title;box.append(b);const sm=document.createElement("small"),path=(r.section_path||[]).join(" › "),pages=r.page_start?(r.page_end&&r.page_end!==r.page_start?`第 ${r.page_start}–${r.page_end} 页`:`第 ${r.page_start} 页`):"";sm.textContent=[r.ref_id,path,pages].filter(Boolean).join(" · ");box.append(sm);if(r.evidence?.length){const q=document.createElement("div");q.className="quote";q.textContent=r.evidence[0];box.append(q)}detail.append(box)}appendTitle("来源 Article");for(const id of c.articles){const a=byArticle[id],row=document.createElement("div");row.className="article";row.innerHTML=`<span class="article-id">${id}</span><span></span>`;row.lastChild.textContent=a?.title||id;detail.append(row)}if(relations.length){appendTitle("相关 Concept");for(const e of relations){const other=byConcept[e.source===c.id?e.target:e.source],row=document.createElement("div");row.className="related";const b=document.createElement("b");b.textContent=other.title;row.append(b);const sm=document.createElement("small");sm.textContent=e.reasons[0];row.append(sm);row.onclick=()=>select(nodeById[other.id]);detail.append(row)}}function appendTitle(text){const x=document.createElement("h3");x.textContent=text;detail.append(x)}}
canvas.addEventListener("mousedown",e=>{drag=true;moved=false;last=[e.clientX,e.clientY];canvas.classList.add("dragging")});window.addEventListener("mouseup",()=>{drag=false;last=null;canvas.classList.remove("dragging")});window.addEventListener("mousemove",e=>{if(!drag)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];if(Math.abs(dx)+Math.abs(dy)>3)moved=true;ox+=dx;oy+=dy;last=[e.clientX,e.clientY];draw()});canvas.addEventListener("wheel",e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,wx=(mx-ox)/scale,wy=(my-oy)/scale,next=Math.max(.18,Math.min(2.8,scale*Math.exp(-e.deltaY*.001)));ox=mx-wx*next;oy=my-wy*next;scale=next;draw()},{passive:false});canvas.addEventListener("click",e=>{if(moved)return;const r=canvas.getBoundingClientRect(),p=worldPoint(e.clientX-r.left,e.clientY-r.top);let hit=null,best=18/scale;for(const n of nodes){if(!visible(n))continue;const d=Math.hypot(n.x-p.x,n.y-p.y);if(d<best){best=d;hit=n}}if(hit)select(hit)});
const types=document.getElementById("types");for(const[type,count]of Object.entries(DATA.stats.types)){const row=document.createElement("label");row.className="type-row";row.innerHTML=`<input type="checkbox" checked><span><i class="swatch" style="display:inline-block;background:${COLORS[type]};margin-right:8px"></i>${type}</span><small>${count}</small>`;const input=row.querySelector("input");input.onchange=()=>{input.checked?enabled.add(type):enabled.delete(type);draw()};types.append(row)}
document.getElementById("search").oninput=e=>{query=e.target.value.trim();draw()};document.getElementById("multiOnly").onchange=e=>{multiOnly=e.target.checked;draw()};document.getElementById("showRelations").onchange=e=>{showRelations=e.target.checked;draw()};document.getElementById("fit").onclick=fit;document.getElementById("reset").onclick=()=>{selected=null;detail.classList.remove("open");detail.innerHTML='<div class="empty"><div><div class="empty-icon">⌁</div><div>选择一个 Concept</div><small>查看正文、Ref 证据和来源文章</small></div></div>';draw()};
document.getElementById("mArticles").textContent=DATA.stats.articles;document.getElementById("mRefs").textContent=DATA.stats.refs;document.getElementById("mConcepts").textContent=DATA.stats.concepts;document.getElementById("mRelations").textContent=DATA.stats.relations;
</script>
</body>
</html>"""
