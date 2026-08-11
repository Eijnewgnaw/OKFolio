"""Bright government-style renderer for the full provenance graph."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any


def build_government_graph(
    articles: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    judgements: list[dict[str, Any]],
) -> str:
    """Build one self-contained HTML graph with layered relation views."""
    graph_refs = [
        {**item, "evidence": list(item.get("evidence", []))[:1]}
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

    semantic_edges = [
        {"source": source, "target": target, **edge}
        for (source, target), edge in sorted(grouped.items())
    ]
    relation_concepts = {
        endpoint
        for edge in semantic_edges
        for endpoint in (edge["source"], edge["target"])
    }
    data = {
        "articles": articles,
        "refs": graph_refs,
        "concepts": concepts,
        "semantic_edges": semantic_edges,
        "stats": {
            "articles": len(articles),
            "refs": len(refs),
            "concepts": len(concepts),
            "multi_source": sum(
                len(concept["ref_ids"]) > 1 for concept in concepts
            ),
            "relations": len(semantic_edges),
            "relation_evidence": sum(
                len(edge["source_refs"]) for edge in semantic_edges
            ),
            "relation_concepts": len(relation_concepts),
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
:root{color-scheme:light;--bg:#f4f8fc;--panel:#fff;--ink:#183758;--muted:#647b93;--line:#d6e2ee;--blue:#13599f;--blue2:#2c73b8;--blue-soft:#eaf3fb;--red:#bd3535;--gold:#bc8b2f;--green:#20836f;--shadow:0 8px 28px rgba(28,65,102,.09)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;overflow:hidden}button,input,select{font:inherit}.app{height:100vh;display:grid;grid-template-rows:4px auto 1fr}.flag{background:linear-gradient(90deg,var(--red) 0 18%,var(--gold) 18% 22%,var(--blue) 22% 100%)}.topbar{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:15px 24px;background:var(--panel);border-bottom:1px solid var(--line);box-shadow:0 2px 10px rgba(28,65,102,.04);z-index:3}.brand{display:flex;align-items:center;gap:12px}.seal{width:42px;height:42px;border-radius:8px;background:var(--blue);display:grid;place-items:center;color:#fff;box-shadow:inset 0 0 0 1px rgba(255,255,255,.25)}.seal svg{width:25px;height:25px}.brand h1{font-size:19px;font-weight:500;letter-spacing:.06em;margin:0}.brand p{font-size:12px;color:var(--muted);margin:2px 0 0}.metrics{display:flex;align-items:center;gap:0}.metric{min-width:83px;padding:0 18px;border-left:1px solid var(--line)}.metric strong{display:block;color:var(--blue);font-size:19px;font-weight:500;line-height:1.15}.metric span{display:block;color:var(--muted);font-size:11px;margin-top:2px}.shell{min-height:0;display:grid;grid-template-columns:250px minmax(480px,1fr) 366px}.left,.detail{background:var(--panel);overflow:auto}.left{border-right:1px solid var(--line);padding:18px}.detail{border-left:1px solid var(--line);padding:22px}.section{margin-bottom:21px}.section-title{color:var(--blue);font-size:12px;font-weight:500;margin:0 0 8px;padding-left:9px;border-left:3px solid var(--red)}.search,.select{width:100%;height:38px;border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--ink);padding:0 10px;outline:none}.search:focus,.select:focus{border-color:var(--blue2);box-shadow:0 0 0 3px rgba(44,115,184,.12)}.type-list{display:grid;gap:4px}.type-row{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:8px;padding:6px 5px;border-radius:4px;color:var(--ink)}.type-row:hover{background:var(--blue-soft)}.type-row input{accent-color:var(--blue)}.type-row small{color:var(--muted)}.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}.btns{display:grid;grid-template-columns:1fr 1fr;gap:7px}.btn{height:35px;border:1px solid #b9ccdf;border-radius:5px;background:#fff;color:var(--blue);cursor:pointer}.btn:hover{background:var(--blue-soft);border-color:var(--blue2)}.summary{padding:10px 11px;background:var(--blue-soft);border-left:3px solid var(--blue);color:var(--muted);font-size:12px}.summary b{color:var(--blue);font-weight:500}.legend{display:grid;gap:8px;color:var(--muted);font-size:12px}.legend-row{display:flex;align-items:center;gap:8px}.legend-line{width:24px;height:2px;background:var(--blue2)}.legend-line.double{height:5px;border-top:1px solid var(--blue2);border-bottom:1px solid var(--blue2);background:none}.legend-line.dash{height:1px;background:repeating-linear-gradient(90deg,var(--green) 0 4px,transparent 4px 7px)}.legend-square{width:9px;height:9px;background:var(--gold);border-radius:2px}.stage{position:relative;min-width:0;overflow:hidden;background-color:#f8fbfe;background-image:linear-gradient(rgba(19,89,159,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(19,89,159,.035) 1px,transparent 1px);background-size:28px 28px}.stage canvas{display:block;width:100%;height:100%;cursor:grab}.stage canvas.dragging{cursor:grabbing}.status{position:absolute;left:16px;bottom:15px;background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:5px;padding:7px 10px;color:var(--muted);font-size:12px;box-shadow:var(--shadow);pointer-events:none}.mode-badge{position:absolute;right:16px;top:15px;background:#fff;border:1px solid var(--line);border-left:3px solid var(--red);border-radius:5px;padding:7px 10px;color:var(--blue);font-size:12px;box-shadow:var(--shadow);pointer-events:none}.empty{min-height:100%;display:grid;place-content:center;text-align:center;color:var(--muted)}.empty-mark{width:50px;height:50px;border:1px solid var(--line);border-radius:8px;margin:0 auto 12px;display:grid;place-items:center;color:var(--blue);background:var(--blue-soft)}.detail h2{font-size:19px;font-weight:500;line-height:1.4;margin:0 0 9px;color:var(--ink)}.meta{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}.pill{padding:2px 8px;border-radius:3px;background:var(--blue-soft);color:var(--blue);font-size:11px}.description{color:var(--muted);margin-bottom:18px}.detail h3{font-size:12px;font-weight:500;color:var(--blue);margin:20px 0 8px;padding-bottom:5px;border-bottom:1px solid var(--line)}.body{white-space:pre-wrap;font:12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f7f9fb;border:1px solid var(--line);border-radius:5px;padding:12px;max-height:230px;overflow:auto}.ref{border-top:1px solid var(--line);padding:10px 0}.ref:first-of-type{border-top:0}.ref b{display:block;font-size:13px;font-weight:500}.ref small{color:var(--muted)}.quote{border-left:2px solid var(--green);padding-left:9px;margin-top:6px;color:var(--muted);font-size:12px}.article{display:grid;grid-template-columns:auto 1fr;gap:8px;padding:6px 0}.article-id{color:var(--gold);font:11px ui-monospace}.related{border-left:3px solid var(--blue2);background:#f7fafd;padding:8px 10px;margin:7px 0;cursor:pointer}.related:hover{background:var(--blue-soft)}.related b{display:block;font-weight:500}.related small{color:var(--muted)}.edge-count{color:var(--red);font-size:11px;margin-left:5px}
@media(max-width:1080px){.shell{grid-template-columns:220px 1fr}.detail{position:absolute;right:0;top:76px;bottom:0;width:350px;box-shadow:var(--shadow);transform:translateX(100%);transition:.2s;z-index:5}.detail.open{transform:none}.metrics .metric:nth-child(-n+2){display:none}}@media(max-width:740px){.topbar{padding:11px 13px}.shell{grid-template-columns:1fr}.left{display:none}.detail{top:66px;width:min(94vw,360px)}.brand p{display:none}.metric{min-width:64px;padding:0 9px}.metrics .metric:nth-child(-n+3){display:none}}
</style>
</head>
<body>
<div class="app">
  <div class="flag"></div>
  <header class="topbar">
    <div class="brand">
      <div class="seal" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none"><circle cx="8" cy="16" r="3" fill="currentColor"/><circle cx="24" cy="8" r="3" fill="currentColor"/><circle cx="24" cy="24" r="3" fill="currentColor"/><path d="M11 15l10-6M11 17l10 6" stroke="currentColor" stroke-width="2"/></svg></div>
      <div><h1>决策参考知识图谱</h1><p>全库联合编译 · Concept / ConceptRef / Article 全链路溯源</p></div>
    </div>
    <div class="metrics">
      <div class="metric"><strong id="mArticles"></strong><span>原始文章</span></div>
      <div class="metric"><strong id="mRefs"></strong><span>ConceptRef</span></div>
      <div class="metric"><strong id="mConcepts"></strong><span>知识概念</span></div>
      <div class="metric"><strong id="mRelations"></strong><span>关系组</span></div>
    </div>
  </header>
  <main class="shell">
    <aside class="left">
      <section class="section"><h2 class="section-title">展示层级</h2><select id="viewMode" class="select"><option value="relations">关系网络（推荐）</option><option value="multi">多来源 Concept</option><option value="all">全部知识资产</option></select></section>
      <section class="section"><h2 class="section-title">知识检索</h2><input id="search" class="search" type="search" placeholder="搜索概念或来源…"></section>
      <section class="section"><h2 class="section-title">概念类型</h2><div id="types" class="type-list"></div></section>
      <section class="section"><h2 class="section-title">视图操作</h2><div class="btns"><button id="fit" class="btn">适应画布</button><button id="reset" class="btn">清除选择</button></div></section>
      <section class="section summary"><b>关系网络</b>默认仅展示存在语义边的 Concept，避免 392 个节点同时铺开。选择节点后展开完整证据链。</section>
      <section class="legend">
        <div class="legend-row"><span class="legend-line"></span>Concept 语义关系</div>
        <div class="legend-row"><span class="legend-line double"></span>多条 Ref 依据形成的关系</div>
        <div class="legend-row"><span class="legend-line dash"></span>ConceptRef 证据链</div>
        <div class="legend-row"><span class="legend-square"></span>来源 Article</div>
      </section>
    </aside>
    <section class="stage"><canvas id="graph" aria-label="可交互知识关系图"></canvas><div id="modeBadge" class="mode-badge"></div><div id="status" class="status"></div></section>
    <aside id="detail" class="detail"><div class="empty"><div><div class="empty-mark">关系</div><div>选择一个 Concept</div><small>查看正文、关系依据和来源证据</small></div></div></aside>
  </main>
</div>
<script>
const DATA=""" + payload + """;
const COLORS={"分析框架":"#1769aa","政策建议":"#bd4545","数据口径":"#20836f","国际比较":"#bc8b2f","术语解释":"#687c99"};
const canvas=document.getElementById("graph"),ctx=canvas.getContext("2d"),detail=document.getElementById("detail"),status=document.getElementById("status"),modeBadge=document.getElementById("modeBadge");
const byRef=Object.fromEntries(DATA.refs.map(x=>[x.ref_id,x])),byArticle=Object.fromEntries(DATA.articles.map(x=>[x.article_id,x])),byConcept=Object.fromEntries(DATA.concepts.map(x=>[x.id,x]));
const relationIds=new Set(DATA.semantic_edges.flatMap(e=>[e.source,e.target])),enabled=new Set(Object.keys(COLORS));
let dpr=1,w=0,h=0,scale=.7,ox=0,oy=0,drag=false,moved=false,last=null,selected=null,query="",viewMode="relations";
const nodes=DATA.concepts.map(c=>({...c,r:c.ref_ids.length>1?9:6,allX:0,allY:0,relX:0,relY:0,multiX:0,multiY:0}));
const nodeById=Object.fromEntries(nodes.map(n=>[n.id,n]));
function clusterLayout(list,xKey,yKey,centers){for(const type of Object.keys(COLORS)){const group=list.filter(x=>x.type===type).sort((a,b)=>b.ref_ids.length-a.ref_ids.length||a.title.localeCompare(b.title)),center=centers[type];group.forEach((n,i)=>{const a=i*2.3999632297,r=22+Math.sqrt(i)*25;n[xKey]=center[0]+Math.cos(a)*r;n[yKey]=center[1]+Math.sin(a)*r})}}
clusterLayout(nodes,"allX","allY",{"分析框架":[560,700],"政策建议":[1740,700],"数据口径":[1130,170],"国际比较":[1130,1240],"术语解释":[1130,700]});
clusterLayout(nodes.filter(n=>n.ref_ids.length>1),"multiX","multiY",{"分析框架":[500,470],"政策建议":[1500,470],"数据口径":[1000,130],"国际比较":[1000,850],"术语解释":[1000,470]});
function relationLayout(){const ids=[...relationIds],adj=Object.fromEntries(ids.map(id=>[id,[]]));for(const e of DATA.semantic_edges){adj[e.source].push(e.target);adj[e.target].push(e.source)}const seen=new Set(),components=[];for(const id of ids){if(seen.has(id))continue;const stack=[id],part=[];seen.add(id);while(stack.length){const now=stack.pop();part.push(now);for(const next of adj[now])if(!seen.has(next)){seen.add(next);stack.push(next)}}components.push(part)}components.sort((a,b)=>b.length-a.length);const cols=Math.min(4,Math.ceil(Math.sqrt(components.length))),cellW=520,cellH=360;components.forEach((part,index)=>{const col=index%cols,row=Math.floor(index/cols),cx=260+col*cellW,cy=190+row*cellH,r=Math.max(58,Math.min(142,part.length*22));part.sort((a,b)=>adj[b].length-adj[a].length||a.localeCompare(b)).forEach((id,i)=>{const a=2*Math.PI*i/part.length-Math.PI/2,node=nodeById[id];node.relX=cx+Math.cos(a)*r;node.relY=cy+Math.sin(a)*r})})}
relationLayout();
function position(n){if(query)return{x:n.allX,y:n.allY};return viewMode==="relations"?{x:n.relX,y:n.relY}:viewMode==="multi"?{x:n.multiX,y:n.multiY}:{x:n.allX,y:n.allY}}
function visible(n){if(!enabled.has(n.type))return false;if(query){const q=query.toLowerCase();return(n.title+" "+n.description+" "+n.articles.join(" ")).toLowerCase().includes(q)}if(viewMode==="relations")return relationIds.has(n.id);if(viewMode==="multi")return n.ref_ids.length>1;return true}
function visibleNodes(){return nodes.filter(visible)}
function resize(){const r=canvas.getBoundingClientRect();dpr=Math.min(devicePixelRatio||1,2);w=r.width;h=r.height;canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);fit()}new ResizeObserver(resize).observe(canvas);
function fit(){const list=visibleNodes();if(!list.length)return;const ps=list.map(position),minX=Math.min(...ps.map(p=>p.x)),maxX=Math.max(...ps.map(p=>p.x)),minY=Math.min(...ps.map(p=>p.y)),maxY=Math.max(...ps.map(p=>p.y)),bw=Math.max(320,maxX-minX+220),bh=Math.max(260,maxY-minY+180);scale=Math.min(w/bw,h/bh,.95);ox=(w-(minX+maxX)*scale)/2;oy=(h-(minY+maxY)*scale)/2;draw()}
function screen(n){const p=position(n);return{x:ox+p.x*scale,y:oy+p.y*scale}}function worldPoint(x,y){return{x:(x-ox)/scale,y:(y-oy)/scale}}
function straight(a,b,color,width=1,dash=[]){ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();ctx.setLineDash([])}
function curve(a,b,color,width,offset){const dx=b.x-a.x,dy=b.y-a.y,len=Math.max(1,Math.hypot(dx,dy)),mx=(a.x+b.x)/2-dy/len*offset,my=(a.y+b.y)/2+dx/len*offset;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.quadraticCurveTo(mx,my,b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function draw(){ctx.clearRect(0,0,w,h);const list=visibleNodes(),ink="#183758",muted="#7890a6",edgeColor="#5f87ad";if(viewMode!=="relations"&&!query){const centers=viewMode==="multi"?{"分析框架":[500,470],"政策建议":[1500,470],"数据口径":[1000,130],"国际比较":[1000,850],"术语解释":[1000,470]}:{"分析框架":[560,700],"政策建议":[1740,700],"数据口径":[1130,170],"国际比较":[1130,1240],"术语解释":[1130,700]};for(const[type,p]of Object.entries(centers)){if(!enabled.has(type))continue;const q={x:ox+p[0]*scale,y:oy+p[1]*scale};ctx.fillStyle=muted;ctx.font="500 12px system-ui";ctx.textAlign="center";ctx.fillText(type,q.x,q.y-100*scale)}}for(const e of DATA.semantic_edges){const a=nodeById[e.source],b=nodeById[e.target];if(!visible(a)||!visible(b))continue;const pa=screen(a),pb=screen(b),count=Math.max(1,e.source_refs.length),active=selected&&(selected.id===a.id||selected.id===b.id);for(let i=0;i<count;i++){const offset=(i-(count-1)/2)*8;curve(pa,pb,active?"#bd3535":edgeColor,active?2.2:1.25,offset)}}for(const n of list){const p=screen(n),active=selected&&selected.id===n.id,neighbor=selected&&DATA.semantic_edges.some(e=>(e.source===selected.id&&e.target===n.id)||(e.target===selected.id&&e.source===n.id)),dim=selected&&!active&&!neighbor;ctx.globalAlpha=dim?.2:1;const radius=Math.max(viewMode==="all"?3.2:5,n.r*scale*1.35);ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fillStyle=COLORS[n.type];ctx.fill();if(n.ref_ids.length>1){ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.stroke();ctx.beginPath();ctx.arc(p.x,p.y,radius+3,0,Math.PI*2);ctx.strokeStyle=active?"#bd3535":"#244f78";ctx.lineWidth=active?2:1;ctx.stroke()}if(active){ctx.beginPath();ctx.arc(p.x,p.y,radius+7,0,Math.PI*2);ctx.strokeStyle="#bc8b2f";ctx.lineWidth=2;ctx.stroke()}const showLabel=viewMode==="relations"||active||scale>1.05||query;if(showLabel){ctx.globalAlpha=1;ctx.font=active?"500 13px system-ui":"12px system-ui";ctx.textAlign="center";ctx.fillStyle=ink;ctx.fillText(n.title.slice(0,20),p.x,p.y-radius-8)}ctx.globalAlpha=1}if(selected&&visible(selected))drawProvenance(selected,ink,muted);const label=query?"检索结果":viewMode==="relations"?"关系网络":viewMode==="multi"?"多来源 Concept":"全部知识资产";modeBadge.textContent=label;status.textContent=`显示 ${list.length} / ${nodes.length} 个 Concept · ${DATA.stats.relations} 组关系 · ${DATA.stats.relation_evidence} 条 Ref 依据`}
function drawProvenance(c,ink,muted){const center=screen(c),refs=c.ref_ids.map(id=>byRef[id]).filter(Boolean),articleIds=[...new Set(c.articles)],refPos={},articlePos={};refs.forEach((r,i)=>{const a=2*Math.PI*i/Math.max(1,refs.length)-Math.PI/2;refPos[r.ref_id]={x:center.x+Math.cos(a)*78,y:center.y+Math.sin(a)*78}});articleIds.forEach((id,i)=>{const a=2*Math.PI*i/Math.max(1,articleIds.length)-Math.PI/2;articlePos[id]={x:center.x+Math.cos(a)*148,y:center.y+Math.sin(a)*148}});for(const r of refs){straight(center,refPos[r.ref_id],"#20836f",1,[4,4]);straight(refPos[r.ref_id],articlePos[r.article_id],"#bc8b2f",1,[4,4])}for(const r of refs){const p=refPos[r.ref_id];ctx.save();ctx.translate(p.x,p.y);ctx.rotate(Math.PI/4);ctx.fillStyle="#20836f";ctx.fillRect(-4,-4,8,8);ctx.restore()}for(const id of articleIds){const p=articlePos[id];ctx.fillStyle="#bc8b2f";ctx.fillRect(p.x-6,p.y-5,12,10);ctx.fillStyle=ink;ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText(id,p.x,p.y+19)}}
function selectNode(n){selected=n;detail.classList.add("open");renderDetail(n);draw()}
function renderDetail(c){const refs=c.ref_ids.map(id=>byRef[id]).filter(Boolean),relations=DATA.semantic_edges.filter(e=>e.source===c.id||e.target===c.id);detail.innerHTML="";const h=document.createElement("h2");h.textContent=c.title;detail.append(h);const meta=document.createElement("div");meta.className="meta";for(const text of[c.type,c.ref_ids.length>1?`${c.articles.length} 篇联合编译`:"单来源",`${refs.length} 个 Ref`,`${relations.length} 组关系`]){const s=document.createElement("span");s.className="pill";s.textContent=text;meta.append(s)}detail.append(meta);const desc=document.createElement("div");desc.className="description";desc.textContent=c.description;detail.append(desc);appendTitle("Concept 正文");const body=document.createElement("div");body.className="body";body.textContent=c.body;detail.append(body);if(relations.length){appendTitle("关联 Concept");for(const e of relations){const other=byConcept[e.source===c.id?e.target:e.source],row=document.createElement("div");row.className="related";const b=document.createElement("b");b.textContent=other.title;const count=document.createElement("span");count.className="edge-count";count.textContent=`${e.source_refs.length} 条 Ref 依据`;b.append(count);row.append(b);const sm=document.createElement("small");sm.textContent=e.reasons.join("；");row.append(sm);row.onclick=()=>selectNode(nodeById[other.id]);detail.append(row)}}appendTitle("ConceptRef 证据");for(const r of refs){const box=document.createElement("div");box.className="ref";const b=document.createElement("b");b.textContent=r.title;box.append(b);const sm=document.createElement("small");sm.textContent=r.ref_id;box.append(sm);if(r.evidence?.length){const q=document.createElement("div");q.className="quote";q.textContent=r.evidence[0];box.append(q)}detail.append(box)}appendTitle("来源 Article");for(const id of c.articles){const a=byArticle[id],row=document.createElement("div");row.className="article";const code=document.createElement("span");code.className="article-id";code.textContent=id;const title=document.createElement("span");title.textContent=a?.title||id;row.append(code,title);detail.append(row)}function appendTitle(text){const x=document.createElement("h3");x.textContent=text;detail.append(x)}}
canvas.addEventListener("mousedown",e=>{drag=true;moved=false;last=[e.clientX,e.clientY];canvas.classList.add("dragging")});window.addEventListener("mouseup",()=>{drag=false;last=null;canvas.classList.remove("dragging")});window.addEventListener("mousemove",e=>{if(!drag)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];if(Math.abs(dx)+Math.abs(dy)>3)moved=true;ox+=dx;oy+=dy;last=[e.clientX,e.clientY];draw()});canvas.addEventListener("wheel",e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,wx=(mx-ox)/scale,wy=(my-oy)/scale,next=Math.max(.2,Math.min(3,scale*Math.exp(-e.deltaY*.001)));ox=mx-wx*next;oy=my-wy*next;scale=next;draw()},{passive:false});canvas.addEventListener("click",e=>{if(moved)return;const r=canvas.getBoundingClientRect(),world=worldPoint(e.clientX-r.left,e.clientY-r.top);let hit=null,best=20/scale;for(const n of visibleNodes()){const p=position(n),d=Math.hypot(p.x-world.x,p.y-world.y);if(d<best){best=d;hit=n}}if(hit)selectNode(hit)});
const types=document.getElementById("types");for(const[type,count]of Object.entries(DATA.stats.types)){const row=document.createElement("label");row.className="type-row";row.innerHTML=`<input type="checkbox" checked><span><i class="swatch" style="background:${COLORS[type]}"></i>${type}</span><small>${count}</small>`;const input=row.querySelector("input");input.onchange=()=>{input.checked?enabled.add(type):enabled.delete(type);fit()};types.append(row)}
document.getElementById("viewMode").onchange=e=>{viewMode=e.target.value;selected=null;detail.classList.remove("open");resetDetail();fit()};document.getElementById("search").oninput=e=>{query=e.target.value.trim();fit()};document.getElementById("fit").onclick=fit;document.getElementById("reset").onclick=()=>{selected=null;detail.classList.remove("open");resetDetail();draw()};function resetDetail(){detail.innerHTML='<div class="empty"><div><div class="empty-mark">关系</div><div>选择一个 Concept</div><small>查看正文、关系依据和来源证据</small></div></div>'}
document.getElementById("mArticles").textContent=DATA.stats.articles;document.getElementById("mRefs").textContent=DATA.stats.refs;document.getElementById("mConcepts").textContent=DATA.stats.concepts;document.getElementById("mRelations").textContent=DATA.stats.relations;
</script>
</body>
</html>"""
