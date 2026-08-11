"""Offline spatial-weave and sphere renderer for the provenance graph."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any


def build_graph_data(
    articles: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    judgements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the portable graph payload shared by all static viewers."""
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
            {
                "source_refs": [],
                "reasons": [],
                "relation_types": [],
                "relation_evidence": [],
            },
        )
        relation_type = str(item.get("relation_type") or "related")
        evidence_ref_ids = list(item.get("evidence_ref_ids") or ())
        if not evidence_ref_ids:
            evidence_ref_ids = [item["left_ref_id"], item["right_ref_id"]]
        edge["source_refs"].append(evidence_ref_ids)
        edge["reasons"].append(item["reason"])
        edge["relation_types"].append(relation_type)
        edge["relation_evidence"].append(
            {
                "edge_id": item.get("edge_id", ""),
                "relation_type": relation_type,
                "direction": str(item.get("direction") or "bidirectional"),
                "ref_ids": evidence_ref_ids,
                "reason": item["reason"],
            }
        )

    semantic_edges = [
        {
            "source": source,
            "target": target,
            **{
                **edge,
                "relation_types": sorted(set(edge["relation_types"])),
            },
        }
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
            "relation_types": dict(
                sorted(
                    Counter(
                        relation_type
                        for edge in semantic_edges
                        for relation_type in edge["relation_types"]
                    ).items()
                )
            ),
            "types": dict(
                sorted(Counter(concept["type"] for concept in concepts).items())
            ),
        },
    }
    return data


def build_spatial_graph(
    articles: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    judgements: list[dict[str, Any]],
) -> str:
    """Build a bright 3D relation weave and a full-asset knowledge sphere."""
    payload = json.dumps(
        build_graph_data(articles, refs, concepts, judgements),
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>决策参考知识图谱</title>
<style>
:root{color-scheme:light;--bg:#f3f8fc;--panel:#fff;--ink:#173858;--muted:#667f98;--line:#d6e3ef;--blue:#145b9f;--blue2:#3b7db8;--blue-soft:#eaf3fa;--red:#ba3d3d;--gold:#b8892e;--green:#208471;--shadow:0 10px 32px rgba(27,67,104,.1)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;overflow:hidden}button,input,select{font:inherit}.app{height:100vh;display:grid;grid-template-rows:4px auto 1fr}.flag{background:linear-gradient(90deg,var(--red) 0 18%,var(--gold) 18% 22%,var(--blue) 22% 100%)}.topbar{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:15px 24px;background:var(--panel);border-bottom:1px solid var(--line);box-shadow:0 2px 10px rgba(27,67,104,.04);z-index:3}.brand{display:flex;align-items:center;gap:12px}.seal{width:42px;height:42px;border-radius:8px;background:var(--blue);display:grid;place-items:center;color:#fff;box-shadow:inset 0 0 0 1px rgba(255,255,255,.25)}.seal svg{width:25px;height:25px}.brand h1{font-size:19px;font-weight:500;letter-spacing:.06em;margin:0}.brand p{font-size:12px;color:var(--muted);margin:2px 0 0}.metrics{display:flex}.metric{min-width:83px;padding:0 18px;border-left:1px solid var(--line)}.metric strong{display:block;color:var(--blue);font-size:19px;font-weight:500;line-height:1.15}.metric span{display:block;color:var(--muted);font-size:11px;margin-top:2px}.shell{min-height:0;display:grid;grid-template-columns:248px minmax(500px,1fr) 366px}.left,.detail{background:var(--panel);overflow:auto}.left{border-right:1px solid var(--line);padding:18px}.detail{border-left:1px solid var(--line);padding:22px}.section{margin-bottom:21px}.section-title{color:var(--blue);font-size:12px;font-weight:500;margin:0 0 8px;padding-left:9px;border-left:3px solid var(--red)}.search,.select{width:100%;height:38px;border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--ink);padding:0 10px;outline:none}.search:focus,.select:focus{border-color:var(--blue2);box-shadow:0 0 0 3px rgba(59,125,184,.12)}.type-list{display:grid;gap:4px}.type-row{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:8px;padding:6px 5px;border-radius:4px}.type-row:hover{background:var(--blue-soft)}.type-row input{accent-color:var(--blue)}.type-row small{color:var(--muted)}.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}.btns{display:grid;grid-template-columns:1fr 1fr;gap:7px}.btn{height:35px;border:1px solid #b9ccdf;border-radius:5px;background:#fff;color:var(--blue);cursor:pointer}.btn:hover{background:var(--blue-soft);border-color:var(--blue2)}.summary{padding:10px 11px;background:var(--blue-soft);border-left:3px solid var(--blue);color:var(--muted);font-size:12px}.summary b{color:var(--blue);font-weight:500}.legend{display:grid;gap:8px;color:var(--muted);font-size:12px}.legend-row{display:flex;align-items:center;gap:8px}.legend-line{width:25px;height:2px;background:linear-gradient(90deg,rgba(59,125,184,.3),var(--blue2),rgba(59,125,184,.3));border-radius:2px}.legend-line.double{height:5px;border-top:1px solid var(--blue2);border-bottom:1px solid var(--blue2);background:none}.legend-line.dash{height:1px;background:repeating-linear-gradient(90deg,var(--green) 0 4px,transparent 4px 7px)}.legend-core{width:11px;height:11px;border:2px solid var(--blue);border-radius:50%;box-shadow:0 0 0 3px rgba(20,91,159,.1)}.stage{position:relative;min-width:0;overflow:hidden;background:radial-gradient(circle at 50% 48%,#fff 0,#f8fbfe 48%,#edf5fb 100%)}.stage canvas{display:block;width:100%;height:100%;cursor:grab}.stage canvas.dragging{cursor:grabbing}.catalog{display:none;position:absolute;inset:0;overflow:auto;padding:38px 34px 64px;background:linear-gradient(145deg,rgba(255,255,255,.94),rgba(239,247,253,.92))}.catalog.show{display:block}.catalog-head{max-width:1120px;margin:0 auto 28px}.catalog-kicker{color:var(--red);font-size:11px;letter-spacing:.12em}.catalog-head h2{margin:4px 0 5px;color:var(--ink);font-size:24px;font-weight:500}.catalog-head p{margin:0;color:var(--muted)}.catalog-section{max-width:1120px;margin:0 auto 30px}.catalog-section-head{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin-bottom:10px;padding-bottom:7px;border-bottom:1px solid var(--line)}.catalog-section-head h3{margin:0;color:var(--blue);font-size:14px;font-weight:500}.catalog-section-head span{color:var(--muted);font-size:11px}.catalog-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:11px}.catalog-card{position:relative;min-height:82px;padding:14px 15px 12px;text-align:left;border:1px solid #c9ddef;border-radius:8px;background:rgba(255,255,255,.86);color:var(--ink);cursor:pointer;box-shadow:0 4px 14px rgba(27,67,104,.04);transition:transform .15s,border-color .15s,box-shadow .15s}.catalog-card:hover{transform:translateY(-2px);border-color:var(--blue2);box-shadow:0 8px 20px rgba(27,67,104,.1)}.catalog-card.selected{border-color:var(--gold);box-shadow:0 0 0 2px rgba(184,137,46,.16)}.catalog-card-title{display:block;padding-left:14px;font-size:14px;font-weight:500;line-height:1.45}.catalog-dot{position:absolute;left:14px;top:20px;width:7px;height:7px;border-radius:50%}.catalog-card-meta{display:block;margin:8px 0 0 14px;color:var(--muted);font-size:11px}.catalog-empty{max-width:520px;margin:100px auto;text-align:center;color:var(--muted)}.status,.mode-badge,.tooltip{position:absolute;background:rgba(255,255,255,.95);border:1px solid var(--line);border-radius:5px;box-shadow:var(--shadow);pointer-events:none}.status{left:16px;bottom:15px;padding:7px 10px;color:var(--muted);font-size:12px}.mode-badge{right:16px;top:15px;border-left:3px solid var(--red);padding:7px 10px;color:var(--blue);font-size:12px}.tooltip{display:none;max-width:260px;padding:8px 10px;color:var(--ink);font-size:12px;z-index:4}.tooltip b{display:block;font-weight:500}.tooltip span{display:block;color:var(--muted);margin-top:2px}.empty{min-height:100%;display:grid;place-content:center;text-align:center;color:var(--muted)}.empty-mark{width:50px;height:50px;border:1px solid var(--line);border-radius:50%;margin:0 auto 12px;display:grid;place-items:center;color:var(--blue);background:var(--blue-soft)}.detail h2{font-size:19px;font-weight:500;line-height:1.4;margin:0 0 9px}.meta{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}.pill{padding:2px 8px;border-radius:3px;background:var(--blue-soft);color:var(--blue);font-size:11px}.description{color:var(--muted);margin-bottom:18px}.detail h3{font-size:12px;font-weight:500;color:var(--blue);margin:20px 0 8px;padding-bottom:5px;border-bottom:1px solid var(--line)}.body{white-space:pre-wrap;font:12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f7f9fb;border:1px solid var(--line);border-radius:5px;padding:12px;max-height:230px;overflow:auto}.ref{border-top:1px solid var(--line);padding:10px 0}.ref:first-of-type{border-top:0}.ref b{display:block;font-size:13px;font-weight:500}.ref small{color:var(--muted)}.quote{border-left:2px solid var(--green);padding-left:9px;margin-top:6px;color:var(--muted);font-size:12px}.article{display:grid;grid-template-columns:auto 1fr;gap:8px;padding:6px 0}.article-id{color:var(--gold);font:11px ui-monospace}.related{border-left:3px solid var(--blue2);background:#f7fafd;padding:8px 10px;margin:7px 0;cursor:pointer}.related:hover{background:var(--blue-soft)}.related b{display:block;font-weight:500}.related small{color:var(--muted)}.edge-count{color:var(--red);font-size:11px;margin-left:5px}
@media(max-width:1080px){.shell{grid-template-columns:220px 1fr}.detail{position:absolute;right:0;top:76px;bottom:0;width:350px;box-shadow:var(--shadow);transform:translateX(100%);transition:.2s;z-index:5}.detail.open{transform:none}.metrics .metric:nth-child(-n+2){display:none}}@media(max-width:740px){.topbar{padding:11px 13px}.shell{grid-template-columns:1fr}.left{display:none}.detail{top:66px;width:min(94vw,360px)}.brand p{display:none}.metric{min-width:64px;padding:0 9px}.metrics .metric:nth-child(-n+3){display:none}}
</style>
</head>
<body>
<div class="app">
  <div class="flag"></div>
  <header class="topbar">
    <div class="brand"><div class="seal" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none"><circle cx="8" cy="16" r="3" fill="currentColor"/><circle cx="24" cy="8" r="3" fill="currentColor"/><circle cx="24" cy="24" r="3" fill="currentColor"/><path d="M11 15l10-6M11 17l10 6" stroke="currentColor" stroke-width="2"/></svg></div><div><h1>决策参考知识图谱</h1><p>全库联合编译 · Concept / ConceptRef / Article 全链路溯源</p></div></div>
    <div class="metrics"><div class="metric"><strong id="mArticles"></strong><span>原始文章</span></div><div class="metric"><strong id="mRefs"></strong><span>ConceptRef</span></div><div class="metric"><strong id="mConcepts"></strong><span>知识概念</span></div><div class="metric"><strong id="mRelations"></strong><span>关系组</span></div></div>
  </header>
  <main class="shell">
    <aside class="left">
      <section class="section"><h2 class="section-title">展示层级</h2><select id="viewMode" class="select"><option value="relations">三维关系编织网（推荐）</option><option value="sphere">三维知识球（全部资产）</option><option value="catalog">全部概念查阅（清单）</option></select></section>
      <section class="section"><h2 class="section-title">知识检索</h2><input id="search" class="search" type="search" placeholder="搜索并定位 Concept…"></section>
      <section class="section"><h2 class="section-title">概念类型</h2><div id="types" class="type-list"></div></section>
      <section class="section"><h2 class="section-title">关系类型</h2><div id="relationTypes" class="type-list"></div></section>
      <section class="section"><h2 class="section-title">视图操作</h2><div class="btns"><button id="fit" class="btn">适应画布</button><button id="reset" class="btn">清除选择</button></div></section>
      <section id="summary" class="section summary"></section>
      <section class="legend"><div class="legend-row"><span class="legend-line"></span>柔性 Concept 关系</div><div class="legend-row"><span class="legend-line double"></span>Concept → Article 来源线</div><div class="legend-row"><span class="legend-core"></span>多来源 Concept 内核</div><div class="legend-row"><span class="legend-line dash"></span>选择后展开 Ref 溯源</div></section>
    </aside>
    <section class="stage"><canvas id="graph" aria-label="三维关系编织网和三维知识球"></canvas><div id="catalog" class="catalog" aria-label="全部概念查阅清单"></div><div id="modeBadge" class="mode-badge"></div><div id="status" class="status"></div><div id="tooltip" class="tooltip"></div></section>
    <aside id="detail" class="detail"><div class="empty"><div><div class="empty-mark">关系</div><div>选择一个 Concept</div><small>查看正文、关联依据和来源证据</small></div></div></aside>
  </main>
</div>
<script>
const DATA=""" + payload + """;
const COLORS={"分析框架":"#1769aa","政策建议":"#bd4545","数据口径":"#20836f","国际比较":"#bc8b2f","术语解释":"#687c99"};
const RELATION_LABELS={defines:"定义/口径",supports:"证据支撑",constrains:"约束条件",causes:"因果影响",recommends:"问题到建议",compares:"比较对标",extends:"补充展开",related:"实质关联"};
const GOLDEN=2.399963229728653;
const canvas=document.getElementById("graph"),ctx=canvas.getContext("2d"),catalog=document.getElementById("catalog"),detail=document.getElementById("detail"),status=document.getElementById("status"),modeBadge=document.getElementById("modeBadge"),summary=document.getElementById("summary"),tooltip=document.getElementById("tooltip");
const byRef=Object.fromEntries(DATA.refs.map(x=>[x.ref_id,x])),byArticle=Object.fromEntries(DATA.articles.map(x=>[x.article_id,x])),byConcept=Object.fromEntries(DATA.concepts.map(x=>[x.id,x])),enabled=new Set(Object.keys(COLORS)),enabledRelations=new Set(Object.keys(DATA.stats.relation_types||{related:DATA.stats.relations}));
const relationIds=new Set(DATA.semantic_edges.flatMap(e=>[e.source,e.target])),degree=Object.fromEntries(DATA.concepts.map(c=>[c.id,0]));
for(const e of DATA.semantic_edges){degree[e.source]++;degree[e.target]++}
let dpr=1,w=0,h=0,sphereZoom=.88,yaw=-.45,pitch=.2,drag=false,moved=false,last=null,selected=null,hovered=null,query="",viewMode="relations";
const nodes=DATA.concepts.map(c=>({...c,wx:0,wy:0,wz:0,sx:0,sy:0,sz:0,px:0,py:0,pz:0,pr:0}));
const nodeById=Object.fromEntries(nodes.map(n=>[n.id,n]));
const relationArticles=[...new Set(nodes.filter(n=>relationIds.has(n.id)).flatMap(n=>n.articles))].map((id,i,all)=>{const y=1-2*(i+.5)/all.length,r=Math.sqrt(Math.max(0,1-y*y)),a=i*GOLDEN+.7;return{id,x:Math.cos(a)*r*.56,y:y*.56,z:Math.sin(a)*r*.56,px:0,py:0,pz:0}});
function stableHash(text){let h=2166136261;for(let i=0;i<text.length;i++){h^=text.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}
function buildWeave(){const list=nodes.filter(n=>relationIds.has(n.id)).sort((a,b)=>stableHash(a.id)-stableHash(b.id));list.forEach((n,i)=>{const y=1-2*(i+.5)/list.length,r=Math.sqrt(Math.max(0,1-y*y)),a=i*GOLDEN;n.wx=Math.cos(a)*r;n.wy=y;n.wz=Math.sin(a)*r})}
function buildSphere(){const list=nodes.slice().sort((a,b)=>stableHash(a.id)-stableHash(b.id));list.forEach((n,i)=>{const y=1-2*(i+.5)/list.length,r=Math.sqrt(Math.max(0,1-y*y)),a=i*GOLDEN,radius=n.ref_ids.length>1?.54:1;n.sx=Math.cos(a)*r*radius;n.sy=y*radius;n.sz=Math.sin(a)*r*radius})}
buildWeave();buildSphere();
function rgba(hex,a){const v=parseInt(hex.slice(1),16);return`rgba(${v>>16},${v>>8&255},${v&255},${a})`}
function matches(n){if(!query)return true;const q=query.toLowerCase();return(n.title+" "+n.description+" "+n.articles.join(" ")).toLowerCase().includes(q)}
function edgeVisible(e){return(e.relation_types||["related"]).some(t=>enabledRelations.has(t))}
function relationVisible(n){return relationIds.has(n.id)&&enabled.has(n.type)&&DATA.semantic_edges.some(e=>edgeVisible(e)&&(e.source===n.id||e.target===n.id))}
function resize(){const r=canvas.getBoundingClientRect();dpr=Math.min(devicePixelRatio||1,2);w=r.width;h=r.height;canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);fit()}new ResizeObserver(resize).observe(canvas);
function fit(){sphereZoom=.92;draw()}
function projectPoint(x,y,z){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=x*cy+z*sy,z1=-x*sy+z*cy,y2=y*cp-z1*sp,z2=y*sp+z1*cp,R=Math.min(w,h)*.4*sphereZoom,perspective=1+z2*.2;return{x:w/2+x1*R*perspective,y:h/2-y2*R*perspective,z:z2,scale:perspective}}
function spherePoint(n){return projectPoint(n.sx,n.sy,n.sz)}
function weavePoint(n){return projectPoint(n.wx,n.wy,n.wz)}
function spatialCurve(pa,pb,active,color,baseAlpha=.26,offset=0){if(!active&&pa.z<-.35&&pb.z<-.35)return;const dx=pb.x-pa.x,dy=pb.y-pa.y,len=Math.max(1,Math.hypot(dx,dy)),nx=-dy/len,ny=dx/len,lift=Math.min(58,len*.18)+offset,mx=(pa.x+pb.x)/2+nx*lift,my=(pa.y+pb.y)/2+ny*lift,depth=Math.max(.12,Math.min(1,(pa.z+pb.z+2)/4));ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.quadraticCurveTo(mx,my,pb.x,pb.y);ctx.strokeStyle=active?"rgba(186,61,61,.88)":rgba(color,baseAlpha*(.45+depth*.8));ctx.lineWidth=active?2.4:1.1;ctx.lineCap="round";ctx.stroke()}
function sphereCurve(a,b,active){spatialCurve(spherePoint(a),spherePoint(b),active,"#3b7db8",.16)}
function drawWeave(){drawSphereGuide();const conceptPoints={};for(const n of nodes)if(relationVisible(n)){const p=weavePoint(n);conceptPoints[n.id]=p;n.px=p.x;n.py=p.y;n.pz=p.z;n.pr=(n.ref_ids.length>1?7.5:5.2)*p.scale}const articlePoints={};for(const a of relationArticles){const p=projectPoint(a.x,a.y,a.z);articlePoints[a.id]=p;a.px=p.x;a.py=p.y;a.pz=p.z}for(const n of nodes){if(!relationVisible(n))continue;for(const articleId of n.articles){const pa=conceptPoints[n.id],pb=articlePoints[articleId];if(!pb)continue;const active=selected&&selected.id===n.id;spatialCurve(pa,pb,active,"#b8892e",selected&&!active?.035:.18,(stableHash(n.id+articleId)%13)-6)}}for(const e of DATA.semantic_edges){const a=nodeById[e.source],b=nodeById[e.target];if(!edgeVisible(e)||!relationVisible(a)||!relationVisible(b))continue;const active=selected&&(selected.id===a.id||selected.id===b.id);for(let i=0;i<Math.max(1,e.source_refs.length);i++)spatialCurve(conceptPoints[a.id],conceptPoints[b.id],active,"#3b7db8",selected&&!active?.05:.42,(i-(e.source_refs.length-1)/2)*11)}const marks=[];for(const a of relationArticles)marks.push({kind:"article",z:a.pz,item:a});for(const n of nodes)if(relationVisible(n))marks.push({kind:"concept",z:n.pz,item:n});marks.sort((a,b)=>a.z-b.z);for(const mark of marks){if(mark.kind==="article"){const a=mark.item,depth=(a.pz+1)/2,active=selected&&selected.articles.includes(a.id);ctx.globalAlpha=selected&&!active?.12:.3+depth*.5;ctx.save();ctx.translate(a.px,a.py);ctx.rotate(Math.PI/4);ctx.fillStyle="#b8892e";ctx.fillRect(-3.5,-3.5,7,7);ctx.restore();ctx.globalAlpha=1;continue}const n=mark.item,active=selected&&selected.id===n.id,neighbor=selected&&DATA.semantic_edges.some(e=>edgeVisible(e)&&((e.source===selected.id&&e.target===n.id)||(e.target===selected.id&&e.source===n.id))),dim=selected&&!active&&!neighbor,match=matches(n),depth=(n.pz+1)/2;ctx.globalAlpha=dim?.1:query&&!match?.08:.28+depth*.72;ctx.shadowColor=rgba(COLORS[n.type],.28+depth*.2);ctx.shadowBlur=n.ref_ids.length>1?11:depth*6;ctx.beginPath();ctx.arc(n.px,n.py,Math.max(3,n.pr),0,Math.PI*2);ctx.fillStyle=COLORS[n.type];ctx.fill();ctx.shadowBlur=0;if(n.ref_ids.length>1){ctx.beginPath();ctx.arc(n.px,n.py,n.pr+3.5,0,Math.PI*2);ctx.strokeStyle=active?"#ba3d3d":`rgba(20,91,159,${.28+depth*.42})`;ctx.lineWidth=active?2:1.1;ctx.stroke()}if(active){ctx.beginPath();ctx.arc(n.px,n.py,n.pr+7,0,Math.PI*2);ctx.strokeStyle="#b8892e";ctx.lineWidth=2;ctx.stroke()}if(active||hovered===n||query&&match||degree[n.id]>2){ctx.globalAlpha=1;ctx.font=active?"500 13px system-ui":"12px system-ui";ctx.textAlign="center";ctx.textBaseline="bottom";ctx.fillStyle="#173858";ctx.fillText(n.title.slice(0,14),n.px,n.py-n.pr-8)}ctx.globalAlpha=1}if(selected&&relationVisible(selected))drawProvenance(selected,{x:selected.px,y:selected.py},{x:w/2,y:h/2})}
function drawSphereGuide(){const cx=w/2,cy=h/2,R=Math.min(w,h)*.39*sphereZoom,g=ctx.createRadialGradient(cx-R*.2,cy-R*.25,R*.08,cx,cy,R);g.addColorStop(0,"rgba(255,255,255,.78)");g.addColorStop(.72,"rgba(213,232,247,.22)");g.addColorStop(1,"rgba(20,91,159,.08)");ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);ctx.fillStyle=g;ctx.fill();ctx.strokeStyle="rgba(20,91,159,.18)";ctx.lineWidth=1.3;ctx.stroke();ctx.save();ctx.translate(cx,cy);ctx.rotate(-pitch*.7);for(const ratio of[.35,.65,.9]){ctx.beginPath();ctx.ellipse(0,0,R*ratio,R*ratio*.22,0,0,Math.PI*2);ctx.strokeStyle="rgba(20,91,159,.07)";ctx.lineWidth=1;ctx.stroke()}ctx.restore()}
function drawSphere(){drawSphereGuide();for(const e of DATA.semantic_edges){const a=nodeById[e.source],b=nodeById[e.target];if(!edgeVisible(e)||!enabled.has(a.type)||!enabled.has(b.type))continue;const active=selected&&(selected.id===a.id||selected.id===b.id);sphereCurve(a,b,active)}const projected=nodes.filter(n=>enabled.has(n.type)).map(n=>{const p=spherePoint(n);n.px=p.x;n.py=p.y;n.pz=p.z;n.pr=(n.ref_ids.length>1?5.5:2.6)*p.scale;return n}).sort((a,b)=>a.pz-b.pz);for(const n of projected){const active=selected&&selected.id===n.id,neighbor=selected&&DATA.semantic_edges.some(e=>edgeVisible(e)&&((e.source===selected.id&&e.target===n.id)||(e.target===selected.id&&e.source===n.id))),match=matches(n),depth=(n.pz+1)/2,dim=selected&&!active&&!neighbor;ctx.globalAlpha=dim?.1:query&&!match?.07:.2+depth*.8;ctx.shadowColor=rgba(COLORS[n.type],.28+depth*.18);ctx.shadowBlur=n.ref_ids.length>1?10:depth*5;ctx.beginPath();ctx.arc(n.px,n.py,Math.max(1.8,n.pr),0,Math.PI*2);ctx.fillStyle=COLORS[n.type];ctx.fill();ctx.shadowBlur=0;if(n.ref_ids.length>1){ctx.beginPath();ctx.arc(n.px,n.py,n.pr+3.2,0,Math.PI*2);ctx.strokeStyle=active?"#ba3d3d":`rgba(20,91,159,${.25+depth*.45})`;ctx.lineWidth=active?2:1;ctx.stroke()}if(active){ctx.beginPath();ctx.arc(n.px,n.py,n.pr+7,0,Math.PI*2);ctx.strokeStyle="#b8892e";ctx.lineWidth=2;ctx.stroke()}if(active||hovered===n||query&&match){ctx.globalAlpha=1;ctx.font=active?"500 13px system-ui":"12px system-ui";ctx.textAlign="center";ctx.textBaseline="bottom";ctx.fillStyle="#173858";ctx.fillText(n.title.slice(0,16),n.px,n.py-n.pr-8)}ctx.globalAlpha=1}if(selected&&enabled.has(selected.type))drawProvenance(selected,{x:selected.px,y:selected.py},{x:w/2,y:h/2})}
function drawProvenance(c,anchor,center){const refs=c.ref_ids.map(id=>byRef[id]).filter(Boolean),articleIds=[...new Set(c.articles)],vx=center.x-anchor.x,vy=center.y-anchor.y,len=Math.max(1,Math.hypot(vx,vy)),ux=vx/len,uy=vy/len,tx=-uy,ty=ux,refPos={},articlePos={};refs.forEach((r,i)=>{const spread=(i-(refs.length-1)/2)*18;refPos[r.ref_id]={x:anchor.x+ux*72+tx*spread,y:anchor.y+uy*72+ty*spread}});articleIds.forEach((id,i)=>{const spread=(i-(articleIds.length-1)/2)*24;articlePos[id]={x:anchor.x+ux*145+tx*spread,y:anchor.y+uy*145+ty*spread}});for(const r of refs){const p=refPos[r.ref_id],a=articlePos[r.article_id];ctx.beginPath();ctx.setLineDash([4,4]);ctx.moveTo(anchor.x,anchor.y);ctx.quadraticCurveTo((anchor.x+p.x)/2+tx*6,(anchor.y+p.y)/2+ty*6,p.x,p.y);ctx.strokeStyle="rgba(32,132,113,.72)";ctx.lineWidth=1;ctx.stroke();ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.quadraticCurveTo((p.x+a.x)/2-tx*6,(p.y+a.y)/2-ty*6,a.x,a.y);ctx.strokeStyle="rgba(184,137,46,.72)";ctx.stroke();ctx.setLineDash([])}for(const r of refs){const p=refPos[r.ref_id];ctx.save();ctx.translate(p.x,p.y);ctx.rotate(Math.PI/4);ctx.fillStyle="#208471";ctx.fillRect(-4,-4,8,8);ctx.restore()}for(const id of articleIds){const p=articlePos[id];ctx.fillStyle="#b8892e";ctx.fillRect(p.x-6,p.y-5,12,10);ctx.fillStyle="#173858";ctx.font="11px system-ui";ctx.textAlign="center";ctx.textBaseline="top";ctx.fillText(id,p.x,p.y+8)}}
function renderCatalog(){const visible=nodes.filter(n=>enabled.has(n.type)&&matches(n)).sort((a,b)=>b.articles.length-a.articles.length||b.ref_ids.length-a.ref_ids.length||a.title.localeCompare(b.title)),joint=visible.filter(n=>n.ref_ids.length>1),single=visible.filter(n=>n.ref_ids.length===1);catalog.innerHTML="";const head=document.createElement("header");head.className="catalog-head";const kicker=document.createElement("div");kicker.className="catalog-kicker";kicker.textContent="ALL CONCEPT CATALOG";const title=document.createElement("h2");title.textContent="全部概念查阅";const intro=document.createElement("p");intro.textContent=`${visible.length} 个 Concept，其中 ${joint.length} 个跨文档联合概念；选择条目查看 Concept → ConceptRef → Article 完整溯源。`;head.append(kicker,title,intro);catalog.append(head);if(!visible.length){const empty=document.createElement("div");empty.className="catalog-empty";empty.textContent="当前筛选条件下没有概念。";catalog.append(empty);return}appendGroup("跨文档联合概念",joint,"由多篇 Article 的 ConceptRef 联合编译");appendGroup("单文档概念",single,"由单篇 Article 的 ConceptRef 编译");function appendGroup(label,list,hint){if(!list.length)return;const section=document.createElement("section");section.className="catalog-section";const sectionHead=document.createElement("header");sectionHead.className="catalog-section-head";const h=document.createElement("h3");h.textContent=`${label} · ${list.length}`;const note=document.createElement("span");note.textContent=hint;sectionHead.append(h,note);const grid=document.createElement("div");grid.className="catalog-grid";for(const n of list){const card=document.createElement("button");card.type="button";card.className="catalog-card"+(selected&&selected.id===n.id?" selected":"");card.setAttribute("aria-label",`${n.title}，${n.articles.length} 篇来源文章，${n.ref_ids.length} 个 ConceptRef`);const dot=document.createElement("i");dot.className="catalog-dot";dot.style.background=COLORS[n.type];const cardTitle=document.createElement("span");cardTitle.className="catalog-card-title";cardTitle.textContent=n.title;const meta=document.createElement("span");meta.className="catalog-card-meta";meta.textContent=`${n.type} · ${n.articles.length} 篇 Article · ${n.ref_ids.length} 个 Ref`;card.append(dot,cardTitle,meta);card.onclick=()=>selectNode(n);grid.append(card)}section.append(sectionHead,grid);catalog.append(section)}}
function draw(){const catalogMode=viewMode==="catalog";canvas.style.display=catalogMode?"none":"block";catalog.classList.toggle("show",catalogMode);tooltip.style.display="none";const matchesCount=query?nodes.filter(n=>enabled.has(n.type)&&matches(n)).length:0;if(catalogMode){const visibleCount=nodes.filter(n=>enabled.has(n.type)&&matches(n)).length;renderCatalog();modeBadge.textContent=query?`检索命中 ${visibleCount}`:"全部概念查阅";status.textContent=`显示 ${visibleCount} 个 Concept · 点击查看完整溯源`;summary.innerHTML=`<b>全部概念查阅</b>清单收录 ${DATA.stats.concepts} 个 Concept，其中 ${DATA.stats.multi_source} 个为跨文档联合概念；选择条目后可查阅正文、Ref 证据与来源 Article。`;return}ctx.clearRect(0,0,w,h);if(viewMode==="relations")drawWeave();else drawSphere();modeBadge.textContent=query?`检索命中 ${matchesCount}`:viewMode==="relations"?"三维关系编织网":"三维知识球";status.textContent=viewMode==="relations"?`显示 ${[...relationIds].filter(id=>relationVisible(nodeById[id])).length} 个关联 Concept · ${relationArticles.length} 篇来源 Article · 拖动旋转 · 滚轮缩放`:`显示 ${nodes.filter(n=>enabled.has(n.type)).length} 个 Concept · 拖动旋转 · 滚轮缩放`;summary.innerHTML=viewMode==="relations"?`<b>三维关系编织网</b>Concept 均匀分布在外层，来源 Article 位于内层；蓝线表示语义关系，金线表示真实来源，旋转可观察交叉深度。`:`<b>三维知识球</b>${DATA.stats.multi_source} 个多来源 Concept 位于内核，其余 Concept 均匀充盈整个球面；默认仅显示已确认的语义关系。`}
function pick(x,y){const list=nodes.filter(n=>viewMode==="relations"?relationVisible(n):enabled.has(n.type)).map(n=>({n,p:{x:n.px,y:n.py},z:n.pz})).sort((a,b)=>b.z-a.z);for(const item of list){const radius=viewMode==="relations"?Math.max(10,item.n.pr+6):Math.max(8,item.n.pr+5);if(Math.hypot(item.p.x-x,item.p.y-y)<=radius)return item.n}return null}
function showTooltip(n,x,y){if(!n){tooltip.style.display="none";return}tooltip.innerHTML="";const b=document.createElement("b");b.textContent=n.title;const s=document.createElement("span");s.textContent=`${n.type} · ${n.ref_ids.length} 个 Ref · ${n.articles.length} 篇来源`;tooltip.append(b,s);tooltip.style.display="block";tooltip.style.left=Math.min(w-275,x+14)+"px";tooltip.style.top=Math.min(h-70,y+14)+"px"}
function selectNode(n){selected=n;detail.classList.add("open");renderDetail(n);draw()}
function renderDetail(c){const refs=c.ref_ids.map(id=>byRef[id]).filter(Boolean),relations=DATA.semantic_edges.filter(e=>e.source===c.id||e.target===c.id);detail.innerHTML="";const h=document.createElement("h2");h.textContent=c.title;detail.append(h);const meta=document.createElement("div");meta.className="meta";for(const text of[c.type,c.ref_ids.length>1?`${c.articles.length} 篇联合编译`:"单来源",`${refs.length} 个 Ref`,`${relations.length} 组关系`]){const s=document.createElement("span");s.className="pill";s.textContent=text;meta.append(s)}detail.append(meta);const desc=document.createElement("div");desc.className="description";desc.textContent=c.description;detail.append(desc);appendTitle("Concept 正文");const body=document.createElement("div");body.className="body";body.textContent=c.body;detail.append(body);if(relations.length){appendTitle("关联 Concept");for(const e of relations){const other=byConcept[e.source===c.id?e.target:e.source],row=document.createElement("div");row.className="related";const b=document.createElement("b");b.textContent=other.title;const count=document.createElement("span");count.className="edge-count";count.textContent=`${(e.relation_types||["related"]).map(t=>RELATION_LABELS[t]||t).join(" / ")} · ${e.source_refs.length} 条 Ref 依据`;b.append(count);row.append(b);const sm=document.createElement("small");sm.textContent=e.reasons.join("；");row.append(sm);row.onclick=()=>selectNode(nodeById[other.id]);detail.append(row)}}appendTitle("ConceptRef 证据");for(const r of refs){const box=document.createElement("div");box.className="ref";const b=document.createElement("b");b.textContent=r.title;box.append(b);const sm=document.createElement("small"),path=(r.section_path||[]).join(" › "),pages=r.page_start?(r.page_end&&r.page_end!==r.page_start?`第 ${r.page_start}–${r.page_end} 页`:`第 ${r.page_start} 页`):"";sm.textContent=[r.ref_id,path,pages].filter(Boolean).join(" · ");box.append(sm);if(r.evidence?.length){const q=document.createElement("div");q.className="quote";q.textContent=r.evidence[0];box.append(q)}detail.append(box)}appendTitle("来源 Article");for(const id of c.articles){const a=byArticle[id],row=document.createElement("div");row.className="article";const code=document.createElement("span");code.className="article-id";code.textContent=id;const title=document.createElement("span");title.textContent=a?.title||id;row.append(code,title);detail.append(row)}function appendTitle(text){const x=document.createElement("h3");x.textContent=text;detail.append(x)}}
function focusSphere(n){const targetYaw=Math.atan2(-n.sx,n.sz),cy=Math.cos(targetYaw),sy=Math.sin(targetYaw),z1=-n.sx*sy+n.sz*cy;yaw=targetYaw;pitch=Math.atan2(n.sy,Math.max(.01,z1))}
canvas.addEventListener("mousedown",e=>{drag=true;moved=false;last=[e.clientX,e.clientY];canvas.classList.add("dragging");tooltip.style.display="none"});window.addEventListener("mouseup",()=>{drag=false;last=null;canvas.classList.remove("dragging")});window.addEventListener("mousemove",e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(drag){const dx=e.clientX-last[0],dy=e.clientY-last[1];if(Math.abs(dx)+Math.abs(dy)>3)moved=true;yaw+=dx*.008;pitch=Math.max(-1.15,Math.min(1.15,pitch+dy*.006));last=[e.clientX,e.clientY];draw();return}const next=pick(x,y);if(next!==hovered){hovered=next;canvas.style.cursor=next?"pointer":"grab";draw()}showTooltip(next,x,y)});canvas.addEventListener("mouseleave",()=>{hovered=null;tooltip.style.display="none";draw()});canvas.addEventListener("wheel",e=>{e.preventDefault();sphereZoom=Math.max(.55,Math.min(1.55,sphereZoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});canvas.addEventListener("click",e=>{if(moved)return;const r=canvas.getBoundingClientRect(),n=pick(e.clientX-r.left,e.clientY-r.top);if(n)selectNode(n)});
const types=document.getElementById("types");for(const[type,count]of Object.entries(DATA.stats.types)){const row=document.createElement("label");row.className="type-row";row.innerHTML=`<input type="checkbox" checked><span><i class="swatch" style="background:${COLORS[type]}"></i>${type}</span><small>${count}</small>`;const input=row.querySelector("input");input.onchange=()=>{input.checked?enabled.add(type):enabled.delete(type);draw()};types.append(row)}
const relationTypes=document.getElementById("relationTypes");for(const[type,count]of Object.entries(DATA.stats.relation_types||{related:DATA.stats.relations})){const row=document.createElement("label");row.className="type-row";row.innerHTML=`<input type="checkbox" checked><span>${RELATION_LABELS[type]||type}</span><small>${count}</small>`;const input=row.querySelector("input");input.onchange=()=>{input.checked?enabledRelations.add(type):enabledRelations.delete(type);draw()};relationTypes.append(row)}
document.getElementById("viewMode").onchange=e=>{viewMode=e.target.value;selected=null;hovered=null;detail.classList.remove("open");resetDetail();fit()};document.getElementById("search").oninput=e=>{query=e.target.value.trim();if(query){const found=nodes.find(n=>enabled.has(n.type)&&matches(n));if(found){if(viewMode==="relations"&&!relationIds.has(found.id)){viewMode="sphere";document.getElementById("viewMode").value="sphere"}if(viewMode==="sphere")focusSphere(found);selected=found;renderDetail(found);detail.classList.add("open")}}draw()};document.getElementById("fit").onclick=fit;document.getElementById("reset").onclick=()=>{selected=null;hovered=null;query="";document.getElementById("search").value="";detail.classList.remove("open");resetDetail();fit()};function resetDetail(){detail.innerHTML='<div class="empty"><div><div class="empty-mark">关系</div><div>选择一个 Concept</div><small>查看正文、关联依据和来源证据</small></div></div>'}
document.getElementById("mArticles").textContent=DATA.stats.articles;document.getElementById("mRefs").textContent=DATA.stats.refs;document.getElementById("mConcepts").textContent=DATA.stats.concepts;document.getElementById("mRelations").textContent=DATA.stats.relations;
</script>
</body>
</html>"""
