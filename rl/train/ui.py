"""Tiny local web UI to watch the trained agent play.

Pick a checkpoint, choose a scenario (random or a specific encounter, optionally
normal-only), set the search budget, and see a turn-by-turn transcript of the greedy
agent. Stdlib only (no Flask) -- run it and open the printed URL.

    PYTHONPATH=. python -m rl.train.ui            # then open http://localhost:8000

The dataset (~1.8M fights) and each net are loaded once and cached in-process.
"""

import os
import glob
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from ..core.scenario import DatasetSampler, CombatConfig, NON_NORMAL_ENCOUNTERS
from ..core.session import CombatSession
from ..algos.mcts import MCTS, MCTSConfig
from ..algos.net import CombatNet, NeuralEvaluator
from .playthrough import play_structured

_LOCK = threading.Lock()      # MCTS/torch are single-threaded here; serialize requests
_FIGHTS = None
_NET_CACHE = {}               # ckpt path -> (net, d_model)


# ----- data / model loading (cached) ----------------------------------------

def _fights():
    global _FIGHTS
    if _FIGHTS is None:
        path = "data/ironclad_a0_fights.json.gz"
        _FIGHTS = DatasetSampler.from_gzip(path, rng=random.Random(0)).fights
    return _FIGHTS


def _find_checkpoints():
    paths = set()
    for pat in ("*.pt", "checkpoints*/*.pt", "checkpoints_from_ec2/*.pt"):
        paths.update(glob.glob(pat))
    return sorted(paths)


def _infer_d_model(sd) -> int:
    # state_mlp first linear: weight shape (d_model, d_model*8)
    w = sd.get("state_mlp.0.weight")
    return int(w.shape[0]) if w is not None else 128


def _load_net(ckpt):
    if ckpt not in _NET_CACHE:
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("net", sd)
        d = _infer_d_model(sd)
        net = CombatNet(d_model=d)
        net.load_state_dict(sd)
        net.eval()
        _NET_CACHE[ckpt] = (net, d)
    return _NET_CACHE[ckpt]


def _encounter_list():
    seen = {}
    for f in _fights():
        seen.setdefault(f["enemies"], 0)
        seen[f["enemies"]] += 1
    out = []
    for name in sorted(seen):
        tier = "boss/elite" if name in NON_NORMAL_ENCOUNTERS else "normal"
        out.append({"name": name, "tier": tier, "count": seen[name]})
    return out


# ----- play a scenario -------------------------------------------------------

def _run(ckpt, sims, mode, encounter, seed, normal_only):
    net, d = _load_net(ckpt)
    evaluator = NeuralEvaluator(net, device="cpu")
    mcts = MCTS(evaluator, MCTSConfig(num_simulations=int(sims), temperature=0.0,
                                      add_root_noise=False, seed=int(seed)))
    pool = _fights()
    if mode == "encounter" and encounter:
        pool = [f for f in pool if f["enemies"] == encounter]
    elif normal_only:
        pool = [f for f in pool if f["enemies"] not in NON_NORMAL_ENCOUNTERS]
    if not pool:
        return {"error": "no fights match that filter"}
    sampler = DatasetSampler(pool, rng=random.Random(int(seed)))
    result = play_structured(mcts, CombatSession(sampler=sampler))
    result["model"] = {"checkpoint": ckpt, "d_model": d, "sims": int(sims)}
    return result


# ----- HTTP ------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            return self._send(200, _PAGE, "text/html")
        if self.path == "/api/init":
            return self._send(200, json.dumps({
                "checkpoints": _find_checkpoints(),
                "encounters": _encounter_list(),
            }))
        self._send(404, "{}")

    def do_POST(self):
        if self.path != "/api/play":
            return self._send(404, "{}")
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or "{}")
        try:
            with _LOCK:
                out = _run(req.get("ckpt"), req.get("sims", 64), req.get("mode", "random"),
                           req.get("encounter"), req.get("seed", 0),
                           bool(req.get("normal_only", True)))
            self._send(200, json.dumps(out))
        except (RuntimeError, KeyError) as e:
            # Most often an old-architecture checkpoint (pre-v5 observation/net layout).
            self._send(200, json.dumps({"error":
                "Could not load this checkpoint into the current network -- it is likely "
                "from an older observation/architecture version. Use a current model "
                f"(e.g. net_v5_final.pt). [{e}]"}))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(200, json.dumps({"error": str(e)}))


_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>STS Agent Viewer</title><style>
 body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#15171c;color:#dde}
 header{background:#1d2027;padding:14px 20px;border-bottom:1px solid #333;display:flex;
   gap:14px;align-items:center;flex-wrap:wrap;position:sticky;top:0}
 h1{font-size:16px;margin:0 16px 0 0;color:#fff}
 select,input,button{background:#262a33;color:#dde;border:1px solid #3a3f4b;border-radius:6px;
   padding:6px 9px;font:inherit}
 button{background:#3b6fd6;border-color:#3b6fd6;color:#fff;cursor:pointer;font-weight:600}
 button:disabled{opacity:.5;cursor:wait}
 label{font-size:12px;color:#9aa;margin-right:4px}
 #main{padding:18px 22px;max-width:1000px;margin:0 auto}
 .setup{background:#1d2027;border:1px solid #2c313b;border-radius:10px;padding:14px 18px;margin-bottom:16px}
 .setup b{color:#fff} .k{color:#8af} .pill{display:inline-block;background:#2a2f3a;border-radius:20px;
   padding:2px 10px;margin:2px;font-size:12px}
 .turn{background:#1d2027;border:1px solid #2c313b;border-radius:10px;padding:12px 16px;margin-bottom:12px}
 .turnh{font-weight:700;color:#fff;border-bottom:1px solid #2c313b;padding-bottom:6px;margin-bottom:8px}
 .row{margin:3px 0} .lab{color:#8aa;display:inline-block;min-width:74px}
 .enemy{margin:2px 0 2px 10px;color:#f9c} .intent-atk{color:#f87} .intent-non{color:#8c9}
 .act{margin:2px 0 2px 12px;color:#bdf} .end{color:#fc8}
 .st{color:#9c8;font-size:12px} .pst{color:#8cf;font-size:12px}
 .win{color:#7e7;font-weight:700} .loss{color:#f77;font-weight:700}
 .muted{color:#889}
</style></head><body>
<header>
 <h1>STS Agent Viewer</h1>
 <span><label>Model</label><select id=ckpt></select></span>
 <span><label>Scenario</label><select id=enc></select></span>
 <span><label>Sims</label><select id=sims>
   <option>32</option><option selected>64</option><option>128</option><option>256</option></select></span>
 <span><label>Seed</label><input id=seed type=number value=0 style=width:64px></span>
 <button id=go onclick=play()>Play</button>
 <span id=status class=muted></span>
</header>
<div id=main><div class=muted>Pick a model and scenario, then Play.</div></div>
<script>
let enc=[];
async function init(){
 const r=await fetch('/api/init').then(x=>x.json());
 const cs=document.getElementById('ckpt');
 r.checkpoints.forEach(c=>cs.add(new Option(c,c)));
 enc=r.encounters;
 const es=document.getElementById('enc');
 es.add(new Option('Random (normal only)','__normal__'));
 es.add(new Option('Random (any encounter)','__any__'));
 enc.forEach(e=>es.add(new Option(`${e.name}  [${e.tier}]`,e.name)));
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function play(){
 const ck=document.getElementById('ckpt').value;
 const e=document.getElementById('enc').value;
 const body={ckpt:ck,sims:+document.getElementById('sims').value,seed:+document.getElementById('seed').value};
 if(e==='__normal__'){body.mode='random';body.normal_only=true;}
 else if(e==='__any__'){body.mode='random';body.normal_only=false;}
 else{body.mode='encounter';body.encounter=e;}
 const go=document.getElementById('go'),st=document.getElementById('status');
 go.disabled=true;st.textContent='thinking...';
 try{
  const r=await fetch('/api/play',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)}).then(x=>x.json());
  render(r);
 }catch(err){document.getElementById('main').innerHTML='<div class=loss>'+esc(''+err)+'</div>';}
 go.disabled=false;st.textContent='';
}
function render(r){
 const m=document.getElementById('main');
 if(r.error){m.innerHTML='<div class=loss>'+esc(r.error)+'</div>';return;}
 const s=r.setup;let h='';
 h+='<div class=setup>';
 h+=`<div class=row><span class=lab>Enemy</span><b>${esc(s.encounter)}</b></div>`;
 h+=`<div class=row><span class=lab>Player HP</span>${s.player_hp}/${s.player_max_hp} `
   +`<span class=muted>(human entered ${s.human_entering_hp}, lost ${s.human_hp_loss})</span></div>`;
 h+=`<div class=row><span class=lab>Energy/turn</span>${s.energy_per_turn}</div>`;
 h+=`<div class=row><span class=lab>Relics</span>${s.relics.map(x=>'<span class=pill>'+esc(x)+'</span>').join('')||'<span class=muted>none</span>'}</div>`;
 h+=`<div class=row><span class=lab>Potions</span>${s.potions.map(x=>'<span class=pill>'+esc(x)+'</span>').join('')||'<span class=muted>none</span>'}</div>`;
 h+=`<div class=row><span class=lab>Deck (${s.deck_size})</span>${esc(s.deck)}</div>`;
 h+=`<div class=row muted>model ${esc(r.model.checkpoint)} · d_model ${r.model.d_model} · ${r.model.sims} sims</div>`;
 h+='</div>';
 r.turns.forEach(t=>{
  h+='<div class=turn>';
  h+=`<div class=turnh>Turn ${t.turn}</div>`;
  h+=`<div class=row><span class=lab>Player</span>HP ${t.player_hp}/${t.player_max_hp} · block ${t.block} · energy ${t.energy}/${t.energy_per_turn}`;
  if(t.statuses.length)h+=` <span class=pst>[${t.statuses.map(esc).join(', ')}]</span>`;
  h+='</div>';
  h+=`<div class=row><span class=lab>Hand</span>${esc(t.hand)} <span class=muted>· draw ${t.draw} discard ${t.discard} exhaust ${t.exhaust}</span></div>`;
  h+='<div class=row><span class=lab>Enemies</span></div>';
  t.enemies.forEach(en=>{
   const atk=en.intent.startsWith('ATTACK');
   h+=`<div class=enemy>${esc(en.name)}: HP ${en.hp}/${en.max_hp}`+(en.block?` block ${en.block}`:'')
     +` · <span class=${atk?'intent-atk':'intent-non'}>${esc(en.intent)}</span>`
     +(en.statuses.length?` <span class=st>[${en.statuses.map(esc).join(', ')}]</span>`:'')+'</div>';
  });
  h+='<div class=row><span class=lab>Plays</span></div>';
  t.actions.forEach(a=>{
   const end=a.startsWith('End turn');
   h+=`<div class="act ${end?'end':''}">${end?'■ ':'▸ '}${esc(a)}</div>`;
  });
  h+='</div>';
 });
 const res=r.result;
 h+=`<div class=setup><span class="${res.won?'win':'loss'}">${res.won?'VICTORY':(res.done?'DEFEAT':'truncated')}</span>`
  +` — final HP ${res.final_hp}/${res.max_hp} · lost ${res.hp_lost} this combat `
  +`<span class=muted>(human lost ${res.human_hp_loss})</span></div>`;
 m.innerHTML=h;
}
init();
</script></body></html>"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"Loading dataset...", flush=True)
    _fights()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  STS Agent Viewer running at  http://localhost:{args.port}\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
