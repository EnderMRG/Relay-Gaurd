"""
RelayGuard -- Dashboard page (loaded via st.navigation in app.py)
"""
import streamlit as st
import plotly.graph_objects as go
import math, random, time
from datetime import datetime

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""<style>
footer { visibility: hidden; }
[data-testid="stAppViewContainer"]  { background: #0d1117; }
[data-testid="stSidebar"]           { background: #161b22 !important;
    border-right: 1px solid rgba(255,255,255,0.1); }
.block-container { padding: 4.5rem 1.5rem 1rem; }
p, div, span, label { color: #c9d1d9; }
hr { border-color: rgba(255,255,255,0.1) !important; }
.stButton > button {
    background: #21262d; color: #c9d1d9;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px; width: 100%;
}
.stButton > button:hover { background: rgba(255,255,255,0.06); }
@keyframes glow-orange {
    0%,100% { box-shadow: 0 0 6px rgba(210,153,34,0.3); }
    50%      { box-shadow: 0 0 18px rgba(210,153,34,0.7); }
}
@keyframes glow-red {
    0%,100% { box-shadow: 0 0 8px rgba(218,54,51,0.4); }
    50%      { box-shadow: 0 0 28px rgba(218,54,51,0.85); }
}
@keyframes glow-dark {
    0%,100% { box-shadow: 0 0 10px rgba(248,81,73,0.5); }
    50%      { box-shadow: 0 0 35px rgba(248,81,73,1.0); }
}
</style>""", unsafe_allow_html=True)

# ── Physics ────────────────────────────────────────────────────────────────────
W_MAX=5000.; T_MAX=105.; R_MAX=0.20; N_MAX=100_000.
EA=0.7; KB=8.617e-5; T_REF=298.15

def arrhenius(Tc):
    return math.exp(min(50., (EA/KB)*(1/T_REF - 1/(Tc+273.15))))

def compute_hi(r):
    d=max(0.,1-r['W']/W_MAX); t=max(0.,1-r['T']/T_MAX)
    rs=max(0.,1-r['R']/R_MAX); c=max(0.,1-r['N']/N_MAX)
    return min(100., max(0., (0.4*d+0.25*t+0.25*rs+0.1*c)*100))

def alarm_lvl(hi):
    return 0 if hi>70 else 1 if hi>40 else 2 if hi>20 else 3

LABEL = ['Healthy','Warning','Critical','Failure Imminent']
EMO   = ['\u2705','\u26a0\ufe0f','\U0001f6a8','\U0001f480']
COL   = ['#2ea043','#d29922','#da3633','#f85149']
BG    = ['rgba(46,160,67,.15)','rgba(210,153,34,.15)',
         'rgba(218,54,51,.15)','rgba(248,81,73,.15)']
ANIM  = ['','animation:glow-orange 2s infinite;',
         'animation:glow-red 1.4s infinite;',
         'animation:glow-dark 0.8s infinite;']

def hcol(hi):
    return '#2ea043' if hi>70 else '#d29922' if hi>40 else '#da3633' if hi>20 else '#f85149'

# ── State init ─────────────────────────────────────────────────────────────────
def make_relay(id_, label, loc, W, N, T, R, hi_ov=None):
    r=dict(id=id_,label=label,loc=loc,W=float(W),N=float(N),T=float(T),
           R=float(R),I=5+random.random(),sim_t=0,
           anomaly=False,alarm=0,HI=0.,wo_done=False,hi_h=[],r_h=[])
    r['HI']=float(hi_ov) if hi_ov else compute_hi(r)
    r['alarm']=alarm_lvl(r['HI'])
    for _ in range(80):
        r['hi_h'].append(r['HI']+random.gauss(0,.3))
        r['r_h'].append(r['R']*1000+random.gauss(0,.05))
    return r

def boot():
    ss=st.session_state
    if ss.get('rdy'): return
    ss.rl={
        'K1':make_relay('K1','Main Contactor','Panel A', 180, 6000,42,0.008,93.),
        'K2':make_relay('K2','Load Switch',   'Panel B',2800,45000,68,0.065,55.),
        'K3':make_relay('K3','HVAC Relay',    'Panel C',  50, 1800,38,0.003,98.),
        'K4':make_relay('K4','Exhaust Fan',   'Panel D',3900,72000,78,0.135,28.),
    }
    ss.rl['K2']['anomaly']=True; ss.rl['K2']['alarm']=1
    ss.rl['K4']['alarm']=2; ss.rl['K4']['wo_done']=True
    ss.sel='K3'; ss.mode='idle'; ss.tick=0; ss.speed=3
    ss.wo_cnt=2; ss.events=[]
    ss.WOs=[dict(id='WO-0001',rid='K4',label='Exhaust Fan',loc='Panel D',
                 hi=28.,alarm='\U0001f6a8 Critical',eta='~19 h',
                 ts=datetime.now().strftime('%H:%M:%S'),acked=False)]
    ss.rdy=True

boot()
ss=st.session_state; rl=ss.rl

# ── Simulation ─────────────────────────────────────────────────────────────────
def sim_tick():
    r=rl['K3']; mode=ss.mode
    if mode!='idle':
        r['sim_t']+=1; ph=r['sim_t']/500.
        rate={'normal':.4,'stress':1.5}.get(mode,.4)
        r['I']=max(.5,(5+3*ph+random.gauss(0,.4))*(1.4 if mode=='stress' else 1))
        r['T']=min(T_MAX-1,40+35*(ph**1.4)+random.gauss(0,1.5)+(15 if mode=='stress' else 0))
        E=r['I']**2*230
        r['W']=min(W_MAX,r['W']+E*arrhenius(r['T'])*0.0002*rate)
        r['N']=r['N']+3*rate; r['R']=max(0.001,0.001+1e-7*(r['N']**.55))
        prev=r['HI']; r['HI']=compute_hi(r); delta=prev-r['HI']
        if not r['anomaly'] and r['HI']>70 and delta>0.05:
            r['anomaly']=True
            push(f'\U0001f916 LSTM Anomaly \u2014 K3: \u0394HI = \u2212{delta:.3f}%/tick \u2014 AI alert BEFORE rule threshold!')
        new_a=alarm_lvl(r['HI'])
        if new_a>r['alarm']:
            r['alarm']=new_a
            push(f'{EMO[new_a]} K3 \u2192 {LABEL[new_a]}  (HI = {r["HI"]:.1f}%)')
        if r['alarm']>=2 and not r['wo_done']:
            r['wo_done']=True
            eta=max(1,int(r['HI']/1.5))
            ss.WOs.append(dict(id=f'WO-{ss.wo_cnt:04d}',rid='K3',label='HVAC Relay',
                               loc='Panel C',hi=r['HI'],
                               alarm=f"{EMO[r['alarm']]} {LABEL[r['alarm']]}",
                               eta=f'~{eta} h',ts=datetime.now().strftime('%H:%M:%S'),
                               acked=False))
            ss.wo_cnt+=1
    r['hi_h'].append(r['HI']); r['r_h'].append(r['R']*1000)
    if len(r['hi_h'])>100: r['hi_h'].pop(0)
    if len(r['r_h'])>100:  r['r_h'].pop(0)
    for rid,(lo,hi) in {'K1':(85,97),'K2':(38,65),'K4':(15,32)}.items():
        rel=rl[rid]
        rel['HI']=max(lo,min(hi,rel['HI']+random.gauss(0,.04)))
        rel['alarm']=alarm_lvl(rel['HI'])
        rel['hi_h'].append(rel['HI']); rel['r_h'].append(rel['R']*1000+random.gauss(0,.05))
        if len(rel['hi_h'])>100: rel['hi_h'].pop(0)
        if len(rel['r_h'])>100:  rel['r_h'].pop(0)
    ss.tick+=1

def push(msg):
    ss.events.insert(0,{'ts':datetime.now().strftime('%H:%M:%S'),'msg':msg})
    if len(ss.events)>5: ss.events.pop()

# ── Chart ──────────────────────────────────────────────────────────────────────
def chart_hi(r):
    x=list(range(len(r['hi_h']))); y=r['hi_h']
    lc=hcol(r['HI'])
    fc=BG[r['alarm']].replace('.15','.10')
    fig=go.Figure()
    fig.add_hrect(y0=70,y1=104,fillcolor='rgba(46,160,67,.05)',line_width=0)
    fig.add_hrect(y0=40,y1=70, fillcolor='rgba(210,153,34,.06)',line_width=0)
    fig.add_hrect(y0=20,y1=40, fillcolor='rgba(218,54,51,.08)',line_width=0)
    fig.add_hrect(y0=0, y1=20, fillcolor='rgba(248,81,73,.12)',line_width=0)
    fig.add_trace(go.Scatter(x=x,y=y,mode='lines',name='Health Index %',
        line=dict(color=lc,width=3),
        fill='tozeroy',fillcolor=fc,
        hovertemplate='HI: %{y:.1f}%<extra></extra>'))
    if r['anomaly']:
        pts=[(i,v) for i,v in enumerate(y) if 67<v<85]
        if pts:
            ax2,ay2=pts[0]
            fig.add_trace(go.Scatter(
                x=[ax2],y=[ay2],mode='markers',name='\U0001f916 LSTM Anomaly',
                marker=dict(symbol='triangle-up',size=20,color='#f85149',
                            line=dict(color='white',width=2)),
                hovertemplate='<b>\U0001f916 LSTM Anomaly</b><br>AI flag fires BEFORE rule threshold<extra></extra>'))
    for yv,col,lbl in [(70,'rgba(210,153,34,0.9)','70% Warn'),
                        (40,'rgba(218,54,51,0.9)','40% Crit'),
                        (20,'rgba(248,81,73,0.8)','20% Fail')]:
        fig.add_hline(y=yv,line=dict(dash='dot',color=col,width=1.5),
                      annotation_text=lbl,
                      annotation_font=dict(color=col,size=9),
                      annotation_position='right')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b949e',size=10),
        margin=dict(l=35,r=68,t=6,b=6),height=300,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#21262d',bordercolor='rgba(255,255,255,0.1)'),
        xaxis=dict(showticklabels=False,gridcolor='rgba(255,255,255,0.04)'),
        yaxis=dict(range=[0,104],ticksuffix='%',gridcolor='rgba(255,255,255,0.04)'),
        legend=dict(x=0.01,y=0.99,bgcolor='rgba(0,0,0,0)',font=dict(size=10)),
    )
    return fig

# ── HTML helpers ───────────────────────────────────────────────────────────────
def fleet_card(r):
    al=r['alarm']; col=hcol(r['HI'])
    sel=r['id']==ss.sel
    bdr=f"2px solid #1f6feb" if sel else f"1px solid {col}55"
    shd=f"0 0 0 2px #1f6feb40" if sel else ''
    return f"""
<div style="background:#161b22;border:{bdr};border-radius:10px;
  padding:14px 12px;text-align:center;{ANIM[al]}box-shadow:{shd}">
  <div style="font-size:10px;color:#8b949e;letter-spacing:.5px">
    <b style="color:{col}">{r['id']}</b> &middot; {r['label']}</div>
  <div style="font-size:36px;font-weight:800;color:{col};
    font-family:monospace;line-height:1.1;margin:4px 0">{r['HI']:.0f}%</div>
  <div style="font-size:11px;font-weight:600;color:{col}">{EMO[al]} {LABEL[al]}</div>
  <div style="background:rgba(255,255,255,0.06);height:4px;border-radius:2px;margin-top:8px">
    <div style="background:{col};height:4px;border-radius:2px;width:{r['HI']:.0f}%"></div>
  </div>
</div>"""

def trend_str(r):
    h=r['hi_h']
    if len(h)<5: return 'Stable','#8b949e'
    d=h[-1]-h[-5]
    if d < -0.4: return '\u2198 Declining','#da3633'
    if d >  0.1: return '\u2197 Recovering','#2ea043'
    return '\u2192 Stable','#8b949e'

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    mc={'idle':'#8b949e','normal':'#d29922','stress':'#da3633'}.get(ss.mode,'#8b949e')
    st.markdown(f"""
<div style="text-align:center;padding:8px 0 4px">
  <div style="font-size:26px">&#9889;</div>
  <div style="font-size:18px;font-weight:800;color:#e6edf3">RelayGuard</div>
  <div style="font-size:10px;color:#8b949e;margin-top:2px">Predictive Maintenance</div>
  <div style="font-size:11px;font-weight:700;color:{mc};margin-top:6px">
    &#9679; {ss.mode.upper()}</div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Demo Target: Relay K3**")
    if st.button("Run Demo",      use_container_width=True, type="primary"):  ss.mode='normal'
    if st.button("Inject Stress", use_container_width=True):                  ss.mode='stress'

    c1,c2 = st.columns(2)
    with c1:
        if st.button("Overload", use_container_width=True):
            r3=rl['K3']
            r3['W']=min(W_MAX,r3['W']+400); r3['T']=min(T_MAX-1,r3['T']+20)
            r3['R']+=0.015; r3['HI']=compute_hi(r3)
            r3['alarm']=alarm_lvl(r3['HI']); ss.mode='stress'
    with c2:
        if st.button("Hot Spell", use_container_width=True):
            rl['K3']['T']=min(T_MAX-1,rl['K3']['T']+25)

    if st.button("Reset Demo", use_container_width=True):
        r3=rl['K3']
        r3.update(W=50.,N=1800.,T=38.,R=0.003,HI=98.,alarm=0,
                  anomaly=False,wo_done=False,sim_t=0,I=5.2)
        r3['hi_h']=[98+random.gauss(0,.3) for _ in range(80)]
        r3['r_h'] =[3.0+random.gauss(0,.05) for _ in range(80)]
        ss.mode='idle'; ss.events=[]
        ss.WOs=[w for w in ss.WOs if w['rid']!='K3']

    st.markdown("---")
    st.markdown("**Speed**")
    s1,s2,s3=st.columns(3)
    for col,v,l in [(s1,1,'1x'),(s2,3,'3x'),(s3,10,'10x')]:
        with col:
            t="primary" if ss.speed==v else "secondary"
            if st.button(l,use_container_width=True,key=f'sp{v}',type=t): ss.speed=v

    st.markdown("---")
    st.markdown("**Select Relay**")
    for rid,r in rl.items():
        icon="> " if rid==ss.sel else "  "
        lbl=f"{icon}{rid}  {r['HI']:.0f}%  {LABEL[r['alarm']]}"
        if st.button(lbl, use_container_width=True, key=f'sr{rid}'):
            ss.sel=rid

    st.markdown("---")
    st.markdown(f"<div style='font-size:10px;color:#484f58'>Tick {ss.tick} "
                f"&middot; WOs open: {sum(1 for w in ss.WOs if not w['acked'])}</div>",
                unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────────────────
ha,hb=st.columns([5,2])
with ha:
    avg=sum(r['HI'] for r in rl.values())/4
    alerts=sum(1 for r in rl.values() if r['alarm']>0)
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px">
  <div style="background:linear-gradient(135deg,#1f6feb,#0d419d);
    width:38px;height:38px;border-radius:9px;display:flex;align-items:center;
    justify-content:center;font-size:20px;flex-shrink:0">&#9889;</div>
  <div>
    <span style="font-size:20px;font-weight:800;color:#e6edf3">RelayGuard</span>
    <span style="font-size:11px;color:#8b949e;margin-left:10px">
      Fleet avg {avg:.0f}% &middot; {alerts} alert{'s' if alerts!=1 else ''} active
    </span>
  </div>
</div>""", unsafe_allow_html=True)
with hb:
    st.markdown(f"<div style='text-align:right;padding-top:6px;font-size:11px;"
                f"color:#8b949e'>&#x1F7E2; MODBUS &middot; MQTT &middot; "
                f"{datetime.now().strftime('%H:%M:%S')}</div>",
                unsafe_allow_html=True)

st.markdown("<hr style='margin:8px 0 12px'>", unsafe_allow_html=True)

# Work-order banner
open_wos=[w for w in ss.WOs if not w['acked']]
if open_wos:
    wo=open_wos[-1]
    hc2=hcol(wo['hi'])
    ba,bb=st.columns([6,1])
    with ba:
        st.markdown(f"""
<div style="background:rgba(218,54,51,.12);border:1px solid rgba(218,54,51,.5);
  border-radius:10px;padding:10px 16px;display:flex;align-items:center;gap:12px">
  <span style="font-size:22px">&#x1F6A8;</span>
  <div>
    <b style="color:#f85149">{wo['id']}</b>
    <span style="color:#c9d1d9;margin-left:8px">
      Replace {wo['label']} ({wo['rid']}) &mdash; {wo['loc']}</span><br>
    <span style="font-size:12px;color:#8b949e">
      HI: <b style="color:{hc2}">{wo['hi']:.1f}%</b> &middot;
      {wo['alarm']} &middot; Est. {wo['eta']} to failure</span>
  </div>
</div>""", unsafe_allow_html=True)
    with bb:
        if st.button("Acknowledge", use_container_width=True, key=f"ack{wo['id']}"):
            wo['acked']=True; st.rerun()
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Fleet cards
cols=st.columns(4)
for i,(rid,r) in enumerate(rl.items()):
    with cols[i]:
        st.markdown(fleet_card(r), unsafe_allow_html=True)
        if st.button("Monitor", key=f"mon{rid}", use_container_width=True):
            ss.sel=rid

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Selected relay panel
sel=rl[ss.sel]; sc=hcol(sel['HI']); al=sel['alarm']
ts,tc=trend_str(sel)
hi_h=sel['hi_h']; delta=(hi_h[-1]-hi_h[-2]) if len(hi_h)>1 else 0.
dc='#da3633' if delta<-0.01 else '#2ea043'

if sel['anomaly']:
    st.markdown(f"""
<div style="background:rgba(248,81,73,.1);border:1px dashed rgba(248,81,73,.6);
  border-radius:9px;padding:9px 16px;margin-bottom:10px;
  display:flex;align-items:center;gap:10px">
  <span style="font-size:20px">&#x1F916;</span>
  <div>
    <b style="color:#f85149">LSTM Anomaly Detected</b>
    <span style="color:#8b949e;font-size:12px;margin-left:8px">
      AI alert fired BEFORE the 70% rule-based threshold &mdash;
      this is the early-warning advantage</span>
  </div>
</div>""", unsafe_allow_html=True)

left, right = st.columns([2, 3])

with left:
    st.markdown(f"""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.1);
  border-radius:12px;padding:20px;height:100%">
  <div style="font-size:10px;color:#8b949e;text-transform:uppercase;
    letter-spacing:.8px">Live Monitor</div>
  <div style="font-size:16px;font-weight:700;color:#1f6feb;
    font-family:monospace;margin:2px 0">{sel['id']}</div>
  <div style="font-size:13px;color:#8b949e">{sel['label']} &middot; {sel['loc']}</div>
  <div style="margin:16px 0 4px">
    <span style="font-size:58px;font-weight:900;color:{sc};
      font-family:monospace;line-height:1">{sel['HI']:.1f}</span>
    <span style="font-size:20px;color:{sc};font-weight:700">%</span>
  </div>
  <div style="font-size:13px;color:{tc};font-weight:600;margin-bottom:8px">{ts}</div>
  <div style="display:inline-block;font-size:12px;font-weight:700;
    padding:4px 14px;background:{BG[al]};color:{sc};
    border-radius:20px;border:1px solid {sc}44;margin-bottom:16px">
    {EMO[al]} {LABEL[al]}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px">
    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);
      border-radius:8px;padding:8px 10px">
      <div style="color:#8b949e;font-size:10px">Temp</div>
      <div style="font-weight:600;font-family:monospace">{sel['T']:.1f} &deg;C</div>
    </div>
    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);
      border-radius:8px;padding:8px 10px">
      <div style="color:#8b949e;font-size:10px">Current</div>
      <div style="font-weight:600;font-family:monospace">{sel['I']:.1f} A</div>
    </div>
    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);
      border-radius:8px;padding:8px 10px">
      <div style="color:#8b949e;font-size:10px">Contact R</div>
      <div style="font-weight:600;font-family:monospace">{sel['R']*1000:.1f} m&Omega;</div>
    </div>
    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);
      border-radius:8px;padding:8px 10px">
      <div style="color:#8b949e;font-size:10px">Cycles</div>
      <div style="font-weight:600;font-family:monospace">{int(sel['N']):,}</div>
    </div>
  </div>
  <div style="background:#0d1117;border:1px solid rgba(31,111,235,.3);
    border-radius:9px;padding:10px 12px;margin-top:12px">
    <div style="font-size:10px;color:#8b949e;text-transform:uppercase;
      letter-spacing:.6px">Delta-HI &middot; XGBoost</div>
    <div style="font-size:24px;font-weight:800;color:{dc};
      font-family:monospace">{delta:+.3f}%<span style="font-size:12px;
      color:#8b949e">/tick</span></div>
    <div style="font-size:10px;color:#8b949e">
      Predicts change in HI &middot; eliminates covariate shift</div>
  </div>
</div>""", unsafe_allow_html=True)

with right:
    anom_note = ('&nbsp;&nbsp;<span style="color:#f85149">'
                 '&#9650; = LSTM anomaly (fires before 70% threshold)</span>'
                 if sel['anomaly'] else '')
    st.markdown(f"""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.1);
  border-radius:12px;padding:16px 16px 8px">
  <div style="font-size:11px;color:#8b949e;text-transform:uppercase;
    letter-spacing:.7px;margin-bottom:4px">
    Health Index Timeline{anom_note}
  </div>""", unsafe_allow_html=True)
    st.plotly_chart(chart_hi(sel), use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# Recent events
if ss.events:
    st.markdown("<hr style='margin:14px 0 8px'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#8b949e;text-transform:uppercase;"
                "letter-spacing:.7px;margin-bottom:6px'>Recent events</div>",
                unsafe_allow_html=True)
    ecols=st.columns(min(len(ss.events),3))
    for i,ev in enumerate(ss.events[:3]):
        with ecols[i]:
            is_anom='anomaly' in ev['msg'].lower() or 'lstm' in ev['msg'].lower()
            bg='rgba(248,81,73,.1)' if is_anom else 'rgba(210,153,34,.1)'
            bdr='rgba(248,81,73,.4)' if is_anom else 'rgba(210,153,34,.3)'
            st.markdown(f"""
<div style="background:{bg};border:1px solid {bdr};
  border-radius:8px;padding:8px 12px;font-size:11px">
  <span style="color:#8b949e">{ev['ts']}</span><br>
  <span>{ev['msg']}</span>
</div>""", unsafe_allow_html=True)

# ── Tick + auto-refresh ────────────────────────────────────────────────────────
for _ in range(ss.speed):
    sim_tick()
time.sleep(0.3)
st.rerun()
