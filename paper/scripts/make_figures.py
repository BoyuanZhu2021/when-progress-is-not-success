"""Generate vector PGFPlots figures and LaTeX tables from the JSON snapshot.

PGFPlots is a standard scientific plotting package. All figures are rendered
by the same TeX engine as the paper; no external font or raster dependency.

Outputs
  figures/training_dynamics.tex  three core evidence panels, matched seeds
  figures/training_diagnostics.tex  five auxiliary panels (appendix)
  figures/budget_effects.tex     three budget/effect panels
  正文/tables/master.tex          paired outcomes + method-family citations
  正文/tables/diagnostics.tex     secondary metrics (appendix)
  正文/tables/pairing.tex         9B attacker / 27B victim
  正文/tables/transfer.tex        Slack transfer (appendix)
  正文/tables/per_seed.tex        per-seed strict success (appendix)

Shared visual encoding (tables): blue header with white text; light blue
bands for row groups; a coloured square before each method marks its family;
paired effects are tinted by sign (blue = better than Sparse, rust = worse);
bold is reserved for the OFC rows; the Wins cell carries a bar showing the
share of seeds that improved; OFC rows are shaded blue. The colour values
live in 正文/main.tex and figures/diagram_style.tex.
"""
import json
from pathlib import Path
import numpy as np

PAPER=Path(__file__).resolve().parents[1]
D=json.loads((PAPER/'results/paper_data.json').read_text(encoding='utf-8'))
FIG=PAPER/'figures'; FIG.mkdir(exist_ok=True)
TAB=PAPER/'tables'; TAB.mkdir(exist_ok=True)
NAMES={'sparse':'Sparse','dense':'Dense','sft':'OFC','rl_pos':'Positive-only RL',
       'sft_all':'Unfiltered CE','hindsight':'Lower threshold','curriculum':'Threshold curriculum',
       'vip':'Adaptive allocation','sil':'Success replay','G12':'More rollouts ($G{=}12$)'}
COLORS={'sparse':'sparsecolor','dense':'densecolor','sft':'ofccolor','rl_pos':'poscolor','sft_all':'gray'}
# Method families (colour). Colours other than samplecolor follow the figures.
FAMILY={'sparse':'sparsecolor','dense':'densecolor','hindsight':'densecolor','curriculum':'densecolor',
        'vip':'samplecolor','sil':'samplecolor','G12':'samplecolor',
        'rl_pos':'poscolor','sft_all':'gray','sft':'ofccolor'}
FAMILY_LEGEND=[('sparsecolor','Sparse reference'),('densecolor','denser/relaxed signal'),
               ('samplecolor','more/reused experience'),('poscolor','positive-only RL'),
               ('gray','unfiltered CE'),('ofccolor','OFC (ours)')]

# ----------------------------------------------------------------------------- formatting

def signed(value, percent=True, digits=2):
    """Typeset a signed number with a proper math minus/plus sign.

    Plain text '-' renders as a hyphen in LaTeX; the table columns mix
    Times digits with a math-mode sign, which is the standard workaround.
    Exact zeros carry no sign.
    """
    v=value*(100 if percent else 1)
    if abs(v) < 0.5*10**(-digits):
        return f'{0:.{digits}f}'
    return ('$+$' if v > 0 else '$-$')+f'{abs(v):.{digits}f}'

def ci_text(ci, percent=True):
    return f'[{signed(ci[0],percent)}, {signed(ci[1],percent)}]'

def p_text(p):
    if p is None: return '--'
    return '$<$0.0001' if p < 1e-4 else f'{p:.4f}'

def seed_range(seeds):
    s=sorted(map(int,seeds))
    if len(s)==1: return str(s[0])
    return f'{s[0]}--{s[-1]}' if s[-1]-s[0]==len(s)-1 else ','.join(map(str,s))

def hdr(*lines):
    """Stacked column header (keeps wide labels from widening narrow columns)."""
    return r'\begin{tabular}[b]{@{}c@{}}'+r'\\'.join(lines)+r'\end{tabular}'

def write(name,text):
    (TAB/f'{name}.tex').write_text(text+'\n',encoding='utf-8')

def tabular(colspec,header,rows,footer=None):
    body=[r'\begin{tabular}{'+colspec+'}',r'\toprule',header+r' \\',r'\midrule',*rows,r'\bottomrule']
    if footer: body.append(footer)
    body.append(r'\end{tabular}')
    return '\n'.join(body)

def coords(xs,ys):
    return ' '.join(f'({x:.5f},{y:.5f})' for x,y in zip(xs,ys) if y is not None)

def save(name,body):
    (FIG/f'{name}.tex').write_text(body+'\n',encoding='utf-8')

def arm_runs(campaign,arm,seeds=None):
    runs=D[campaign][arm]
    if seeds is None: return [runs[s] for s in sorted(runs,key=int)]
    return [runs[str(s)] for s in seeds]

def pct(x): return f'{100*x:.2f}'
def pct1(x): return f'{100*x:.1f}'

# ----------------------------------------------------------------------------- tables

OFC_ROW=r'\rowcolor{ofccolor!20}'
HEAD_ROW=r'\rowcolor{headcolor}'
GROUP_ROW=r'\rowcolor{headcolor!10}'

# Column specs keep the default \tabcolsep padding at BOTH ends (no @{}). colortbl paints
# each coloured panel \tabcolsep beyond the cell, so with @{} ends a coloured row stuck out
# of the tabular -- and past the text margin -- by 3pt per side. With the padding kept, the
# panel fills exactly the padding, and \resizebox{\linewidth} scales band, rules and text
# block to the same edges.

def sd_pct(campaign,arm,step,seeds=None):
    """Sample standard deviation (pp) of held-out ASR across seeds; None when n<2."""
    v=[r['eval'][str(step)]['ood_asr'] for r in arm_runs(campaign,arm,seeds)]
    return 100*float(np.std(v,ddof=1)) if len(v)>1 else None

def mean_sd(mean,sd):
    """'mean +- sd' in percent, or the bare mean for a single run."""
    return pct(mean) if sd is None else f"{pct(mean)} $\\pm$ {sd:.1f}"

def head_row(cells):
    """Header row: slate band, white text."""
    return HEAD_ROW+' & '.join(r'\textcolor{white}{'+c+'}' for c in cells)

def chip(color):
    """Small coloured square marking the method family."""
    return r'\raisebox{0.3ex}{\textcolor{'+color+r'}{\rule{5pt}{5pt}}}'

def group_title(ncol,title):
    """Row-group band; the group letter is bold, the rest upright."""
    letter,rest=title.split(' ',1)
    return (r'\addlinespace[3pt]'+'\n'+GROUP_ROW+r'\multicolumn{'+str(ncol)+r'}{l}{\textbf{'+letter+'} '+rest+r'} \\')

def family_legend(ncol,per_row=6):
    """Family legend under the table, split into rows so it never widens the tabular."""
    items=[chip(c)+'~'+t for c,t in FAMILY_LEGEND]
    rows=[r'\multicolumn{'+str(ncol)+r'}{l}{\scriptsize '+r'\qquad '.join(items[i:i+per_row])+r'} \\'
          for i in range(0,len(items),per_row)]
    return r'\addlinespace[2pt]'+'\n'+'\n'.join(rows)

def effect_cell(c, tint=True):
    """Paired effect as 'D +- h' where h is the 95% paired-t half-width.

    Colour by sign (teal = better than Sparse, orange = worse); the cell is
    tinted by sign unless the row already carries the OFC shading. No bold
    here: bold is reserved for the OFC (ours) rows. Values are never altered.
    """
    eff=c['effect']; h=(c['t_ci'][1]-c['t_ci'][0])/2
    txt=f"{signed(eff)} $\\pm$ {100*h:.2f}"
    col='ofccolor' if eff > 0 else 'densecolor'
    return (r'\cellcolor{'+col+'!14}' if tint else '')+r'\textcolor{'+col+'}{'+txt+'}'

def rel_cell(c):
    rel=c['effect']/c['base_mean']
    col='ofccolor' if rel > 0 else 'densecolor'
    return r'\textcolor{'+col+'}{'+signed(rel,digits=1)+'}'

def wins_cell(c):
    """'wins/pairs' coloured by majority, followed by a 12pt bar of the winning share."""
    w,n=c['sign_dense_gt_sparse'],c['n']
    filled=12*w/n
    bar=(r'\raisebox{0.3ex}{\textcolor{ofccolor}{\rule{'+f'{filled:.1f}'+r'pt}{4pt}}'
         r'\textcolor{gray!35}{\rule{'+f'{12-filled:.1f}'+r'pt}{4pt}}}')
    col='ofccolor' if 2*w > n else ('densecolor' if 2*w < n else 'diagramink')
    return r'\textcolor{'+col+'}{'+f'{w}/{n}'+r'}\,'+bar

def summary(campaign,arm,step,seeds=None):
    """Held-out metrics at the read point; training metrics averaged over iterations 1..step."""
    rr=arm_runs(campaign,arm,seeds)
    def upto(r,k):
        v=[x for x in r['series'][k][:step] if x is not None]
        return float(np.mean(v))
    return {'asr':np.mean([r['eval'][str(step)]['ood_asr'] for r in rr]),
            'phi':np.mean([r['eval'][str(step)]['ood_phi'] for r in rr]),
            'slack':np.mean([r['eval'][str(step)]['transfer_asr'] for r in rr]),
            'zero':np.mean([upto(r,'all_zero_group_frac') for r in rr]),
            'succ':np.mean([upto(r,'train_success') for r in rr])}

def metric_cells(s):
    return [f"{s['phi']:.3f}",pct(s['slack']),pct1(s['zero']),pct1(s['succ'])]

def emphasize(cells):
    """OFC (ours) rows: teal method name, shaded row, bold in every cell.

    Bold is reserved for our method across all tables; no other cell is bold.
    """
    out=[]
    for c in cells:
        c=c.replace('OFC',r'\textcolor{ofccolor}{OFC}',1)
        out.append(c if c=='--' else r'\textbf{'+c+'}')
    return out

def method_label(arm, cite=True):
    """Family square + one line per method; implementation qualifications stay in the appendix."""
    label={'hindsight':'Lower Threshold','curriculum':'Threshold Curriculum',
           'vip':'Adaptive Allocation','sil':'Success Replay',
           'G12':'More Rollouts'}.get(arm,NAMES[arm])
    sources={
        'sparse':'shao2024deepseekmath',
        'dense':'shao2024deepseekmath',
        'curriculum':'bengio2009curriculum',
        'vip':'nguyen2026vip',
        'sil':'oh2018sil',
        'rl_pos':'oh2018sil',
        'G12':'hu2025brorl',
    }
    if arm=='sft':
        label=label+' (ours)'
    elif cite and arm in sources:
        label=label+r' \citep{'+sources[arm]+'}'
    return chip(FAMILY[arm])+'~'+label

def master_table():
    """Core paired outcomes in the main table; secondary readouts in the appendix."""
    header=head_row(['Method',
        hdr('Sparse ASR (\\%)','mean $\\pm$ sd'),hdr('ASR (\\%)','mean $\\pm$ sd'),
        hdr('$\\Delta\\pm95\\%$','(pp)'),hdr('Wins /','pairs')])
    rows=[]; diagnostics=[]
    diagheader=head_row(['Method',hdr('Rel.\\ $\\Delta$','(\\%)'),
        '$p$',hdr('Held-out','$\\Phi$'),hdr('Cross-domain','ASR, Slack (\\%)'),
        hdr('Zero-signal','groups (\\%)'),hdr('Train succ.','(\\%)')])
    groups=[('fullft',16,['sparse','dense','hindsight','curriculum','vip','sil','G12','sft'],
             '(A) Original campaign, iteration 16'),
            ('dynamics',8,['sparse','sft','rl_pos','sft_all'],
             '(B) Mechanism study, iteration 8'),
            ('dynamics',32,['sparse','sft','rl_pos','sft_all'],
             '(C) Same mechanism runs, iteration 32')]
    for campaign,step,arms,title in groups:
        rows.append(group_title(5,title))
        diagnostics.append(group_title(7,title))
        for arm in arms:
            s=summary(campaign,arm,step)
            if campaign=='fullft':
                c=D['fullft_comparisons'].get(arm)
            else:
                c=D['dynamics_comparisons'].get(arm,{}).get(str(step))
            label=method_label(arm); short=method_label(arm,cite=False)
            if c:
                # Means +- sample sd over the matched seed subset, for the arm and for Sparse.
                cells=[label,
                       mean_sd(c['base_mean'],sd_pct(campaign,'sparse',step,c['seeds'])),
                       mean_sd(c['arm_mean'],sd_pct(campaign,arm,step,c['seeds'])),
                       effect_cell(c,tint=arm!='sft'),wins_cell(c)]
                dr=[short,rel_cell(c),p_text(c['p_two_sided']),*metric_cells(s)]
            else:
                cells=[label,'--',mean_sd(s['asr'],sd_pct(campaign,arm,step)),'--','--']
                dr=[short,'--','--',*metric_cells(s)]
            if arm=='sft':
                # Colour the name and shade the row; bold stays reserved for intervals excluding zero.
                cells=emphasize(cells)
                dr=emphasize(dr)
            rows.append((OFC_ROW if arm=='sft' else '')+' & '.join(cells)+r' \\')
            diagnostics.append((OFC_ROW if arm=='sft' else '')+' & '.join(dr)+r' \\')
    write('master',r'\setlength{\tabcolsep}{3pt}\small'+'\n'+
          r'\resizebox{\linewidth}{!}{'+tabular('lrrrr',header,rows,footer=family_legend(5))+'}')
    write('diagnostics',r'\setlength{\tabcolsep}{4pt}\small'+'\n'+
          r'\resizebox{\linewidth}{!}{'+tabular('lrrrrrr',diagheader,diagnostics)+'}')

def pairing_table():
    """9B attacker / 27B-FP8 victim, same layout as Table 1, read at iterations 8 and 16.

    Slack successes are zero for every arm, so that column is replaced by the
    number of nonzero-weight examples per iteration (averaged up to the read point).
    """
    cols='l r r r r r r r r r'
    head=head_row(['Method',hdr('Held-out ASR (\\%)','mean $\\pm$ sd'),hdr('$\\Delta$ vs \\sparse{}','$\\pm$\\,95\\% (pp)'),
                   hdr('Rel.\\ $\\Delta$','(\\%)'),'$p$','Wins',hdr('Held-out','$\\Phi$'),
                   hdr('Examples','/ iteration'),hdr('Zero-signal','groups (\\%)'),hdr('Train','succ.\\ (\\%)')])
    def n_ex(arm,step):
        rr=arm_runs('scale',arm)
        return f"{np.mean([np.mean([v for v in r['series']['n_examples'][:step] if v is not None]) for r in rr]):.0f}"
    rows=[]
    for step in (8,16):
        rows.append(group_title(10,f'Iteration {step}'))
        for arm in ['sparse','dense','sft']:
            s=summary('scale',arm,step)
            if arm=='sparse':
                stats=['--']*4; label=chip(FAMILY[arm])+r'~\sparse{} (ref.)'
            else:
                c=D['scale_comparisons_by_step'][arm][str(step)]; label=chip(FAMILY[arm])+'~'+NAMES[arm]
                stats=[effect_cell(c,tint=arm!='sft'),rel_cell(c),p_text(c['p_two_sided']),wins_cell(c)]
            cells=[label,mean_sd(s['asr'],sd_pct('scale',arm,step)),*stats,f"{s['phi']:.3f}",n_ex(arm,step),pct1(s['zero']),pct1(s['succ'])]
            if arm=='sft':
                cells=emphasize(cells)
            rows.append((OFC_ROW if arm=='sft' else '')+' & '.join(cells)+r' \\')
    write('pairing',r'\setlength{\tabcolsep}{3pt}\small'+'\n'+r'\resizebox{\linewidth}{!}{'+tabular(cols,head,rows)+'}')

def appendix_tables():
    rows=[]
    for key,r in D['transfer_comparisons'].items():
        campaign,arm=key.split('_',1)
        label={'fullft':'Original, 16','dynamics':'Mechanism, 32','scale':'Pairing, 16'}[campaign]
        eff=effect_cell(r,tint=arm!='sft') if r['sd_between'] > 0 else '--'
        tc=[label,f"{chip(FAMILY[arm])}~{NAMES[arm]}",f"{100*r['base_mean']:.2f}",f"{100*r['arm_mean']:.2f}",eff]
        rows.append((OFC_ROW if arm=='sft' else '')+' & '.join(emphasize(tc) if arm=='sft' else tc)+r' \\')
    write('transfer',tabular(r'llrrr',
        head_row(['Study, iteration','Method',r'Sparse (\%)',r'Method (\%)',r'$\Delta$ $\pm$ 95\% (pp)']),rows))
    rows=[]
    for seed in map(str,range(2,18)):
        entries=[]
        for a in ['sparse','dense','hindsight','curriculum','vip','sil','G12','sft']:
            rr=D['fullft'][a].get(seed)
            v=f"{100*rr['eval']['16']['ood_asr']:.2f}" if rr else '--'
            entries.append(r'\textbf{'+v+'}' if (a=='sft' and rr) else v)
        rows.append(seed+' & '+' & '.join(entries)+r' \\')
    write('per_seed',tabular(r'rrrrrrrr>{\columncolor{ofccolor!12}}r',
        head_row(['Seed','Sparse','Dense','Lower','Curric.','Adaptive','Replay','$G=12$','OFC']),rows))

# ----------------------------------------------------------------------------- figures

PT_PER_CM=28.4528
# Main-text figures use fixed axis boxes (scale only axis). Four boxes plus
# gutters come to about 1.05 linewidth, so \resizebox leaves fonts near their
# nominal size: \small headers, \footnotesize ticks and endpoint labels.
# Endpoint labels sit in the gutter right of each axis, so HSEP must hold one
# label plus the next panel's tick labels.
AXIS_W=2.45; HSEP=1.25

def groupplot_head(cols,height):
    """Open a one-row groupplot with `cols` fixed-size axis boxes."""
    return (r'\begin{tikzpicture}\begin{groupplot}[group style={group size='+f'{cols} by 1,horizontal sep={HSEP}cm'+'},'
            f'width={AXIS_W}cm,height={height}cm,')

# Panels: no grid, a light panel background, footnote-size tick labels, and a
# tight gap between the two-line header and the axis frame. Per-run spread is
# drawn as shaded bands (see density_band), not as hairlines.
PANEL_STYLE=(r'axis background/.style={fill=diagramlight},tick label style={font=\footnotesize},'
             r'title style={font=\small,yshift=-2pt},label style={font=\footnotesize},'
             r'every axis plot/.append style={line join=round}')

MAIN_STYLE=(r'scale only axis,clip mode=individual,axis background/.style={fill=diagramlight},'
            r'axis line style={gray!55},tick style={gray!55},'
            r'tick label style={font=\footnotesize},'
            r'title style={at={(0,1)},anchor=south west,align=left,font=\small,yshift=1pt,inner ysep=1pt,inner xsep=0pt},'
            r'xlabel style={font=\footnotesize,yshift=2pt},'
            r'every axis plot/.append style={line join=round},'
            r'legend style={font=\fontsize{8pt}{9.5pt}\selectfont,draw=none,fill=none,'
            r'/tikz/every even column/.append style={column sep=0.8em}}')

def shared_legend(name,cols=4,legend_drop='0.45cm',label_drop='0.85cm'):
    """Legend centred directly under the full panel span, then the shared
    x-axis label 'Training iteration' beneath the legend."""
    mid=r'($(group c1r1.south west)!0.5!(group c'+str(cols)+r'r1.south east)'
    return (r'\node[anchor=north] at '+mid+f'+(0,-{legend_drop})$) '+r'{\pgfplotslegendfromname{'+name+r'}};'+'\n'+
            r'\node[anchor=north,font=\footnotesize] at '+mid+f'+(0,-{label_drop})$) '+r'{Training iteration};')

def header(letter,title,subtitle):
    """Two-line panel header: bold letter and title, then the metric in muted grey."""
    return (r'title={\textbf{('+letter+r')} '+title+r'\\[-1.5pt]{\scriptsize\color{diagrammuted}'+subtitle+'}}')

def seed_lines(color,series,width='.35pt',shade=38):
    """Thin per-seed traces; None entries are skipped."""
    out=[]
    for xs,ys in series:
        c=coords(xs,ys)
        if c: out.append(r'\addplot['+color+f'!{shade},line width={width},forget plot] coordinates '+'{'+c+'};')
    return out

def density_band(color,series,name):
    """Spread across runs as two nested shaded bands: min--max (lightest) and
    mean +- sd (darker). Replaces per-seed hairlines. `name` must be unique
    within the figure because it names the fill-between paths."""
    xs=series[0][0]
    arr=np.array([[np.nan if v is None else v for v in ys] for _,ys in series],dtype=float)
    if arr.shape[0] < 2: return []
    with np.errstate(all='ignore'):
        mean=np.nanmean(arr,0); sd=np.nanstd(arr,0,ddof=1); lo=np.nanmin(arr,0); hi=np.nanmax(arr,0)
    clean=lambda a: [None if np.isnan(v) else float(v) for v in a]
    out=[]
    for tag,(a,b),op in (('mm',(lo,hi),0.10),('sd',(mean-sd,mean+sd),0.18)):
        out+=[r'\addplot[name path='+f'{name}{tag}a'+',draw=none,forget plot] coordinates {'+coords(xs,clean(a))+'};',
              r'\addplot[name path='+f'{name}{tag}b'+',draw=none,forget plot] coordinates {'+coords(xs,clean(b))+'};',
              r'\addplot[fill='+color+f',fill opacity={op},draw=none,forget plot] fill between[of={name}{tag}a and {name}{tag}b];']
    return out

def mean_line(color,xs,ys,legend=None,mark='*',size='1.1pt',width='1.05pt',extra=''):
    line=r'\addplot['+color+f',line width={width},mark={mark},mark size={size},mark options={{solid}}{extra}'+'] coordinates {'+coords(xs,ys)+'};'
    return [line]+([r'\addlegendentry{'+legend+'}'] if legend else [])

# The Sparse reference is dashed so that it separates from OFC, whose colour
# sits in the same blue family.
LINE_EXTRA={'sparse':',densely dashed','dense':'','sft':'','rl_pos':'','sft_all':''}

def gap_shade(xs,ya,yb,color,name,opacity=0.22):
    """Shade the region between two mean curves: the visible size of an effect."""
    return [r'\addplot[name path='+name+'a,draw=none,forget plot] coordinates {'+coords(xs,ya)+'};',
            r'\addplot[name path='+name+'b,draw=none,forget plot] coordinates {'+coords(xs,yb)+'};',
            r'\addplot[fill='+color+f',fill opacity={opacity},draw=none,forget plot] fill between[of={name}a and {name}b];']

def vmarker(x,lo,hi,color='diagrammuted'):
    """Dotted vertical guide at a read-out iteration discussed in the text."""
    return [r'\draw[densely dotted,'+color+',line width=.6pt] (axis cs:'+f'{x},{lo}'+') -- (axis cs:'+f'{x},{hi}'+');']

def callout(color,x,y,text,anchor='west'):
    """Free-standing annotation inside a panel, in data coordinates."""
    return [r'\node[font=\scriptsize,text='+color+f',anchor={anchor},inner sep=1pt,align=left] at (axis cs:'
            f'{x:.2f},{y:.5f}'+r') {'+text+'};']

def nanmean(rows):
    """Column-wise mean ignoring None."""
    arr=np.array([[np.nan if v is None else v for v in r] for r in rows],dtype=float)
    with np.errstate(all='ignore'):
        m=np.nanmean(arr,axis=0)
    return [None if np.isnan(v) else float(v) for v in m]

def arm_series(rr,metric,scale):
    """Per-run (xs, ys) traces and the run-mean trace for a training or EVAL: metric."""
    if metric.startswith('EVAL:'):
        m=metric.split(':',1)[1]
        xs=sorted(map(int,rr[0]['eval_all']))
        series=[(sorted(map(int,r['eval_all'])),
                 [scale*r['eval_all'][str(s)][m] for s in sorted(map(int,r['eval_all']))]) for r in rr]
        ys=nanmean([[scale*r['eval_all'][str(s)][m] for s in xs] for r in rr])
    else:
        xs=rr[0]['series']['steps']
        series=[(r['series']['steps'],[None if v is None else scale*v for v in r['series'][metric]]) for r in rr]
        ys=nanmean([[None if v is None else scale*v for v in r['series'][metric]] for r in rr])
    return xs,series,ys

def dodge(values,lo,hi,height_cm,min_pt=8.5):
    """Label positions (data units) at least min_pt apart, kept inside [lo, hi].

    Labels are pushed apart symmetrically around their true values so that a
    cluster stays centred on the points it annotates.
    """
    gap=(hi-lo)*min_pt/(height_cm*PT_PER_CM)
    order=sorted(range(len(values)),key=lambda i: values[i])
    ys=[values[i] for i in order]
    out=[]
    for y in ys:
        out.append(max(y,out[-1]+gap) if out else y)
    shift=sum(o-v for o,v in zip(out,ys))/len(ys)
    out=[o-shift for o in out]
    if out[0] < lo: out=[o+(lo-out[0]) for o in out]
    if out[-1] > hi: out=[o-(out[-1]-hi) for o in out]
    pos=[None]*len(values)
    for i,y in zip(order,out): pos[i]=y
    return pos

def endpoint_labels(entries,x,lo,hi,height_cm,fmt):
    """Direct labels right of the last point (in the gutter), dodged vertically."""
    ys=dodge([v for _,v in entries],lo,hi,height_cm)
    return [r'\node[font=\footnotesize,text='+c+r',anchor=west,inner sep=1pt] at (axis cs:'+
            f'{x:.2f},{y:.5f}'+r') {'+fmt(v)+'};' for (c,v),y in zip(entries,ys)]

def training_dynamics():
    """Figure 3: four panels on matched seeds 2--7.

    (a) and (d) are per-iteration training series (all 16 iterations);
    (b) and (c) are the held-out read-outs taken every two iterations. The
    Dense--Sparse gap is shaded in the held-out panels, the iteration-16
    read-out used by Table 1 is marked, and each panel carries direct
    endpoint labels."""
    seeds=range(2,8); H=2.6; READOUT=16
    arms=[('sparse','Sparse'),('dense','Dense'),('sft','OFC')]
    marker={'sparse':'*','dense':'square*','sft':'triangle*'}
    panels=[('all_zero_group_frac',100,'a','Training signal',r'Zero-signal goal groups (\%)',(0,85),'{:.1f}'),
            ('EVAL:ood_phi',1,'b','Process score',r'Held-out process score $\Phi$',(0.4,0.72),'{:.3f}'),
            ('EVAL:ood_asr',100,'c','Strict success',r'Held-out strict ASR (\%)',(0,65),'{:.1f}'),
            ('train_success',100,'d','Training success',r'In-sample success (\%)',(0,65),'{:.1f}')]
    body=[groupplot_head(4,H)+r'xmin=0.5,xmax=16.5,xtick={4,8,12,16},xlabel={},'+MAIN_STYLE+']']
    for i,(metric,scale,letter,title,unit,(lo,hi),fmt) in enumerate(panels):
        opts=[header(letter,title,unit),f'ymin={lo},ymax={hi}']
        if i==len(panels)-1: opts.append('legend to name=coreleg,legend columns=3')
        body.append(r'\nextgroupplot['+','.join(opts)+']')
        held_out=metric.startswith('EVAL:')
        if held_out:
            body+=vmarker(READOUT,lo,hi)
        ends=[]; means={}
        for arm,label in arms:
            rr=arm_runs('fullft',arm,seeds); col=COLORS[arm]
            xs,series,ys=arm_series(rr,metric,scale)
            means[arm]=(xs,ys)
            body+=density_band(col,series,f'{arm}{letter}')
            ends.append((col,ys[-1]))
        if held_out:
            # The Dense-minus-Sparse region, drawn before the lines so that it sits behind them.
            body+=gap_shade(means['sparse'][0],means['sparse'][1],means['dense'][1],'densecolor',f'gap{letter}')
        for arm,label in arms:
            xs,ys=means[arm]
            body+=mean_line(COLORS[arm],xs,ys,legend=(label if i==len(panels)-1 else None),
                            mark=(marker[arm] if held_out else 'none'),size='1.6pt',width='1.15pt',extra=LINE_EXTRA[arm])
        body+=endpoint_labels(ends,16.75,lo,hi,H,fmt.format)
    body.append(r'\end{groupplot}')
    body.append(shared_legend('coreleg',cols=4))
    body.append(r'\end{tikzpicture}')
    save('training_dynamics','\n'.join(body))

def training_diagnostics():
    """Appendix figure: four auxiliary panels on the same matched seeds (2x2 grid).
    Training success itself is panel (d) of the main-text Figure 3."""
    seeds=range(2,8)
    arms=[('sparse','Sparse'),('dense','Dense'),('sft','OFC')]
    panels=[
        ('train_mean_phi','(a) Training progress',1,r'$\Phi$',''),
        ('n_examples','(b) Nonzero-weight examples',1,'Count',''),
        ('pg_loss','(c) Training loss',1,'Loss',r'scaled y ticks=false,yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2}'),
        ('grad_norm','(d) Gradient norm',1,'Norm','')]
    body=[r'\begin{tikzpicture}\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.8cm,vertical sep=1.25cm},'
          r'width=6.1cm,height=4.0cm,xmin=0.5,xmax=17.0,xtick={2,8,16},xlabel={Training iteration},'+PANEL_STYLE+
          r',tick label style={font=\small},label style={font=\small},'
          r'legend style={font=\small,draw=none,fill=none,/tikz/every even column/.append style={column sep=1em}},legend columns=3]']
    for i,(metric,title,scale,unit,extra) in enumerate(panels):
        opts=[f'title={{{title}}}',r'ylabel={'+unit+'}','xlabel={}']
        if extra: opts.append(extra)
        if i==len(panels)-1: opts.append('legend to name=auxleg')
        body.append(r'\nextgroupplot['+','.join(opts)+']')
        for arm,label in arms:
            rr=arm_runs('fullft',arm,seeds); col=COLORS[arm]
            xs,series,ys=arm_series(rr,metric,scale)
            body+=density_band(col,series,f'{arm}aux{i}')
            body+=mean_line(col,xs,ys,legend=(label if i==len(panels)-1 else None),mark='none',size='1.6pt',extra=LINE_EXTRA[arm])
    body.append(r'\end{groupplot}')
    body.append(r'\node[anchor=north] at ($(group c1r2.south)!0.5!(group c2r2.south)+(0,-0.6cm)$) {\pgfplotslegendfromname{auxleg}};')
    body.append(r'\node[anchor=north,font=\small] at ($(group c1r2.south)!0.5!(group c2r2.south)+(0,-1.1cm)$) {Training iteration};')
    body.append(r'\end{tikzpicture}')
    save('training_diagnostics','\n'.join(body))

def band(xs,ys,lo,hi,color,name,label=None,width='1.15pt'):
    lines=[
      r'\addplot[name path='+name+'lo,draw=none,forget plot] coordinates {'+coords(xs,lo)+'};',
      r'\addplot[name path='+name+'hi,draw=none,forget plot] coordinates {'+coords(xs,hi)+'};',
      r'\addplot['+color+r'!15,forget plot] fill between[of='+name+'lo and '+name+'hi];',
      r'\addplot['+color+f',line width={width},mark=*,mark size=1.4pt'+('' if label else ',forget plot')+'] coordinates {'+coords(xs,ys)+'};']
    if label: lines.append(r'\addlegendentry{'+label+'}')
    return '\n'.join(lines)

def point_label(color,x,y,text,anchor,shift='3pt'):
    """Label next to a single point; 'north*' anchors place it below, 'south*' above."""
    sign='-' if anchor.startswith('north') else ''
    xshift=',xshift=2pt' if anchor.endswith('west') else ''
    return (r'\node[font=\footnotesize,text='+color+f',anchor={anchor},inner sep=1pt,yshift={sign}{shift}{xshift}] at (axis cs:'
            f'{x:.2f},{y:.5f}'+r') {'+text+'};')

def budget_grid():
    """Figure 5: paired gaps (original campaign, mechanism study), the
    per-iteration training-success curves of the mechanism cohort (all 32
    iterations), and its held-out levels. The read-outs discussed in the
    text (iterations 16; 8 and 32) are marked with dotted guides."""
    H=2.5; gap_lo,gap_hi=-9,18
    body=[groupplot_head(4,H)+r'xlabel={},'+MAIN_STYLE+']']
    below_zero=lambda x0,x1: r'\fill[highlightcolor!40] (axis cs:'+f'{x0},{gap_lo}'+r') rectangle (axis cs:'+f'{x1},0'+');'
    # (a) original campaign: OFC - Sparse across 16 matched pairs, every second iteration
    rows=list(D['fullft_gap'].values()); xs=list(map(int,D['fullft_gap'])); n=rows[0]['n']
    gap_ticks=r',ytick={-5,0,5,10,15}'
    body.append(r'\nextgroupplot['+header('a','Original campaign',f'OFC $-$ Sparse (pp), {n} pairs')+
                f',ymin={gap_lo},ymax={gap_hi}'+gap_ticks+r',xmin=1,xmax=17,xtick={4,8,12,16}]')
    body.append(below_zero(1,17))
    body+=vmarker(16,gap_lo,gap_hi)
    eff=[100*r['effect'] for r in rows]
    body.append(band(xs,eff,[100*r['bootstrap']['ci95_lower'] for r in rows],
                     [100*r['bootstrap']['ci95_upper'] for r in rows],'ofccolor','orig'))
    body.append(r'\addplot[gray,dashed,forget plot] coordinates {(1,0)(17,0)};')
    k=int(np.argmax(eff))
    body.append(point_label('ofccolor',xs[k],eff[k],signed(eff[k]/100,digits=1),'south'))
    body+=endpoint_labels([('ofccolor',eff[-1])],17.25,gap_lo,gap_hi,H,lambda v: signed(v/100,digits=1))
    # (b) mechanism study: paired gaps for OFC and Positive-only RL
    # The axis starts left of the first read-out so that the iteration-8 labels fit inside the frame.
    body.append(r'\nextgroupplot['+header('b','Mechanism study','Gap to matched Sparse (pp)')+
                f',ymin={gap_lo},ymax={gap_hi}'+gap_ticks+r',xmin=-7,xmax=34,xtick={8,16,24,32}]')
    body.append(below_zero(-7,34))
    body+=vmarker(8,gap_lo,gap_hi)+vmarker(32,gap_lo,gap_hi)
    ends=[]
    # Start labels sit left of the first point: OFC's below its height, Positive-only RL's above.
    for name,arm,dx,anchor in [('_T12-full','sft',-0.3,'north east'),('_side-full','rl_pos',-0.3,'south east')]:
        dd=D['published_dynamics'][name]['per_step']; rows=list(dd.values()); xs=list(map(int,dd))
        eff=[100*r['paired_t']['effect'] for r in rows]
        body.append(band(xs,eff,[100*r['bootstrap']['ci95_lower'] for r in rows],
                         [100*r['bootstrap']['ci95_upper'] for r in rows],COLORS[arm],arm))
        body.append(point_label(COLORS[arm],xs[0]+dx,eff[0],signed(eff[0]/100,digits=1),anchor))
        ends.append((COLORS[arm],eff[-1]))
    body.append(r'\addplot[gray,dashed,forget plot] coordinates {(-7,0)(34,0)};')
    body+=endpoint_labels(ends,34.4,gap_lo,gap_hi,H,lambda v: signed(v/100,digits=1))
    # (c) mechanism study: in-sample training success at every one of the 32 iterations
    body.append(r'\nextgroupplot['+header('c','Training success',r'In-sample success (\%)')+
                r',ymin=0,ymax=65,xmin=0.5,xmax=32.5,xtick={8,16,24,32}]')
    body+=vmarker(8,0,65)+vmarker(32,0,65)
    ends=[]
    for arm in ['sparse','sft','rl_pos','sft_all']:
        rr=arm_runs('dynamics',arm); col=COLORS[arm]
        xs,series,ys=arm_series(rr,'train_success',100)
        body+=density_band(col,series,f'{arm}trn')
        body+=mean_line(col,xs,ys,mark='none',width='1.15pt',extra=LINE_EXTRA[arm])
        ends.append((col,ys[-1]))
    body+=endpoint_labels(ends,32.9,0,65,H,'{:.1f}'.format)
    # (d) mechanism study: absolute held-out ASR for every complete run
    body.append(r'\nextgroupplot['+header('d','Held-out success',r'Held-out strict ASR (\%)')+
                r',ymin=0,ymax=65,xmin=2.5,xmax=34,xtick={8,16,24,32},legend to name=budgetleg,legend columns=4]')
    body+=vmarker(8,0,65)+vmarker(32,0,65)
    ends=[]
    for arm in ['sparse','sft','rl_pos','sft_all']:
        rr=arm_runs('dynamics',arm); col=COLORS[arm]; steps=[8,16,24,32]
        body+=density_band(col,[(steps,[100*r['eval'][str(s)]['ood_asr'] for s in steps]) for r in rr],f'{arm}lvl')
        means=[100*np.mean([r['eval'][str(s)]['ood_asr'] for r in rr]) for s in steps]
        body+=mean_line(col,steps,means,legend=f'{NAMES[arm]} ($n{{=}}{len(rr)}$)',size='1.4pt',width='1.15pt',extra=LINE_EXTRA[arm])
        ends.append((col,means[-1]))
    body+=endpoint_labels(ends,34.4,0,65,H,'{:.1f}'.format)
    body.append(r'\end{groupplot}')
    body.append(shared_legend('budgetleg',cols=4))
    body.append(r'\end{tikzpicture}')
    save('budget_effects','\n'.join(body))

if __name__=='__main__':
    for stale in ['reward_diagnostics.tex','mechanism_levels.tex']:
        p=FIG/stale
        if p.exists(): p.unlink()
    for stale in ['campaign.tex','mechanism.tex','scale.tex']:
        p=TAB/stale
        if p.exists(): p.unlink()
    training_dynamics(); training_diagnostics(); budget_grid()
    master_table(); pairing_table(); appendix_tables()
    print('Generated 3 vector PGFPlots figures and 5 LaTeX tables.')
