from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import beta, halfnorm, lognorm, norm, spearmanr, uniform


MODEL_LABELS={
    'competing_risks':'constant competing risks',
    'effort_threshold':'effort threshold',
    'age_hazards':'age-structured hazards',
    'project_frailty':'project frailty',
    'dynamic_context':'dynamic effort and soil'
}

PARAMETER_ROLES={
    'lambda_seed':'population seed-arrival rate; never scholar-specific',
    'entry_stock':'mean latent project stock before the first observed roster year',
    'k_base':'baseline hard capacity in projects',
    'p_h':'constant annual project harvest probability before competing-risk conversion',
    'p_d':'constant annual project death probability before competing-risk conversion',
    'w_total':'population-median annual effort budget in maintenance units',
    'sigma_w':'stationary log-scale volatility of annual effort',
    'allocation_alpha':'normalized-Gamma allocation concentration; low values concentrate effort',
    'beta_effort':'elasticity of harvest intensity with respect to effort above maintenance',
    'harvest_scale':'discrete Weibull harvest time scale',
    'harvest_shape':'discrete Weibull harvest shape; one is geometric',
    'death_scale':'discrete Weibull death time scale',
    'death_shape':'discrete Weibull death shape; one is geometric',
    'sigma_h':'project-level log-harvest-intensity frailty scale',
    'sigma_d':'project-level log-death-intensity frailty scale',
    'phi_w':'AR(1) persistence of log effort',
    'phi_z':'AR(1) persistence of standardized soil quality',
    'beta_k_z':'log-capacity response to standardized soil quality',
    'rho_wz':'correlation of annual effort and soil innovations'
}


def _common_priors():
    return {
        'lambda_seed':{'family':'lognormal','median':4.0,'sigma':0.55,'low':0.5,'high':15.0},
        'entry_stock':{'family':'lognormal','median':8.0,'sigma':0.60,'low':0.2,'high':60.0},
        'k_base':{'family':'lognormal','median':40.0,'sigma':0.55,'low':8.0,'high':220.0}
    }


def prior_specs(model):
    specs=_common_priors()
    if model in ['competing_risks','effort_threshold','project_frailty','dynamic_context']:
        specs.update({
            'p_h':{'family':'beta','a':2.0,'b':6.0,'low':0.02,'high':0.80},
            'p_d':{'family':'beta','a':1.5,'b':10.0,'low':0.002,'high':0.50}
        })
    if model in ['effort_threshold','dynamic_context']:
        specs.update({
            'w_total':{'family':'lognormal','median':28.0,'sigma':0.55,'low':3.0,'high':140.0},
            'sigma_w':{'family':'halfnormal','scale':0.45,'low':0.01,'high':1.25},
            'allocation_alpha':{'family':'lognormal','median':1.5,'sigma':0.75,'low':0.10,'high':20.0},
            'beta_effort':{'family':'halfnormal','scale':1.0,'low':0.0,'high':4.0}
        })
    if model=='age_hazards':
        specs.update({
            'harvest_scale':{'family':'lognormal','median':4.0,'sigma':0.55,'low':0.75,'high':25.0},
            'harvest_shape':{'family':'lognormal','median':1.0,'sigma':0.45,'low':0.30,'high':3.5},
            'death_scale':{'family':'lognormal','median':10.0,'sigma':0.60,'low':1.0,'high':50.0},
            'death_shape':{'family':'lognormal','median':1.0,'sigma':0.45,'low':0.30,'high':3.5}
        })
    if model=='project_frailty':
        specs.update({
            'sigma_h':{'family':'halfnormal','scale':0.8,'low':0.0,'high':2.5},
            'sigma_d':{'family':'halfnormal','scale':0.8,'low':0.0,'high':2.5}
        })
    if model=='dynamic_context':
        specs.update({
            'phi_w':{'family':'beta','a':4.0,'b':2.0,'low':0.05,'high':0.98},
            'phi_z':{'family':'beta','a':4.0,'b':2.0,'low':0.05,'high':0.98},
            'beta_k_z':{'family':'halfnormal','scale':0.45,'low':0.0,'high':1.5},
            'rho_wz':{'family':'uniform','low':-0.60,'high':0.80}
        })
    return specs


def _distribution(spec):
    family=spec['family']
    if family=='lognormal':
        return lognorm(s=spec['sigma'],scale=spec['median'])
    if family=='beta':
        return beta(spec['a'],spec['b'],loc=spec['low'],scale=spec['high']-spec['low'])
    if family=='halfnormal':
        return halfnorm(scale=spec['scale'])
    if family=='uniform':
        return uniform(loc=spec['low'],scale=spec['high']-spec['low'])
    raise ValueError(family)


def prior_to_z(value,spec):
    dist=_distribution(spec)
    f0=dist.cdf(spec['low'])
    f1=dist.cdf(spec['high'])
    u=(dist.cdf(value)-f0)/(f1-f0)
    return norm.ppf(np.clip(u,1e-9,1-1e-9))


def z_to_prior(value,spec):
    dist=_distribution(spec)
    f0=dist.cdf(spec['low'])
    f1=dist.cdf(spec['high'])
    return float(dist.ppf(f0+norm.cdf(value)*(f1-f0)))


def sample_prior(model,rng,scale=1.0):
    draw={}
    for name,spec in prior_specs(model).items():
        z=rng.normal(0,scale)
        draw[name]=z_to_prior(z,spec)
    return draw


def theta_to_z(model,theta):
    return np.array([prior_to_z(theta[name],spec) for name,spec in prior_specs(model).items()])


def z_to_theta(model,z):
    return {name:z_to_prior(value,spec) for (name,spec),value in zip(prior_specs(model).items(),z)}


def prior_table():
    rows=[]
    for model in MODEL_LABELS:
        for parameter,spec in prior_specs(model).items():
            rows.append({'model':model,'parameter':parameter,'role':PARAMETER_ROLES[parameter],**spec})
    return pd.DataFrame(rows)


def hypothesis_edges():
    rows=[]
    def edge(model,child,parents,relationship):
        rows.append({'model':model,'child':child,'parents':parents,'relationship':relationship})
    for model in MODEL_LABELS:
        edge(model,'N_it','lambda_seed','Poisson arrivals with one population-level rate')
        edge(model,'accepted_N_it','N_it, G_i,t-, K_it','min(N_it, max(K_it-G_i,t-, 0)); blocked seeds are not deaths')
        edge(model,'G_i,t','G_i,t-, accepted_N_it, H_it, D_it','surviving stock identity')
        edge(model,'H_it','project harvest outcomes','sum of project-level harvest indicators; only observed process')
        edge(model,'D_it','project death outcomes, capacity contraction','sum of permanent project deaths; latent')
    for model in ['competing_risks','project_frailty']:
        edge(model,'project outcome','p_h, p_d','multinomial competing risks: remain, harvest, or die')
    edge('effort_threshold','w_ig,t','W_it, allocation_alpha','W_it times normalized Gamma allocation weights')
    edge('effort_threshold','project death','w_ig,t','death is certain below the fixed maintenance unit')
    edge('effort_threshold','project harvest','p_h, w_ig,t, beta_effort','harvest intensity rises with effort above maintenance')
    edge('age_hazards','project harvest','project age, harvest_scale, harvest_shape','discrete Weibull cause-specific hazard')
    edge('age_hazards','project death','project age, death_scale, death_shape','discrete Weibull cause-specific hazard')
    edge('project_frailty','project harvest','p_h, project harvest frailty','persistent project-level lognormal intensity multiplier')
    edge('project_frailty','project death','p_d, project death frailty','persistent project-level lognormal intensity multiplier')
    edge('dynamic_context','W_it','W_i,t-1, phi_w, sigma_w','stationary log-AR(1) effort state')
    edge('dynamic_context','Z_it','Z_i,t-1, phi_z','standardized stationary AR(1) soil state')
    edge('dynamic_context','K_it','k_base, Z_it, beta_k_z','rounded k_base exp(beta_k_z Z_it)')
    edge('dynamic_context','project hazards','W_it, allocation_alpha, beta_effort','effort affects maintenance and harvest; Z has no direct hazard edge')
    edge('dynamic_context','W/Z innovations','rho_wz','correlated annual shocks without a scholar creativity trait')
    return pd.DataFrame(rows)


def load_bundle(path):
    path=Path(path)
    return {p.stem.replace('solo_',''):pd.read_csv(p) for p in path.glob('solo_*.csv')}


def read_trajectory_matrix(path,n_years=12):
    df=pd.read_csv(path)
    years=[str(x) for x in range(n_years)]
    missing=[x for x in years if x not in df.columns]
    if missing:
        raise ValueError(f'missing trajectory columns: {missing}')
    id_columns=[x for x in df.columns if x not in years]
    ids=df[id_columns[0]].astype(str).to_numpy() if id_columns else np.arange(len(df)).astype(str)
    return ids,df[years].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)


def inverse_ecdf(df,metric,probabilities):
    part=df.loc[df['metric']==metric].sort_values(['cdf','x'])
    values={}
    for probability in probabilities:
        eligible=part.loc[part['cdf']>=probability,'x']
        values[probability]=float(eligible.iloc[0] if len(eligible) else part['x'].iloc[-1])
    return values


def global_age_moments(age):
    n=age['n'].sum()
    total=(age['positive_n']*age['positive_mean']).sum()
    second=((age['positive_n']-1).clip(lower=0)*age['positive_variance']+age['positive_n']*age['positive_mean']**2).sum()
    mean=total/n
    variance=(second-n*mean**2)/(n-1)
    return float(mean),float(variance)


def metric_scale(metric,target):
    if any(x in metric for x in ['fraction','probability','hazard','transition.','p_zero','p_positive','p_negative']):
        return 0.05
    if metric.startswith('lag.') or metric.startswith('decay.'):
        return 0.08
    if 'variance' in metric:
        return max(1.0,0.25*abs(target))
    if 'increment.mean' in metric:
        return 0.25
    return max(0.5,0.20*abs(target))


def build_empirical_targets(bundle,raw_q=None):
    rows=[]
    def add(metric,block,target,source):
        if np.isfinite(target):
            rows.append({'metric':metric,'block':block,'target':float(target),'scale':metric_scale(metric,target),'source':source})
    age=bundle['age_distribution_table']
    mean_q,var_q=global_age_moments(age)
    add('global.mean_q','annual_distribution',mean_q,'solo_age_distribution_table')
    add('global.variance_q','annual_distribution',var_q,'solo_age_distribution_table')
    add('global.zero_fraction','annual_distribution',np.average(age['zero_fraction'],weights=age['n']),'solo_age_distribution_table')
    annual=inverse_ecdf(bundle['ecdfs'],'annual q',[0.5,0.75,0.9,0.95,0.99])
    for p,value in annual.items():
        add(f'global.q_p{int(100*p):02d}','annual_distribution',value,'solo_ecdfs')
    for career_metric in ['cumulative q','career mean q','zero-year fraction','career qmax','age of first qmax']:
        values=inverse_ecdf(bundle['ecdfs'],career_metric,[0.5,0.9,0.99])
        stem=career_metric.replace(' ','_')
        for p,value in values.items():
            add(f'career.{stem}_p{int(100*p):02d}','career_distribution',value,'solo_ecdfs')
    selected_ages=[0,4,7,11]
    for _,row in age.loc[age['age'].isin(selected_ages)].iterrows():
        t=int(row['age'])
        add(f'age.{t}.zero_fraction','age_profile',row['zero_fraction'],'solo_age_distribution_table')
        add(f'age.{t}.positive_mean','age_profile',row['positive_mean'],'solo_age_distribution_table')
        add(f'age.{t}.positive_variance','age_profile',row['positive_variance'],'solo_age_distribution_table')
        add(f'age.{t}.q_p90','age_profile',row['p90'],'solo_age_distribution_table')
        add(f'age.{t}.q_p99','age_profile',row['p99'],'solo_age_distribution_table')
    for _,row in bundle['transition_counts'].iterrows():
        transition=row['transition'].replace('→','_to_')
        add(f'transition.{transition}','transitions',row['fraction'],'solo_transition_counts')
    inc=bundle['increment_summary'].set_index('group').loc['all']
    symmetry=bundle['increment_symmetry'].iloc[0]
    add('increment.mean','increments',inc['mean'],'solo_increment_summary')
    add('increment.variance','increments',inc['variance'],'solo_increment_summary')
    add('increment.p_zero','increments',symmetry['p_delta_zero'],'solo_increment_symmetry')
    add('increment.p_positive','increments',symmetry['p_delta_positive'],'solo_increment_symmetry')
    add('increment.p_negative','increments',symmetry['p_delta_negative'],'solo_increment_symmetry')
    absolute=inverse_ecdf(bundle['absolute_increment_ecdf'],'absolute raw increment',[0.9,0.95,0.99])
    for p,value in absolute.items():
        add(f'increment.abs_p{int(100*p):02d}','increments',value,'solo_absolute_increment_ecdf')
    by_age=bundle['dropout_restart_by_age']
    for _,row in by_age.loc[by_age['age'].isin([0,4,7,10])].iterrows():
        t=int(row['age'])
        add(f'dropout.age.{t}','transitions',row['dropout_probability'],'solo_dropout_restart_by_age')
        add(f'restart.age.{t}','transitions',row['restart_probability'],'solo_dropout_restart_by_age')
    hazards=bundle['dropout_restart_duration_hazards']
    for _,row in hazards.loc[hazards['duration'].isin([1,2,3,5,8,10])].iterrows():
        add(f"{row['event']}.hazard.{int(row['duration'])}",'spell_hazards',row['hazard'],'solo_dropout_restart_duration_hazards')
    for _,row in bundle['dropout_by_within_age_q_quintile'].iterrows():
        add(f"dropout.{row['q_quintile']}",'dropout_gradient',row['probability'],'solo_dropout_by_within_age_q_quintile')
    if raw_q is not None:
        observed=panel_metrics(raw_q)
        for metric,value in observed.items():
            if metric.startswith('lag.') or metric.startswith('decay.'):
                add(metric,'dependence',value,'scientist_trajectories')
    return pd.DataFrame(rows)


def followup_distribution(age_coverage):
    coverage=age_coverage.sort_values('age')
    survival=coverage['scientists'].to_numpy(dtype=float)/coverage['scientists'].iloc[0]
    exact=np.r_[survival[:-1]-survival[1:],survival[-1]]
    exact=np.clip(exact,0,None)
    exact=exact/exact.sum()
    return pd.DataFrame({'observed_years':np.arange(1,len(exact)+1),'probability':exact})


def draw_followup_mask(followup,n,rng,n_years=12):
    lengths=rng.choice(followup['observed_years'].to_numpy(),size=n,p=followup['probability'].to_numpy())
    return np.arange(n_years)[None,:]<lengths[:,None]


def initialize_state(n,model,theta,rng,max_projects):
    alive=np.zeros((n,max_projects),dtype=bool)
    state={
        'alive':alive,
        'age':np.zeros((n,max_projects),dtype=np.int16),
        'frailty_h':np.zeros((n,max_projects),dtype=np.float32),
        'frailty_d':np.zeros((n,max_projects),dtype=np.float32),
        'logw_deviation':np.zeros(n),
        'z':np.zeros(n)
    }
    if model=='dynamic_context':
        state['logw_deviation']=rng.normal(0,theta['sigma_w'],n)
        state['z']=rng.normal(0,1,n)
    k=np.minimum(max_projects,np.maximum(1,np.rint(theta['k_base']).astype(int)))
    k=np.repeat(k,n) if np.ndim(k)==0 else k
    initial=np.minimum(rng.poisson(theta['entry_stock'],n),k)
    _add_projects(state,initial,model,theta,rng)
    initial_mask=state['alive']
    state['age'][initial_mask]=np.minimum(rng.geometric(0.4,initial_mask.sum())-1,8)
    return state


def _add_projects(state,counts,model,theta,rng):
    alive=state['alive']
    free=~alive
    rank=np.cumsum(free,axis=1)
    new=free&(rank<=counts[:,None])
    alive[new]=True
    state['age'][new]=0
    if model=='project_frailty':
        state['frailty_h'][new]=rng.normal(0,theta['sigma_h'],new.sum())
        state['frailty_d'][new]=rng.normal(0,theta['sigma_d'],new.sum())
    else:
        state['frailty_h'][new]=0
        state['frailty_d'][new]=0
    return new


def _capacity(model,theta,state,max_projects):
    if model=='dynamic_context':
        k=theta['k_base']*np.exp(theta['beta_k_z']*state['z'])
    else:
        k=np.repeat(theta['k_base'],state['alive'].shape[0])
    return np.clip(np.rint(k),1,max_projects).astype(int)


def _contract_to_capacity(state,k,rng):
    alive=state['alive']
    excess=alive.sum(axis=1)-k
    deaths=np.zeros(len(k),dtype=int)
    for i in np.flatnonzero(excess>0):
        candidates=np.flatnonzero(alive[i])
        drop=rng.choice(candidates,size=excess[i],replace=False)
        alive[i,drop]=False
        deaths[i]=len(drop)
    return deaths


def _effort(model,theta,state,rng):
    alive=state['alive']
    n=alive.shape[0]
    if model not in ['effort_threshold','dynamic_context']:
        return np.where(alive,1.0,0.0),np.full(n,np.nan)
    if model=='effort_threshold':
        logw=np.log(theta['w_total'])+rng.normal(0,theta['sigma_w'],n)
    else:
        logw=np.log(theta['w_total'])+state['logw_deviation']
    total=np.exp(logw)
    raw=rng.gamma(theta['allocation_alpha'],1.0,size=alive.shape)*alive
    denominator=raw.sum(axis=1,keepdims=True)
    weights=np.divide(raw,denominator,out=np.zeros_like(raw),where=denominator>0)
    return total[:,None]*weights,total


def _rates(model,theta,state,effort):
    alive=state['alive']
    age=state['age'].astype(float)
    if model=='age_hazards':
        h=((age+1)/theta['harvest_scale'])**theta['harvest_shape']-(age/theta['harvest_scale'])**theta['harvest_shape']
        d=((age+1)/theta['death_scale'])**theta['death_shape']-(age/theta['death_scale'])**theta['death_shape']
    else:
        h=np.full(alive.shape,-np.log1p(-theta['p_h']))
        d=np.full(alive.shape,-np.log1p(-theta['p_d']))
    if model=='project_frailty':
        h=h*np.exp(state['frailty_h'])
        d=d*np.exp(state['frailty_d'])
    if model in ['effort_threshold','dynamic_context']:
        h=h*np.maximum(effort,1e-6)**theta['beta_effort']
    return h*alive,d*alive


def garden_step(state,model,theta,rng,max_projects):
    n=state['alive'].shape[0]
    if model=='dynamic_context':
        phi_w=theta['phi_w']
        phi_z=theta['phi_z']
        eta_w=rng.normal(size=n)
        eta_z=theta['rho_wz']*eta_w+np.sqrt(1-theta['rho_wz']**2)*rng.normal(size=n)
        state['logw_deviation']=phi_w*state['logw_deviation']+theta['sigma_w']*np.sqrt(1-phi_w**2)*eta_w
        state['z']=phi_z*state['z']+np.sqrt(1-phi_z**2)*eta_z
    k=_capacity(model,theta,state,max_projects)
    capacity_deaths=_contract_to_capacity(state,k,rng)
    g_before=state['alive'].sum(axis=1)
    planted=rng.poisson(theta['lambda_seed'],n)
    accepted=np.minimum(planted,np.maximum(k-g_before,0))
    overflow=planted-accepted
    _add_projects(state,accepted,model,theta,rng)
    effort,total_effort=_effort(model,theta,state,rng)
    alive=state['alive']
    forced_death=alive&(effort<1.0) if model in ['effort_threshold','dynamic_context'] else np.zeros_like(alive)
    h_rate,d_rate=_rates(model,theta,state,effort)
    eligible=alive&~forced_death
    total_rate=(h_rate+d_rate)*eligible
    event_probability=1-np.exp(-total_rate)
    event=rng.random(alive.shape)<event_probability
    harvest_probability=np.divide(h_rate,h_rate+d_rate,out=np.zeros_like(h_rate),where=(h_rate+d_rate)>0)
    harvest=event&(rng.random(alive.shape)<harvest_probability)&eligible
    random_death=event&~harvest&eligible
    death=forced_death|random_death
    h=harvest.sum(axis=1)
    d=death.sum(axis=1)+capacity_deaths
    removed=harvest|death
    alive[removed]=False
    state['age'][removed]=0
    state['frailty_h'][removed]=0
    state['frailty_d'][removed]=0
    state['age'][alive]+=1
    diagnostics={
        'H':h,'D':d,'N':planted,'accepted':accepted,'overflow':overflow,
        'G':alive.sum(axis=1),'K':k,'W':total_effort
    }
    return state,diagnostics


def simulate_garden(model,theta,n,n_years,rng,max_projects,mask=None):
    state=initialize_state(n,model,theta,rng,max_projects)
    q=np.zeros((n,n_years),dtype=float)
    latent={name:np.zeros((n,n_years),dtype=float) for name in ['D','N','accepted','overflow','G','K','W']}
    for t in range(n_years):
        state,diagnostics=garden_step(state,model,theta,rng,max_projects)
        q[:,t]=diagnostics['H']
        for name in latent:
            latent[name][:,t]=diagnostics[name]
    if mask is not None:
        q=np.where(mask,q,np.nan)
        for name in latent:
            latent[name]=np.where(mask,latent[name],np.nan)
    return q,latent


def _spell_hazards(q,state_value,max_duration=12):
    risk=np.zeros(max_duration+1,dtype=int)
    events=np.zeros(max_duration+1,dtype=int)
    for row in q:
        observed=row[np.isfinite(row)]
        if len(observed)==0:
            continue
        active=observed>0
        start=0
        while start<len(active):
            value=active[start]
            end=start+1
            while end<len(active) and active[end]==value:
                end+=1
            duration=end-start
            if value==state_value:
                risk[1:min(duration,max_duration)+1]+=1
                if end<len(active) and duration<=max_duration:
                    events[duration]+=1
            start=end
    return np.divide(events,risk,out=np.full(max_duration+1,np.nan),where=risk>0)


def _dropout_quintiles(q):
    at_risk=np.zeros(5)
    events=np.zeros(5)
    for t in range(q.shape[1]-1):
        valid=np.isfinite(q[:,t])&np.isfinite(q[:,t+1])&(q[:,t]>0)
        values=q[valid,t]
        if len(values)==0:
            continue
        pct=pd.Series(values).rank(method='average',pct=True).to_numpy()
        group=np.minimum((pct*5).astype(int),4)
        dropout=q[valid,t+1]==0
        for k in range(5):
            select=group==k
            at_risk[k]+=select.sum()
            events[k]+=dropout[select].sum()
    return np.divide(events,at_risk,out=np.full(5,np.nan),where=at_risk>0)


def _safe_correlation(x,y,kind='pearson'):
    valid=np.isfinite(x)&np.isfinite(y)
    if valid.sum()<10 or np.nanstd(x[valid])==0 or np.nanstd(y[valid])==0:
        return np.nan
    if kind=='spearman':
        return float(spearmanr(x[valid],y[valid]).statistic)
    return float(np.corrcoef(x[valid],y[valid])[0,1])


def panel_metrics(q):
    q=np.asarray(q,dtype=float)
    metrics={}
    observed=q[np.isfinite(q)]
    metrics['global.mean_q']=np.mean(observed)
    metrics['global.variance_q']=np.var(observed,ddof=1)
    metrics['global.zero_fraction']=np.mean(observed==0)
    for p in [0.5,0.75,0.9,0.95,0.99]:
        metrics[f'global.q_p{int(100*p):02d}']=np.quantile(observed,p)
    career_sum=np.nansum(q,axis=1)
    career_n=np.isfinite(q).sum(axis=1)
    valid_career=career_n>0
    career_mean=np.divide(career_sum,career_n,out=np.zeros_like(career_sum),where=valid_career)
    zero_fraction=np.divide(np.nansum(q==0,axis=1),career_n,out=np.zeros_like(career_sum),where=valid_career)
    qmax=np.nanmax(q,axis=1)
    first_qmax=np.array([np.nanargmax(row) if np.isfinite(row).any() else np.nan for row in q])
    career_values={'cumulative_q':career_sum,'career_mean_q':career_mean,'zero-year_fraction':zero_fraction,'career_qmax':qmax,'age_of_first_qmax':first_qmax}
    for name,values in career_values.items():
        values=values[np.isfinite(values)&valid_career]
        for p in [0.5,0.9,0.99]:
            metrics[f'career.{name}_p{int(100*p):02d}']=np.quantile(values,p)
    for t in [0,4,7,11]:
        if t>=q.shape[1]:
            continue
        values=q[:,t]
        values=values[np.isfinite(values)]
        positive=values[values>0]
        metrics[f'age.{t}.zero_fraction']=np.mean(values==0)
        metrics[f'age.{t}.positive_mean']=np.mean(positive) if len(positive) else np.nan
        metrics[f'age.{t}.positive_variance']=np.var(positive,ddof=1) if len(positive)>1 else np.nan
        metrics[f'age.{t}.q_p90']=np.quantile(values,0.9)
        metrics[f'age.{t}.q_p99']=np.quantile(values,0.99)
    current=q[:,:-1]
    future=q[:,1:]
    valid=np.isfinite(current)&np.isfinite(future)
    active=current>0
    future_active=future>0
    denominator=valid.sum()
    transitions={
        'active_to_active':valid&active&future_active,
        'active_to_zero':valid&active&~future_active,
        'zero_to_active':valid&~active&future_active,
        'zero_to_zero':valid&~active&~future_active
    }
    for name,select in transitions.items():
        metrics[f'transition.{name}']=select.sum()/denominator
    increments=(future-current)[valid]
    metrics['increment.mean']=np.mean(increments)
    metrics['increment.variance']=np.var(increments,ddof=1)
    metrics['increment.p_zero']=np.mean(increments==0)
    metrics['increment.p_positive']=np.mean(increments>0)
    metrics['increment.p_negative']=np.mean(increments<0)
    for p in [0.9,0.95,0.99]:
        metrics[f'increment.abs_p{int(100*p):02d}']=np.quantile(np.abs(increments),p)
    for t in [0,4,7,10]:
        if t>=q.shape[1]-1:
            continue
        valid=np.isfinite(q[:,t])&np.isfinite(q[:,t+1])
        active=q[:,t]>0
        inactive=~active
        metrics[f'dropout.age.{t}']=np.mean(q[valid&active,t+1]==0) if np.any(valid&active) else np.nan
        metrics[f'restart.age.{t}']=np.mean(q[valid&inactive,t+1]>0) if np.any(valid&inactive) else np.nan
    dropout=_dropout_quintiles(q)
    for k,value in enumerate(dropout,1):
        metrics[f'dropout.Q{k}']=value
    active_hazard=_spell_hazards(q,True)
    inactive_hazard=_spell_hazards(q,False)
    for duration in [1,2,3,5,8,10]:
        metrics[f'dropout.hazard.{duration}']=active_hazard[duration]
        metrics[f'restart.hazard.{duration}']=inactive_hazard[duration]
    raw_corr=[]
    for lag in range(1,min(7,q.shape[1])):
        x=q[:,:-lag].ravel()
        y=q[:,lag:].ravel()
        raw=_safe_correlation(x,y,'spearman')
        activity=_safe_correlation((x>0).astype(float),(y>0).astype(float))
        metrics[f'lag.raw_spearman.{lag}']=raw
        metrics[f'lag.activity_phi.{lag}']=activity
        raw_corr.append(raw)
    valid_decay=np.isfinite(raw_corr)&(np.array(raw_corr)>0)
    if valid_decay.sum()>=3:
        slope=np.polyfit(np.arange(1,len(raw_corr)+1)[valid_decay],np.log(np.array(raw_corr)[valid_decay]),1)[0]
        metrics['decay.raw_rho']=np.exp(slope)
    return pd.Series(metrics,dtype=float)


def target_distance(simulated,targets,block_weights):
    compare=targets.copy()
    compare['simulated']=compare['metric'].map(simulated)
    compare['standardized_error']=(compare['simulated']-compare['target']).abs()/compare['scale']
    compare['standardized_error']=compare['standardized_error'].fillna(10.0).clip(upper=10.0)
    blocks=compare.groupby('block',as_index=False)['standardized_error'].median().rename(columns={'standardized_error':'distance'})
    blocks['weight']=blocks['block'].map(block_weights).fillna(1.0)
    distance=np.average(blocks['distance'],weights=blocks['weight'])
    return float(distance),blocks


def simulate_summary(model,theta,config,followup,targets,rng,n=None):
    n=config['n_scholars'] if n is None else n
    mask=draw_followup_mask(followup,n,rng,config['n_years'])
    q,latent=simulate_garden(model,theta,n,config['n_years'],rng,config['max_projects'],mask)
    summary=panel_metrics(q)
    distance,blocks=target_distance(summary,targets,config['target_block_weights'])
    latent_summary={f'latent.{name}.mean':float(np.nanmean(value)) if np.isfinite(value).any() else np.nan for name,value in latent.items()}
    return summary,distance,blocks,latent_summary


def prior_predictive(model,config,followup,targets,rng,draws=None,prior_scale=1.0):
    draws=config['prior_draws'] if draws is None else draws
    parameter_rows=[]
    metric_rows=[]
    block_rows=[]
    for draw in range(draws):
        theta=sample_prior(model,rng,prior_scale)
        summary,distance,blocks,latent=simulate_summary(model,theta,config,followup,targets,rng)
        parameter_rows.append({'model':model,'draw':draw,'distance':distance,**theta,**latent})
        metric_rows.extend({'model':model,'draw':draw,'metric':metric,'value':value} for metric,value in summary.items())
        blocks=blocks.assign(model=model,draw=draw)
        block_rows.append(blocks)
    return pd.DataFrame(parameter_rows),pd.DataFrame(metric_rows),pd.concat(block_rows,ignore_index=True)


def prior_predictive_check(metrics,targets,model):
    quantiles=metrics.groupby('metric')['value'].quantile([0.05,0.5,0.95]).unstack()
    quantiles.columns=['prior_q05','prior_median','prior_q95']
    check=targets.merge(quantiles,left_on='metric',right_index=True,how='left')
    check['model']=model
    check['covered_90']=(check['target']>=check['prior_q05'])&(check['target']<=check['prior_q95'])
    check['median_standardized_error']=(check['prior_median']-check['target']).abs()/check['scale']
    return check


def _simulate_candidate(model,theta,config,followup,targets,rng):
    _,distance,blocks,_=simulate_summary(model,theta,config,followup,targets,rng)
    return distance,{row.block:row.distance for row in blocks.itertuples()}


def abc_smc(model,config,followup,targets,rng):
    n_particles=config['abc_particles']
    n_candidates=config['abc_candidates_per_generation']
    names=list(prior_specs(model))
    records=[]
    previous=None
    for generation in range(config['abc_generations']):
        candidates=[]
        if generation==0:
            for attempt in range(n_candidates):
                theta=sample_prior(model,rng)
                distance,blocks=_simulate_candidate(model,theta,config,followup,targets,rng)
                candidates.append({'theta':theta,'z':theta_to_z(model,theta),'distance':distance,'blocks':blocks,'attempt':attempt})
            candidates=sorted(candidates,key=lambda x:x['distance'])[:n_particles]
            epsilon_target=candidates[-1]['distance']
            epsilon_actual=epsilon_target
            weights=np.repeat(1/len(candidates),len(candidates))
            relaxed=False
        else:
            prev_z=np.vstack(previous['z'])
            prev_w=previous['weight'].to_numpy(dtype=float)
            center=np.sum(prev_z*prev_w[:,None],axis=0)
            variance=np.sum(prev_w[:,None]*(prev_z-center)**2,axis=0)
            kernel_sd=np.maximum(np.sqrt(2*variance)*config['abc_kernel_scale'],0.15)
            epsilon_target=float(np.quantile(previous['distance'],config['abc_epsilon_quantile']))
            proposals=[]
            for attempt in range(n_candidates):
                parent=rng.choice(len(previous),p=prev_w)
                z=prev_z[parent]+rng.normal(0,kernel_sd)
                theta=z_to_theta(model,z)
                distance,blocks=_simulate_candidate(model,theta,config,followup,targets,rng)
                proposals.append({'theta':theta,'z':z,'distance':distance,'blocks':blocks,'attempt':attempt})
            accepted=[x for x in proposals if x['distance']<=epsilon_target]
            relaxed=len(accepted)<n_particles
            candidates=sorted(accepted if not relaxed else proposals,key=lambda x:x['distance'])[:n_particles]
            epsilon_actual=max(x['distance'] for x in candidates)
            log_weights=[]
            for candidate in candidates:
                log_numerator=norm.logpdf(candidate['z']).sum()
                log_kernel=norm.logpdf((candidate['z'][None,:]-prev_z)/kernel_sd).sum(axis=1)-np.log(kernel_sd).sum()
                log_denominator=logsumexp(np.log(prev_w)+log_kernel)
                log_weights.append(log_numerator-log_denominator)
            weights=np.exp(np.array(log_weights)-logsumexp(log_weights))
        generation_rows=[]
        for particle,(candidate,weight) in enumerate(zip(candidates,weights)):
            row={'model':model,'generation':generation,'particle':particle,'weight':weight,'distance':candidate['distance'],'epsilon_target':epsilon_target,'epsilon_actual':epsilon_actual,'relaxed':relaxed,'attempt':candidate['attempt'],**candidate['theta']}
            row.update({f'distance.{name}':value for name,value in candidate['blocks'].items()})
            records.append(row)
            generation_rows.append({'z':candidate['z'],'weight':weight,'distance':candidate['distance']})
        previous=pd.DataFrame(generation_rows)
    return pd.DataFrame(records)


def weighted_sample(frame,n,rng):
    weights=frame['weight'].to_numpy(dtype=float)
    weights=weights/weights.sum()
    return frame.iloc[rng.choice(len(frame),size=n,replace=True,p=weights)].reset_index(drop=True)


def posterior_predictive(particles,config,followup,targets,rng):
    metric_rows=[]
    draw_rows=[]
    for model,part in particles.groupby('model'):
        final=part.loc[part['generation']==part['generation'].max()]
        sampled=weighted_sample(final,config['posterior_draws'],rng)
        names=list(prior_specs(model))
        for draw,row in sampled.iterrows():
            theta={name:row[name] for name in names}
            summary,distance,blocks,latent=simulate_summary(model,theta,config,followup,targets,rng)
            draw_rows.append({'model':model,'draw':draw,'distance':distance,**theta,**latent})
            metric_rows.extend({'model':model,'draw':draw,'metric':metric,'value':value} for metric,value in summary.items())
    return pd.DataFrame(draw_rows),pd.DataFrame(metric_rows)


def posterior_predictive_check(metrics,targets):
    quantiles=metrics.groupby(['model','metric'])['value'].quantile([0.05,0.5,0.95]).unstack().reset_index()
    quantiles=quantiles.rename(columns={0.05:'posterior_q05',0.5:'posterior_median',0.95:'posterior_q95'})
    check=quantiles.merge(targets,on='metric',how='inner')
    check['covered_90']=(check['target']>=check['posterior_q05'])&(check['target']<=check['posterior_q95'])
    check['median_standardized_error']=(check['posterior_median']-check['target']).abs()/check['scale']
    return check


def model_screening(check):
    by_block=check.groupby(['model','block'],as_index=False).agg(block_error=('median_standardized_error','median'),coverage_90=('covered_90','mean'),metrics=('metric','size'))
    overall=by_block.groupby('model',as_index=False).agg(median_block_error=('block_error','median'),mean_block_error=('block_error','mean'),mean_coverage_90=('coverage_90','mean'),blocks=('block','size'))
    overall['rank']=overall['median_block_error'].rank(method='min').astype(int)
    return overall.sort_values(['rank','mean_block_error']),by_block


def _resample_state(state,index):
    return {name:value[index].copy() for name,value in state.items()}


def systematic_resample(weights,rng):
    positions=(rng.random()+np.arange(len(weights)))/len(weights)
    cumulative=np.cumsum(weights)
    return np.searchsorted(cumulative,positions)


def abc_particle_filter(y,model,theta,config,rng):
    particles=config['particle_count']
    state=initialize_state(particles,model,theta,rng,config['max_projects'])
    weights=np.repeat(1/particles,particles)
    epsilon=config['particle_filter_epsilon']
    ratio=np.exp(-1/epsilon)
    kernel_constant=(1-ratio)/(1+ratio)
    log_score=0.0
    rows=[]
    for t,observed in enumerate(y):
        state,diagnostics=garden_step(state,model,theta,rng,config['max_projects'])
        if np.isfinite(observed):
            kernel=kernel_constant*ratio**np.abs(diagnostics['H']-observed)
            increment=np.sum(weights*kernel)
            log_score+=np.log(max(increment,1e-300))
            weights=weights*kernel/max(increment,1e-300)
        ess=1/np.sum(weights**2)
        rows.append({
            'age':t,'observed_h':observed,'filtered_h':np.sum(weights*diagnostics['H']),
            'filtered_g':np.sum(weights*diagnostics['G']),'filtered_k':np.sum(weights*diagnostics['K']),
            'filtered_w':np.nansum(weights*diagnostics['W']) if np.isfinite(diagnostics['W']).any() else np.nan,
            'ess':ess,'log_score':log_score
        })
        if ess<config['particle_filter_ess_fraction']*particles:
            index=systematic_resample(weights,rng)
            state=_resample_state(state,index)
            weights=np.repeat(1/particles,particles)
    return pd.DataFrame(rows),log_score


def hypothesis_text(model):
    text={
        'competing_risks':'Poisson planting, a hard garden capacity, and constant project-level competing harvest/death hazards. Constant harvest hazard is the geometric waiting-time baseline.',
        'effort_threshold':'Annual effort is lognormal and allocated with normalized Gamma weights. Allocation below the fixed unit maintenance requirement causes death; surplus effort raises harvest intensity.',
        'age_hazards':'Harvest and death follow discrete Weibull cause-specific hazards. Shape one recovers the constant geometric hazard; shapes below or above one permit decreasing or increasing project-age hazards.',
        'project_frailty':'Each project receives persistent lognormal harvest and death frailties at planting. Mixing constant hazards across projects can generate non-geometric population decay without scholar-specific creativity.',
        'dynamic_context':'Effort and soil follow correlated AR(1) states. Effort controls maintenance and harvest; standardized soil changes capacity only. The population seed-arrival rate remains fixed.'
    }
    return text[model]
