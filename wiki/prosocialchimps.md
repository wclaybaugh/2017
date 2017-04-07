---
title: Prosocial Chimps
shorttitle: prosocialchimps
notebook: prosocialchimps.ipynb
noline: 1
summary: ""
layout: wiki
---
{% assign links = site.data.wikilinks %}

![](images/pchimps.png)

>Chimpanzee prosociality experiment, as seen from the perspective of the focal animal. The left and right levers are indicated in the foreground. Pulling either expands an accordion device in the center, pushing the food trays towards both ends of the table. Both food trays close to the focal animal have food in them. Only one of the food trays on the other side contains food. The partner condition means another animal, as pictured, sits on the other end of the table. Otherwise, the other end was empty. (McElreath 293)





```python
%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import pymc3 as pm
```




## Data



```python
df=pd.read_csv("chimpanzees2.csv", sep=";")
df.head()
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actor</th>
      <th>recipient</th>
      <th>condition</th>
      <th>block</th>
      <th>trial</th>
      <th>prosoc_left</th>
      <th>chose_prosoc</th>
      <th>pulled_left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



>We're going to focus on `pulled_left` as the outcome to predict, with `prosoc_left` and `condition` as predictor variables. The outcome `pulled_left` is a 0 or 1 indicator that the focal animal pulled the left-hand lever. The predictor `prosoc_left` is a 0/1 indicator that the left-hand lever was (1) or was not (0) attached to the prosocial option, the side with two pieces of food. The `condition` predictor is another 0/1 indicator, with value 1 for the partner condition and value 0 for the control condition. (McElreath 293)



```python
df.shape
```





    (504, 8)





```python
gd={}
for k, v in df.groupby('actor'):
    temp = v.groupby(['condition', 'prosoc_left'])['pulled_left'].mean()
    gd[k] = temp.values
    #print(k, ldf.values)
```




```python
gd
```





    {1: array([ 0.33333333,  0.5       ,  0.27777778,  0.55555556]),
     2: array([1, 1, 1, 1]),
     3: array([ 0.27777778,  0.61111111,  0.16666667,  0.33333333]),
     4: array([ 0.33333333,  0.5       ,  0.11111111,  0.44444444]),
     5: array([ 0.33333333,  0.55555556,  0.27777778,  0.5       ]),
     6: array([ 0.77777778,  0.61111111,  0.55555556,  0.61111111]),
     7: array([ 0.77777778,  0.83333333,  0.94444444,  1.        ])}



## 3 different models

![](images/modelfull.png)



```python
def full_model():
    with pm.Model() as ps1:
        betapc = pm.Normal("betapc", 0, 10)
        betap = pm.Normal("betap", 0, 10)
        alpha = pm.Normal('alpha', 0, 10)
        logitpi = alpha + (betap + betapc*df.condition)*df.prosoc_left
        o = pm.Bernoulli("pulled_left", p=pm.math.invlogit(logitpi), observed=df.pulled_left)
        
    return ps1
```


![](images/modelicept.png)



```python
def ionly_model():
    with pm.Model() as ps0:
        alpha = pm.Normal('alpha', 0, 10)
        logitpi = alpha 
        o = pm.Bernoulli("pulled_left", p=pm.math.invlogit(logitpi), observed=df.pulled_left)
    return ps0
```


![](images/modelnocong.png)



```python
def plonly_model():
    with pm.Model() as plonly:
        betap = pm.Normal("betap", 0, 10)
        alpha = pm.Normal('alpha', 0, 10)
        logitpi = alpha + betap*df.prosoc_left
        o = pm.Bernoulli("pulled_left", p=pm.math.invlogit(logitpi), observed=df.pulled_left)
    return plonly
```




```python
ionly = ionly_model()
with ionly:
    trace_ionly=pm.sample(2000)
```


      6%|▌         | 11544/200000 [00:01<00:18, 10017.93it/s]| 763/200000 [00:00<00:26, 7628.10it/s]
    100%|██████████| 2000/2000 [00:00<00:00, 2043.77it/s]




```python
pm.autocorrplot(trace_ionly)
```





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x129aa2470>]], dtype=object)




![png](prosocialchimps_files/prosocialchimps_17_1.png)




```python
def invlogit(x):
    return np.exp(x) / (1 + np.exp(x))
```




```python
plt.hist(invlogit(trace_ionly['alpha']), bins=50);
```



![png](prosocialchimps_files/prosocialchimps_19_0.png)




```python
pm.quantiles(invlogit(trace_ionly['alpha'])), np.mean(invlogit(trace_ionly['alpha']))
```





    ({2.5: 0.53479792470734233,
      25: 0.56532086101628731,
      50: 0.57873578697599359,
      75: 0.5939058480054924,
      97.5: 0.62205924869921014},
     0.57907787686874779)





```python
full = full_model()
with full:
    trace_full=pm.sample(2000)
```


     10%|▉         | 19533/200000 [00:04<00:40, 4497.62it/s] | 140/200000 [00:00<02:22, 1398.67it/s]
    100%|██████████| 2000/2000 [00:04<00:00, 429.90it/s]




```python
plonly = plonly_model()
with plonly:
    trace_plonly=pm.sample(2000)
```


      0%|          | 0/200000 [00:00<?, ?it/s]  0%|          | 601/200000 [00:00<00:33, 6004.71it/s]
    100%|██████████| 2000/2000 [00:02<00:00, 713.59it/s]


## Model Comparison



```python
def make_compare(names, traces, models, ic='WAIC'):
    comparedf=pm.compare(traces,models)
    temp=comparedf.sort_index()
    temp['Model']=names
    comparedf = temp.sort(ic).set_index('Model')
    return comparedf
```




```python
names=['intercept only', 'full', 'prosocial left only']
dfc=make_compare(names, [trace_ionly, trace_full, trace_plonly],[ionly, full, plonly])
dfc
```


    //anaconda/envs/py35/lib/python3.5/site-packages/ipykernel/__main__.py:5: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAIC</th>
      <th>pWAIC</th>
      <th>dWAIC</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>prosocial left only</th>
      <td>680.662</td>
      <td>2.08242</td>
      <td>0</td>
      <td>0.70398</td>
      <td>9.3148</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>full</th>
      <td>682.521</td>
      <td>3.09051</td>
      <td>1.85955</td>
      <td>0.277821</td>
      <td>9.41291</td>
      <td>0.0322596</td>
      <td>0</td>
    </tr>
    <tr>
      <th>intercept only</th>
      <td>687.972</td>
      <td>1.01602</td>
      <td>7.31072</td>
      <td>0.0181994</td>
      <td>7.09938</td>
      <td>1.70941</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





```python
def compare_plot(comp_df, ax=None):
    """
    Model comparison summary plot in the style of the one used in the book
    Statistical Rethinking by Richard McElreath.
    Parameters
    ----------
    comp_df: DataFrame
        The result of the pm.compare() function
    ax : axes
        Matplotlib axes. Defaults to None.
    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots()

    yticks_pos, step = np.linspace(0, -1, (comp_df.shape[0] * 2) - 1, retstep=True)
    yticks_pos[1::2] = yticks_pos[1::2] + step / 2

    yticks_labels = [''] * len(yticks_pos)
    yticks_labels[0] = comp_df.index[0]
    yticks_labels[1::2] = comp_df.index[1:]

    data = comp_df.values
    min_ic = data[0, 0]

    ax.errorbar(x=data[:, 0], y=yticks_pos[::2], xerr=data[:, 4],
                fmt='ko', mfc='None', mew=1)
    ax.errorbar(x=data[1:, 0], y=yticks_pos[1::2],
                xerr=data[1:, 5], fmt='^', color='grey')

    ax.plot(data[:, 0] - (2 * data[:, 1]), yticks_pos[::2], 'ko')
    ax.axvline(min_ic, ls='--', color='grey')

    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(yticks_labels)
    ax.set_xlabel('Deviance')
    lims=ax.get_ylim()
    ax.set_ylim(lims[0] + step/2., lims[1] - step/2.)
    return ax
```




```python
compare_plot(dfc)
```





    <matplotlib.axes._subplots.AxesSubplot at 0x129509048>




![png](prosocialchimps_files/prosocialchimps_27_1.png)




```python
pm.summary(trace_full)
```


    
    betapc:
    
      Mean             SD               MC Error         95% HPD interval
      -------------------------------------------------------------------
      
      -0.110           0.271            0.007            [-0.670, 0.378]
    
      Posterior quantiles:
      2.5            25             50             75             97.5
      |--------------|==============|==============|--------------|
      
      -0.649         -0.301         -0.105         0.077          0.417
    
    
    betap:
    
      Mean             SD               MC Error         95% HPD interval
      -------------------------------------------------------------------
      
      0.616            0.225            0.006            [0.165, 1.049]
    
      Posterior quantiles:
      2.5            25             50             75             97.5
      |--------------|==============|==============|--------------|
      
      0.175          0.467          0.613          0.768          1.062
    
    
    alpha:
    
      Mean             SD               MC Error         95% HPD interval
      -------------------------------------------------------------------
      
      0.047            0.127            0.003            [-0.221, 0.281]
    
      Posterior quantiles:
      2.5            25             50             75             97.5
      |--------------|==============|==============|--------------|
      
      -0.209         -0.035         0.048          0.135          0.307
    


## Posteriors and Posterior predictives



```python
def trace_or_zero(trace, name):
    if name in trace.varnames:
        return trace[name]
    else:
        return np.zeros(len(trace))
```




```python
def model_pp(gridx, tracedict):
    temp = tracedict['alpha'] + gridx['P']*(tracedict['betap'] + tracedict['betapc']*gridx['C'])
    return temp
```




```python
def compute_pp(lpgrid, trace, paramnames, sampdistrib, invlink, inner_pp):
    tdict={}
    for pn in paramnames:
        tdict[pn] = trace_or_zero(trace, pn)
    print(tdict.keys())
    tl=len(trace)
    gl=len(lpgrid)
    pp = np.empty((gl, tl))
    for i, v in enumerate(lpgrid):
        temp = inner_pp(lpgrid[i], tdict)
        pp[i,:] = sampdistrib(invlink(temp))
    return pp
```




```python
import itertools
psleft = [0,1]
condition = [0,1]
xgrid = [{'C':v[0], 'P':v[1]} for v in itertools.product(condition, psleft)]
```




```python
xgrid
```





    [{'C': 0, 'P': 0}, {'C': 0, 'P': 1}, {'C': 1, 'P': 0}, {'C': 1, 'P': 1}]





```python
from scipy.stats import bernoulli, binom
```




```python
ppdivisor=7
def like_sample(p_array):
    ppdivisor=7
    return binom.rvs(ppdivisor, p=p_array)
```




```python
ppfull = compute_pp(xgrid, trace_full, trace_full.varnames, like_sample, invlogit, model_pp)
```




```python
ppfull
```





    array([[ 5.,  5.,  5., ...,  2.,  3.,  2.],
           [ 6.,  4.,  7., ...,  2.,  6.,  5.],
           [ 4.,  4.,  5., ...,  2.,  1.,  3.],
           [ 6.,  3.,  3., ...,  6.,  5.,  4.]])





```python
meanpp, stdpp = ppfull.mean(axis=1), ppfull.std(axis=1)
```




```python
meanpp, stdpp
```





    (array([ 3.6065,  4.5925,  3.5595,  4.392 ]),
     array([ 1.34746345,  1.25277442,  1.33957447,  1.31428155]))





```python
fmt = lambda d: ",".join([e+"="+str(d[e]) for e in d])
plt.plot(range(4),meanpp/ppdivisor, lw=3, color="black")
for chimp in gd:
    plt.plot(range(4), gd[chimp], label=)
plt.fill_between(range(4), (meanpp-stdpp)/ppdivisor, (meanpp+stdpp)/ppdivisor, alpha=0.1, color="gray")
plt.ylim([0,1])
plt.xticks(range(4),[fmt(e) for e in xgrid])
```





    ([<matplotlib.axis.XTick at 0x1299bf550>,
      <matplotlib.axis.XTick at 0x129346860>,
      <matplotlib.axis.XTick at 0x1251acf60>,
      <matplotlib.axis.XTick at 0x1247df358>],
     <a list of 4 Text xticklabel objects>)




![png](prosocialchimps_files/prosocialchimps_41_1.png)


## Ensemble the model



```python
for m in dfc.index:
    print(m)
```


    prosocial left only
    full
    intercept only




```python
modeldict={
    "prosocial left only": trace_plonly,
    "full": trace_full,
    "intercept only": trace_ionly
}
```




```python
def ensemble(grid, modeldict, paramnames, comparedf):
    accum_pp=0
    accum_weight=0
    for m in comparedf.index:
        weight = comparedf.ix[m]['weight']
        pp=compute_pp(grid, modeldict[m], paramnames, like_sample, invlogit, model_pp)
        print(m, weight, np.median(pp))
        accum_pp += pp*weight
        accum_weight +=weight
    return accum_pp/accum_weight
```




```python
ppens = ensemble(xgrid, modeldict, ['alpha', 'betap', 'betapc'], dfc)
```


    dict_keys(['alpha', 'betap', 'betapc'])
    prosocial left only 0.703979782921 4.0
    dict_keys(['alpha', 'betap', 'betapc'])
    full 0.27782078442 4.0
    dict_keys(['alpha', 'betap', 'betapc'])
    intercept only 0.0181994326587 4.0




```python
meanpp, stdpp = ppens.mean(axis=1), ppens.std(axis=1)
fmt = lambda d: ",".join([e+"="+str(d[e]) for e in d])
plt.plot(range(4),meanpp/ppdivisor, lw=3, color="black")
for i, chimp in enumerate(gd):
    plt.plot(range(4), gd[chimp], label="actor{}".format(i), lw=3)
plt.fill_between(range(4), (meanpp-stdpp)/ppdivisor, (meanpp+stdpp)/ppdivisor, alpha=0.1, color="gray")
plt.ylim([0,1.1])
plt.xticks(range(4),[fmt(e) for e in xgrid])
plt.legend();
```



![png](prosocialchimps_files/prosocialchimps_47_0.png)


## A heirarchical model

Here we have a varying intercepts model

![](images/multichimp.png)



```python
def vi_model():
    with pm.Model() as vi:
        betapc = pm.Normal("betapc", 0, 10)
        betap = pm.Normal("betap", 0, 10)
        alpha = pm.Normal('alpha', 0, 10)
        sigma_actor = pm.HalfCauchy("sigma_actor", 1)
        alpha_actor = pm.Normal('alpha_actor', 0, sigma_actor, shape=7)
        logitpi = alpha + alpha_actor[df.index//72] + (betap + betapc*df.condition)*df.prosoc_left
        o = pm.Bernoulli("pulled_left", p=pm.math.invlogit(logitpi), observed=df.pulled_left)
        
    return vi
```




```python
vi = vi_model()
with vi:
    vi_trace=pm.sample(10000)
```


    Average ELBO = -287.67:  26%|██▌       | 51940/200000 [00:15<00:39, 3736.14it/s]22, 2435.62it/s]
    100%|██████████| 10000/10000 [01:16<00:00, 131.55it/s]




```python
pm.traceplot(vi_trace);
```



![png](prosocialchimps_files/prosocialchimps_52_0.png)




```python
pm.autocorrplot(vi_trace);
```



![png](prosocialchimps_files/prosocialchimps_53_0.png)




```python
pm.plot_posterior(vi_trace, kde_plot=True);
```



![png](prosocialchimps_files/prosocialchimps_54_0.png)




```python
pm.forestplot(vi_trace);
```



![png](prosocialchimps_files/prosocialchimps_55_0.png)




```python
trace.varnames
```





    ['betapc', 'betap', 'alpha', 'sigma_actor_log_', 'alpha_actor', 'sigma_actor']



### Predictives are on individuals now



```python
def like_sample_hier(p_array):
    return bernoulli.rvs(p=p_array)
```




```python
def model_pp_hier(gridx, tracedict, chimp):
    temp = tracedict['alpha'] + tracedict['alpha_actor'][:,chimp]+gridx['P']*(tracedict['betap'] + tracedict['betapc']*gridx['C'])
    return temp
```




```python
def compute_pp2(lpgrid, trace, paramnames, sampdistrib, invlink, inner_pp, chimp):
    tdict=trace
    tl=len(trace)
    gl=len(lpgrid)
    pp = np.empty((gl, tl))
    for i, v in enumerate(lpgrid):
        temp = inner_pp(lpgrid[i], tdict, chimp)
        pp[i,:] = invlink(temp)
    return pp
```




```python
vnames=['betapc', 'betap', 'alpha', 'alpha_actor']
pphier0=compute_pp2(xgrid, vi_trace, vnames, like_sample_hier, invlogit, model_pp_hier, 0)
```




```python
ppdivisor=1
meanpp, stdpp = pphier0.mean(axis=1), pphier0.std(axis=1)
fmt = lambda d: ",".join([e+"="+str(d[e]) for e in d])
plt.plot(range(4),meanpp/ppdivisor, lw=3, color="black")
plt.plot(range(4), gd[1], label="actor{}".format(1), lw=3)
plt.fill_between(range(4), (meanpp-stdpp)/ppdivisor, (meanpp+stdpp)/ppdivisor, alpha=0.1, color="gray")
plt.ylim([0,1.1])
plt.xticks(range(4),[fmt(e) for e in xgrid])
plt.legend();
```



![png](prosocialchimps_files/prosocialchimps_62_0.png)




```python
pphier6=compute_pp2(xgrid, vi_trace, vnames, like_sample_hier, invlogit, model_pp_hier, 6)
```




```python
ppdivisor=1
meanpp, stdpp = pphier6.mean(axis=1), pphier6.std(axis=1)
fmt = lambda d: ",".join([e+"="+str(d[e]) for e in d])
plt.plot(range(4),meanpp/ppdivisor, lw=3, color="black")
plt.plot(range(4), gd[7], label="actor{}".format(7), lw=3)
plt.fill_between(range(4), (meanpp-stdpp)/ppdivisor, (meanpp+stdpp)/ppdivisor, alpha=0.1, color="gray")
plt.ylim([0,1.1])
plt.xticks(range(4),[fmt(e) for e in xgrid])
plt.legend();
```



![png](prosocialchimps_files/prosocialchimps_64_0.png)


## Varying experimental blocks as well

>The kind of data structure here is usually called a CROSS-CLASSIFIED multilevel model. It is cross-classified, because actors are not nested within unique blocks. If each chimpanzee had instead done all of his or her pulls on a single day, within a single block, then the data structure would instead be hierarchical. However, the model specification would typically be the same. So the model structure and code you'll see below will apply both to cross-classified designs and hierarchical designs. Other software sometimes forces you to treat these differently, on account of using a conditioning engine substantially less capable than MCMC. There are other types of “hierarchical” multilevel models, types that make adaptive priors for adaptive priors. It's turtles all the way down, recall (page 13). You'll see an example in the next chapter. But for the most part, people (or their software) nearly always use the same kind of model in both cases. (McElreath 371)

>Each pull is within a cluster of pulls belonging to an individual chimpanzee. But each pull is also within an experimental block, which represents a collection of observations that happened on the same day. So each observed pull belongs to both an actor (1 to 7) and a block (1 to 6). There may be unique intercepts for each actor as well as for each block. (McElreath 370-371)

![](images/multichimpcluster.png)



```python
df.head(50)
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actor</th>
      <th>recipient</th>
      <th>condition</th>
      <th>block</th>
      <th>trial</th>
      <th>prosoc_left</th>
      <th>chose_prosoc</th>
      <th>pulled_left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>24</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>36</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>38</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>44</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>46</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>50</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>56</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>58</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>62</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>64</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>66</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>68</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>70</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>72</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>21</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>27</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





```python
def viplusblock_model():
    with pm.Model() as vipb:
        betapc = pm.Normal("betapc", 0, 10)
        betap = pm.Normal("betap", 0, 10)
        alpha = pm.Normal('alpha', 0, 10)
        sigma_actor = pm.HalfCauchy("sigma_actor", 1)
        sigma_block = pm.HalfCauchy("sigma_block", 1)
        alpha_actor = pm.Normal('alpha_actor', 0, sigma_actor, shape=7)
        alpha_block = pm.Normal('alpha_block', 0, sigma_block, shape=6)
        logitpi = alpha + alpha_actor[df.index//72] + alpha_block[df.block.values//6]+ (betap + betapc*df.condition)*df.prosoc_left
        o = pm.Bernoulli("pulled_left", p=pm.math.invlogit(logitpi), observed=df.pulled_left)
        
    return vipb
```




```python
vipb=viplusblock_model()
with vipb:
    trace_vipb = pm.sample(20000)
```


    Average ELBO = -290.76: 100%|██████████| 200000/200000 [01:07<00:00, 2957.33it/s]0, 3287.83it/s]
    100%|██████████| 20000/20000 [05:35<00:00, 59.59it/s] 




```python
pm.autocorrplot(trace_vipb);
```



![png](prosocialchimps_files/prosocialchimps_70_0.png)




```python
pm.traceplot(trace_vipb);
```



![png](prosocialchimps_files/prosocialchimps_71_0.png)




```python
sns.distplot(trace_vipb['sigma_block'])
sns.distplot(trace_vipb['sigma_actor']);
plt.xlim([-1,10])
```


    //anaconda/envs/py35/lib/python3.5/site-packages/statsmodels/nonparametric/kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j





    (-1, 10)




![png](prosocialchimps_files/prosocialchimps_72_2.png)




```python
pm.forestplot(trace_vipb);
```



![png](prosocialchimps_files/prosocialchimps_73_0.png)


Do this and compare the WAICs.



```python
dfc2=pm.compare([vi_trace, trace_vipb],[vi, vipb])
```




```python
compare_plot(dfc2)
```





    <matplotlib.axes._subplots.AxesSubplot at 0x12f48e588>




![png](prosocialchimps_files/prosocialchimps_76_1.png)

