from src.utils import calibrated_twin, intermittent_policy

"""OncoEvo Research Console — offline computational research, not clinical advice."""

from pathlib import Path

import numpy as np

import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import streamlit as st

from src.data import load_patients, calibration_prefix

from src.tumor_model import fit_prefix, historical_prediction, state_at_end

from src.evaluate import replay

from src.utils import load_config, ContinuousPolicy, IntermittentPolicy



st.set_page_config(
    page_title="OncoEvo Research Console",
    page_icon="◈",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: #ffffff;
        color: #1e293b;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #172554;
        letter-spacing: -0.035em;
    }

    .stMetric {
        background: #f1f5f9;
        border: 1px solid #dbe3ec;
        border-radius: 14px;
        padding: 15px;
    }

    [data-testid="stSidebar"] {
        background: #f8fafc;
    }

    .eyebrow {
        color: #0d9488;
        letter-spacing: 0.18em;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .muted {
        color: #64748b;
    }

    .hero {
        padding: 10px 0 24px;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="eyebrow">
        EVOLUTIONARY ONCOLOGY · LOCAL RESEARCH PROTOTYPE
    </div>
    <h1>OncoEvo Research Console</h1>
    <div class="muted">
        Patient informed simulation of adaptive androgen deprivation schedules
        · Offline computational research only
    </div>
</div>
""", unsafe_allow_html=True)



@st.cache_data

def data():

    return load_patients()



@st.cache_data

def results():

    files = [Path("results") / n for n in ("policy_metrics.csv", "traces.csv", "observed_forecasts.csv")]

    return tuple(pd.read_csv(f) if f.exists() else pd.DataFrame() for f in files)



try:

    patients, audit = data()

except (FileNotFoundError, ValueError) as exc:

    st.error(f"Raw dataset unavailable or invalid: {exc}")

    st.stop()

config = load_config()

metrics, traces, forecasts = results()

colors={"PPO":"#52d8c9","Continuous":"#a590ef","Intermittent threshold":"#e9bb76"}

section = st.sidebar.radio("Workspace", ["Patient Explorer", "Digital Tumor Twin", "RL Treatment Laboratory", "Research Results"])

patient_id = st.sidebar.selectbox("Patient series", sorted(patients))

frame = patients[patient_id]

prefix = calibration_prefix(frame, config["calibration_fraction"])



def styled(fig, height=480):

    fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1624", plot_bgcolor="#0d1624",

                      font=dict(color="#c8d8e4"), height=height, margin=dict(l=30,r=25,t=55,b=35),

                      hovermode="x unified", legend=dict(orientation="h", y=1.12),

                      xaxis=dict(gridcolor="#233347"), yaxis=dict(gridcolor="#233347"))

    return fig



def bands(fig, series, row=None):

    days = series.day.to_numpy()

    status = series.treatment_on.to_numpy()

    for i, on in enumerate(status):

        if on:

            end = days[i+1] if i+1 < len(days) else days[i]

            fig.add_vrect(x0=days[i], x1=end, fillcolor="#5c61d9", opacity=.12,

                          line_width=0, layer="below", row=row, col=1) if row else fig.add_vrect(

                          x0=days[i], x1=end, fillcolor="#5c61d9", opacity=.12, line_width=0, layer="below")



if section == "Patient Explorer":

    st.subheader("Observed patient timeline")

    a,b,c,d = st.columns(4)

    a.metric("Measurements", len(frame)); b.metric("Follow-up", f"{frame.day.max():.0f} days")

    c.metric("Initial PSA", f"{frame.psa_ng_ml.iloc[0]:.2f} ng/mL")

    d.metric("Recorded ON visits", int(frame.treatment_on.sum()))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.09,

                        subplot_titles=("Observed PSA · ng/mL", "Reported androgen · source units unspecified"))

    fig.add_trace(go.Scatter(x=frame.day,y=frame.psa_ng_ml,mode="lines+markers",name="Observed PSA",

                             line_color="#52d8c9"),row=1,col=1)

    fig.add_trace(go.Scatter(x=frame.day,y=frame.androgen_reported,mode="lines+markers",name="Reported androgen",

                             line_color="#a590ef"),row=2,col=1)

    bands(fig, frame, 1); bands(fig, frame, 2)

    fig.update_xaxes(title_text="Days since first observation",row=2,col=1)

    st.plotly_chart(styled(fig,570),width="stretch")

    st.caption("Violet bands: recorded treatment ON from each visit until the next. Androgen units are not stated by the source repository; values are shown without unit conversion.")

    with st.expander("Source quality audit"):

        st.json(audit)



elif section == "Digital Tumor Twin":

    st.subheader("Digital tumor twin")

    twin, fit = calibrated_twin(prefix, config)

    prediction = historical_prediction(frame,twin)

    a,b,c = st.columns(3)

    a.metric("Calibration observations",len(prefix)); b.metric("Prefix log RMSE",f"{fit['prefix_log_rmse']:.3f}")

    c.metric("Forecast observations",len(frame)-len(prefix))

    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.09,

                        subplot_titles=("Historical PSA fit and forward forecast", "Latent simulated populations under recorded treatment"))

    fig.add_trace(go.Scatter(x=frame.day,y=frame.psa_ng_ml,mode="markers",name="Observed PSA",marker_color="#52d8c9"),row=1,col=1)

    fig.add_trace(go.Scatter(x=frame.day,y=prediction,name="Model PSA",line_color="#a590ef"),row=1,col=1)

    fig.add_vline(x=float(prefix.day.iloc[-1]),line_dash="dot",line_color="#e9bb76",row=1,col=1)

    from src.tumor_model import advance

    state=twin.initial_state(); latent=[state.copy()]

    for i in range(len(frame)-1):

        state=advance(state,float(frame.day.iloc[i+1]-frame.day.iloc[i]),int(frame.treatment_on.iloc[i]),twin)

        latent.append(state.copy())

    latent=np.asarray(latent)

    fig.add_trace(go.Scatter(x=frame.day,y=latent[:,0],name="Sensitive · simulated",line_color="#52d8c9"),row=2,col=1)

    fig.add_trace(go.Scatter(x=frame.day,y=latent[:,1],name="Resistant · simulated",line_color="#ee86b5"),row=2,col=1)

    bands(fig,frame,1);bands(fig,frame,2)

    st.plotly_chart(styled(fig,620),width="stretch")

    st.info("Effective growth and treatment kill are fitted to the early PSA prefix when both ON and OFF intervals exist; ON-only prefixes fit kill with growth fixed. Resistance fraction, competition, carrying capacity and PSA scale are assumed; latent cell curves are not measurements. Later observed PSA is used only for forecast assessment under the recorded treatment schedule.")



elif section == "RL Treatment Laboratory":

    st.subheader("Counterfactual policy replay")

    twin,_=calibrated_twin(prefix, config);state=state_at_end(prefix,twin)

    available=["Continuous","Intermittent threshold"]

    checkpoint=Path("models/ppo_oncoevo.zip")

    if checkpoint.exists():available.insert(0,"PPO")

    else:st.warning("PPO checkpoint missing. Run `python -m src.train` to enable PPO replay.")

    selected=st.multiselect("Policies",available,default=available)

    st.caption("Biological sliders change the simulated scenario only. Saved PPO weights and observation preprocessing stay fixed.")

    factor=st.slider("Assumed sensitive growth multiplier",0.7,1.3,1.0,0.05)

    replay_day=st.slider("Replay through day",config["decision_days"],config["horizon_days"],config["horizon_days"],config["decision_days"])

    from dataclasses import replace

    variant=replace(twin,growth=twin.growth*factor)

    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.09,

                      subplot_titles=("Simulated PSA · matched initial state", "Simulated sensitive and resistant burden"))

    for name in selected:

        if name=="PPO":

            from src.experiment import load_ppo

            policy=load_ppo(checkpoint)

        else:policy=ContinuousPolicy() if name=="Continuous" else intermittent_policy(config)

        if name == selected[0]:

            from src.experiment import ExperimentRunner

            import json

            signature=(patient_id,name,factor,json.dumps(config,sort_keys=True))

            if st.session_state.get("experiment_signature") != signature:

                st.session_state.experiment = ExperimentRunner(variant,state,config,policy,config["evaluation_seeds"][0])

                st.session_state.experiment_signature = signature

            runner=st.session_state.experiment

            with st.expander(f"Decision controls and trace · {name}"):

                controls=st.columns(6)

                if controls[0].button("Reset"): runner.reset()

                if controls[1].button("Pause"): runner.pause()

                if controls[2].button("Resume"): runner.resume()

                if controls[3].button("Step",disabled=runner.paused or runner.done): runner.step()

                if controls[4].button("Advance 5",disabled=runner.paused or runner.done): runner.advance(5)

                if controls[5].button("Replay"): runner.replay()

                st.caption(f"Day {runner.env.elapsed} · paused={runner.paused} · finished={runner.done}")

                if runner.trace: st.dataframe(pd.DataFrame(runner.trace),width="stretch")

        rows,summary=replay(variant,state,config,policy,config["evaluation_seeds"][0])

        sub=pd.DataFrame(rows);sub=sub[sub.day<=replay_day]

        envelope=[]

        for gf in config["sensitivity_growth_factors"]:

            for rf in config["sensitivity_resistance_factors"]:

                perturbed=replace(twin,growth=twin.growth*gf,resistant_growth=twin.resistant_growth*rf)

                sample,_=replay(perturbed,state,config,policy,config["evaluation_seeds"][0])

                envelope.append(pd.DataFrame(sample).set_index("day").psa_ng_ml.rename(f"{gf}_{rf}"))

        uncertainty=pd.concat(envelope,axis=1).sort_index().loc[:replay_day]

        low=uncertainty.min(axis=1); high=uncertainty.max(axis=1)

        fig.add_trace(go.Scatter(x=uncertainty.index,y=high,mode="lines",line=dict(width=0),

                                 showlegend=False,hoverinfo="skip"),row=1,col=1)

        fig.add_trace(go.Scatter(x=uncertainty.index,y=low,mode="lines",line=dict(width=0),

                                 fill="tonexty",fillcolor="rgba(82,216,201,0.09)",

                                 name=f"{name} sensitivity range",hoverinfo="skip"),row=1,col=1)

        fig.add_trace(go.Scatter(x=sub.day,y=sub.psa_ng_ml,name=name,line_color=colors[name]),row=1,col=1)

        fig.add_trace(go.Scatter(x=sub.day,y=sub.sensitive_simulated,name=f"{name} sensitive",line_color=colors[name]),row=2,col=1)

        fig.add_trace(go.Scatter(x=sub.day,y=sub.resistant_simulated,name=f"{name} resistant",line_color=colors[name],line_dash="dot"),row=2,col=1)

        st.caption(f"{name}: exposure {summary['exposure_fraction']:.0%} · progression {bool(summary['progression'])} · safety overrides {summary['safety_overrides']}")

    st.plotly_chart(styled(fig,600),width="stretch")

    st.caption("All replay curves are counterfactual ODE simulations from the calibrated prefix. Shaded ranges vary assumed growth and resistant growth rates; they are sensitivity ranges, not calibrated confidence intervals. ON/OFF decisions occur every 28 simulated days; safety overrides force ON above 1.5× starting PSA.")



else:

    st.subheader("Research results")

    if metrics.empty:

        st.warning("Evaluation outputs are missing. Run `python -m src.train` then `python -m src.evaluate`.")

    else:

        baseline=metrics[(metrics.growth_factor==1.0)&(metrics.resistance_factor==1.0)]

        a,b,c=st.columns(3)

        a.metric("Held-out patients",baseline.patient_id.nunique())

        b.metric("Virtual replays",len(metrics))

        if not forecasts.empty:

            err=np.sqrt(np.mean((np.log1p(forecasts.forecast_psa_ng_ml)-np.log1p(forecasts.observed_psa_ng_ml))**2))

            c.metric("Historical forecast log RMSE",f"{err:.3f}")

        summary=baseline.groupby("policy",as_index=False).agg(progression_rate=("progression","mean"),

            exposure_fraction=("exposure_fraction","mean"),safety_overrides=("safety_overrides","mean"))

        st.dataframe(summary.style.format({"progression_rate":"{:.1%}","exposure_fraction":"{:.1%}","safety_overrides":"{:.2f}"}),width="stretch")

        fig=go.Figure()

        for name,group in metrics.groupby("policy"):

            per_patient=group.groupby("patient_id").exposure_fraction.mean()

            low,high=np.quantile(per_patient,[.025,.975])

            fig.add_trace(go.Bar(x=[name],y=[per_patient.mean()],error_y=dict(type="data",symmetric=False,

                                  array=[high-per_patient.mean()],arrayminus=[per_patient.mean()-low]),name=name))

        fig.update_yaxes(title="Treatment exposure fraction")

        st.plotly_chart(styled(fig,430),width="stretch")

        st.caption("Bars: mean across held-out patients; whiskers: 2.5–97.5% empirical patient quantiles across seeds and growth/resistance-rate sensitivity. Progression is a modeled PSA threshold, not a clinical event. Time to progression is right-censored at the simulation horizon.")

        if patient_id in set(traces.patient_id):

            subset=traces[traces.patient_id==patient_id]

            fig=go.Figure()

            for name,group in subset.groupby("policy"):

                fig.add_trace(go.Scatter(x=group.day,y=group.psa_ng_ml,name=name,line_color=colors.get(name)))

            st.plotly_chart(styled(fig,420),width="stretch")

        if not forecasts.empty:

            st.caption(f"Observed future PSA points used solely for forecast assessment: {len(forecasts)}. No observed outcomes under PPO or alternative policies are available.")

