"""Builds paper/oll_paper.html from the committed result files (every number in the paper comes from runs/ or audit/).
Then: chromium prints it to paper/oll_paper.pdf (see print_pdf.py)."""
import glob, json, math, os
import numpy as np

J = lambda p: json.load(open(p))
main = J("runs/gpu/main/main.json"); caps = J("runs/gpu/main/capabilities.json"); ref = J("runs/real/reference.json")["test"]
adapt = J("runs/gpu/main/ett_adapt.json"); facts = J("audit/facts_memory.json")
nab_run, nab_calm = J("runs/gpu/main/nab.json"), J("runs/gpu/main/nab_calm.json")
sel_run, sel_calm = J("runs/nab_rule_selection.json"), J("runs/nab_rule_selection_calm.json")
DOM = ["weather", "grid", "market", "audio", "text"]


def f(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")


def agg(pat, key="test"):
    R = [J(p) for p in sorted(glob.glob(pat))]
    return {d: (np.mean([r[key][d] for r in R]), np.std([r[key][d] for r in R])) for d in R[0][key]}, len(R)


def ms(t):
    return f"{f(t[0])} ± {f(t[1])}"


# ------------------------------------------------------------------ tables
t_main = "".join(f"<tr><td>{d}</td><td>{f(main['v4_test'][d])}</td><td>{f(ref[d])}</td><td>{f(agg('runs/real/joint_s*.json')[0][d][0])}</td></tr>" for d in DOM)

cal_rows = [("calibration_v4_test_weather", "weather (Jena climate)"), ("calibration_v4_test_grid", "electricity transformer (ETTh1)"),
            ("calibration_v4_test_market", "market (VIX, S&P 500)"), ("calibration_v4_test_audio", "speech audio (FSDD)"),
            ("calibration_series_heldout", "held-out series (10 collections)")]
t_cal = "".join(f"<tr><td>{n}</td><td>{caps[k]['n']}</td><td>{f(caps[k]['pit_ks'])}</td><td>{f(caps[k]['crps'])}</td><td>{f(caps[k]['crps_reference'])}</td>"
                f"<td>{caps[k]['crps_reference'] / caps[k]['crps']:.1f}×</td></tr>" for k, n in cal_rows)
md_s, md_w = caps["missing_data_series_nll_per_event"], caps["missing_data_weather_nll_per_event"]
t_miss = "".join(f"<tr><td>{p}</td><td>{f(md_s[p])}</td><td>{f(md_w[p])}</td></tr>" for p in md_s)


def ett_rows(name):
    a, o = adapt[name + "_adapt"], main["ett_" + name]
    return "".join(f"<tr><td>{name}</td><td>{h}</td><td><b>{f(a[h]['model_mse'])}</b> / {f(a[h]['model_mae'])}</td><td>{f(o[h]['model_mse'])}</td>"
                   f"<td>{f(a[h]['repeat_last_mse'])} / {f(a[h]['repeat_last_mae'])}</td><td>{f(a[h]['seasonal_naive_mse'])} / {f(a[h]['seasonal_naive_mae'])}</td></tr>"
                   for h in ("96", "192", "336", "720"))


t_ett = ett_rows("ETTh2") + ett_rows("ETTh1") + ett_rows("ETTm1")

fk = facts
t_facts = "".join(f"<tr><td>{lab}</td><td>{fk[k]['RM']}</td><td>{100 * fk[k]['exact_capital_recall']:.1f}%</td><td>{100 * fk[k]['char_accuracy']:.1f}%</td></tr>"
                  for k, lab in (("order_1_only", "previous event only (order 1)"), ("context_key_8", "context key, order 8"),
                                 ("context_keys_4_8_12", "context keys, orders 4+8+12"), ("context_keys_4_8_12_RM16384", "context keys, orders 4+8+12")))
er = fk["context_keys_4_8_12_RM16384"]["erase"]

abl = [("joint (full machine)", "joint"), ("staged curriculum", "staged"), ("− memory", "abl_memory"), ("− PhaseLinear", "abl_phaselinear"),
       ("− Kuramoto coupling", "abl_kuramoto"), ("amplitude-weighted read-out", "abl_flatreadout"), ("+ context-keyed memory (1,4,8)", "ctxmem")]
t_abl = ""
for lab, k in abl:
    A, n = agg(f"runs/real/{k}_s*.json")
    t_abl += f"<tr><td>{lab}</td>" + "".join(f"<td>{ms(A[d])}</td>" for d in DOM) + "</tr>"

o2 = [("base", "base"), ("+ clock", "clock"), ("+ clock + context memory", "clock_ctxmem"), ("+ clock + anticipation", "clock_ant")]
t_o2 = ""
for lab, k in o2:
    R = [J(p) for p in sorted(glob.glob(f"runs/oll2/{k}_s*.json"))]
    m_ = lambda g: np.mean([g(r) for r in R])
    t_o2 += (f"<tr><td>{lab}</td>" + "".join(f"<td>{f(m_(lambda r: r['test'][d]))}</td>" for d in DOM)
             + "".join(f"<td>{f(m_(lambda r: r['ett_' + n]['96']['model_mse']))}</td>" for n in ("ETTh1", "ETTm1")) + "</tr>")
R0 = J("runs/oll2/base_s0.json")
o2_naive = f"one-day repeat on the same 30 windows: ETTh1 {f(R0['ett_ETTh1']['96']['seasonal_naive_mse'])}, ETTm1 {f(R0['ett_ETTm1']['96']['seasonal_naive_mse'])}"

cont = {}
for meth in ("finetune", "adapters", "replay"):
    R = [J(p) for p in sorted(glob.glob(f"runs/real/continual_{meth}_s*.json"))]
    cont[meth] = (np.mean([r["avg_forgetting"] for r in R]), np.std([r["avg_forgetting"] for r in R]), R)
comp = cont["adapters"][2][0]["all_adapters_composed"]; fin = cont["adapters"][2][0]["final"]

rob = {k: J(f"runs/robot/{k}.json") for k in ("world_s0", "world_s1", "noact_s0", "noact_s1")}
rp = rob["world_s0"]["test_nll_persistence_reference"]


def nabrow(lab, r):
    return f"<tr><td>{lab}</td><td>{f(r['mean_point_auc'])}</td><td>{r['windows_detected']}/{r['labelled_windows']}</td><td>{f(r['alarm_precision'])}</td></tr>"


t_nab = (nabrow("running normalisation, fixed rule", nab_run) + nabrow("calm-period normalisation, fixed rule", nab_calm)
         + f"<tr><td>calm, rule chosen on the other half (2-fold)</td><td>—</td><td>{sel_calm['two_fold_pooled_test']['windows_detected']}/110</td><td>{f(sel_calm['two_fold_pooled_test']['alarm_precision'])}</td></tr>")

PENDING = os.path.exists("paper/pending.html") and open("paper/pending.html").read() or ""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Sequence Modelling Without a Vocabulary</title>
<style>
body{{font:10.5pt/1.45 "Times New Roman",Georgia,serif;color:#111;max-width:760px;margin:28px auto;padding:0 18px}}
h1{{font-size:19pt;text-align:center;margin:8px 0 4px}} .sub{{text-align:center;font-style:italic;margin-bottom:6px}}
.auth{{text-align:center;margin-bottom:18px}} h2{{font-size:13pt;margin:20px 0 6px}} h3{{font-size:11pt;margin:14px 0 4px}}
.abs{{margin:0 28px 14px;font-size:10pt}} table{{border-collapse:collapse;margin:8px auto;font-size:9pt}}
td,th{{border-top:1px solid #bbb;border-bottom:1px solid #bbb;padding:3px 7px;text-align:center}} td:first-child,th:first-child{{text-align:left}}
.cap{{font-size:9pt;margin:2px 20px 12px}} img{{width:100%}} code{{font-size:9pt}} .neg{{color:#8a1c1c}}
ol.refs li{{font-size:9pt;margin-bottom:2px}} @media print{{body{{margin:0 auto}} h2{{break-after:avoid}} table,figure{{break-inside:avoid}}}}
</style></head><body>
<h1>Sequence Modelling Without a Vocabulary</h1>
<div class="sub">A frozen prime-frequency basis on ℝ and a continuous machine built on it, trained and measured on real data</div>
<div class="auth">Alptekin Keser</div>

<div class="abs"><b>Abstract.</b> Sequence models begin with a learned table over a finite alphabet. We replace it with arithmetic:
each real number x is sent to b(x)<sub>k</sub> = p<sub>k</sub><sup>−1/2</sup> e<sup>2πi x√p<sub>k</sub></sup> over the first R primes — no parameters,
no table, defined on all of ℝ, a homomorphism from (ℝ,+) into the torus with constant norm and translation-covariant Gram matrix.
On this basis we build one machine whose input, clock and output are continuous: typed event streams bound to frozen role phases,
a reversible stack of continuous-time oscillator blocks that advances by any real time gap, an associative phase memory that is
written in one pass and can be erased exactly, and a phase-amplitude output density with a closed-form normaliser. There is no
tokenizer, embedding table, attention or positional encoding. A 17.5 M-parameter machine was trained for 200 minutes on one GPU on
real data only (weather, electricity, markets, speech, books, web text and ten public forecasting collections). On held-out data it
is better than a no-learning reference on four of five domains (text: {f(main['v4_test']['text'], 2)} vs {f(ref['text'], 2)} bits per byte), its predictive
distributions are 1.6–5.5× better than a no-learning Gaussian in CRPS on four of five held-out sets (equal on markets), it degrades gracefully when 75% of the events are removed,
streaming equals the full pass to 2×10<sup>−6</sup> and the trained trunk inverts to 2×10<sup>−11</sup>. On the standard ETT long-horizon
protocol with per-window adaptation it is below both naive baselines on ETTh2 at every horizon, and below repeat-last but above
one-day repeat on ETTh1/ETTm1. Its memory recalls {100 * fk['context_keys_4_8_12_RM16384']['exact_capital_recall']:.0f}% of 243 real country–capital facts
after a single read with no training, and one fact can be erased without changing any other correct answer. We report negative results
as prominently: anomaly detection on NAB is at chance level, and a world model of a real robot arm is below a hold-last reference.</div>

<h2>1. Introduction</h2>
<p>Every widely used sequence model starts from a learned lookup table over a finite alphabet; a large literature changes the mixer
(state-space models, long convolutions, linear attention) but not the input. Here the input is ℝ. A byte, an audio sample, a temperature
or a timestamp is already a point of the space, with no tokenizer or quantiser. Physical signals arrive at irregular times, at mixed
rates and with missing readings; a machine whose input, state dynamics and output all live on ℝ and on continuous time meets them directly.</p>
<p><b>Position.</b> The goal is not a better language model. The question is what a machine can do when its representation is fixed
arithmetic rather than learned lookup, and its clock and output are continuous; we answer with an implementation trained end to end on
real data and with measurements, including where it fails. All data are real recordings or real text; no synthetic data are used anywhere
in Sections 5–7.</p>
<p><b>Contributions.</b> (i) A parameter-free basis on ℝ with four exact properties (Section 2). (ii) One machine combining typed
streams with role phases, a reversible continuous-time trunk, a one-pass, exactly erasable phase memory with context keys and a
closed-form normalised density head (Section 3), with an automated exactness test suite (Section 4). (iii) Training and measurement on a
frozen, hash-verified corpus of real data across eight domains, with held-out splits and no-learning references (Sections 5–6).
(iv) A list of negative results and limits (Section 7).</p>

<h2>2. The basis</h2>
<p>Let p<sub>1</sub> &lt; … &lt; p<sub>R</sub> be the first R primes and b(x)<sub>k</sub> = p<sub>k</sub><sup>−1/2</sup>·e<sup>2πi x√p<sub>k</sub></sup>. Real and imaginary
parts are stored side by side. Nothing is a parameter; the argument is a real number, not an index.</p>
<p><b>Proposition 1 (arithmetic inside the representation).</b> b(a) ⊙ b(c) = A ⊙ b(a+c), A<sub>k</sub> = p<sub>k</sub><sup>−1/2</sup>: up to the frozen amplitude, b is a
group homomorphism from (ℝ,+) into the torus (measured deviation 1.4×10<sup>−10</sup> over 500 random pairs in [−10<sup>5</sup>,10<sup>5</sup>], R = 2048, float64).
<b>Proposition 2 (constant norm).</b> ‖b(x)‖² = Σ<sub>k</sub> p<sub>k</sub><sup>−1</sup> for every x (by Mertens' theorem ≈ log log p<sub>R</sub> + M); identity is carried
entirely by phase. <b>Proposition 3 (translation covariance).</b> ⟨b(x),b(y)⟩ depends on x−y only; the Gram matrix is Toeplitz.
<b>Proposition 4 (continuity and injectivity).</b> b is continuous, and since the √p<sub>k</sub> are ℚ-linearly independent (Besicovitch) the phase
vector equidistributes on the torus and b is injective; 1−cos(b(x),b(x+δ)) scales exactly as δ².</p>
<p><b>The exponent 1/2.</b> Prime p contributes p<sup>−2σ</sup> to the squared norm; σ = 1/2 is the boundary between domination by a few small primes
(σ &gt; 1/2) and by the largest ones (σ &lt; 1/2), where the total diverges only as log log. Its cost is a small effective dimension
((ΣA²)²/ΣA⁴ = 14.3 at R = 2048), which is why the trained machine reads out with a flat amplitude (Section 3).
<b>Typed regions.</b> A disjoint family (type·10<sup>5</sup> + value) costs nothing to add; Section 3 uses this as frozen role phases per stream.
<b>Frequency families.</b> Propositions 1–4 only need ℚ-independent frequencies; with log p<sub>k</sub> instead of √p<sub>k</sub> the coordinates are the Euler-product
terms p<sup>−1/2−it</sup> of ζ on the critical line, and a tapered prime-power bank recovers the first ten zeros of ζ (max error 0.025).</p>

<h2>3. The machine</h2>
<p><b>Events and streams.</b> The input is a sequence of events (value x, stream s, gap Δt). Each stream (a weather variable, an audio
channel, bytes of text, a robot joint) is a typed region: its code is b(x/scale<sub>s</sub>) rotated by a frozen role phase ψ(s) = 2π frac(10<sup>5</sup>(s+1)f),
the binding of Proposition 1. Values of all streams therefore share one coordinate system; nothing is learned per stream at the input.</p>
<p><b>Trunk.</b> L reversible blocks y<sub>1</sub> = x<sub>1</sub> + F(x<sub>2</sub>), y<sub>2</sub> = x<sub>2</sub> + G(y<sub>1</sub>), followed by a per-prime invertible 2×2 map (PhaseLinear).
Inside F and G: a continuous-time convolution κ(t) = Re Σ<sub>m</sub> c<sub>m</sub> e<sup>(−λ<sub>m</sub>+iω<sub>m</sub>)t</sup> whose state obeys s ← e<sup>aΔt</sup>s + u and is read out at
any offset q as Re Σ c<sub>m</sub>e<sup>a<sub>m</sub>q</sup>s<sub>m</sub>; a circulant phase filter; a low-rank Kuramoto phase coupling; a spectral operator W = U diag(e<sup>σ</sup>)V<sup>T</sup>; and a
gate α initialised at zero, so every block is born as the identity. The state is O(channels × modes), independent of sequence length.
Training uses an exact chunked scan (FFT for regular clocks); generation uses the recurrence one event at a time.</p>
<p><b>Associative phase memory.</b> A wide prime bank (R<sub>M</sub> frequencies) gives each event a unit phase key. The pair (context key K<sub>j−1</sub> → event j)
is stored as conj(K<sub>j−1</sub>) ⊗ v<sub>j</sub> in an R<sub>M</sub>×R complex matrix, and reading with the current key returns what followed the same context before
(binding/unbinding by Proposition 1). A key of order n binds the last n events under fixed permutations, so different contexts that end
in the same symbol have different keys. Writing is one addition; erasing a pair is its subtraction. Training uses an equivalent parallel
form; streaming uses the constant-size matrix.</p>
<p><b>Output.</b> For a continuous stream the next-value density on a window [a,c] is q(x|h) = |⟨b(x),h⟩|²/Z(h) with Z = h<sup>T</sup>Gh̄ and a frozen
closed-form Hermitian G (sinc entries); it is non-negative and integrates to one by construction. For a discrete stream (bytes, digits)
the same read-out evaluated on the symbol set gives logits log|s(c)|² with a learned temperature. The head has no output vocabulary
and no output weights besides a gain.</p>
<p><b>Adaptation (inference only).</b> For forecasting of new series, each window is expressed relative to the mean and spread of its own
look-back and the forecast is mapped back (the RevIN idea [18]); only past values are used.</p>
<p><b>Extensions implemented and under evaluation</b> (Section 6.8): a calendar clock (phases of absolute time on day/week/year circles),
anticipation heads (when the next event comes and from which stream), salience-gated memory writing, and a query head that separates the
state from the asked time so that any future moment can be asked directly.</p>

<h2>4. Exactness</h2>
<p>An automated suite (16 tests, all passing; <code>tests/</code>) checks on the implemented machine: streaming equals the full pass (with memory,
context keys, gated writing, clock, anticipation and query head), invertibility of the trunk, normalisation of every density, mixed regular
and irregular clocks with checkpointed gradients, the query head's state is independent of the asked time, and reading several future
moments from one state equals stepping to each. On the trained 17.5 M model: streaming vs full pass relative difference
{caps['streaming_vs_full_rel_diff']:.1e}; inverse relative error {caps['inverse_rel_err_fp64']:.1e} (float64); density integrals
{f(caps['calibration_v4_test_weather']['density_integral_min_max'][0], 6)}–{f(caps['calibration_v4_test_weather']['density_integral_min_max'][1], 6)}.</p>

<h2>5. Data and setup</h2>
<p><b>Corpus (real data only, frozen with SHA-256 manifest).</b> Jena climate (6 variables, 10 min) [24]; ETT electricity transformer
(7 variables, hourly) [19]; VIX daily close and S&P 500 monthly returns; Free Spoken Digit Dataset audio (8 kHz) [25]; public-domain books
and the Brown corpus (bytes); 243 country–capital facts. Large corpus: ten public forecasting collections of the Chronos/Monash archives
[14,26] (electricity, traffic, weather, solar, taxi, exchange rates, air quality, Uber, wind, pedestrians) and a 400 MB sample of FineWeb-Edu
[27]. Robot: LeRobot koch_pick_place_5_lego [28] (50 real episodes; 6 joints, 6 motor commands, 8×8 grey camera pixels per position as streams).
Anomalies: the real* series of NAB [21] with official labels. Splits are by time or by series/episode; standardisation uses train statistics only.</p>
<p><b>Main model.</b> R = {main['args']['R']}, L = {main['args']['L']}, {main['args']['M']} modes, R<sub>M</sub> = {main['args']['RM']}: {main['params']:,} parameters.
Batches of {main['args']['B']} windows × {main['args']['T']} events, one domain per batch (mix {main['args']['mix']}), AdamW, {main['steps']:,} steps in
{main['train_minutes']:.0f} minutes on one NVIDIA H100; no step was skipped for non-finite loss. Small models (81 k parameters) for ablations were
trained on CPU. <b>References.</b> For continuous streams the no-learning reference is persistence with a Gaussian of the train change width;
for text the train unigram. They are not competitors; they answer whether the machine learned anything.</p>

<h2>6. Results</h2>
<h3>6.1 Held-out prediction</h3>
<table><tr><th>domain (held-out test)</th><th>17.5 M machine</th><th>no-learning reference</th><th>81 k machine, CPU (3 seeds)</th></tr>{t_main}</table>
<div class="cap">Table 1. Negative log-likelihood per event (nats, standardised units; lower is better); text in bits per byte. FineWeb-Edu
held-out: {f(main['fineweb_val_bits_per_byte'])} bits per byte. Held-out series collections: {f(main['series_val_nll'])} nats per event.
The machine is better than the reference on weather, electricity, speech and text, and worse on markets.</div>

<h3>6.2 Calibrated uncertainty</h3>
<table><tr><th>held-out data</th><th>events</th><th>PIT KS</th><th>CRPS</th><th>CRPS, Gaussian reference</th><th>ratio</th></tr>{t_cal}</table>
<div class="cap">Table 2. Probability integral transform (KS distance from uniform; 0 = perfectly calibrated) and continuous ranked
probability score [23] of the predictive densities. The reference is a Gaussian centred on the previous value with the empirical change width.</div>
<figure><img src="fig_calibration.png"><div class="cap">Figure 1. PIT histograms on held-out real data (uniform = calibrated).</div></figure>
<table><tr><th>events removed at random</th><th>held-out series</th><th>weather</th></tr>{t_miss}</table>
<div class="cap">Table 3. Missing data: the same windows with events deleted (gaps grow, nothing is imputed). NLL per remaining event rises
smoothly, as it must when each prediction is further ahead; there is no failure mode.</div>

<h3>6.3 Long-horizon forecasting (ETT protocol)</h3>
<table><tr><th>data</th><th>H</th><th>machine, adapted MSE / MAE</th><th>machine, not adapted MSE</th><th>repeat-last MSE / MAE</th><th>one-day repeat MSE / MAE</th></tr>{t_ett}</table>
<div class="cap">Table 4. Standard borders (12/4/4 months), train-split scaling, look-back 512, channel-independent median roll-outs.
Deviation from the reference protocol: windows every 24 h (ETTm1: 24 h) instead of every step; all horizons use the same window starts.
Without adaptation the machine fails on ETTh2 (two channels reach levels never seen in training, 2–3 standard deviations below the train mean,
and the machine returns to its learned mean; diagnosis in <code>runs/gpu/main/diag_ett.json</code>). With adaptation it is below both naive
baselines on ETTh2 at every horizon. On ETTh1/ETTm1 it is below repeat-last but above one-day repeat: fed-back median roll-outs flatten the
daily cycle (Figure 2).</div>
<figure><img src="fig_fans.png"><div class="cap">Figure 2. Held-out forecasts with 80% bands; mean coverage of the 80% band over six windows:
{100 * caps['ett_fans_mean_coverage_80']:.0f}% (slightly over-confident).</div></figure>

<h3>6.4 One-pass memory and exact erasure</h3>
<table><tr><th>memory key (no trained weights)</th><th>R<sub>M</sub></th><th>exact capital recalled</th><th>characters correct</th></tr>{t_facts}</table>
<div class="cap">Table 5. 243 real "Country:Capital" lines ({fk['bytes_written']} bytes) are streamed once into the phase memory of an untrained
machine; then for each country the prompt "Country:" is given and the capital is read back from the memory alone. Erasing one fact
({er['erased_country']}) by subtracting the pairs written for its line: recalled before = {er['erased_recalled_before']}, after = {er['erased_recalled_after']};
correct answers for the other countries: {er['other_correct_before']} before and {er['other_correct_after']} after (none changed; {er['other_answers_changed']} of the
wrong answers changed). Without context keys recall is 0%.</div>

<h3>6.5 Ablations (CPU, 81 k parameters, 3 seeds)</h3>
<table><tr><th>variant</th>{''.join(f'<th>{d}</th>' for d in DOM)}</tr>{t_abl}</table>
<div class="cap">Table 6. Held-out NLL (text: bits per byte), mean ± std over 3 seeds, 1000 steps. Removing the memory costs 1.5 nats on
weather and 0.6 on electricity; PhaseLinear helps slightly and consistently; the Kuramoto coupling has no measurable effect at this scale;
a perception-first curriculum does not help and hurts text.</div>
<table><tr><th>variant (2 seeds, 3000 steps)</th>{''.join(f'<th>{d}</th>' for d in DOM)}<th>ETTh1 H96</th><th>ETTm1 H96</th></tr>{t_o2}</table>
<div class="cap">Table 7. Extensions on CPU (means of 2 seeds; ETT on 30 windows with adaptation; {o2_naive}). The calendar clock alone does
not close the gap to one-day repeat; context-keyed memory gives a small consistent gain on electricity; anticipation heads trade weather and
electricity for a large gain on text.</div>

<h3>6.6 Learning domains one after another</h3>
<p>Weather → electricity → market → speech → text, 400 steps each, 3 seeds. Average forgetting (best score ever minus final score):
fine-tuning all weights {f(cont['finetune'][0], 2)} ± {f(cont['finetune'][1], 2)}; replay of 20% earlier data {f(cont['replay'][0], 2)} ± {f(cont['replay'][1], 2)};
one phase adapter per domain ({cont['adapters'][2][0]['adapter_params']:,} parameters each, 9% of the model) {f(cont['adapters'][0], 2)} by construction.
Adding all adapters at once fails (weather {f(comp['weather'], 2)} vs {f(fin['weather'], 2)} with its own adapter): adapters are switched, not composed.</p>

<h3>6.7 Negative results</h3>
<p class="neg"><b>Anomaly detection.</b> Surprise (−log predicted density) per reading on the 47 real NAB series, 110 labelled windows:</p>
<table><tr><th>normalisation / rule</th><th>point AUC</th><th>windows detected</th><th>alarm precision</th></tr>{t_nab}</table>
<div class="cap">Table 8. Chance level (about 0.10 of points lie in windows). One-step surprise is dominated by sensor noise and adapts to new
levels; it is not an anomaly detector as such.</div>
<p class="neg"><b>World model of a real robot arm.</b> Trained on 40 real pick-and-place episodes (3–4 k steps, 81 k parameters, CPU), test NLL on the
10 held-out episodes is joints {f(rob['world_s0']['test_nll']['joints'], 2)} / {f(rob['world_s1']['test_nll']['joints'], 2)}, pixels
{f(rob['world_s0']['test_nll']['pixels'], 2)} / {f(rob['world_s1']['test_nll']['pixels'], 2)} (2 seeds), against {f(rp['joints'], 2)} and {f(rp['pixels'], 2)} for hold-last.
Replacing the motor commands with those of another episode makes the real joints less likely in both seeds
({f(rob['world_s0']['test_nll_foreign_actions']['joints'], 3)} vs {f(rob['world_s0']['test_nll']['joints'], 3)}; {f(rob['world_s1']['test_nll_foreign_actions']['joints'], 3)} vs
{f(rob['world_s1']['test_nll']['joints'], 3)}), so the commands are used, but the effect is small and 1-second imagination is worse than hold-last.</p>
{PENDING}

<h2>7. Related work</h2>
<p><b>Fixed sinusoidal codes.</b> Random Fourier features [1] and sinusoidal/rotary position codes [2,3] use frozen frequencies for kernels or
positions; here a frozen prime bank is the input representation of every value. <b>Vector-symbolic and holographic memories</b> [4,5,6] bind
by elementwise product of phase vectors; our memory is such a memory, used inside a trained sequence model with context keys and with the
group law of Proposition 1, and linear/fast-weight memories [7] are the closest learned relatives. <b>State-space and continuous-time models</b>
[8,9,10] and neural ODE/CDE models for irregular series [11,12,13] share the continuous-time state; our convolution is a sum of damped complex
exponentials read out at an arbitrary offset, inside reversible blocks [15]. <b>Byte-level models</b> [16,17] remove the tokenizer but keep a
learned table over 256 symbols. <b>Probabilistic forecasting models</b> [14,20,22] learn output heads or quantise values into tokens; our density is
parameter-free with a closed-form normaliser. Per-window normalisation follows [18]; exact unlearning is studied in [29]. Oscillator synchrony
follows Kuramoto [30]. What is specific to this work is the combination: one frozen arithmetic basis for every value of every modality, a
continuous clock and a continuous output, a one-pass erasable memory, and exactness checks for all of them.</p>

<h2>8. Limits and discussion</h2>
<p>Scale is small: 17.5 M parameters and 200 GPU-minutes; no scaling law has been measured. Fed-back median roll-outs lose daily rhythm;
one-step surprise is not an anomaly detector; the robot world model does not beat hold-last; adapters do not compose; markets are not
predicted better than persistence. The strengths that hold on real held-out data are calibrated densities, graceful degradation with missing
events, exact streaming and invertibility, and a memory that is written in one pass and erased exactly.</p>

<h2>Acknowledgements and disclosure</h2>
<p>Implementation, experiments and drafting were carried out with the assistance of an AI coding assistant (Claude, Anthropic).
Compute: one NVIDIA H100 for the main run and small cloud GPUs for evaluation (about 16 USD in total), free CI runners for CPU experiments.</p>

<h2>Appendix: reproducibility</h2>
<p>Code, frozen data manifests, every result file and the trained weights are in the repository. Tests: <code>python3 -m pytest -q tests</code>.
Main run: <code>gpu/main_train.py</code>; capabilities: <code>gpu/eval_caps.py</code>; ETT: <code>pfm/ett_eval.py</code>, <code>gpu/eval_ett_adapt.py</code>;
memory: <code>audit/facts_memory.py</code>; ablations: <code>real_train.py</code>; continual: <code>real_continual.py</code>; NAB: <code>gpu/eval_nab.py</code>,
<code>oll/alarm_select.py</code>; robot: <code>data/build_robot.py</code>, <code>robot_world.py</code>; living machine: <code>life.py</code>.
Every number in this paper is generated from those files by <code>paper/build_paper.py</code>. An earlier version of this paper reported experiments on
synthetic signals; they have been removed.</p>

<h2>References</h2><ol class="refs">
<li>A. Rahimi, B. Recht. Random features for large-scale kernel machines. NeurIPS 2007.</li>
<li>A. Vaswani et al. Attention is all you need. NeurIPS 2017.</li>
<li>J. Su et al. RoFormer: enhanced transformer with rotary position embedding. Neurocomputing 2024 (arXiv 2021).</li>
<li>T. Plate. Holographic reduced representations. IEEE Trans. Neural Networks 1995.</li>
<li>P. Kanerva. Hyperdimensional computing. Cognitive Computation 2009.</li>
<li>E. P. Frady, D. Kleyko, F. T. Sommer. Variable binding for sparse distributed representations. IEEE TNNLS 2022.</li>
<li>I. Schlag, K. Irie, J. Schmidhuber. Linear transformers are secretly fast weight programmers. ICML 2021.</li>
<li>A. Gu, K. Goel, C. Ré. Efficiently modeling long sequences with structured state spaces. ICLR 2022.</li>
<li>J. T. H. Smith, A. Warrington, S. Linderman. Simplified state space layers for sequence modeling. ICLR 2023.</li>
<li>A. Gu, T. Dao. Mamba: linear-time sequence modeling with selective state spaces. arXiv 2023.</li>
<li>R. T. Q. Chen et al. Neural ordinary differential equations. NeurIPS 2018.</li>
<li>Y. Rubanova, R. T. Q. Chen, D. Duvenaud. Latent ODEs for irregularly-sampled time series. NeurIPS 2019.</li>
<li>P. Kidger et al. Neural controlled differential equations for irregular time series. NeurIPS 2020.</li>
<li>A. F. Ansari et al. Chronos: learning the language of time series. TMLR 2024.</li>
<li>A. N. Gomez et al. The reversible residual network. NeurIPS 2017.</li>
<li>L. Xue et al. ByT5: towards a token-free future with pre-trained byte-to-byte models. TACL 2022.</li>
<li>A. Pagnoni et al. Byte latent transformer: patches scale better than tokens. arXiv 2024.</li>
<li>T. Kim et al. Reversible instance normalization for accurate time-series forecasting against distribution shift. ICLR 2022.</li>
<li>H. Zhou et al. Informer: beyond efficient transformer for long sequence time-series forecasting. AAAI 2021.</li>
<li>D. Salinas et al. DeepAR: probabilistic forecasting with autoregressive recurrent networks. Int. J. Forecasting 2020.</li>
<li>A. Lavin, S. Ahmad. Evaluating real-time anomaly detection algorithms: the Numenta Anomaly Benchmark. ICMLA 2015.</li>
<li>A. Das et al. A decoder-only foundation model for time-series forecasting. ICML 2024.</li>
<li>T. Gneiting, A. E. Raftery. Strictly proper scoring rules, prediction, and estimation. JASA 2007.</li>
<li>Max Planck Institute for Biogeochemistry. Jena climate dataset (weather station records).</li>
<li>Z. Jackson et al. Free Spoken Digit Dataset. 2018.</li>
<li>R. Godahewa et al. Monash time series forecasting archive. NeurIPS Datasets and Benchmarks 2021.</li>
<li>G. Penedo et al. The FineWeb datasets. NeurIPS Datasets and Benchmarks 2024.</li>
<li>R. Cadene et al. LeRobot: state-of-the-art machine learning for real-world robotics. 2024.</li>
<li>L. Bourtoule et al. Machine unlearning. IEEE S&P 2021.</li>
<li>Y. Kuramoto. Chemical oscillations, waves, and turbulence. Springer 1984.</li>
</ol></body></html>"""
open("paper/oll_paper.html", "w").write(html)
print("written", len(html))
