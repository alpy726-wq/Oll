# Google TPU Research Cloud (TRC) — application text (English)

Apply at the TRC page (sites.research.google/trc). The form asks for a short description of the research and how TPUs will be used.

**Project title.** Sequence modelling without a vocabulary: scaling a continuous-time, tokenizer-free signal model

**Research description.** We study a sequence model whose input representation is a frozen prime-frequency basis on the real
line (b(x)_k = p_k^{-1/2} exp(2πi x √p_k)), with a reversible continuous-time oscillator trunk, a one-pass erasable phase memory and a
parameter-free output density with a closed-form normaliser. A 17.5M-parameter model trained for 200 GPU-minutes on real data only
(weather, electricity, markets, speech, books, web text, ten public forecasting collections) gives calibrated predictive densities
(CRPS 1.6–5.5× better than a no-learning reference on four of five held-out sets), is below naive baselines on the ETTh2
long-horizon benchmark, degrades gracefully with 75% missing events, and recalls 80% of 243 real facts after a single read with
exact erasure. Paper: [arXiv link]. Code and weights: [GitHub / Hugging Face links].

**Planned use of TPUs.** (1) A scaling study at 4–5 model sizes (10M–1B parameters) on the same frozen real corpus to measure whether
held-out likelihood follows a power law; (2) a larger multi-domain run with the calendar clock, anticipation heads and the query head
that answers any future moment directly; (3) continual "living" runs where the model learns online from an uninterrupted real event
stream. All results, including negative ones, will be published openly with code and weights.

**Researcher.** [Name], independent researcher, [email].
