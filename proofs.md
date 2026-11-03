# Formal results — germline vs chained inheritance of task specifications

Math is written in LaTeX inline ($…$) so it ports straight into the paper.
Every result is stated against explicit assumptions A1–A4. Section 7 says which
assumptions are mathematical and which are empirical claims about LLMs that the
experiments must establish.

---

## 1. Setting

- $(S, D)$: a metric space of task specifications. $\sigma_0 \in S$ is the spec
  induced by the user's original statement $\tau$.
- $Y$: output space. For each $\sigma \in S$, $A_\sigma : Y \to \{0,1\}$ is the
  acceptance predicate. $A := A_{\sigma_0}$ is the true objective.
- **Derivation kernel.** An agent given input $x$ forms a working spec
  $\Phi(x) \in S$, a random variable (the model samples). Noises across agents are
  independent given their inputs.
- **Chained lineage (Lamarckian).** $\sigma_d = \Phi(\sigma_{d-1})$ for $d \ge 1$.
  Each agent derives its spec from its parent's working spec.
- **Anchored lineage (Weismannian).** $\sigma_d = \Phi(g, e_{d-1})$ where
  $g \equiv \sigma_0$ verbatim and $e_{d-1}$ is arbitrary somatic/epigenetic state
  that may depend on the entire past of the lineage.
- Agent context in the anchored case: $c_d = (g, e_{d-1}, s_d)$.

### Assumptions

**A1 (bounded anchored error — the barrier property).**
There exists $\varepsilon \ge 0$ such that for every somatic state $e$,
$$\mathbb{E}\big[\,D(\Phi(g, e), \sigma_0) \,\big|\, e\,\big] \le \varepsilon .$$
*Meaning:* given the verbatim original, the derived spec is $\varepsilon$-close to it
no matter what else is in context. This is the instruction-hierarchy property
(germline $\succ$ soma). It is the **training target**, not a given.

**A2 (relaxation bias).**
There is a 1-Lipschitz functional $\rho : S \to \mathbb{R}$ ("relaxation level",
e.g. minus the number of original constraints retained, or the pass rate of a fixed
reference generator) and $\beta > 0$ such that for every reachable $\sigma$,
$$\mathbb{E}\big[\rho(\Phi(\sigma)) - \rho(\sigma)\big] \ge \beta .$$
*Meaning:* each re-derivation, on average, makes the task at least $\beta$ easier.

**A3 (ergodic rewriting).**
The chained kernel $\sigma \mapsto \Phi(\sigma)$ on $S$ has a unique stationary law
$\pi$ and $\mathrm{Law}(\sigma_d \mid \sigma_0 = s) \to \pi$ in total variation for
every $s$. (See §7: this holds automatically for bounded-length specs sampled at
any temperature $T>0$ without top-$k$/top-$p$ truncation.)

**A4 (criterion Lipschitz).**
For the generator's output law $G$ on $Y$ and all $\sigma \in S$,
$$\delta(\sigma) := \Pr_{y \sim G}\big[A_\sigma(y) \ne A_{\sigma_0}(y)\big] \le \kappa\, D(\sigma, \sigma_0).$$
*Meaning:* nearby specs disagree on few outputs.

---

## 2. Theorem 1 (drift)

**(i) Chained.** Under A2, for every $d \ge 1$,
$$\mathbb{E}\,D(\sigma_d, \sigma_0) \;\ge\; \beta\, d .$$

**(ii) Anchored.** Under A1, for every $d \ge 1$,
$$\mathbb{E}\,D(\sigma_d, \sigma_0) \;\le\; \varepsilon .$$

*Proof.* (i) By the tower property and A2,
$$\mathbb{E}[\rho(\sigma_d)] - \rho(\sigma_0)
 = \sum_{k=1}^{d} \mathbb{E}\Big[\,\mathbb{E}\big[\rho(\sigma_k)-\rho(\sigma_{k-1}) \,\big|\, \sigma_{k-1}\big]\Big]
 \;\ge\; \beta d .$$
Since $\rho$ is 1-Lipschitz, $D(\sigma_d,\sigma_0) \ge \rho(\sigma_d)-\rho(\sigma_0)$
pointwise. Take expectations.
(ii) Condition on $e_{d-1}$, apply A1, take expectations. $\blacksquare$

**Remarks.**
0. *Noise model (revised after pilot F2, notes.md §8).* Empirically the per-hop loss is not a small
   constant; it is a heavy-tailed mixture. With probability $1-q$ a hop loses $\approx 0$
   constraints (re-statement by a capable model is nearly lossless, F1), and with probability $q$
   the brief *collapses* — the agent passes a result, a pointer, or a one-line summary instead of
   the spec, losing a large fraction $\ell$ of constraints at once. Then A2 holds with
   $\beta = q\,\mathbb{E}[\ell \mid \text{collapse}]$, so Thm 1(i) is unchanged in expectation, but
   drift is *bursty*: the per-lineage distance at depth $d$ is approximately
   $\ell \cdot \mathrm{Bin}(d, q)$, and the fraction of lineages that have lost anything is
   $1-(1-q)^d$, which is the quantity to plot. In the anchored lineage a collapsed brief is
   harmless because the child re-derives from $g$: this is the mechanism behind (ii), observed
   directly in the pilot (B2 briefs collapse at similar rates; the next hop's restatement is intact).
1. (i) is a *lower* bound: chained drift is at least linear. (ii) is an *upper*
   bound: anchored drift is at most $\varepsilon$, independent of $d$. The gap grows
   without bound.
2. If $\beta = 0$ (unbiased rewriting) but the per-step variance is $v>0$ in a
   Hilbert embedding of $S$, then $\mathbb{E}\|\sigma_d-\sigma_0\|^2 = v d$, so RMS
   drift is $\sqrt{vd}$: still unbounded, only slower. The biased case is the
   realistic one (§7, A2).
3. Dead-reckoning analogy: (i) is error proportional to distance travelled; (ii) is
   a position fix available at every node.

**Corollary 1.1 (leaf success, anchored).** If output $y_d$ generated under spec
$\sigma_d$ satisfies $\Pr[A(y_d)=1] \ge 1 - \lambda D(\sigma_d,\sigma_0)$, then in
the anchored lineage $\Pr[A(y_d)=1] \ge 1 - \lambda\varepsilon$ for all $d$.
*(One-sided on purpose: the chained case needs Theorem 2 to say where success goes.)*

---

## 3. Theorem 2 (information)

**(i) Chained.** $\sigma_0 \to \sigma_1 \to \cdots \to \sigma_d$ is a Markov chain,
hence
$$H(\sigma_0) \;\ge\; I(\sigma_0;\sigma_1) \;\ge\; I(\sigma_0;\sigma_2) \;\ge\; \cdots \;\ge\; I(\sigma_0;\sigma_d),$$
with equality at step $k$ iff $\sigma_0 \perp \sigma_{k-1} \mid \sigma_k$.

**(ii) Regression to the prior.** Under A3 with $S$ finite and $\pi$ of full
support, $I(\sigma_0;\sigma_d) \to 0$ as $d\to\infty$: the leaf's working spec
becomes independent of what was asked.

**(iii) Anchored.** For every $d$, $I(\sigma_0; c_d) = H(\sigma_0)$.

*Proof.* (i) Data processing inequality (Cover & Thomas, Thm 2.8.1), applied
step by step; the equality condition is the standard one.
(ii) Write $P_d(\cdot\mid s) = \mathrm{Law}(\sigma_d\mid\sigma_0=s)$ and
$P_d = \sum_s p(s)P_d(\cdot\mid s)$. Then
$I(\sigma_0;\sigma_d) = \sum_s p(s)\,\mathrm{KL}\big(P_d(\cdot\mid s)\,\|\,P_d\big)$.
By A3 both $P_d(\cdot\mid s)\to\pi$ and $P_d\to\pi$ in TV; on a finite alphabet
with $\pi$ of full support, KL is jointly continuous at $(\pi,\pi)$, so each term
$\to 0$.
(iii) $g$ is a deterministic injective function of $\sigma_0$ (its bytes), so
$H(g)=H(\sigma_0)$ and $H(\sigma_0) \ge I(\sigma_0;c_d) \ge I(\sigma_0;g) = H(\sigma_0)$. $\blacksquare$

**Interpretation.** (ii) is the iterated-learning result of Griffiths & Kalish
(2007) in our setting: a chain of re-derivers converges to the kernel's stationary
law, which for Bayesian learners is exactly their prior. $\pi$ is "the task the model
would have liked to be asked". The anchored lineage does not break the DPI; it
removes the Markov structure by giving every node a direct edge from the root. The
working spec $\sigma_d = \Phi(g,e)$ is still lossy, but the loss is a one-step
quantity, not a product of $d$ kernels.

---

## 4. Theorem 3 (horizon under soma reset)

**Setup.** The task needs $L$ units of work. Context window $W$. Work proceeds in
epochs $k=1,\dots,n$, $n = \lceil L/W \rceil$. At an epoch boundary everything is
discarded except a carry $c_k$.
- *Chained carry:* $c_k = \Psi(\text{context}_k)$, a summary; the next epoch's working
  spec is derived from it. In the spec coordinate this is a chained lineage of length
  $n$.
- *Anchored carry:* $c_k = (g, p_k)$ with $p_k$ a progress state.

**Definition (germline-compressible).** Task $T$ is $(\alpha,\varepsilon)$-germline-
compressible if there is $g$ with $|g| \le \alpha W$ such that $A_{\sigma_0}$ is a
function of $g$ alone and A1 holds with error $\varepsilon$.

**(i)** Chained, under A2: $\mathbb{E}\,D(\sigma_n,\sigma_0) \ge \beta\lceil L/W\rceil \to \infty$ as $L\to\infty$.

**(ii)** Anchored, $T$ germline-compressible with $\alpha<1$:
$\mathbb{E}\,D(\sigma_k,\sigma_0) \le \varepsilon$ for every epoch $k$ and every $L$.

*Proof.* Theorem 1 with $d=n$. Compressibility guarantees $g$ fits and leaves
$(1-\alpha)W$ for somatic work per epoch. $\blacksquare$

**Remark (invariant drift vs progress loss).** A lossy progress carry $p_k$ costs
redone work. It never changes what counts as success. A chained summary conflates
the two: the invariant rides inside the lossy summary and drifts with it. The
structural correction is to separate the two kinds of state and apply lossless
transport only to the small one. This is what makes the horizon claim honest: the
method does not extend context; it guarantees that unbounded $L$ never changes the
problem being solved, for tasks whose invariants are $O(1)$ in $L$.

---

## 5. Proposition 4 (best-of-$N$ with a verifier)

**Setup.** $y_1,\dots,y_N \sim G$ i.i.d., each correct ($A(y)=1$) with probability
$p\in(0,1)$. The verifier accepts a correct sample with probability $a$ and an
incorrect one with probability $b$, independently given correctness. Select
uniformly among accepted samples; uniformly among all if none accepted.

**(i)** As $N\to\infty$,
$$\Pr[\text{selected is correct}] \;\to\; \Lambda(a,b) := \frac{p\,a}{p\,a + (1-p)\,b}.$$

**(ii)** $\Lambda(a,b) > p \iff a > b$, and $\Lambda(a,b) < p \iff a < b$.
*A verifier that accepts wrong outputs more readily than right ones makes
best-of-$N$ worse than a single sample.*

**(iii) Drifted criterion.** Suppose the verifier is exact for its own spec
$\sigma_d$ (accepts iff $A_{\sigma_d}(y)=1$). With $\delta_d := \delta(\sigma_d)$
from A4,
$$a \ge 1 - \frac{\delta_d}{p}, \qquad b \le \frac{\delta_d}{1-p}, \qquad
\Lambda \ge \frac{p-\delta_d}{p+\delta_d}.$$
Anchored: $\delta_d \le \kappa\varepsilon$, so $\Lambda \ge (p-\kappa\varepsilon)/(p+\kappa\varepsilon)$
for all $d$ — close to 1 for small $\varepsilon$, independent of depth. Chained:
$\delta_d$ grows with $d$ (Thm 1(i) + A4 is only an upper bound on $\delta$, so
state this as: the guarantee degrades to vacuous once $\delta_d \ge p$, and by (ii)
the verifier can then actively harm).

*Proof.* (i) Let $C = \#\{\text{accepted correct}\}\sim\mathrm{Bin}(N,pa)$ and
$I = \#\{\text{accepted incorrect}\}\sim\mathrm{Bin}(N,(1-p)b)$. Given $(C,I)$ with
$C+I>0$, the selected sample is correct with probability $C/(C+I)$. By the SLLN,
$C/N\to pa$ and $I/N\to(1-p)b$ a.s., so $C/(C+I)\to\Lambda$ a.s.; the event
$C+I=0$ has probability $(1-pa-(1-p)b)^N\to 0$; conclude by bounded convergence.
(ii) $\Lambda > p \iff a > pa+(1-p)b \iff a(1-p) > (1-p)b \iff a>b$.
(iii) $a = \Pr[A_{\sigma_d}=1\mid A=1] \ge 1 - \Pr[A_{\sigma_d}\ne A]/\Pr[A=1] = 1-\delta_d/p$;
$b = \Pr[A_{\sigma_d}=1\mid A=0] \le \Pr[A_{\sigma_d}\ne A]/\Pr[A=0] = \delta_d/(1-p)$.
Then $pa \ge p-\delta_d$ and $pa+(1-p)b \le p+\delta_d$. $\blacksquare$

**Interpretation.** The generator–verifier gap is only exploitable against a fixed
criterion. The germline is what fixes it. This is the formal content of "clonal
selection needs an unchanged germline".

---

## 6. Proposition 5 (majority voting across a lineage)

**Setup.** A binary question whose correct answer is defined by $A$. $N$ leaf agents
vote independently; voter $j$ is correct with probability $q_j$.

**(i) Condorcet, both directions.** If $q_j \ge q > 1/2$ for all $j$,
$\Pr[\text{majority correct}] \ge 1 - e^{-2N(q-1/2)^2} \to 1$.
If $q_j \le q < 1/2$ for all $j$, $\Pr[\text{majority correct}] \le e^{-2N(1/2-q)^2} \to 0$.

**(ii) Drift enters through the question.** Voter $j$ works under $\sigma^{(j)}$ and
is correct w.r.t. *its own* question with probability $q$. Its accuracy w.r.t. the
original question is $q_j = q - \eta_j$ with $\eta_j \le \kappa' D(\sigma^{(j)},\sigma_0)$
(A4-type assumption on the voter). Anchored: $\eta_j \le \kappa'\varepsilon$, so
$q - \kappa'\varepsilon > 1/2$ suffices and holds uniformly in depth. Chained: once
drift pushes $q_j$ below $1/2$, (i) runs in reverse and the swarm converges with
exponential confidence on the answer to the wrong question.

*Proof.* (i) Hoeffding on the sum of independent Bernoullis. (ii) substitution. $\blacksquare$

**Remark (correlation).** Shared weights make votes correlated; the Condorcet limit
under correlation is bounded away from 1 (Boland 1989; Ladha 1992 — verify refs).
The germline does not fix this and the paper must not claim it does. It fixes the
*premise* of the theorem (one question), not the independence of the voters.

---

## 7. Which assumptions are math, which are claims about LLMs

| Assumption | Status | How to establish |
|---|---|---|
| A1 bounded anchored error | **Empirical; it is the training target.** Expected to FAIL for the system-prompt arm and HOLD after germline training. | Feed $g$ + adversarial somatic context, measure $D(\Phi(g,e),\sigma_0)$. This is experiment E3 and the ablation. |
| A2 relaxation bias $\beta>0$ | **Empirical, and a standalone finding.** | One-step rewriting: ask the model to restate a spec for a subagent, measure constraint retention. Report sign and size of $\beta$. |
| A3 ergodicity | **Mathematical, nearly free.** Specs are bounded-length token strings (finite $S$). Softmax sampling at any $T>0$ without top-$k$/top-$p$ gives every transition positive probability ⇒ irreducible and aperiodic ⇒ unique $\pi$ with full support. With truncated sampling it can fail; say so. | None needed beyond stating the sampling regime. |
| A4 criterion Lipschitz | **Empirical, mild.** | Evaluate $A_{\sigma_d}$ vs $A_{\sigma_0}$ on generated outputs across $d$. |
| Hash integrity | Mathematical. Collision-resistant $H$: $H(g_i)=h_{\text{root}}$ at every node certifies $g_i=g_{\text{root}}$ except with negligible probability. This is what makes the propagation reward deterministic. | — |

**Honest gaps.**
- Thm 1(i) lower-bounds *distance*, not failure. To say chained success → low you
  need Thm 2(ii) plus "a $\pi$-typical spec is not $\sigma_0$", which is generic but
  not a theorem.
- Prop 4(iii) and Prop 5(ii) bound the anchored case tightly and the chained case
  only qualitatively. That is fine for the paper: the claim is that the anchored
  guarantee is depth-independent, and the chained guarantee is absent.
- Nothing here formalizes the reasoning class $\kappa$ as a species trait. It enters
  only through A1's $\varepsilon$. Open question 2 in notes.md stands.
