//! The two supported training regimes (ADR-003 §1) and N3 regularization
//! (§2), each as a per-positive step that reads embeddings from [`Tables`],
//! accumulates loss gradients into [`Grads`], and returns its scalar loss.
//!
//! Scores are "higher is more plausible". The self-adversarial loss is the
//! RotatE form (arXiv:1902.10197) rewritten for that convention; the
//! 1-vs-all loss is softmax cross-entropy over every entity (Lacroix et al.
//! 2018), run on both sides.

use super::grad::Differentiable;
use super::negatives::sample_corruptions;
use super::optim::Grads;
use crate::data::Rng;
use crate::{Result, Tables, Triple};

/// Numerically stable sigmoid.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable `log(sigmoid(x)) = -softplus(-x)`.
fn log_sigmoid(x: f32) -> f32 {
    // softplus(z) = max(z,0) + ln(1 + e^{-|z|}); here z = -x.
    let z = -x;
    -(z.max(0.0) + (1.0 + (-z.abs()).exp()).ln())
}

/// Softmax in place (max-shifted). Returns nothing; `v` becomes probabilities.
fn softmax_inplace(v: &mut [f32]) {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// N3 (nuclear 3-norm) penalty for one embedding row and its gradient.
/// Penalty `= lambda * sum |x_i|^3`; gradient `= 3 * lambda * x_i * |x_i|`.
/// Returns the penalty and fills `grad_out` (same length as `row`).
pub(crate) fn n3_grad(row: &[f32], lambda: f32, grad_out: &mut Vec<f32>) -> f32 {
    grad_out.clear();
    grad_out.reserve(row.len());
    let mut penalty = 0.0f32;
    for &x in row {
        let ax = x.abs();
        penalty += ax * ax * ax;
        grad_out.push(3.0 * lambda * x * ax);
    }
    lambda * penalty
}

/// One self-adversarial step for a positive triple. Corrupts both the tail and
/// the head with `neg_count` samples each. Returns the loss.
#[allow(clippy::too_many_arguments)]
pub(crate) fn self_adversarial_step(
    tables: &Tables,
    scorer: &dyn Differentiable,
    t: Triple,
    neg_count: usize,
    temperature: f32,
    margin: f32,
    rng: &mut Rng,
    grads: &mut Grads,
) -> Result<f32> {
    let mut buf = Vec::new();
    let mut loss = 0.0;
    // Tail corruption: (s, r, o'), open slot = object.
    loss += sided_step(
        tables,
        scorer,
        t,
        Side::Tail,
        neg_count,
        temperature,
        margin,
        rng,
        grads,
        &mut buf,
    )?;
    // Head corruption: (s', r, o), open slot = subject.
    loss += sided_step(
        tables,
        scorer,
        t,
        Side::Head,
        neg_count,
        temperature,
        margin,
        rng,
        grads,
        &mut buf,
    )?;
    Ok(loss)
}

#[derive(Clone, Copy)]
enum Side {
    Head,
    Tail,
}

#[allow(clippy::too_many_arguments)]
fn sided_step(
    tables: &Tables,
    scorer: &dyn Differentiable,
    t: Triple,
    side: Side,
    neg_count: usize,
    temperature: f32,
    margin: f32,
    rng: &mut Rng,
    grads: &mut Grads,
    negs: &mut Vec<u32>,
) -> Result<f32> {
    let s = tables.entity(t.s)?;
    let r = tables.relation(t.r)?;
    let o = tables.entity(t.o)?;

    let f_pos = scorer.score(s, r, o);

    sample_corruptions(rng, tables.num_entities(), neg_count, negs);

    // Scores of the negatives, and self-adversarial weights over them.
    let mut f_neg: Vec<f32> = Vec::with_capacity(negs.len());
    for &c in negs.iter() {
        let ce = tables.entity(c)?;
        let fn_i = match side {
            Side::Tail => scorer.score(s, r, ce),
            Side::Head => scorer.score(ce, r, o),
        };
        f_neg.push(fn_i);
    }
    let mut w: Vec<f32> = f_neg.iter().map(|&f| temperature * f).collect();
    if !w.is_empty() {
        softmax_inplace(&mut w);
    }

    // Loss.
    let mut loss = -log_sigmoid(margin + f_pos);
    for (&fn_i, &wi) in f_neg.iter().zip(&w) {
        loss -= wi * log_sigmoid(-(margin + fn_i));
    }

    // Positive gradient: dL/df_pos = sigmoid(margin + f_pos) - 1.
    let coeff_pos = sigmoid(margin + f_pos) - 1.0;
    accumulate_triple_grad(scorer, grads, t.s, t.r, t.o, s, r, o, coeff_pos);

    // Negative gradients: dL/df_neg_i = w_i * sigmoid(margin + f_neg_i).
    for (&c, (&fn_i, &wi)) in negs.iter().zip(f_neg.iter().zip(&w)) {
        let coeff = wi * sigmoid(margin + fn_i);
        let ce = tables.entity(c)?;
        match side {
            Side::Tail => accumulate_triple_grad(scorer, grads, t.s, t.r, c, s, r, ce, coeff),
            Side::Head => accumulate_triple_grad(scorer, grads, c, t.r, t.o, ce, r, o, coeff),
        }
    }
    Ok(loss)
}

/// One 1-vs-all cross-entropy step over every entity, on both the tail and the
/// head side. Returns the summed loss.
pub(crate) fn one_vs_all_step(
    tables: &Tables,
    scorer: &dyn Differentiable,
    t: Triple,
    grads: &mut Grads,
) -> Result<f32> {
    let n = tables.num_entities();
    let s = tables.entity(t.s)?;
    let r = tables.relation(t.r)?;
    let o = tables.entity(t.o)?;

    // ---- Tail side: classify the object among all entities. ----
    let mut probs: Vec<f32> = Vec::with_capacity(n);
    for e in 0..n as u32 {
        probs.push(scorer.score(s, r, tables.entity(e)?));
    }
    softmax_inplace(&mut probs);
    let mut loss = -(probs[t.o as usize].max(1e-30)).ln();
    // Gradients: dL/df_e = p_e - 1{e==o}. Accumulate into s, r (summed) and e.
    let mut gs_acc = vec![0.0f32; scorer.dims()];
    let mut gr_acc = vec![0.0f32; scorer.dims()];
    for e in 0..n as u32 {
        let coeff = probs[e as usize] - if e == t.o { 1.0 } else { 0.0 };
        if coeff == 0.0 {
            continue;
        }
        let ee = tables.entity(e)?;
        let (gs, gr, go) = scorer.grad(s, r, ee);
        axpy(&mut gs_acc, coeff, &gs);
        axpy(&mut gr_acc, coeff, &gr);
        grads.add_entity(e, &scaled(coeff, &go));
    }
    grads.add_entity(t.s, &gs_acc);
    grads.add_relation(t.r, &gr_acc);

    // ---- Head side: classify the subject among all entities. ----
    let mut probs_h: Vec<f32> = Vec::with_capacity(n);
    for e in 0..n as u32 {
        probs_h.push(scorer.score(tables.entity(e)?, r, o));
    }
    softmax_inplace(&mut probs_h);
    loss += -(probs_h[t.s as usize].max(1e-30)).ln();
    let mut gr_acc2 = vec![0.0f32; scorer.dims()];
    let mut go_acc = vec![0.0f32; scorer.dims()];
    for e in 0..n as u32 {
        let coeff = probs_h[e as usize] - if e == t.s { 1.0 } else { 0.0 };
        if coeff == 0.0 {
            continue;
        }
        let ee = tables.entity(e)?;
        let (gs, gr, go) = scorer.grad(ee, r, o);
        grads.add_entity(e, &scaled(coeff, &gs));
        axpy(&mut gr_acc2, coeff, &gr);
        axpy(&mut go_acc, coeff, &go);
    }
    grads.add_relation(t.r, &gr_acc2);
    grads.add_entity(t.o, &go_acc);

    Ok(loss)
}

/// Per-batch precomputation for [`one_vs_all_step_batched`]: every entity's
/// index vector (row-major, `num_entities × index_dims`) plus a dense
/// accumulator for the per-entity gradients of the batch.
///
/// Built once per mini-batch — the tables do not change inside a batch
/// (the optimiser applies after it), so the index vectors stay valid.
pub(crate) struct BatchedOneVsAll {
    index: Vec<f32>,
    index_dims: usize,
    dense: Vec<f32>,
    dims: usize,
    logits: Vec<f32>,
}

impl BatchedOneVsAll {
    pub(crate) fn new(tables: &Tables, scorer: &dyn Differentiable) -> Result<Self> {
        let n = tables.num_entities();
        let dd = scorer.index_dims();
        let mut index = Vec::with_capacity(n * dd);
        for e in 0..n as u32 {
            index.extend_from_slice(&scorer.index_vector(tables.entity(e)?));
        }
        Ok(Self {
            index,
            index_dims: dd,
            dense: vec![0.0; n * scorer.dims()],
            dims: scorer.dims(),
            logits: vec![0.0; n],
        })
    }

    fn score_all(&mut self, q: &[f32]) {
        for (e, lg) in self.logits.iter_mut().enumerate() {
            let row = &self.index[e * self.index_dims..(e + 1) * self.index_dims];
            *lg = row.iter().zip(q).map(|(a, b)| a * b).sum();
        }
    }

    /// Move the accumulated per-entity gradients into `grads`.
    pub(crate) fn flush(&mut self, grads: &mut Grads) {
        let d = self.dims;
        for (e, row) in self.dense.chunks(d).enumerate() {
            if row.iter().any(|&x| x != 0.0) {
                grads.add_entity(e as u32, row);
            }
        }
        self.dense.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// Batched 1-vs-all cross-entropy for multilinear scorers
/// ([`Differentiable::multilinear`]). Same loss and gradients as
/// [`one_vs_all_step`], computed without a per-entity `score`/`grad` call:
///
/// - every entity is scored with one dot product against the precomputed
///   index vectors (`query · index = score` exactly);
/// - because the score is linear in the open slot, `Σ_e c_e ∇_{s,r} score(s,r,e)`
///   equals `∇_{s,r} score(s, r, Σ_e c_e e)`, so the anchor/relation gradients
///   need one `grad` call on the coefficient-weighted entity;
/// - the open-slot gradient does not depend on the open entity, so entity `e`
///   receives `c_e · g` for a single vector `g`.
///
/// Cost per positive: `O(|E|·d)` multiply-adds plus two `grad` calls, instead
/// of `4·|E|` FFT-based score/grad calls.
pub(crate) fn one_vs_all_step_batched(
    tables: &Tables,
    scorer: &dyn Differentiable,
    t: Triple,
    batch: &mut BatchedOneVsAll,
    grads: &mut Grads,
) -> Result<f32> {
    let n = tables.num_entities();
    let d = scorer.dims();
    let s = tables.entity(t.s)?;
    let r = tables.relation(t.r)?;
    let o = tables.entity(t.o)?;
    let mut loss = 0.0f32;

    for side in [crate::Side::Tail, crate::Side::Head] {
        let (anchor, anchor_id, gold) = match side {
            crate::Side::Tail => (s, t.s, t.o),
            crate::Side::Head => (o, t.o, t.s),
        };
        let q = scorer.query_vector(r, anchor, side);
        batch.score_all(&q);
        softmax_inplace(&mut batch.logits);
        loss += -(batch.logits[gold as usize].max(1e-30)).ln();
        // Coefficient-weighted open entity: Σ_e (p_e − 1{e = gold}) · E_e.
        let mut weighted = vec![0.0f32; d];
        for e in 0..n {
            let c = batch.logits[e] - if e as u32 == gold { 1.0 } else { 0.0 };
            if c != 0.0 {
                axpy(&mut weighted, c, tables.entity(e as u32)?);
            }
        }
        let (unit, g_rel, g_anchor) = match side {
            crate::Side::Tail => {
                let (gs, gr, go) = scorer.grad(s, r, &weighted);
                (go, gr, gs)
            }
            crate::Side::Head => {
                let (gs, gr, go) = scorer.grad(&weighted, r, o);
                (gs, gr, go)
            }
        };
        grads.add_entity(anchor_id, &g_anchor);
        grads.add_relation(t.r, &g_rel);
        // `unit` is ∂score/∂(open slot); it is the same for every open entity.
        for e in 0..n {
            let c = batch.logits[e] - if e as u32 == gold { 1.0 } else { 0.0 };
            if c != 0.0 {
                axpy(&mut batch.dense[e * d..(e + 1) * d], c, &unit);
            }
        }
    }
    Ok(loss)
}

/// Accumulate `coeff * grad(score)` into the three rows of a triple.
#[allow(clippy::too_many_arguments)]
fn accumulate_triple_grad(
    scorer: &dyn Differentiable,
    grads: &mut Grads,
    s_id: u32,
    r_id: u32,
    o_id: u32,
    s: &[f32],
    r: &[f32],
    o: &[f32],
    coeff: f32,
) {
    if coeff == 0.0 {
        return;
    }
    let (gs, gr, go) = scorer.grad(s, r, o);
    grads.add_entity(s_id, &scaled(coeff, &gs));
    grads.add_relation(r_id, &scaled(coeff, &gr));
    grads.add_entity(o_id, &scaled(coeff, &go));
}

fn scaled(a: f32, v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| a * x).collect()
}

fn axpy(dst: &mut [f32], a: f32, v: &[f32]) {
    for (d, &x) in dst.iter_mut().zip(v) {
        *d += a * x;
    }
}

#[cfg(test)]
mod batched_tests {
    use super::*;
    use crate::scorer::HolE;
    use crate::train::grad::testing::DistMult;
    use crate::train::optim::Grads;

    fn max_abs_diff(
        a: &std::collections::BTreeMap<u32, Vec<f32>>,
        b: &std::collections::BTreeMap<u32, Vec<f32>>,
    ) -> f32 {
        assert_eq!(
            a.keys().collect::<Vec<_>>(),
            b.keys().collect::<Vec<_>>(),
            "same rows touched"
        );
        a.iter()
            .flat_map(|(k, va)| va.iter().zip(&b[k]).map(|(x, y)| (x - y).abs()))
            .fold(0.0, f32::max)
    }

    fn check(scorer: &dyn Differentiable, dims: usize) {
        let (ne, nr) = (37, 4);
        let tables = Tables::new(ne, nr, dims, 11);
        let triples = [
            Triple::new(0, 1, 5),
            Triple::new(7, 0, 7),
            Triple::new(36, 3, 2),
            Triple::new(12, 2, 30),
        ];
        let mut exact = Grads::new(dims);
        let mut fast = Grads::new(dims);
        let mut batch = BatchedOneVsAll::new(&tables, scorer).unwrap();
        let (mut l_exact, mut l_fast) = (0.0f32, 0.0f32);
        for &t in &triples {
            l_exact += one_vs_all_step(&tables, scorer, t, &mut exact).unwrap();
            l_fast += one_vs_all_step_batched(&tables, scorer, t, &mut batch, &mut fast).unwrap();
        }
        batch.flush(&mut fast);
        assert!(
            (l_exact - l_fast).abs() < 1e-4 * l_exact.abs().max(1.0),
            "loss {l_exact} vs {l_fast}"
        );
        let scale = exact
            .entity_rows()
            .values()
            .flatten()
            .fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(
            scale > 1e-3,
            "gradients are non-trivial (max |g| = {scale})"
        );
        let de = max_abs_diff(exact.entity_rows(), fast.entity_rows());
        let dr = max_abs_diff(exact.relation_rows(), fast.relation_rows());
        assert!(
            de < 1e-4 * scale.max(1.0) && dr < 1e-4 * scale.max(1.0),
            "entity diff {de}, relation diff {dr} (scale {scale})"
        );
    }

    #[test]
    fn batched_matches_exact_for_hole() {
        check(&HolE::new(16).unwrap(), 16);
    }

    #[test]
    fn batched_matches_exact_for_distmult() {
        check(&DistMult::new(8), 8);
    }
}
