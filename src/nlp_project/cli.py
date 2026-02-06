#!/usr/bin/env python3
"""
Project CLI entrypoint.

Design:
- Thin wrappers only (load config -> call library function -> print results)
- All parameters come from YAML, so UI can reuse the same keys.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nlp_project.common.config import load_config_as

# scrape
from nlp_project.scrape.qz_inform import ScrapeConfig, scrape

# preprocess
from nlp_project.preprocess.clean import CleanConfig, clean_corpus
from nlp_project.preprocess.metadata import MetadataConfig, normalize_metadata

# qc
from nlp_project.qc.audit_raw import AuditRawConfig, audit_raw, format_audit_report
from nlp_project.qc.corpus_summary import CorpusSummaryConfig, make_corpus_summary


# tasks
from nlp_project.tasks.bpe import (
    BpeTrainConfig, train_bpe, format_bpe_train_report,
    BpeApplyExamplesConfig, run_bpe_apply_examples, format_bpe_apply_examples_report,
)
from nlp_project.tasks.vocab import VocabConfig, build_vocab, format_vocab_report
from nlp_project.tasks.heaps import HeapsConfig, fit_heaps, format_heaps_report
from nlp_project.tasks.heaps_plot import HeapsPlotConfig, plot_heaps_loglog
from nlp_project.tasks.tokenize import TokenizeConfig, run_tokenize, format_tokenize_report
from nlp_project.tasks.sentseg import SentSegConfig, run_sentseg, format_sentseg_report
from nlp_project.tasks.sentseg_eval import SentSegEvalConfig, run_sentseg_eval, format_sentseg_eval_report
from nlp_project.tasks.spell_lev import SpellLevDemoConfig, run_spell_lev_demo
from nlp_project.tasks.spell_lev_eval import SpellLevEvalConfig, run_spell_lev_eval, format_spell_lev_eval_report
from nlp_project.tasks.confusion_synth import ConfusionSynthConfig, build_confusion_synthetic, format_confusion_synth_report
from nlp_project.tasks.spell_weighted import SpellWeightedConfig, run_spell_weighted_demo
from nlp_project.tasks.spell_weighted_eval import SpellWeightedEvalConfig, run_spell_weighted_eval, format_spell_weighted_eval_report
from nlp_project.tasks.confusion_top import ConfusionTopConfig, run_confusion_top




def cmd_scrape(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, ScrapeConfig)
    res = scrape(cfg)
    print(f"Planned URLs: {res.planned_urls}")
    print(f"Saved:       {res.saved}")
    print(f"Trafilatura: {res.used_trafilatura}")
    print(f"Skipped robots: {res.skipped_robots}")
    print(f"Skipped fetch:  {res.skipped_fetch}")
    print(f"Skipped short:  {res.skipped_too_short}")
    print(f"JSONL: {cfg.out_jsonl}")
    print(f"TXT:   {cfg.out_txt}")
    print(f"STATS: {cfg.out_stats}")
    return 0


def cmd_preprocess_clean(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, CleanConfig)
    n = clean_corpus(cfg)
    print(f"Cleaned {n} documents")
    print(f"Output: {cfg.out_jsonl}")
    return 0


def cmd_preprocess_metadata(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, MetadataConfig)
    stats = normalize_metadata(cfg)
    print(f"Input:  {cfg.inp_jsonl}")
    print(f"Output: {cfg.out_jsonl}")
    print(f"Docs processed: {stats['docs_processed']}")
    print(f"published_at_iso parsed:  {stats['published_parsed']} / {stats['docs_processed']}")
    print(f"published_at_iso missing: {stats['published_missing']} / {stats['docs_processed']}")
    return 0


def cmd_qc_audit_raw(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, AuditRawConfig)
    res = audit_raw(cfg)
    print(format_audit_report(cfg, res))
    return 0


def cmd_qc_corpus_summary(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, CorpusSummaryConfig)
    out = make_corpus_summary(cfg)
    print(f"Wrote: {out.resolve()}")
    return 0


def cmd_task_tokenize(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, TokenizeConfig)
    r = run_tokenize(cfg)
    print(format_tokenize_report(cfg, r))
    return 0


def cmd_task_vocab(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, VocabConfig)
    r = build_vocab(cfg)
    print(format_vocab_report(cfg, r))
    return 0


def cmd_task_heaps(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, HeapsConfig)
    r = fit_heaps(cfg)
    print(format_heaps_report(cfg, r))
    return 0


def cmd_task_heaps_plot(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, HeapsPlotConfig)
    out = plot_heaps_loglog(cfg)
    print(f"Saved plot: {out.resolve()}")
    return 0


def cmd_task_bpe_train(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, BpeTrainConfig)
    r = train_bpe(cfg)
    print(format_bpe_train_report(cfg, r))
    return 0


def cmd_task_bpe_apply_examples(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, BpeApplyExamplesConfig)
    pairs = run_bpe_apply_examples(cfg)
    print(format_bpe_apply_examples_report(pairs))
    return 0

def cmd_task_sentseg(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, SentSegConfig)
    r = run_sentseg(cfg)
    print(format_sentseg_report(r))
    return 0


def cmd_task_sentseg_eval(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, SentSegEvalConfig)
    r = run_sentseg_eval(cfg)
    print(format_sentseg_eval_report(r))
    return 0

def cmd_task_spell_lev_demo(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, SpellLevDemoConfig)
    print(run_spell_lev_demo(cfg))
    return 0


def cmd_task_spell_lev_eval(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, SpellLevEvalConfig)
    r = run_spell_lev_eval(cfg)
    print(format_spell_lev_eval_report(r))
    return 0

def cmd_task_confusion_synth(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, ConfusionSynthConfig)
    stats = build_confusion_synthetic(cfg)
    print(format_confusion_synth_report(stats))
    return 0


def cmd_task_spell_weighted_demo(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, SpellWeightedConfig)
    print(run_spell_weighted_demo(cfg))
    return 0


def cmd_task_spell_weighted_eval(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, SpellWeightedEvalConfig)
    r = run_spell_weighted_eval(cfg)
    print(format_spell_weighted_eval_report(r))
    return 0


def cmd_task_confusion_top(args: argparse.Namespace) -> int:
    cfg = load_config_as(args.config, ConfusionTopConfig)
    r = run_confusion_top(cfg)
    print(f"Wrote: {r['out_md'].resolve()}")
    print(f"Wrote: {r['out_csv'].resolve()}")
    print(f"Wrote: {r['out_tsv'].resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nlp_project", description="Kazakh NLP corpus pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # scrape
    p_scrape = sub.add_parser("scrape", help="Scrape qz.inform.kz using YAML config")
    p_scrape.add_argument("--config", required=True, help="Path to scrape YAML config")
    p_scrape.set_defaults(func=cmd_scrape)

    # preprocess group
    p_prep = sub.add_parser("preprocess", help="Preprocessing steps")
    sub_prep = p_prep.add_subparsers(dest="prep_cmd", required=True)

    p_clean = sub_prep.add_parser("clean", help="Create clean_text from raw text")
    p_clean.add_argument("--config", required=True, help="Path to preprocess_clean YAML config")
    p_clean.set_defaults(func=cmd_preprocess_clean)

    p_meta = sub_prep.add_parser("metadata", help="Normalize metadata (doc_id, published_at_iso)")
    p_meta.add_argument("--config", required=True, help="Path to preprocess_metadata YAML config")
    p_meta.set_defaults(func=cmd_preprocess_metadata)

    # qc group
    p_qc = sub.add_parser("qc", help="Quality control / reports")
    sub_qc = p_qc.add_subparsers(dest="qc_cmd", required=True)

    p_audit = sub_qc.add_parser("audit-raw", help="Audit raw JSONL dataset")
    p_audit.add_argument("--config", required=True, help="Path to qc_audit_raw YAML config")
    p_audit.set_defaults(func=cmd_qc_audit_raw)

    p_sum = sub_qc.add_parser("corpus-summary", help="Generate Markdown corpus summary")
    p_sum.add_argument("--config", required=True, help="Path to qc_corpus_summary YAML config")
    p_sum.set_defaults(func=cmd_qc_corpus_summary)

    # tasks group  
    p_task = sub.add_parser("task", help="NLP tasks (tokenize, vocab, heaps, bpe, etc.)")
    sub_task = p_task.add_subparsers(dest="task_cmd", required=True)

    p_tok = sub_task.add_parser("tokenize", help="Tokenization counts + top tokens")
    p_tok.add_argument("--config", required=True, help="Path to task_tokenize YAML config")
    p_tok.set_defaults(func=cmd_task_tokenize)

    p_vocab = sub_task.add_parser("vocab", help="Build word vocabulary (word -> count)")
    p_vocab.add_argument("--config", required=True, help="Path to task_vocab YAML config")
    p_vocab.set_defaults(func=cmd_task_vocab)

    p_heaps = sub_task.add_parser("heaps", help="Fit Heaps' Law parameters")
    p_heaps.add_argument("--config", required=True, help="Path to task_heaps YAML config")
    p_heaps.set_defaults(func=cmd_task_heaps)

    p_heaps_plot = sub_task.add_parser("heaps-plot", help="Plot Heaps' Law log-log figure")
    p_heaps_plot.add_argument("--config", required=True, help="Path to task_heaps_plot YAML config")
    p_heaps_plot.set_defaults(func=cmd_task_heaps_plot)

    p_bpe_train = sub_task.add_parser("bpe-train", help="Train BPE merges and save merge file")
    p_bpe_train.add_argument("--config", required=True, help="Path to task_bpe_train YAML config")
    p_bpe_train.set_defaults(func=cmd_task_bpe_train)

    p_bpe_apply = sub_task.add_parser("bpe-apply-examples", help="Apply BPE to sample words")
    p_bpe_apply.add_argument("--config", required=True, help="Path to task_bpe_apply_examples YAML config")
    p_bpe_apply.set_defaults(func=cmd_task_bpe_apply_examples)

    p_sentseg = sub_task.add_parser("sentseg", help="Sentence segmentation stats + examples")
    p_sentseg.add_argument("--config", required=True, help="Path to task_sentseg YAML config")
    p_sentseg.set_defaults(func=cmd_task_sentseg)

    p_sentseg_eval = sub_task.add_parser("sentseg-eval", help="Evaluate sentence segmentation using GOLD file")
    p_sentseg_eval.add_argument("--config", required=True, help="Path to task_sentseg_eval YAML config")
    p_sentseg_eval.set_defaults(func=cmd_task_sentseg_eval)

    p_spell_demo = sub_task.add_parser("spell-lev-demo", help="Spellcheck demo (artificial typos)")
    p_spell_demo.add_argument("--config", required=True, help="Path to task_spell_lev_demo YAML config")
    p_spell_demo.set_defaults(func=cmd_task_spell_lev_demo)

    p_spell_eval = sub_task.add_parser("spell-lev-eval", help="Evaluate Levenshtein spellchecker (acc@1/acc@5)")
    p_spell_eval.add_argument("--config", required=True, help="Path to task_spell_lev_eval YAML config")
    p_spell_eval.set_defaults(func=cmd_task_spell_lev_eval)


    p_conf = sub_task.add_parser("confusion-synth", help="Build synthetic confusion matrix (substitutions)")
    p_conf.add_argument("--config", required=True, help="Path to task_confusion_synth YAML config")
    p_conf.set_defaults(func=cmd_task_confusion_synth)

    p_wdemo = sub_task.add_parser("spell-weighted-demo", help="Weighted spellcheck demo")
    p_wdemo.add_argument("--config", required=True, help="Path to task_spell_weighted_demo YAML config")
    p_wdemo.set_defaults(func=cmd_task_spell_weighted_demo)

    p_weval = sub_task.add_parser("spell-weighted-eval", help="Evaluate weighted vs baseline spellcheck")
    p_weval.add_argument("--config", required=True, help="Path to task_spell_weighted_eval YAML config")
    p_weval.set_defaults(func=cmd_task_spell_weighted_eval)

    p_ctop = sub_task.add_parser("confusion-top", help="Generate Top-N confusion table (md/csv/tsv)")
    p_ctop.add_argument("--config", required=True, help="Path to task_confusion_top YAML config")
    p_ctop.set_defaults(func=cmd_task_confusion_top)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
