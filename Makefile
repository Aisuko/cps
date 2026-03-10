SHELL := /bin/bash

ROOT      := $(shell pwd)
export PYTHONPATH := $(ROOT)/src
PYTHON    := $(ROOT)/.venv/bin/python
RUN       := $(PYTHON) -m conflict.run

MODEL     ?= mistralai/Mistral-7B-v0.1
DATASET   ?= NQ-Swap
SEED      ?= 0
SEEDS     ?= 0 1 2
DEVICE    ?=
NUM_MASKS ?= 10
BATCH_SIZE ?= 32
DELTA     ?= 0.3

MODELS   := mistralai/Mistral-7B-v0.1 google/gemma-2-9b meta-llama/Meta-Llama-3.1-8B
DATASETS := NQ-Swap ConflictQA-popQA-llama2-7b ConflictBank PopQA-Swap

BASE = $(RUN) experiment --dataset $(DATASET) --model $(MODEL) --seed $(SEED) \
       --num-masks $(NUM_MASKS) --batch-size $(BATCH_SIZE) $(if $(DEVICE),--device $(DEVICE),)

.PHONY: prepare split main ablation hyperparameter soft delta experiment all help

prepare:        ; $(RUN) prepare --dataset $(DATASET)
split:          ; $(RUN) split   --dataset $(DATASET) --seed $(SEED)

main:           ; $(BASE) --stages main
ablation:       ; $(BASE) --stages ablation
hyperparameter: ; $(BASE) --stages hyperparameter
soft:           ; $(BASE) --stages soft --ablation-delta $(DELTA)
delta:          ; $(BASE) --stages delta
experiment:     ; $(BASE) --stages main ablation hyperparameter

all:
	@set -e; \
	for s in $(SEEDS); do \
		for d in $(DATASETS); do \
			for m in $(MODELS); do \
				echo "=== $$d x $$m (seed $$s) ==="; \
				$(MAKE) --no-print-directory experiment DATASET="$$d" MODEL="$$m" SEED=$$s; \
			done; \
		done; \
	done

help:
	@echo "make <target> [DATASET=$(DATASET)] [MODEL=$(MODEL)] [SEED=$(SEED)] [SEEDS='$(SEEDS)'] [DEVICE=cuda:0]"
	@echo "DATASET : $(DATASETS)"
	@echo "MODEL   : $(MODELS)"
	@echo "targets : prepare split"
	@echo "          main ablation hyperparameter soft delta"
	@echo "          experiment all"
