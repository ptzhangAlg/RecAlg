#!/usr/bin/env bash

nohup python -u -m rec_alg.preprocessing.criteo.criteo_process > process_criteo.log 2>&1 &
