#!/usr/bin/env bash

nohup python -u -m rec_alg.preprocessing.kdd12.kdd12_process > process_kdd12.log 2>&1 &
