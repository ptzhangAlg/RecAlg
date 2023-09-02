#!/usr/bin/env bash

nohup python -u -m rec_alg.preprocessing.avazu.avazu_process > process_avazu.log 2>&1 &
