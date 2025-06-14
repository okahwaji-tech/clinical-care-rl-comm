#!/bin/bash

set -e

if [[ "$1" = 'clinical-care-rl-comm' ]]; then
    shift
    exec python scripts/train.py "$@"
else
    exec "$@"
fi