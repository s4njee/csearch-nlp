#!/bin/sh
set -eu

INPUT_DIR="${EMBEDDED_CHUNKS_DIR:-/home/sanjee/nlp/embedded_chunks}"

has_input_arg=0
for arg in "$@"; do
    case "$arg" in
        --input|-i|--input=*)
            has_input_arg=1
            break
            ;;
    esac
done

if [ "$has_input_arg" -eq 0 ]; then
    set -- --input "$INPUT_DIR" "$@"
fi

exec python /app/upserter.py "$@"
