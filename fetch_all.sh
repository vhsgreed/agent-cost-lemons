#!/bin/bash
# Stream all 9 full Bai et al. archives through extract_bai.py, 3 at a time, 3 tries each.
cd "$(dirname "$0")"
ls_=(gpt-5_4runs claude-sonnet-4.5_new_v2_4runs gpt_5.2_4runs claude-sonnet-3.7_4runs claude-sonnet-4_4runs qwen3-coder-480b-a35b-instruct-4runs gemini-3-pro-preview_4runs claude-sonnet-4.5_4runs kimi-k2_4runs)
printf '%s\n' "${ls_[@]}" | xargs -P3 -I{} bash -c 'for t in 1 2 3; do python3 extract_bai.py {} >> data/fetch_{}.log 2>&1 && exit 0; echo "retry $t" >> data/fetch_{}.log; sleep 30; done; exit 1'
echo "ALL FINISHED"; ls data/*.done | wc -l; cat data/*.done
