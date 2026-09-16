# SWE-agent on SWE-bench Lite

This directory contains the 300 trajectories from the original SWE-agent +
GPT-4 (1106) submission on SWE-bench Lite:

- Paper: <https://arxiv.org/abs/2405.15793>
- Submission: <https://github.com/swe-bench/experiments/tree/main/evaluation/lite/20240402_sweagent_gpt4>
- Data: `s3://swe-bench-submissions/lite/20240402_sweagent_gpt4/trajs`

The downloaded `.traj` files live under `trajs/` and are intentionally ignored
by Git. Each file contains the full system/user/assistant history of one agent
run.
