# Incident (2026-10-22, 22:59): analysis waiters fired before their data existed

Waiters for the locked analyses of G27b, G30, G31b and G31c polled `squeue -j 4268` (the Muse-Glimmer battery) and ran
the analysis when the job left the queue. At 22:59 the job left the queue because it FAILED at startup (vLLM port 8199
was held by a stale local mock server of this project, started 2026-10-15), not because its data existed. Effects:
- G31c's locked analysis ran on gpt-oss-20b only (Qwen3.8's and Muse-Glimmer's batteries did not exist) and wrote a
  result file, whose output was seen. The file is kept, renamed
  results/t1/g31c_memory_PARTIAL_gptoss_only_EARLY_TRIGGER.json. It is not G31c's result: G31c's pooled endpoints are
  defined over the three agents and are computed only when all three batteries exist. Nothing in G31c's code or
  endpoints changes.
- G27b's, G31b's and G30's analyses started and were stopped before printing or writing any outcome (their outputs hold
  only file listings and code hashes).
Fix: the stale server was stopped; the Muse-Glimmer battery is resubmitted; all waiters now wait for the data files to
exist instead of polling job ids.
