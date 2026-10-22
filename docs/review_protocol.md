# Fixed review protocol (so that scores are comparable across versions)

Single reviewer scores are noisy: two reviewers with different prompts gave 6/10 and 4/10 to versions that differed
little. From 2026-10-21 every score we quote comes from this protocol, unchanged:

- 3 independent reviewers (fresh agents, no shared context, read-only), each given the prompt below verbatim.
- Each reads the compiled PDF text and the .tex sources, and checks every headline number against results/t1/*.json
  and docs/PREREGISTRATION_RECORD.md.
- We report the mean and the range of the 3 scores, and every issue that 2 or more reviewers raise.

## Prompt (verbatim)
You are an ICML reviewer. Review the paper in /home/arya/graybox/paper (read the PDF text with
`pdftotext /home/arya/graybox/paper/main.pdf -` and the .tex sources). Check headline numbers against
/home/arya/graybox/results/t1/*.json and /home/arya/graybox/docs/PREREGISTRATION_RECORD.md. Do not modify any file.
Give: (1) an overall score on the ICML 1-10 scale with one sentence of justification; (2) soundness, presentation,
significance and originality, each 1-4; (3) the five most important weaknesses, each with file/section and a quote;
(4) any number in the abstract, introduction, figure captions or conclusion that the results files contradict;
(5) the three strongest points. Under 900 words.
