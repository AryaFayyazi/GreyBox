#!/bin/bash
# After web/data/observable_ref.json exists: rebuild the site locally, record the narrated explainer (live Claude session;
# up to 3 attempts until a run shows a held call; the kept attempt is logged), compose, GIF, stills. Nothing is published.
set -uo pipefail
cd /home/arya/graybox
M=/data/models/graybox_ws/media; E=$M/explainer; PW=/data/models/graybox_ws/venv_pw/bin/python
export PLAYWRIGHT_BROWSERS_PATH=/data/models/graybox_ws/pw-browsers
/data/models/cape_ws/venv/bin/python site/build.py > /dev/null 2>&1 || { echo "BUILD FAILED"; exit 1; }
(cd site/dist && python3 -m http.server 8977 --bind 127.0.0.1 >/dev/null 2>&1 &) ; sleep 2
for A in 1 2 3; do
  rm -rf $E/raw $E/screen.webm
  $PW web/record_explainer.py http://127.0.0.1:8977/selfpolice.html $E none web/explainer_script.json > $E/record_attempt$A.log 2>&1
  H=$(python3 -c "import json;T={s['id']:s for s in json.load(open('$E/timeline.json'))};print(T['session_end'].get('flagged_steps',0))" 2>/dev/null || echo 0)
  echo "attempt $A flagged_steps=$H" | tee -a $E/attempts.log
  [ "$H" -gt 0 ] && break
done
/data/models/graybox_ws/venv_tts/bin/python web/compose_explainer.py $E web/explainer_script.json
F=$($PW -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"); L=$(python3 -c "import json;d=json.load(open('$E/trim.json'));print(d['lead'], d['end'])"); set -- $L
$F -hide_banner -loglevel error -y -ss $1 -t $2 -i $E/screen.webm -i $E/narration.wav -i $E/explainer.srt -map 0:v -map 1:a -map 2:s -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p -r 30 -c:a aac -b:a 128k -c:s mov_text -metadata:s:s:0 language=eng -movflags +faststart -shortest $E/greybox_selfpolicing_explainer.mp4
LEAD=$1; mkdir -p $M/stills; rm -f $M/stills/*.png
python3 - <<PY > $M/st.txt
import json; T=json.load(open("$E/timeline.json")); lead=$LEAD
pick={"intro":0.5,"results":0.6,"agent3d":0.85,"geometry3d":0.8,"session_watch":0.9,"session_warn":0.7,"session_end":0.6,"replay":0.75,"limit":0.7,"law":0.6}
for i,s in enumerate(T):
    if s["id"] == "session_warn" and "released_at" in s: print(f'{i+1:02d}_{s["id"]} {s["released_at"]-lead-2.5:.2f}'); continue
    if s["id"] in pick: print(f'{i+1:02d}_{s["id"]} {s["start"]-lead+pick[s["id"]]*(s["end"]-s["start"]):.2f}')
PY
while read n t; do $F -hide_banner -loglevel error -y -ss $t -i $E/greybox_selfpolicing_explainer.mp4 -frames:v 1 $M/stills/$n.png; done < $M/st.txt
A=$(python3 -c "import json;T={s['id']:s for s in json.load(open('$E/timeline.json'))};print(T['session_watch']['start']-$LEAD+2)"); B=$(python3 -c "import json;T={s['id']:s for s in json.load(open('$E/timeline.json'))};w=T['session_warn'];print(w.get('released_at', w['start']+8)-$LEAD-6)")
$F -hide_banner -loglevel error -y -ss $A -t 8 -i $E/greybox_selfpolicing_explainer.mp4 -ss $B -t 11 -i $E/greybox_selfpolicing_explainer.mp4 -filter_complex "[0:v][1:v]concat=n=2:v=1[c];[c]fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=sierra2_4a" $M/greybox_selfpolicing.gif
echo "MEDIA DONE"; ls -la $E/greybox_selfpolicing_explainer.mp4 $M/greybox_selfpolicing.gif
