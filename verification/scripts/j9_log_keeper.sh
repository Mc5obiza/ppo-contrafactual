#!/bin/bash
# Copies each J9 task's log every 60 s to logs/J9_preserved/<log>.restart<N>,
# N = Slurm restart count, so a requeue (which truncates the .out) loses at most ~60 s.
cd ~/verification
while :; do
  live=0
  for t in 0 1; do
    info=$(scontrol show job 18497970_$t 2>/dev/null) || continue
    st=$(grep -oP 'JobState=\K\S+' <<<"$info"); n=$(grep -oP 'Restarts=\K\d+' <<<"$info")
    case $st in RUNNING|PENDING|REQUEUED|CONFIGURING) live=1;; esac
    for ext in out err; do
      f=logs/verif-retrain-18497970_$t.$ext
      [ -s "$f" ] && [ "$st" = RUNNING ] && cp -p "$f" "logs/J9_preserved/$(basename $f).restart$n"
    done
  done
  [ $live = 1 ] || break
  sleep 60
done
