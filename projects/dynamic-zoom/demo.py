"""Deterministic frame-queue simulation; simulated work, no GPU or video weights."""
from collections import deque
from statistics import mean
import json


def simulate(policy, frames=60, fps=30, work_ms=60):
    if policy not in {'fifo','latest'} or frames < 1 or fps <= 0 or work_ms <= 0:
        raise ValueError('Require fifo/latest, positive frames, frame rate, and work time')
    arrivals = [i*1000/fps for i in range(frames)]
    pending, completed, dropped = deque(), [], []
    cursor, clock, peak_queue = 0, 0., 0
    while cursor < frames or pending:
        if not pending and cursor < frames:clock=max(clock,arrivals[cursor])
        while cursor < frames and arrivals[cursor] <= clock+1e-9:
            if policy == 'latest' and pending:
                dropped.append(pending.pop())
            pending.append(cursor)
            peak_queue=max(peak_queue,len(pending))
            cursor+=1
        frame=pending.popleft()
        started=clock
        clock+=work_ms
        completed.append({'frame_id':frame,'captured_ms':round(arrivals[frame],3),
                          'started_ms':round(started,3),'displayed_ms':round(clock,3),
                          'latency_ms':round(clock-arrivals[frame],3)})
    latencies=sorted(r['latency_ms'] for r in completed)
    return {'policy':policy,'input_frames':frames,'processed_frames':len(completed),
            'dropped_frames':len(dropped),'peak_waiting_frames':peak_queue,
            'mean_latency_ms':round(mean(latencies),3),
            'p95_latency_ms':latencies[max(0,int(.95*len(latencies)+.999999)-1)],
            'max_latency_ms':max(latencies),'frames':completed,'dropped_ids':dropped}


def run():
    return {'evidence':'Simulation: 60 arrivals at 30 fps, fixed 60 ms service; not a hardware benchmark',
            'fifo':simulate('fifo'),'latest':simulate('latest')}


if __name__ == '__main__':print(json.dumps(run(),indent=2))
