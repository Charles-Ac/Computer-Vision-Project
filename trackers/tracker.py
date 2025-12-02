from ultralytics import YOLO
import supervision as sv
import pickle
import os
from utils import get_center_of_box, get_bbox_width,get_foot_position
import cv2
import numpy as np
import pandas as pd
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        # One tracker per class (parameters are defaults in your version)
        self.player_tracker = sv.ByteTrack()
        self.ref_tracker    = sv.ByteTrack()
        self.ball_tracker   = sv.ByteTrack()
        # in trackers/tracker.py

    # trackers/tracker.py
    def add_position_to_tracks(self, tracks):
        for obj, frames_dict in tracks.items():                 
            for frame_num, ids_dict in frames_dict.items():     
                for track_id, track_info in ids_dict.items():
                    bbox = track_info.get('bbox')
                    if not bbox:
                        continue
                    if obj == 'ball':
                        position = get_center_of_box(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[obj][frame_num][track_id]['position'] = position

                    

    def fill_with_velocity_fallback(self, ball_by_frame: dict, max_predict=10, decay=0.85, max_speed=40):
        """
        Accepts either:
        {frame: {'bbox':[...]}}  OR  {frame: {id: {'bbox':[...]}}}
        Fills short gaps by extrapolating with decayed velocity.
        Returns the SAME (nested) shape as the input.
        """
        import math

        # ---- normalize to simple {frame: {'bbox':[...]}} ----
        simple = {}
        for f, v in ball_by_frame.items():
            if v is None:
                continue
            if isinstance(v, dict) and 'bbox' in v:          # flat
                simple[f] = {'bbox': v['bbox']}
            elif isinstance(v, dict) and len(v) > 0:          # nested {id: {...}}
                first = next(iter(v.values()))
                if isinstance(first, dict) and 'bbox' in first:
                    simple[f] = {'bbox': first['bbox']}

        def center(b):
            x1,y1,x2,y2 = b
            return 0.5*(x1+x2), 0.5*(y1+y2)

        frames = sorted(simple.keys())
        if len(frames) < 2:
            return ball_by_frame  # nothing to do

        i = 1
        while i < len(frames):
            f_prev, f_curr = frames[i-1], frames[i]
            gap = f_curr - f_prev - 1
            if gap > 0:
                # velocity from two frames before the gap if possible
                if i-2 >= 0:
                    f_prev2 = frames[i-2]
                    c2 = center(simple[f_prev2]['bbox'])
                    c1 = center(simple[f_prev]['bbox'])
                    vx, vy = (c1[0]-c2[0], c1[1]-c2[1])
                else:
                    vx, vy = (0.0, 0.0)

                # clamp speed
                speed = math.hypot(vx, vy)
                if speed > max_speed and speed > 0:
                    s = max_speed / speed
                    vx *= s; vy *= s

                # predict up to max_predict
                steps = min(gap, max_predict)
                x, y = center(simple[f_prev]['bbox'])
                x1,y1,x2,y2 = simple[f_prev]['bbox']
                w = x2 - x1; h = y2 - y1

                for k in range(1, steps+1):
                    x += vx; y += vy
                    vx *= decay; vy *= decay
                    nx1 = x - 0.5*w; ny1 = y - 0.5*h
                    nx2 = x + 0.5*w; ny2 = y + 0.5*h
                    simple[f_prev + k] = {'bbox': [float(nx1), float(ny1), float(nx2), float(ny2)]}

                # refresh frame order after inserts
                frames = sorted(simple.keys())
                i = frames.index(f_curr) + 1
                continue
            i += 1

        # ---- write predictions back in the original (nested) shape ----
        out = dict(ball_by_frame)  # shallow copy
        for f, entry in simple.items():
            bbox = entry['bbox']
            # If original was nested, keep nested
            if f in ball_by_frame and isinstance(ball_by_frame[f], dict) and 'bbox' not in ball_by_frame[f]:
                # choose id=1 (or existing sole id)
                if len(ball_by_frame[f]) == 1:
                    only_id = next(iter(ball_by_frame[f].keys()))
                    out.setdefault(f, {})[only_id] = {'bbox': bbox}
                else:
                    out.setdefault(f, {})[1] = {'bbox': bbox}
            else:
                out[f] = {'bbox': bbox}
        return out

    def _normalize_ball_tracks(self, ball: dict) -> dict:
        """
        Accepts any of:
        {f: {'bbox':[...]} }
        {f: [x1,y1,x2,y2]}
        {f: {track_id: {'bbox':[...]} } }
        {f: {track_id: [x1,y1,x2,y2]} }
        Returns canonical: {f: {1: {'bbox':[x1,y1,x2,y2]}}}
        """
        out = {}
        for f, val in (ball or {}).items():
            # Case A: {f: {'bbox':[...]} }
            if isinstance(val, dict) and 'bbox' in val:
                out[f] = {1: {'bbox': list(map(float, val['bbox']))}}
                continue

            # Case B: {f: [x1,y1,x2,y2]}
            if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 4:
                out[f] = {1: {'bbox': [float(x) for x in val]}}
                continue

            # Case C: {f: {track_id: {...}}}
            if isinstance(val, dict):
                chosen = None
                for v in val.values():
                    if isinstance(v, dict) and 'bbox' in v:
                        chosen = v['bbox']; break
                    if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 4:
                        chosen = v; break
                if chosen is not None:
                    out[f] = {1: {'bbox': [float(x) for x in chosen]}}
        return out


    def interpolate_ball_positions(self, ball: dict, num_frames: int, max_gap: int = 10) -> dict:
        """
        Linearly fills gaps up to `max_gap` between known frames.
        Input may be in various shapes; it will be normalized first.
        Returns canonical: {frame: {1: {'bbox':[x1,y1,x2,y2]}}}
        """
        # 1) normalize
        simple = self._normalize_ball_tracks(ball)

        if not simple:
            return {}

        # 2) collect known frames in order
        known = sorted(simple.keys())

        # 3) start output with known boxes
        out = {f: {1: {'bbox': simple[f][1]['bbox']}} for f in known}

        # 4) interpolate between pairs
        for i in range(len(known) - 1):
            f0, f1 = known[i], known[i + 1]
            gap = f1 - f0 - 1
            if gap <= 0 or gap > max_gap:
                continue

            b0 = simple[f0][1]['bbox']
            b1 = simple[f1][1]['bbox']

            for t in range(1, gap + 1):
                a = t / (gap + 1.0)
                interp = [b0[j] + (b1[j] - b0[j]) * a for j in range(4)]
                out[f0 + t] = {1: {'bbox': [float(x) for x in interp]}}

        # 5) keep within video range if num_frames is provided
        if isinstance(num_frames, int) and num_frames > 0:
            out = {f: v for f, v in out.items() if 0 <= f < num_frames}

        return out

    



    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # Use predict() for raw detection without internal tracking
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
    
    # tracker.py (inside Tracker)
    def _resolve_class_ids(self, detection):
        # normalize names: lower + strip
        names = {int(k): str(v).strip().lower() for k, v in detection.names.items()}

        # direct lookups
        pid = next((k for k,v in names.items() if v == 'player'), None)
        rid = next((k for k,v in names.items() if v == 'referee'), None)

        # "ball" can be called many things; accept any name containing 'ball'
        ball_candidates = [k for k,v in names.items()
                        if v == 'ball' or 'ball' in v or v in ('soccer-ball','football')]
        bid = ball_candidates[0] if ball_candidates else None

        return pid, rid, bid, names


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"players": {}, "ball": {}, "referees": {}}

        for frame_num, detection in enumerate(detections):
            pid, rid, bid, names = self._resolve_class_ids(detection)

            det_sup = sv.Detections.from_ultralytics(detection)

            players_sup   = det_sup[det_sup.class_id == pid] if pid is not None else sv.Detections.empty()
            referees_sup  = det_sup[det_sup.class_id == rid] if rid is not None else sv.Detections.empty()
            balls_sup     = det_sup[det_sup.class_id == bid] if bid is not None else sv.Detections.empty()

            # --- Track players / referees (defaults are fine)
            players_trk  = self.player_tracker.update_with_detections(players_sup)
            referees_trk = self.ref_tracker.update_with_detections(referees_sup)

            if len(players_trk) > 0:
                tracks['players'].setdefault(frame_num, {})
                for t in players_trk:
                    bbox = t[0].tolist()
                    track_id = int(t[4])
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}

            if len(referees_trk) > 0:
                tracks['referees'].setdefault(frame_num, {})
                for t in referees_trk:
                    bbox = t[0].tolist()
                    track_id = int(t[4])
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            # --- Ball: try ByteTrack first; if it fails, fall back to top-1 detection
            ball_written = False
            if bid is not None:
                balls_trk = self.ball_tracker.update_with_detections(balls_sup)

                if len(balls_trk) > 0:
                    tracks['ball'].setdefault(frame_num, {})
                    for t in balls_trk:
                        bbox = t[0].tolist()
                        track_id = int(t[4])
                        tracks['ball'][frame_num][track_id] = {'bbox': bbox}
                        ball_written = True

                # Fallback: keep best YOLO ball even if ByteTrack dropped it
                if not ball_written and len(balls_sup) > 0:
                    tracks['ball'].setdefault(frame_num, {})
                    # pick the highest-confidence ball box
                    import numpy as np
                    try:
                        j = int(np.argmax(balls_sup.confidence))
                        bbox = balls_sup.xyxy[j].tolist()
                    except Exception:
                        # in case confidence tensor is missing, just take first
                        bbox = balls_sup.xyxy[0].tolist()
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks


    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_box(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame,
                    center=(x_center, y2),
                    axes=(int(width), int(0.45 * width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=245,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA)

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (0, 0, 0),
                        2)

        return frame

    def draw_triangle(self, frame, bbox, color, track_id=None):
        y = int(bbox[1])
        x, _ = get_center_of_box(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame
    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-trasnparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,244,244),-1)
        alpha = 0.4
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of times each team had the ball
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame,f"Team 1 Ball control :{ team_1*100:.2f}% ",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Team 2 Ball control :{ team_2*100:.2f}% ",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame


    def draw_annotations(self, video_frames, tracks,team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'].get(frame_num, {})
            referee_dict = tracks['referees'].get(frame_num, {})
            ball_dict = tracks['ball'].get(frame_num, {})

            for track_id, player in player_dict.items():
                color=player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame,player['bbox'],(0,0,255))

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255), track_id)

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0), track_id)

            #draw team ball control
            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
