from utils.video_utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np 
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer import Viewtransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
def merge_ball_tracks_non_destructive(raw_ball: dict, smooth_ball: dict, num_frames: int) -> dict:
        """
        Prefer smooth when present; otherwise keep raw. Normalize keys to 0..num_frames-1.
        Output format: {frame: {1: {'bbox':[...]}}}
        """
        out = {}
        for f in range(num_frames):
            rb = raw_ball.get(f, {})
            sb = smooth_ball.get(f, {})
            if sb:
                out[f] = {1: sb.get(1, next(iter(sb.values())))}
            elif rb:
                out[f] = {1: rb.get(1, next(iter(rb.values())))}
        return out 

def main(): 
    #Read Video
    video_frames=read_video('input videos/test(15).mp4')


    
    #initialize tracker
    tracker = Tracker('models/best.pt')
    tracks= tracker.get_object_tracks(video_frames,read_from_stub=False,stub_path='stubs/track_stubs.pkl')
    # after tracks = tracker.get_object_tracks(...)
    #get object positions
    tracker.add_position_to_tracks(tracks)
    num_frames = len(video_frames)

    raw_ball = tracks['ball']  # keep a copy

    smooth_ball = tracker.interpolate_ball_positions(raw_ball, num_frames, max_gap=10)

    # Merge so you never lose good raw detections
    tracks['ball'] = merge_ball_tracks_non_destructive(raw_ball, smooth_ball, num_frames)

    # Optional: velocity fallback through gaps
    tracks['ball'] = tracker.fill_with_velocity_fallback(tracks['ball'], max_predict=10, decay=0.85, max_speed=40)

    # Sanity check (prints once)
    raw_frames    = sum(1 for f in range(num_frames) if raw_ball.get(f))
    smooth_frames = sum(1 for f in range(num_frames) if smooth_ball.get(f))
    final_frames  = sum(1 for f in range(num_frames) if tracks['ball'].get(f))
    print(f"Ball frames — raw:{raw_frames} smooth:{smooth_frames} final:{final_frames} / {num_frames}")

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True,  stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # View transformer
    view_transformer = Viewtransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    #Interpolate Ball Positions
    # in main.py
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'],num_frames=len(video_frames))

    #$peed and distance estimator
    speed_and_distance_estimator=SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    
    #Assign player Teams
    team_assigner= TeamAssigner()
    first_player_frame =min(tracks['players'])
    team_assigner.assign_team_color(video_frames[first_player_frame],tracks['players'][first_player_frame])

    for frame_num,player_track in tracks['players'].items():
        for player_id,track in player_track.items():
            team = team_assigner.get_player_team(  video_frames[frame_num],track['bbox'],player_id)
            tracks['players'][frame_num][player_id]['team']= team
            tracks['players'][frame_num][player_id]['team_color']= team_assigner.team_colors[team]
    
    # Assign ball aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control=[]
    for frame_num, player_track in tracks['players'].items():
         ball_entry = tracks['ball'].get(frame_num)
         if not ball_entry:
              continue
         ball_info = ball_entry.get(1) or next(iter (ball_entry.values()))
         ball_bbox= ball_info['bbox']

         assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)
         if assigned_player != -1:
              tracks['players'][frame_num][assigned_player]['has_ball']= True 
              team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
         else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
        
         
    #Draw output
    #Draw oject Tracks
    output_video_frames=tracker.draw_annotations(video_frames,tracks,team_ball_control)

    # Draw Camera Movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ##Draw speed and distance 
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    #Save video
    save_video( output_video_frames,'output_videos/output_test_new_video_5_.avi')
    
if __name__ == '__main__':
    main()