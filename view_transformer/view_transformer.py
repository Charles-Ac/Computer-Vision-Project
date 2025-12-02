# view_transformer.py
import numpy as np
import cv2

class Viewtransformer():
    def __init__(self) -> None:
        court_width  = 68.0
        court_length = 23.32  # if this is your calibrated length

        self.pixel_vertices = np.array([
            [110, 1035],   # BL
            [265,  275],   # TL
            [910,  260],   # TR
            [1640, 915],   # BR
        ], dtype=np.float32)

        self.target_vertices = np.array([
            [0.0,           court_width],  # BL
            [0.0,           0.0],          # TL
            [court_length,  0.0],          # TR
            [court_length,  court_width],  # BR
        ], dtype=np.float32)

        # both must be 4x2 float32 and in corresponding order
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        # point could be tuple/list/np.array
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped, self.perspective_transformer)
        return transformed.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        # iterate properly and use the correct key name 'position_adjusted'
        for obj, frames_dict in tracks.items():  # obj: 'players'|'ball'|'referees'
            for frame_num, ids_dict in frames_dict.items():
                # handle both nested and "flat ball" cases
                if obj == 'ball' and isinstance(ids_dict, dict) and 'bbox' in ids_dict:
                    items = [(1, ids_dict)]  # pretend id=1
                else:
                    items = ids_dict.items()

                for track_id, info in items:
                    pos_adj = info.get('position_adjusted')
                    if pos_adj is None:
                        continue
                    pos_adj = np.array(pos_adj, dtype=np.float32)
                    pos_tf = self.transform_point(pos_adj)
                    if pos_tf is not None:
                        pos_tf = pos_tf.squeeze().tolist()

                    # write back respecting flat vs nested
                    if obj == 'ball' and isinstance(frames_dict[frame_num], dict) and 'bbox' in frames_dict[frame_num]:
                        frames_dict[frame_num]['position_transformed'] = pos_tf
                    else:
                        frames_dict[frame_num][track_id]['position_transformed'] = pos_tf
