import numpy as np
import cv2 as cv
import pywavefront

# The given video and calibration data
video_file = 'Chess.MOV'
K = np.array([[1731.93038, 0, 618.500789],
              [0, 1723.00973, 996.528529],
              [0, 0, 1]])
dist_coeff = np.array([0.14031321, -0.05553018, 0.01209608, 0.0187642, -0.15186641])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

#.obj file 불러오기
license = "pac man by Ricardo Marroquin [CC-BY] via Poly Pizza"
scene = pywavefront.Wavefront('model.obj', collect_faces=True)
vertices = np.array(scene.vertices)
# 정규화, 크기 조절
vertices -= np.mean(vertices, axis=0)
vertices /= np.max(np.linalg.norm(vertices, axis=1))
vertices *= 0.05

#모델 회전
theta = -np.pi / 2
rotation = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta),  np.cos(theta)]])
vertices = vertices @ rotation.T

#모델 투영 위치 설정 
target_position = np.array([4.5 * board_cellsize, 3.5 * board_cellsize, 0.0])
vertices_translated = vertices + target_position

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

fourcc = cv.VideoWriter_fourcc(*'avc1')  #codec specifying
frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv.CAP_PROP_FPS))
output = cv.VideoWriter('myAR.mp4', fourcc, fps, (frame_width , frame_height))

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # 모델 투영
        projected_points, _ = cv.projectPoints(vertices_translated, rvec, tvec, K, dist_coeff)
        projected_points = projected_points.reshape(-1, 2).astype(int)

        # 모델의 각 face 그리기
        for mesh in scene.mesh_list:
            for face in mesh.faces:
                pts = projected_points[face]
                cv.fillPoly(img, [pts], color=(51, 255, 255))
                cv.polylines(img, [pts], isClosed=True, color=(50, 50, 50), thickness=1)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) 
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        cv.putText(img, license, (10, 45), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    output.write(img)
    key = cv.waitKey(3)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()