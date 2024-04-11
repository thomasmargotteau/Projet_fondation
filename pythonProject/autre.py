import cv2

aruco_image = cv2.imread("ArucoBoard.png")
cv2.imshow("img", aruco_image)
cv2.waitKey(0)
Aruco_Dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
Aruco_Params = cv2.aruco.DetectorParameters_create()
(marker_corners, marker_id, rejected_markers) = cv2.aruco.detectMarkers(
    aruco_image, Aruco_Dict, parameters=Aruco_Params
)

if len(marker_corners) > 0:
    marker_id = marker_id.flatten()

    for (markerCorner, markerID) in zip(marker_corners, marker_id):
        marker_corners = markerCorner.reshape((4, 2))
        (top_Left, top_Right, bottom_Right, bottom_Left) = marker_corners
        top_Right = (int(top_Right[0]), int(top_Right[1]))
        bottom_Right = (int(bottom_Right[0]), int(bottom_Right[1]))
        bottom_Left = (int(bottom_Left[0]), int(bottom_Left[1]))
        top_Left = (int(top_Left[0]), int(top_Left[1]))

        cv2.line(aruco_image, top_Left, top_Right, (255, 0, 0), 2)
        cv2.line(aruco_image, top_Right, bottom_Right, (255, 0, 0), 2)
        cv2.line(aruco_image, bottom_Right, bottom_Left, (255, 0, 0), 2)
        cv2.line(aruco_image, bottom_Left, top_Left, (255, 0, 0), 2)
        cX = int((top_Left[0] + bottom_Right[0]) / 2.0)
        cY = int((top_Left[1] + bottom_Right[1]) / 2.0)
        cv2.circle(aruco_image, (cX, cY), 4, (0, 255, 0), -1)
        cv2.putText(
            aruco_image,
            str(markerID),
            (top_Left[0], top_Left[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            2,
        )
        print("Aruco Marker ID: {}".format(markerID))
        cv2.imshow("Image", aruco_image)
        cv2.waitKey(0)
cv2.destroyAllWindows()