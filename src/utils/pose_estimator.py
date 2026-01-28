"""Estimate head pose according to the facial landmarks"""
import numpy as np
import math
import cv2

class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, image_width, image_height):
        """Init a pose estimator.
        Args:
            image_width (int): input image width
            image_height (int): input image height
        """
        self.size = (image_height, image_width)
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)

        self.model_points_5 = np.array([
            [-32.0, -35.0, 30.0],  # left eye
            [ 32.0, -35.0, 30.0],  # right eye
            [  0.0,   0.0, 20.0],  # nose tip
            [-25.0,  35.0, 25.0],  # left mouth
            [ 25.0,  35.0, 25.0],  # right mouth
        ], dtype=np.float32)

        self.X = np.hstack((self.model_points_5, np.ones([self.model_points_5.shape[0],1]))) #n x 4

    #Get Face Bounding Box Coordinates
    def get_coordinates(self,bbox):

        """
            bbox: It gives xmin,ymin,xmax,ymax
        """
        x1, y1, x2, y2 = bbox.astype(int)
        return [max(0, x1), max(0, y1),min(self.size[1], x2), min(self.size[0], y2)]

     
    def estimate_affine_matrix_3d23d(self,Y):
        ''' Using least-squares solution 
        Args:
            Y: [n, 3]. corresponding 3d points(moving). Y = PX
        Returns:
            P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
        '''
        P = np.linalg.lstsq(self.X, Y)[0].T # Affine matrix. 3 x 4
        return P

    def P2sRt(self,P):
        ''' decompositing camera matrix P
        Args: 
            P: (3, 4). Affine Camera Matrix.
        Returns:
            s: scale factor.
            R: (3, 3). rotation matrix.
            t: (3,). translation. 
        '''
        R1 = P[0:1, :3]
        R2 = P[1:2, :3]
        
        r1 = R1/np.linalg.norm(R1)
        r2 = R2/np.linalg.norm(R2)
        r3 = np.cross(r1, r2)

        R = np.concatenate((r1, r2, r3), 0)
        return R

    def matrix2angle(self,R):
        ''' get three Euler angles from Rotation Matrix
        Args:
            R: (3,3). rotation matrix
        Returns:
            x: pitch
            y: yaw
            z: roll
        '''
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6
    
        if  not singular :
            y = math.atan2(-R[2,0], sy)
        else :
            y = math.atan2(-R[2,0], sy)

        return np.rad2deg(y)
    

    def get_directions(self,yaw, face_angle):

        offset = round(math.cos(math.radians(face_angle))*30)

        RIGHT_YAW_THRESHOLD  = -50 - offset
        LEFT_YAW_THRESHOLD =  50 - offset

        return "Idle" if yaw > LEFT_YAW_THRESHOLD or yaw < RIGHT_YAW_THRESHOLD else "Working"

    def visualize(self,image,box,status):

        xmin,ymin,xmax,ymax=box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2,cv2.LINE_AA)
        cv2.putText(image, f"{status}",(box[0], box[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2,cv2.LINE_AA)