import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceIdentification:
    

    def load_model(self):
        
        self.app = FaceAnalysis(name = "buffalo_s", providers = ["CPUExecutionProvider"])
        self.app.prepare(ctx_id = 0, det_size = (640, 640))
     

    def online_identification(self):
        THRESHOLD = 25
        flag = False

        face_bank = np.load("../Face_Identification/face_bank.npy", allow_pickle=True)

        cap=cv2.VideoCapture(0)
        while (True):
            
            ret, frame = cap.read()
            result = self.app.get(frame)

            cv2.rectangle(frame, (int(result[0].bbox[0]), 
                                        int(result[0].bbox[1])), 
                                        (int(result[0].bbox[2]), 
                                        int(result[0].bbox[3])),(0, 0, 255),4)
            

            for person in face_bank:
                face_bank_person_embedding = person["embedding"]
                new_person_embedding = result[0]["embedding"]

                distance = np.sqrt(np.sum((face_bank_person_embedding - new_person_embedding)**2))

                if distance < THRESHOLD:
                    cv2.putText(frame, person["name"],
                    (int(result.bbox[0]) - 40, int(result.bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0),3, cv2.LINE_AA)
                    flag = True
                    break

            else:
                cv2.putText(frame, "Unknown",
                (int(result[0].bbox[0]) - 40, int(result[0].bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3,cv2.LINE_AA)
                flag = False

            cv2.imshow('frame', frame)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        cap.release()
        cv2.destroyAllWindows()

        if flag : 
            return True
        else:
            return False    

