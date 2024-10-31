import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import time
from collections import deque
import warnings
warnings.simplefilter(action='ignore', category=Warning)

#defining the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
model.load_state_dict(torch.load('frisbee_model.pth'))
model.to(device)
model.eval()

#decorator to measure inference time
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f'\nInference took {(end-start):.2f} seconds')
    return wrapper


def draw_boxes(image, boxes,thickness, color=(0,255,0)):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

@time_it
def detect_image(image_path, output_path,thickness=3,threshold=0.5):
    print('[INFO] Detecting Image...')


    #load and preprocess the image
    image = Image.open(image_path)
    transform = T.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

    #run the model
    with torch.no_grad():
        prediction = model(image_tensor)

    #extract the bounding boxes
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    selected_boxes = boxes[scores > threshold]

    #convert the image to a numpy array and draw the boxes
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result_image = draw_boxes(image_np, selected_boxes,thickness)
    cv2.imwrite(output_path, result_image)

    print('[INFO] Image Detection Complete!')


@time_it
def detect_video(video_path, output_path,thickness=3, threshold=0.5):
    print('[INFO] Detecting objects in video...')
    tracking_points = deque(maxlen=50)

    # open the video file and get the video properties
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # define codec and create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #convert the frame to a tensor and run the model
        image_tensor = T.ToTensor()(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(image_tensor)

        # #extract the bounding boxes
        # boxes = prediction[0]['boxes'].cpu().numpy()
        # scores = prediction[0]['scores'].cpu().numpy()
        # selected_boxes = boxes[scores > threshold]
        #
        # result_frame = draw_boxes(frame, selected_boxes,thickness)
        # out.write(result_frame)



        #draw the bounding boxes and trace path
        for i in range(len(prediction[0]['boxes'])):
            score = prediction[0]['scores'][i].cpu().numpy()
            if score > threshold:
                x1, y1, x2, y2 = map(int, prediction[0]['boxes'][i].cpu().tolist())

                #draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), thickness)

                #calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                tracking_points.append((center_x, center_y))

        #draw the path
        for i in range(1, len(tracking_points)):
            if tracking_points[i - 1] is None or tracking_points[i] is None:
                continue
            cv2.line(frame, tracking_points[i - 1], tracking_points[i], (0, 255, 0), thickness)

        out.write(frame)

        #display the frame
        cv2.imshow('Object Detection', cv2.resize(frame, (800, 600)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('[INFO] Object Detection Complete!\n [INFO] Video Saved to:', output_path)


if __name__ == '__main__':

    test_path = 'test2.mp4'
    detect_video(test_path, 'output.avi', threshold=0.5)